import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseSelector, register_selector


@register_selector("graph_diversity")
class GraphDiversitySelector(BaseSelector):
    requires_preselection_metrics = False
    """
    Graph-based diversity-aware client selector.

    Kept components:
    1) reward-based scoring
    2) discovery / primary tradeoff
    3) staleness incentive
    4) blacklist robustness
    5) graph construction
    6) diversity-aware greedy subset selection
    """

    def __init__(
        self,
        total_clients: int,
        clients_per_round: int,

        # -------- discovery --------
        discovery_rate: float = 0.9,
        discovery_decay: float = 0.95,
        discovery_min: float = 0.2,

        # -------- candidate filtering --------
        discovery_window: int = 5,
        score_clip_quantile: float = 0.98,
        score_cutoff_ratio: float = 0.95,

        # -------- robustness --------
        blacklist_count_threshold: int = -1,
        blacklist_max_ratio: float = 0.3,

        # -------- graph construction --------
        graph_update_interval: int = 1,
        graph_topk: int = 5,
        graph_sigma: float = 1.0,
        graph_min_weight: float = 1e-6,

        # -------- diversity --------
        diversity_beta: float = 0.35,
        diversity_mode: str = "max",  # max / sum

        # -------- feature weights --------
        feature_weight_reward: float = 1.0,
        feature_weight_count: float = 0.6,
        feature_weight_staleness: float = 1.0,
        feature_weight_loss: float = 1.0,
        feature_weight_processed: float = 0.7,

        # -------- utility proxy ablation --------
        reward_mode: str = "processed_loss",

        **kwargs,
    ):
        super().__init__(total_clients=total_clients, clients_per_round=clients_per_round, **kwargs)

        # selection state
        self.discovery_rate = float(discovery_rate)
        self.discovery_decay = float(discovery_decay)
        self.discovery_min = float(discovery_min)

        self.discovery_window = int(discovery_window)
        self.score_clip_quantile = float(score_clip_quantile)
        self.score_cutoff_ratio = float(score_cutoff_ratio)

        self.blacklist_count_threshold = int(blacklist_count_threshold)
        self.blacklist_max_ratio = float(blacklist_max_ratio)

        self.reward_mode = str(reward_mode).lower()
        valid_reward_modes = {"processed_loss", "loss", "loss_drop", "gradient_norm"}
        if self.reward_mode not in valid_reward_modes:
            raise ValueError(
                f"Unknown reward_mode={reward_mode!r}; expected one of {sorted(valid_reward_modes)}"
            )

        self.client_stats: Dict[int, Dict[str, Any]] = {}
        self.undiscovered_clients = set()

        for cid in range(total_clients):
            self.client_stats[cid] = {
                "reward": 0.0,
                "loss": 0.0,
                "processed": 0.0,
                "time_stamp": 0,
                "count": 0,
            }
            self.undiscovered_clients.add(cid)

        # graph params
        self.graph_update_interval = max(1, int(graph_update_interval))
        self.graph_topk = max(1, int(graph_topk))
        self.graph_sigma = float(max(1e-6, graph_sigma))
        self.graph_min_weight = float(max(0.0, graph_min_weight))

        # diversity params
        self.diversity_beta = float(max(0.0, diversity_beta))
        self.diversity_mode = str(diversity_mode).lower()
        if self.diversity_mode not in {"max", "sum"}:
            self.diversity_mode = "max"

        # graph feature weights
        self.feature_weights = np.asarray(
            [
                feature_weight_reward,
                feature_weight_count,
                feature_weight_staleness,
                feature_weight_loss,
                feature_weight_processed,
            ],
            dtype=np.float64,
        )

        # graph state
        self.client_features: Dict[int, np.ndarray] = {
            cid: np.zeros(5, dtype=np.float64) for cid in range(total_clients)
        }
        self.graph_neighbors: Dict[int, List[Tuple[int, float]]] = {
            cid: [] for cid in range(total_clients)
        }
        self.graph_edges: Dict[int, Dict[int, float]] = {
            cid: {} for cid in range(total_clients)
        }
        self.graph_last_rebuild_round: int = -1

        self._rebuild_graph(force=True)

    # =========================================================
    # utility helpers
    # =========================================================
    def _compute_norm_stats(
        self,
        values: List[float],
        clip_quantile: float = 0.95,
        min_range: float = 1e-4,
    ):
        if not values:
            return 0.0, 0.0, min_range, 0.0, 0.0

        values = list(values)
        values.sort()

        clip_idx = min(int(len(values) * clip_quantile), len(values) - 1)
        clip_value = values[clip_idx]

        max_value = max(values)
        min_value = min(values) * 0.999
        value_range = max(max_value - min_value, min_range)
        avg_value = sum(values) / max(float(len(values)), 1e-4)

        return (
            float(max_value),
            float(min_value),
            float(value_range),
            float(avg_value),
            float(clip_value),
        )

    def _build_blacklist(self) -> set:
        if self.blacklist_count_threshold == -1:
            return set()

        sorted_client_ids = sorted(
            list(self.client_stats.keys()),
            reverse=True,
            key=lambda cid: self.client_stats[cid]["count"],
        )

        blacklist = []
        for cid in sorted_client_ids:
            if self.client_stats[cid]["count"] > self.blacklist_count_threshold:
                blacklist.append(cid)
            else:
                break

        max_blacklist_len = int(self.blacklist_max_ratio * len(self.client_stats)) if self.client_stats else 0
        if max_blacklist_len >= 0 and len(blacklist) > max_blacklist_len:
            blacklist = blacklist[:max_blacklist_len]

        return set(blacklist)

    def _safe_choice(
        self,
        candidates: List[int],
        k: int,
        probs: Optional[List[float]] = None,
    ) -> List[int]:
        if k <= 0 or not candidates:
            return []

        k = min(k, len(candidates))

        if probs is None:
            return list(np.random.choice(candidates, k, replace=False))

        prob_arr = np.asarray(probs, dtype=np.float64)
        prob_arr = np.nan_to_num(prob_arr, nan=0.0, posinf=0.0, neginf=0.0)
        total_prob = prob_arr.sum()

        if total_prob <= 0:
            return list(np.random.choice(candidates, k, replace=False))

        prob_arr = prob_arr / total_prob
        return list(np.random.choice(candidates, k, replace=False, p=prob_arr))

    # =========================================================
    # feature extraction
    # =========================================================
    def _build_feature_vector(self, cid: int, cur_time: int) -> np.ndarray:
        stat = self.client_stats[cid]

        reward = float(max(0.0, stat.get("reward", 0.0)))
        count = float(max(0, stat.get("count", 0)))
        staleness = float(max(0, cur_time - int(stat.get("time_stamp", 0))))
        loss = float(max(0.0, stat.get("loss", 0.0)))
        processed = float(max(0.0, stat.get("processed", 0.0)))

        return np.asarray(
            [
                reward,
                count,
                staleness,
                loss,
                processed,
            ],
            dtype=np.float64,
        )

    def _update_feature_cache(self, cur_time: int):
        for cid in range(self.total_clients):
            self.client_features[cid] = self._build_feature_vector(cid, cur_time)

    def _normalize_feature_matrix(self, client_ids: List[int]) -> Dict[int, np.ndarray]:
        if not client_ids:
            return {}

        mat = np.asarray([self.client_features[cid] for cid in client_ids], dtype=np.float64)
        if mat.ndim != 2:
            return {cid: np.zeros_like(self.feature_weights) for cid in client_ids}

        med = np.median(mat, axis=0)
        q75 = np.percentile(mat, 75, axis=0)
        q25 = np.percentile(mat, 25, axis=0)
        scale = np.maximum(q75 - q25, 1e-6)

        norm_mat = (mat - med) / scale
        norm_mat = norm_mat * self.feature_weights[None, :]

        return {cid: norm_mat[idx] for idx, cid in enumerate(client_ids)}

    # =========================================================
    # graph construction
    # =========================================================
    def _similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        dist2 = float(np.sum((x - y) ** 2))
        return float(math.exp(-dist2 / (2.0 * self.graph_sigma * self.graph_sigma)))

    def _rebuild_graph(self, force: bool = False):
        """Rebuild the exact top-k Gaussian relation graph.

        The mathematical graph is unchanged from the original implementation.
        Pairwise distances are computed with NumPy vector operations one source
        client at a time, avoiding the Python O(N^2) inner loop and avoiding an
        N x N dense similarity matrix. Memory is therefore O(Nd + Nk_g).
        """
        cur_time = max(1, int(getattr(self, "round", 0)))
        if not force and self.graph_last_rebuild_round >= 0:
            if cur_time - self.graph_last_rebuild_round < self.graph_update_interval:
                return

        self._update_feature_cache(cur_time=cur_time)
        client_ids = list(range(self.total_clients))
        if len(client_ids) <= 1:
            self.graph_edges = {cid: {} for cid in client_ids}
            self.graph_neighbors = {cid: [] for cid in client_ids}
            self.graph_last_rebuild_round = cur_time
            return

        norm_features = self._normalize_feature_matrix(client_ids)
        feature_matrix = np.asarray(
            [norm_features[cid] for cid in client_ids], dtype=np.float64
        )
        topk = min(self.graph_topk, len(client_ids) - 1)
        denom = 2.0 * self.graph_sigma * self.graph_sigma

        local_neighbors: Dict[int, List[Tuple[int, float]]] = {
            cid: [] for cid in client_ids
        }
        id_array = np.asarray(client_ids, dtype=np.int64)

        for row_idx, cid in enumerate(client_ids):
            diff = feature_matrix - feature_matrix[row_idx]
            dist2 = np.einsum("ij,ij->i", diff, diff, optimize=True)
            sims = np.exp(-dist2 / denom)
            sims[row_idx] = -np.inf

            eligible = np.flatnonzero(sims > self.graph_min_weight)
            if eligible.size == 0:
                continue

            if eligible.size > topk:
                part = np.argpartition(-sims[eligible], topk - 1)[:topk]
                chosen = eligible[part]
            else:
                chosen = eligible

            # Stable deterministic ordering for downstream reproducibility.
            ordered = sorted(
                chosen.tolist(),
                key=lambda idx: (-float(sims[idx]), int(id_array[idx])),
            )[:topk]
            local_neighbors[cid] = [
                (int(id_array[idx]), float(sims[idx])) for idx in ordered
            ]

        edges: Dict[int, Dict[int, float]] = {cid: {} for cid in client_ids}
        for cid, neighbors in local_neighbors.items():
            for nbr, weight in neighbors:
                edges[cid][nbr] = max(edges[cid].get(nbr, 0.0), weight)
                edges[nbr][cid] = max(edges[nbr].get(cid, 0.0), weight)

        self.graph_edges = edges
        self.graph_neighbors = {
            cid: sorted(neigh.items(), key=lambda x: (-x[1], x[0]))
            for cid, neigh in edges.items()
        }
        self.graph_last_rebuild_round = cur_time

    def _compute_score_map(self, feasible_clients: set, cur_time: int) -> Dict[int, float]:
        blacklist = self._build_blacklist()

        candidate_ids = [
            cid for cid in self.client_stats.keys()
            if cid in feasible_clients and cid not in blacklist
        ]

        if not candidate_ids:
            return {}

        reward_values = [
            float(self.client_stats[cid]["reward"])
            for cid in candidate_ids
            if float(self.client_stats[cid]["reward"]) > 0.0
        ]

        if not reward_values:
            return {cid: 0.0 for cid in candidate_ids}

        _, min_reward, reward_range, _, clip_value = self._compute_norm_stats(
            reward_values,
            clip_quantile=self.score_clip_quantile,
        )

        score_map: Dict[int, float] = {}
        for cid in candidate_ids:
            reward = 0.0
            if self.client_stats[cid]["count"] > 0:
                reward = min(float(self.client_stats[cid]["reward"]), clip_value)

            last_seen_round = max(1.0, float(self.client_stats[cid]["time_stamp"]))
            score = (
                (reward - min_reward) / reward_range
                + np.sqrt(0.1 * np.log(max(2.0, cur_time)) / last_seen_round)
            )
            score_map[cid] = float(score)

        return score_map

    def _score_discovery_candidates(
        self,
        candidates: List[int],
        score_map: Dict[int, float],
        cur_time: int,
    ) -> Dict[int, float]:
        discovery_scores: Dict[int, float] = {}

        for cid in candidates:
            stale_bonus = np.sqrt(0.1 * np.log(max(2.0, cur_time)))
            score = float(self.client_stats[cid]["reward"])

            if score <= 0.0:
                neighbors = [
                    (nbr, w)
                    for nbr, w in self.graph_neighbors.get(cid, [])
                    if nbr in score_map and self.client_stats[nbr]["count"] > 0
                ]
                if neighbors:
                    weight_sum = sum(w for _, w in neighbors)
                    score = sum(score_map[nbr] * w for nbr, w in neighbors) / max(weight_sum, 1e-8)

            score += float(stale_bonus)
            discovery_scores[cid] = float(max(score, 1e-8))

        return discovery_scores

    # =========================================================
    # diversity-aware selection
    # =========================================================
    def _pair_similarity(self, cid1: int, cid2: int) -> float:
        if cid1 == cid2:
            return 1.0
        return float(self.graph_edges.get(cid1, {}).get(cid2, 0.0))

    def _redundancy_penalty(self, cid: int, selected: List[int]) -> float:
        if not selected or self.diversity_beta <= 0.0:
            return 0.0

        sims = [self._pair_similarity(cid, sid) for sid in selected]
        if not sims:
            return 0.0

        redundancy = float(sum(sims)) if self.diversity_mode == "sum" else float(max(sims))
        return float(self.diversity_beta * redundancy)

    def _greedy_diverse_select(
        self,
        candidates: List[int],
        k: int,
        score_map: Dict[int, float],
    ) -> List[int]:
        if k <= 0 or not candidates:
            return []

        remaining = [cid for cid in candidates if cid in score_map]
        if not remaining:
            return []

        selected: List[int] = []

        while remaining and len(selected) < k:
            best_cid = None
            best_gain = -float("inf")
            best_score = -float("inf")

            for cid in remaining:
                base_score = float(score_map.get(cid, 0.0))
                redundancy = self._redundancy_penalty(cid, selected)
                gain = base_score - redundancy

                if gain > best_gain or (abs(gain - best_gain) <= 1e-12 and base_score > best_score):
                    best_gain = gain
                    best_score = base_score
                    best_cid = cid

            if best_cid is None:
                break

            selected.append(int(best_cid))
            remaining.remove(best_cid)

        return selected

    def _select_with_diversity(
        self,
        candidates: List[int],
        k: int,
        score_map: Dict[int, float],
    ) -> List[int]:
        if k <= 0 or not candidates:
            return []

        candidates = [cid for cid in candidates if cid in score_map]
        if not candidates:
            return []

        return self._greedy_diverse_select(candidates, k, score_map)

    # =========================================================
    # selection core
    # =========================================================
    def _select_topk(self, num_samples: int, cur_time: int, feasible_clients: set) -> List[int]:
        self._rebuild_graph()

        score_map = self._compute_score_map(feasible_clients=feasible_clients, cur_time=cur_time)
        if not score_map:
            fallback = list(feasible_clients)
            return self._safe_choice(fallback, min(num_samples, len(fallback)))

        self.discovery_rate = max(self.discovery_rate * self.discovery_decay, self.discovery_min)

        available_discovery = [
            cid for cid in self.undiscovered_clients
            if cid in feasible_clients and cid in score_map
        ]
        discovery_ratio = self.discovery_rate if available_discovery else 0.0

        primary_candidates = [cid for cid in score_map if cid not in self.undiscovered_clients]
        if not primary_candidates:
            primary_candidates = list(score_map.keys())

        primary_len = min(int(num_samples * (1.0 - discovery_ratio)), len(primary_candidates))

        picked_clients: List[int] = []

        # primary selection
        if primary_len > 0:
            sorted_clients = sorted(primary_candidates, key=lambda c: score_map[c], reverse=True)
            idx = min(primary_len - 1, len(sorted_clients) - 1)
            cutoff = score_map[sorted_clients[idx]] * self.score_cutoff_ratio

            filtered_candidates = []
            for cid in sorted_clients:
                if score_map[cid] < cutoff:
                    break
                filtered_candidates.append(cid)

            if not filtered_candidates:
                filtered_candidates = sorted_clients[:primary_len]

            picked_primary = self._select_with_diversity(
                candidates=filtered_candidates,
                k=min(primary_len, len(filtered_candidates)),
                score_map=score_map,
            )
            picked_clients.extend(picked_primary)

        # discovery selection
        if available_discovery:
            discovery_len = min(len(available_discovery), num_samples - len(picked_clients))
            if discovery_len > 0:
                discovery_scores = self._score_discovery_candidates(
                    candidates=available_discovery,
                    score_map=score_map,
                    cur_time=cur_time,
                )

                top_n = min(int(self.discovery_window * discovery_len), len(available_discovery))
                discovery_candidates = sorted(
                    available_discovery,
                    key=lambda c: discovery_scores[c],
                    reverse=True,
                )[:top_n]

                picked_discovery = self._select_with_diversity(
                    candidates=discovery_candidates,
                    k=min(discovery_len, len(discovery_candidates)),
                    score_map=discovery_scores,
                )
                picked_clients.extend(picked_discovery)

        # fill
        if len(picked_clients) < num_samples:
            remaining = num_samples - len(picked_clients)
            fill_candidates = [cid for cid in score_map if cid not in picked_clients]
            if fill_candidates:
                picked_fill = self._select_with_diversity(
                    candidates=fill_candidates,
                    k=min(remaining, len(fill_candidates)),
                    score_map=score_map,
                )
                picked_clients.extend(picked_fill)

        picked_clients = list(dict.fromkeys(int(cid) for cid in picked_clients))[:num_samples]
        return picked_clients

    def select(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[int]:
        if not available_clients:
            return []

        num_samples = min(self.clients_per_round, len(available_clients))
        feasible_clients = set(int(cid) for cid in available_clients)
        cur_time = self.round + 1

        if all(
            (self.client_stats[cid]["count"] == 0) or (float(self.client_stats[cid]["reward"]) <= 0.0)
            for cid in feasible_clients
        ):
            selected = np.random.choice(list(feasible_clients), num_samples, replace=False).tolist()
            selected = [int(cid) for cid in selected]
            self.selected_clients = selected
            return selected

        selected = self._select_topk(
            num_samples=num_samples,
            cur_time=cur_time,
            feasible_clients=feasible_clients,
        )
        selected = [int(cid) for cid in selected]
        self.selected_clients = selected
        return selected

    # =========================================================
    # update
    # =========================================================
    def update(self, selected_clients: List[int], results: Dict[Any, Any]):
        super().update(selected_clients, results)
        cur_time = self.round

        for client_id, result in results.items():
            try:
                cid = int(client_id)
            except (ValueError, TypeError):
                continue

            if cid not in self.client_stats:
                self.client_stats[cid] = {
                    "reward": 0.0,
                    "loss": 0.0,
                    "processed": 0.0,
                    "time_stamp": 0,
                    "count": 0,
                }

            if not isinstance(result, dict):
                continue

            processed = float(
                result.get(
                    "num_processed",
                    result.get(
                        "num_samples",
                        result.get(
                            "trained_size",
                            result.get("data_size", 0.0),
                        )
                    )
                ) or 0.0
            )
            loss = float(result.get("loss", 0.0) or 0.0)
            loss_drop_raw = result.get("loss_drop", None)
            grad_norm_raw = result.get("gradient_norm", 0.0)
            loss_drop = 0.0 if loss_drop_raw is None else float(loss_drop_raw or 0.0)
            grad_norm = float(grad_norm_raw or 0.0)

            if self.reward_mode == "processed_loss":
                reward = processed * loss
            elif self.reward_mode == "loss":
                reward = loss
            elif self.reward_mode == "loss_drop":
                reward = max(0.0, loss_drop)
            else:  # gradient_norm
                reward = max(0.0, grad_norm)

            self.client_stats[cid]["reward"] = float(max(0.0, reward))
            self.client_stats[cid]["loss"] = float(max(0.0, loss))
            self.client_stats[cid]["processed"] = float(max(0.0, processed))
            self.client_stats[cid]["time_stamp"] = int(cur_time)
            self.client_stats[cid]["count"] = int(self.client_stats[cid]["count"]) + 1

            self.undiscovered_clients.discard(cid)

        self._update_feature_cache(cur_time=max(1, int(self.round)))

        if self.round % self.graph_update_interval == 0:
            self._rebuild_graph(force=True)

    def get_status(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "discovery_rate": self.discovery_rate,
            "num_undiscovered_clients": len(self.undiscovered_clients),
            "graph_last_rebuild_round": self.graph_last_rebuild_round,
            "reward_mode": self.reward_mode,
        }
