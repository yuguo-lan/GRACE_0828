import numpy as np
from typing import List, Dict, Any, Optional
from .base import BaseSelector, register_selector


@register_selector("oort")
class OortFullSelector(BaseSelector):
    """
    Oort training selector. This is the implementation previously named ``oort_core``.

    核心机制：
    1) statistical utility: aggregate training loss approximation
    2) exploration / exploitation
    3) staleness incentive
    4) clipping + blacklist robustness
    """

    def __init__(
        self,
        total_clients: int,
        clients_per_round: int,

        # -------- exploration --------
        exploration_factor: float = 0.95,
        exploration_decay: float = 0.99,
        exploration_min: float = 0.6,

        # -------- exploitation --------
        sample_window: int = 15,
        clip_bound: float = 0.5,
        cut_off_util: float = 0.4,

        # -------- robustness --------
        blacklist_rounds: int = 3,
        blacklist_max_len: float = 0.4,

        **kwargs,
    ):
        super().__init__(total_clients, clients_per_round, **kwargs)

        # exploration
        self.exploration = float(exploration_factor)
        self.exploration_decay = float(exploration_decay)
        self.exploration_min = float(exploration_min)

        # exploitation
        self.sample_window = int(sample_window)
        self.clip_bound = float(clip_bound)
        self.cut_off_util = float(cut_off_util)

        # blacklist
        self.blacklist_rounds = int(blacklist_rounds)
        self.blacklist_max_len = float(blacklist_max_len)
        self.blacklist = set()

        # round status
        self.training_round = 0

        # Oort state
        # total_arms[cid] = {
        #   reward, time_stamp, count
        # }
        self.total_arms: Dict[int, Dict[str, Any]] = {}
        self.unexplored = set()

        self.exploit_clients: List[int] = []
        self.explore_clients: List[int] = []

        for cid in range(total_clients):
            self.total_arms[cid] = {
                "reward": 0.0,
                "time_stamp": 0,
                "count": 0,
            }
            self.unexplored.add(cid)

    # =========================================================
    # utility helpers
    # =========================================================
    def _get_norm(self, values: List[float], clip_bound: float = 0.95, thres: float = 1e-4):
        """
        返回:
        max, min, range, avg, clip_value
        """
        if not values:
            return 0.0, 0.0, thres, 0.0, 0.0

        values = list(values)
        values.sort()

        clip_idx = min(int(len(values) * clip_bound), len(values) - 1)
        clip_value = values[clip_idx]

        vmax = max(values)
        vmin = min(values) * 0.999
        vrange = max(vmax - vmin, thres)
        vavg = sum(values) / max(float(len(values)), 1e-4)

        return float(vmax), float(vmin), float(vrange), float(vavg), float(clip_value)

    def _get_blacklist(self) -> set:
        if self.blacklist_rounds == -1:
            return set()

        sorted_client_ids = sorted(
            list(self.total_arms.keys()),
            reverse=True,
            key=lambda k: self.total_arms[k]["count"],
        )

        blacklist = []
        for cid in sorted_client_ids:
            if self.total_arms[cid]["count"] > self.blacklist_rounds:
                blacklist.append(cid)
            else:
                break

        predefined_max_len = int(self.blacklist_max_len * len(self.total_arms)) if self.total_arms else 0
        if predefined_max_len >= 0 and len(blacklist) > predefined_max_len:
            blacklist = blacklist[:predefined_max_len]

        return set(blacklist)

    def _safe_choice(self, candidates: List[int], k: int, probs: Optional[List[float]] = None) -> List[int]:
        if k <= 0 or not candidates:
            return []

        k = min(k, len(candidates))

        if probs is None:
            return list(np.random.choice(candidates, k, replace=False))

        probs_arr = np.asarray(probs, dtype=np.float64)
        probs_arr = np.nan_to_num(probs_arr, nan=0.0, posinf=0.0, neginf=0.0)
        s = probs_arr.sum()

        if s <= 0:
            return list(np.random.choice(candidates, k, replace=False))

        probs_arr = probs_arr / s
        return list(np.random.choice(candidates, k, replace=False, p=probs_arr))

    # =========================================================
    # core selection
    # =========================================================
    def select(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[int]:
        if not available_clients:
            return []

        num_samples = min(self.clients_per_round, len(available_clients))
        feasible_clients = set(int(cid) for cid in available_clients)

        self.training_round = self.round + 1

        # 温启动：如果全都没有有效 reward，随机选
        if all(
            (self.total_arms[cid]["count"] == 0) or (float(self.total_arms[cid]["reward"]) <= 0.0)
            for cid in feasible_clients
        ):
            selected = np.random.choice(
                list(feasible_clients),
                num_samples,
                replace=False
            ).tolist()
            selected = [int(cid) for cid in selected]
            self.exploit_clients = selected
            self.explore_clients = []
            self.selected_clients = selected
            return selected

        selected = self._get_topk(
            num_samples=num_samples,
            cur_time=self.training_round,
            feasible_clients=feasible_clients,
        )

        selected = [int(cid) for cid in selected]
        self.selected_clients = selected
        return selected

    def _get_topk(self, num_samples: int, cur_time: int, feasible_clients: set) -> List[int]:
        self.training_round = cur_time
        self.blacklist = self._get_blacklist()

        ordered_keys = [
            cid for cid in self.total_arms.keys()
            if cid in feasible_clients and cid not in self.blacklist
        ]

        if not ordered_keys:
            fallback = list(feasible_clients)
            return self._safe_choice(fallback, min(num_samples, len(fallback)))

        # reward stats
        moving_reward = [
            float(self.total_arms[cid]["reward"])
            for cid in ordered_keys
            if float(self.total_arms[cid]["reward"]) > 0.0
        ]

        if not moving_reward:
            selected = self._safe_choice(ordered_keys, min(num_samples, len(ordered_keys)))
            self.exploit_clients = selected
            self.explore_clients = []
            return selected

        _, min_reward, range_reward, _, clip_value = self._get_norm(
            moving_reward,
            clip_bound=self.clip_bound,
        )

        # score = normalized clipped reward + staleness incentive
        scores: Dict[int, float] = {}
        for cid in ordered_keys:
            reward = 0.0
            if self.total_arms[cid]["count"] > 0:
                reward = min(float(self.total_arms[cid]["reward"]), clip_value)

            last_round_seen = max(1.0, float(self.total_arms[cid]["time_stamp"]))
            score = (
                (reward - min_reward) / range_reward
                + np.sqrt(0.1 * np.log(max(2.0, cur_time)) / last_round_seen)
            )

            scores[cid] = float(score)

        # exploration decay
        self.exploration = max(self.exploration * self.exploration_decay, self.exploration_min)

        available_unexplored = [cid for cid in self.unexplored if cid in feasible_clients]
        exploration = self.exploration if len(available_unexplored) > 0 else 0.0

        exploit_len = min(int(num_samples * (1.0 - exploration)), len(scores))

        sorted_clients = sorted(scores, key=scores.get, reverse=True)
        picked_clients: List[int] = []

        # ---------------- exploit ----------------
        if exploit_len > 0:
            idx = min(exploit_len - 1, len(sorted_clients) - 1)
            cutoff = scores[sorted_clients[idx]] * self.cut_off_util

            picked_candidates = []
            for cid in sorted_clients:
                if scores[cid] < cutoff:
                    break
                picked_candidates.append(cid)

            if not picked_candidates:
                picked_clients = sorted_clients[:exploit_len]
            else:
                probs = [scores[cid] for cid in picked_candidates]
                picked_clients = self._safe_choice(
                    picked_candidates,
                    min(exploit_len, len(picked_candidates)),
                    probs=probs,
                )

        self.exploit_clients = list(picked_clients)

        # ---------------- explore ----------------
        self.explore_clients = []

        if available_unexplored:
            init_reward = {
                cid: float(self.total_arms[cid]["reward"])
                for cid in available_unexplored
            }

            explore_len = min(len(available_unexplored), num_samples - len(picked_clients))

            if explore_len > 0 and len(init_reward) > 0:
                top_n = min(int(self.sample_window * explore_len), len(init_reward))
                unexplored_candidates = sorted(
                    init_reward,
                    key=init_reward.get,
                    reverse=True
                )[:top_n]

                probs = [init_reward[cid] for cid in unexplored_candidates]
                picked_unexplored = self._safe_choice(
                    unexplored_candidates,
                    min(explore_len, len(unexplored_candidates)),
                    probs=probs,
                )

                self.explore_clients = picked_unexplored
                picked_clients.extend(picked_unexplored)

        # ---------------- fill ----------------
        if len(picked_clients) < num_samples:
            remaining = num_samples - len(picked_clients)
            candidates = [cid for cid in ordered_keys if cid not in picked_clients]
            if candidates:
                picked_clients.extend(self._safe_choice(candidates, min(remaining, len(candidates))))

        picked_clients = list(dict.fromkeys(picked_clients))[:num_samples]
        return picked_clients

    # =========================================================
    # update
    # =========================================================
    def update(self, selected_clients: List[int], results: Dict[Any, Any]):
        """
        兼容当前框架 results:
        - result["loss"]
        - result["data_size"] / num_processed / num_samples / trained_size

        采用当前框架下的 utility 近似：
        reward = processed * loss
        """
        super().update(selected_clients, results)
        cur_time = self.round

        for client_id, result in results.items():
            try:
                cid = int(client_id)
            except (ValueError, TypeError):
                continue

            if cid not in self.total_arms:
                self.total_arms[cid] = {
                    "reward": 0.0,
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
                            result.get("data_size", 0.0)
                        )
                    )
                ) or 0.0
            )

            loss = float(result.get("loss", 0.0) or 0.0)
            reward = processed * loss

            self.total_arms[cid]["reward"] = float(max(0.0, reward))
            self.total_arms[cid]["time_stamp"] = int(cur_time)
            self.total_arms[cid]["count"] = int(self.total_arms[cid]["count"]) + 1

            self.unexplored.discard(cid)

    def get_status(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "training_round": self.training_round,
            "exploration": self.exploration,
            "num_unexplored": len(self.unexplored),
            "num_exploit_clients_last_round": len(self.exploit_clients),
            "num_explore_clients_last_round": len(self.explore_clients),
            "blacklist_size": len(self.blacklist),
        }