"""
DivFL selector adapted to the current fedselect framework.

Paper idea:
    Balakrishnan et al., "Diverse Client Selection for Federated Learning via
    Submodular Maximization", ICLR 2022.

Framework adaptation notes:
    * The original DivFL selects a subset that represents all clients in gradient
      space by maximizing a facility-location style submodular objective.
    * This framework does not request one-step gradients from every client before
      selection. Therefore this implementation follows the paper's practical
      "no-overhead" variant: reuse stale update vectors from previously selected
      clients, and use pre-selection scalar metrics as a fallback feature for
      clients that have not been observed yet.
    * It keeps the core stochastic greedy facility-location selection rule.

Usage:
    Put this file under fedselect/selection/ and import it in
    fedselect/selection/__init__.py:
        from . import divfl

    Then run with selector name:
        --algorithm divfl
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseSelector, register_selector


_EPS = 1e-12


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return v


def _flatten_update(value: Any) -> Optional[np.ndarray]:
    """Convert torch tensors, numpy arrays, lists, or state-dict-like values to 1-D numpy."""
    if value is None:
        return None

    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            arr = value.detach().float().cpu().reshape(-1).numpy()
            return np.nan_to_num(arr, copy=False).astype(np.float32, copy=False)
    except Exception:
        pass

    if isinstance(value, dict):
        chunks = []
        for key in sorted(value.keys()):
            chunk = _flatten_update(value[key])
            if chunk is not None and chunk.size > 0:
                chunks.append(chunk)
        if not chunks:
            return None
        arr = np.concatenate(chunks, axis=0)
        return np.nan_to_num(arr, copy=False).astype(np.float32, copy=False)

    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return np.nan_to_num(arr, copy=False).astype(np.float32, copy=False)


@register_selector("divfl")
class DivFLSelector(BaseSelector):
    """Diverse client selection via stochastic greedy facility location."""

    def __init__(
        self,
        total_clients: int,
        clients_per_round: int,
        embedding_dim: int = 32,
        greedy_candidate_size: int = 0,
        warmup_rounds: int = 1,
        embedding_ema: float = 0.7,
        metric_fallback_weight: float = 0.35,
        distance_metric: str = "cosine",  # cosine or euclidean
        staleness_weight: float = 0.0,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(total_clients=total_clients, clients_per_round=clients_per_round, **kwargs)

        self.embedding_dim = int(max(2, embedding_dim))
        self.greedy_candidate_size = int(max(0, greedy_candidate_size))
        self.warmup_rounds = int(max(0, warmup_rounds))
        self.embedding_ema = float(np.clip(embedding_ema, 0.0, 0.999))
        self.metric_fallback_weight = float(np.clip(metric_fallback_weight, 0.0, 1.0))
        self.distance_metric = str(distance_metric).lower()
        if self.distance_metric not in {"cosine", "euclidean"}:
            self.distance_metric = "cosine"
        self.staleness_weight = float(max(0.0, staleness_weight))

        self.rng = np.random.default_rng(random_state)
        self.embeddings = np.zeros((self.total_clients, self.embedding_dim), dtype=np.float64)
        self.has_embedding = np.zeros(self.total_clients, dtype=bool)
        self.last_seen_round = np.full(self.total_clients, -1, dtype=np.int64)
        self._projection: Optional[np.ndarray] = None
        self._projection_input_dim: Optional[int] = None

        self.last_selected: List[int] = []
        self.last_gains: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _valid_ids(self, client_ids: List[int]) -> List[int]:
        return [int(cid) for cid in client_ids if 0 <= int(cid) < self.total_clients]

    def _sample_random(self, available_clients: List[int]) -> List[int]:
        available_clients = self._valid_ids(available_clients)
        k = min(self.clients_per_round, len(available_clients))
        if k <= 0:
            return []
        if len(available_clients) <= k:
            return list(available_clients)
        return self.rng.choice(available_clients, size=k, replace=False).astype(int).tolist()

    def _get_projection(self, input_dim: int) -> np.ndarray:
        if self._projection is None or self._projection_input_dim != input_dim:
            scale = 1.0 / math.sqrt(max(1, input_dim))
            self._projection = self.rng.normal(
                loc=0.0,
                scale=scale,
                size=(input_dim, self.embedding_dim),
            ).astype(np.float32)
            self._projection_input_dim = input_dim
        return self._projection

    def _project_update(self, vector: np.ndarray) -> Optional[np.ndarray]:
        if vector is None or vector.size == 0:
            return None
        proj = self._get_projection(int(vector.size))
        emb = vector.astype(np.float32, copy=False) @ proj
        emb = np.asarray(emb, dtype=np.float64)
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm <= _EPS:
            return None
        return emb / norm

    def _update_embedding_from_vector(self, cid: int, vector: np.ndarray) -> None:
        emb = self._project_update(vector)
        if emb is None:
            return
        if self.has_embedding[cid]:
            new = self.embedding_ema * self.embeddings[cid] + (1.0 - self.embedding_ema) * emb
            norm = float(np.linalg.norm(new))
            self.embeddings[cid] = new / max(norm, _EPS)
        else:
            self.embeddings[cid] = emb
            self.has_embedding[cid] = True
        self.last_seen_round[cid] = int(self.round)

    def _metric_vector(
        self,
        cid: int,
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        metrics = client_metrics.get(cid, {}) if isinstance(client_metrics, dict) else {}
        info = self.client_info.get(cid, None)

        loss = _as_float(metrics.get("loss", getattr(info, "last_loss", 0.0) if info else 0.0), 0.0)
        prev_loss = _as_float(metrics.get("prev_loss", loss), loss)
        loss_drop = _as_float(metrics.get("loss_drop", 0.0), 0.0)
        grad_norm = _as_float(metrics.get("gradient_norm", getattr(info, "gradient_norm", 0.0) if info else 0.0), 0.0)
        data_size = _as_float(metrics.get("data_size", getattr(info, "data_size", 1.0) if info else 1.0), 1.0)
        count = _as_float(getattr(info, "participation_count", 0.0) if info else 0.0, 0.0)
        staleness = 0.0
        if info is not None:
            staleness = max(0.0, float(self.round - getattr(info, "last_selected_round", -1)))

        base = np.asarray(
            [
                loss,
                prev_loss,
                loss_drop,
                grad_norm,
                math.log1p(max(0.0, data_size)),
                count,
                staleness,
            ],
            dtype=np.float64,
        )

        out = np.zeros(self.embedding_dim, dtype=np.float64)
        n = min(self.embedding_dim, base.size)
        out[:n] = base[:n]
        return out

    def _build_feature_matrix(
        self,
        client_ids: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        ids = self._valid_ids(client_ids)
        if not ids:
            return np.zeros((0, self.embedding_dim), dtype=np.float64), []

        raw_features = []
        for cid in ids:
            metric_vec = self._metric_vector(cid, client_metrics)
            if self.has_embedding[cid]:
                # Blend stale update-space feature with current scalar metric fallback.
                feature = (
                    (1.0 - self.metric_fallback_weight) * self.embeddings[cid]
                    + self.metric_fallback_weight * metric_vec
                )
            else:
                feature = metric_vec
            raw_features.append(feature)

        X = np.asarray(raw_features, dtype=np.float64)

        # Robust column normalization so scalar metrics do not dominate update embeddings.
        med = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        scale = q75 - q25
        scale = np.where(scale > _EPS, scale, 1.0)
        X = (X - med) / scale

        row_norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(row_norm, _EPS)
        return X, ids

    def _pairwise_distance(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0, 0), dtype=np.float64)
        if self.distance_metric == "euclidean":
            sq = np.sum(X * X, axis=1, keepdims=True)
            D2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
            return np.sqrt(D2)

        # Cosine distance on row-normalized features.
        sim = np.clip(X @ X.T, -1.0, 1.0)
        return 1.0 - sim

    def _client_weights(
        self,
        ids: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        vals = []
        for cid in ids:
            metrics = client_metrics.get(cid, {}) if isinstance(client_metrics, dict) else {}
            info = self.client_info.get(cid, None)
            ds = _as_float(metrics.get("data_size", getattr(info, "data_size", 1.0) if info else 1.0), 1.0)
            vals.append(max(ds, 1.0))
        w = np.asarray(vals, dtype=np.float64)
        total = float(w.sum())
        if total <= 0 or not np.isfinite(total):
            return np.full(len(ids), 1.0 / max(1, len(ids)), dtype=np.float64)
        return w / total

    # ------------------------------------------------------------------
    # Main selector API
    # ------------------------------------------------------------------
    def select(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[int]:
        available_clients = self._valid_ids(available_clients)
        if not available_clients:
            self.last_selected = []
            return []
        if len(available_clients) <= self.clients_per_round:
            self.last_selected = list(available_clients)
            return list(available_clients)

        # Give the no-overhead update cache at least one random round.
        if self.round < self.warmup_rounds and not np.any(self.has_embedding):
            selected = self._sample_random(available_clients)
            self.last_selected = list(selected)
            return selected

        X, ids = self._build_feature_matrix(available_clients, client_metrics)
        if len(ids) <= self.clients_per_round:
            self.last_selected = list(ids)
            return list(ids)

        D = self._pairwise_distance(X)
        weights = self._client_weights(ids, client_metrics)
        id_to_pos = {cid: pos for pos, cid in enumerate(ids)}

        selected: List[int] = []
        remaining = list(ids)
        # Facility-location minimization form: minimize sum_i min_{j in S} d(i,j).
        # Start from a large constant so the first selected client is the most representative.
        init_const = float(np.max(D) + 1.0) if D.size else 1.0
        current_min = np.full(len(ids), init_const, dtype=np.float64)
        self.last_gains = {}

        while len(selected) < self.clients_per_round and remaining:
            if self.greedy_candidate_size > 0 and len(remaining) > self.greedy_candidate_size:
                eval_candidates = self.rng.choice(
                    remaining,
                    size=self.greedy_candidate_size,
                    replace=False,
                ).astype(int).tolist()
            else:
                eval_candidates = list(remaining)

            best_cid = None
            best_gain = None
            best_new_min = None

            base_obj = float(np.dot(weights, current_min))
            for cid in eval_candidates:
                pos = id_to_pos[cid]
                new_min = np.minimum(current_min, D[:, pos])
                new_obj = float(np.dot(weights, new_min))
                gain = base_obj - new_obj

                if self.staleness_weight > 0.0:
                    last_seen = self.last_seen_round[cid]
                    stale = max(0, self.round - int(last_seen)) if last_seen >= 0 else self.round + 1
                    gain *= 1.0 + self.staleness_weight * math.log1p(stale)

                self.last_gains[cid] = float(gain)

                if best_gain is None or gain > best_gain:
                    best_gain = float(gain)
                    best_cid = int(cid)
                    best_new_min = new_min

            if best_cid is None:
                break

            selected.append(best_cid)
            remaining.remove(best_cid)
            current_min = best_new_min if best_new_min is not None else current_min

        if len(selected) < self.clients_per_round:
            extra_pool = [cid for cid in available_clients if cid not in selected]
            if extra_pool:
                extra = self.rng.choice(
                    extra_pool,
                    size=min(self.clients_per_round - len(selected), len(extra_pool)),
                    replace=False,
                ).astype(int).tolist()
                selected.extend(extra)

        self.last_selected = selected[: self.clients_per_round]
        return list(self.last_selected)

    def update(self, selected_clients: List[int], results: Dict[int, Dict[str, Any]]) -> None:
        selected_clients = self._valid_ids(selected_clients)
        for cid in selected_clients:
            result = results.get(cid, {}) if isinstance(results, dict) else {}
            if not isinstance(result, dict):
                continue
            update_vec = _flatten_update(result.get("update_vector", None))
            if update_vec is None:
                update_vec = _flatten_update(result.get("delta", None))
            if update_vec is not None:
                self._update_embedding_from_vector(cid, update_vec)

        super().update(selected_clients, results)
