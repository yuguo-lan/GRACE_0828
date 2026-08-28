"""
FedCor selector adapted to the current fedselect framework.

Paper idea:
    Tang et al., "FedCor: Correlation-Based Active Client Selection Strategy for
    Heterogeneous Federated Learning".

Framework adaptation notes:
    * The original FedCor trains a GP over all clients' loss changes. In this
      selection-only framework, select() only receives pre-selection metrics and
      update() only receives the selected clients' local updates. Therefore this
      implementation uses a low-overhead proxy:
        - cache pre-selection losses in select();
        - after local training, estimate selected-client loss changes from
          local training loss minus cached pre-selection loss;
        - learn a covariance proxy from stale update vectors via random
          projection + a linear kernel, matching the paper's low-rank kernel
          spirit while avoiding Server/Client changes.
    * It keeps the paper's core selection rule: iterative GP posterior update,
      lower confidence bound prediction, and annealing beta^{tau_k}.

Usage:
    Put this file under fedselect/selection/ and import it in
    fedselect/selection/__init__.py:
        from . import fedcor

    Then run with selector name:
        --algorithm fedcor
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseSelector, register_selector


_EPS = 1e-12


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
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

    # torch is optional at selector import time.
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


@register_selector("fedcor")
class FedCorSelector(BaseSelector):
    """
    Correlation-based active client selector.

    Parameters are intentionally conservative so it can run without modifying
    Server, Client, or Experiment.
    """

    def __init__(
        self,
        total_clients: int,
        clients_per_round: int,
        warmup_rounds: int = 5,
        embedding_dim: int = 32,
        covariance_update_interval: int = 5,
        alpha_scale: float = 1.0,
        beta: float = 0.95,
        embedding_ema: float = 0.7,
        loss_ema: float = 0.7,
        min_variance: float = 1e-3,
        jitter: float = 1e-6,
        random_state: Optional[int] = None,
        use_weighted_warmup: bool = True,
        **kwargs: Any,
    ):
        super().__init__(total_clients=total_clients, clients_per_round=clients_per_round, **kwargs)

        self.warmup_rounds = int(max(0, warmup_rounds))
        self.embedding_dim = int(max(2, embedding_dim))
        self.covariance_update_interval = int(max(1, covariance_update_interval))
        self.alpha_scale = float(max(0.0, alpha_scale))
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.embedding_ema = float(np.clip(embedding_ema, 0.0, 0.999))
        self.loss_ema = float(np.clip(loss_ema, 0.0, 0.999))
        self.min_variance = float(max(min_variance, 1e-12))
        self.jitter = float(max(jitter, 1e-12))
        self.use_weighted_warmup = bool(use_weighted_warmup)

        self.rng = np.random.default_rng(random_state)

        # Low-rank client embeddings X. Sigma is built as X X^T after row norm.
        self.embeddings = np.zeros((self.total_clients, self.embedding_dim), dtype=np.float64)
        prior = self.rng.normal(size=(self.total_clients, self.embedding_dim))
        prior_norm = np.linalg.norm(prior, axis=1, keepdims=True) + _EPS
        self.prior_embeddings = prior / prior_norm
        self.has_embedding = np.zeros(self.total_clients, dtype=bool)

        self.loss_change_ema = np.zeros(self.total_clients, dtype=np.float64)
        self.loss_scale_ema = np.ones(self.total_clients, dtype=np.float64)
        self.selection_counts = np.zeros(self.total_clients, dtype=np.int64)  # tau_k since last reset

        self.Sigma = np.eye(self.total_clients, dtype=np.float64)
        self._projection: Optional[np.ndarray] = None
        self._projection_input_dim: Optional[int] = None
        self._last_pre_metrics: Dict[int, Dict[str, Any]] = {}
        self._last_available_clients: List[int] = []
        self._has_covariance_from_updates = False

        # Debug fields useful for logging or plotting.
        self.last_selected: List[int] = []
        self.last_scores: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _valid_ids(self, client_ids: List[int]) -> List[int]:
        return [int(cid) for cid in client_ids if 0 <= int(cid) < self.total_clients]

    def _client_weights(
        self,
        client_ids: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        weights = []
        for cid in client_ids:
            data_size = None
            if client_metrics is not None and cid in client_metrics:
                data_size = client_metrics[cid].get("data_size")
            if data_size is None and cid in self.client_info:
                data_size = getattr(self.client_info[cid], "data_size", None)
            value = _as_float(data_size, None)
            if value is None or value <= 0:
                value = 1.0
            weights.append(value)

        arr = np.asarray(weights, dtype=np.float64)
        total = float(arr.sum())
        if total <= 0 or not np.isfinite(total):
            return np.full(len(client_ids), 1.0 / max(1, len(client_ids)), dtype=np.float64)
        return arr / total

    def _sample_random(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[int]:
        available_clients = self._valid_ids(available_clients)
        k = min(self.clients_per_round, len(available_clients))
        if k <= 0:
            return []
        if len(available_clients) <= k:
            return list(available_clients)

        probs = None
        if self.use_weighted_warmup:
            probs = self._client_weights(available_clients, client_metrics)
        return self.rng.choice(available_clients, size=k, replace=False, p=probs).astype(int).tolist()

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

    def _update_embedding_from_vector(self, cid: int, vector: np.ndarray) -> None:
        if vector is None or vector.size == 0:
            return
        proj = self._get_projection(int(vector.size))
        emb = vector.astype(np.float32, copy=False) @ proj
        emb = np.asarray(emb, dtype=np.float64)
        norm = float(np.linalg.norm(emb))
        if not np.isfinite(norm) or norm <= _EPS:
            return
        emb = emb / norm

        if self.has_embedding[cid]:
            old = self.embeddings[cid]
            new = self.embedding_ema * old + (1.0 - self.embedding_ema) * emb
            new_norm = float(np.linalg.norm(new))
            self.embeddings[cid] = new / max(new_norm, _EPS)
        else:
            self.embeddings[cid] = emb
            self.has_embedding[cid] = True

    def _rebuild_covariance(self) -> None:
        """Build PSD covariance proxy using normalized client embeddings."""
        X = self.embeddings.copy()
        missing = ~self.has_embedding
        if np.any(missing):
            # Small random prior prevents a totally identical covariance before all clients appear.
            X[missing] = 0.05 * self.prior_embeddings[missing]

        row_norm = np.linalg.norm(X, axis=1, keepdims=True)
        row_norm = np.maximum(row_norm, _EPS)
        Xn = X / row_norm

        # Client-specific variance scale comes from observed absolute loss changes.
        scale = np.sqrt(np.maximum(self.loss_scale_ema, self.min_variance))
        Xs = Xn * scale[:, None]

        Sigma = Xs @ Xs.T
        Sigma = 0.5 * (Sigma + Sigma.T)
        diag = np.diag(Sigma).copy()
        diag = np.maximum(diag, self.min_variance)
        np.fill_diagonal(Sigma, diag + self.jitter)

        self.Sigma = Sigma.astype(np.float64, copy=False)
        self._has_covariance_from_updates = bool(np.any(self.has_embedding))

    def _posterior_mean_given(self, mu: np.ndarray, Sigma: np.ndarray, cid: int, observed: float) -> np.ndarray:
        denom = float(Sigma[cid, cid] + self.jitter)
        if denom <= 0 or not np.isfinite(denom):
            return mu.copy()
        return mu + (Sigma[:, cid] / denom) * (observed - mu[cid])

    def _posterior_cov_given(self, Sigma: np.ndarray, cid: int) -> np.ndarray:
        denom = float(Sigma[cid, cid] + self.jitter)
        if denom <= 0 or not np.isfinite(denom):
            return Sigma.copy()
        col = Sigma[:, cid].copy()
        new_sigma = Sigma - np.outer(col, col) / denom
        new_sigma = 0.5 * (new_sigma + new_sigma.T)
        diag = np.maximum(np.diag(new_sigma), self.min_variance)
        np.fill_diagonal(new_sigma, diag + self.jitter)
        return new_sigma

    # ------------------------------------------------------------------
    # Main selector API
    # ------------------------------------------------------------------
    def select(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[int]:
        available_clients = self._valid_ids(available_clients)
        self._last_available_clients = list(available_clients)
        self._last_pre_metrics = dict(client_metrics or {})

        if not available_clients:
            self.last_selected = []
            return []
        if len(available_clients) <= self.clients_per_round:
            self.last_selected = list(available_clients)
            return list(available_clients)

        # Warm-up keeps the paper's "collect information first" phase but without
        # requesting extra full-client post-training loss evaluation.
        if self.round < self.warmup_rounds or not self._has_covariance_from_updates:
            selected = self._sample_random(available_clients, client_metrics)
            self.last_selected = list(selected)
            self.last_scores = {}
            return selected

        objective_ids = list(range(self.total_clients))
        p = self._client_weights(objective_ids, client_metrics)

        mu = np.zeros(self.total_clients, dtype=np.float64)
        Sigma = self.Sigma.copy()
        selected: List[int] = []
        remaining = set(available_clients)
        self.last_scores = {}

        while len(selected) < self.clients_per_round and remaining:
            best_cid = None
            best_score = None
            best_y = None

            for cid in list(remaining):
                std = math.sqrt(max(float(Sigma[cid, cid]), self.min_variance))
                alpha_k = self.alpha_scale * (self.beta ** int(self.selection_counts[cid]))
                predicted_loss_change = float(mu[cid] - alpha_k * std)
                post_mu = self._posterior_mean_given(mu, Sigma, cid, predicted_loss_change)
                score = float(np.dot(p, post_mu))
                self.last_scores[cid] = score

                if best_score is None or score < best_score:
                    best_score = score
                    best_cid = cid
                    best_y = predicted_loss_change

            if best_cid is None:
                break

            selected.append(int(best_cid))
            remaining.remove(int(best_cid))

            # Iteratively condition on the predicted observation, as in FedCor Algorithm 1.
            mu = self._posterior_mean_given(mu, Sigma, int(best_cid), float(best_y))
            Sigma = self._posterior_cov_given(Sigma, int(best_cid))

        # Safety fallback.
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

        # Update embeddings and observed loss-change proxy for selected clients.
        for cid in selected_clients:
            result = results.get(cid, {}) if isinstance(results, dict) else {}
            if not isinstance(result, dict):
                continue

            update_vec = _flatten_update(result.get("update_vector", None))
            if update_vec is None:
                update_vec = _flatten_update(result.get("delta", None))
            if update_vec is not None:
                self._update_embedding_from_vector(cid, update_vec)

            pre_loss = None
            if cid in self._last_pre_metrics:
                pre_loss = _as_float(self._last_pre_metrics[cid].get("loss"), None)
            post_loss = _as_float(result.get("eval_loss", None), None)
            if post_loss is None:
                post_loss = _as_float(result.get("loss", None), None)

            if pre_loss is not None and post_loss is not None:
                loss_change = post_loss - pre_loss
                self.loss_change_ema[cid] = (
                    self.loss_ema * self.loss_change_ema[cid]
                    + (1.0 - self.loss_ema) * float(loss_change)
                )
                self.loss_scale_ema[cid] = (
                    self.loss_ema * self.loss_scale_ema[cid]
                    + (1.0 - self.loss_ema) * max(abs(float(loss_change)), self.min_variance)
                )

        for cid in selected_clients:
            self.selection_counts[cid] += 1

        # Rebuild covariance periodically. During early rounds, rebuild whenever new
        # embeddings arrive so the selector can leave warm-up promptly.
        should_rebuild = (
            self.round < self.warmup_rounds
            or (self.round + 1) % self.covariance_update_interval == 0
        )
        if should_rebuild:
            self._rebuild_covariance()
            # Paper resets alpha/tau after GP training rounds.
            self.selection_counts[:] = 0

        super().update(selected_clients, results)
