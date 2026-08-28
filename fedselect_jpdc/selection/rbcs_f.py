"""
公平选择算法（RBCS-F，按当前实验框架对齐的简化版）。

论文：
Huang et al., "An Efficiency-Boosting Client Selection Scheme for Federated Learning
with Fairness Guarantee", TPDS 2021.

当前框架下的实现原则：
1. 保留论文核心：
   - 长期公平队列 Z_{t+1,n} = [Z_{t,n} + beta - x_{t,n}]^+
   - 每轮目标：min V * max_selected_tau - sum Z_t,n * x_t,n
   - 采用论文 Algorithm 1 的 divide-and-conquer 思想求解
2. 不强行实现 C2MAB 上下文时延预测：
   - 你们当前框架在 select 前没有稳定提供论文需要的 context
   - 因此改为基于历史 observed duration 的在线估计
3. 只改算法文件，不改 Client / Server / Experiment
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .base import BaseSelector, register_selector


@register_selector("rbcs_f")
class FairSelector(BaseSelector):
    """
    RBCS-F aligned to our framework.

    保留：
    - fairness queue Z
    - Lyapunov-style per-round objective
    - divide-and-conquer style selection

    去掉：
    - C2MAB contextual estimation
    - 复杂上下文特征
    """

    def __init__(
        self,
        total_clients: int,
        clients_per_round: int,
        target_participation: float = 0.15,   # 论文中的 beta
        lyapunov_v: float = 10.0,             # 论文中的 V
        default_duration: float = 1.0,        # 无时延信息时的默认估计
        ema_alpha: float = 0.5,               # 历史 duration 的 EMA 更新系数
        use_duration_penalty: bool = True,    # 若无 duration，可设为 False 退化成纯公平队列
        **kwargs
    ):
        super().__init__(total_clients, clients_per_round, **kwargs)

        self.beta = float(target_participation)
        self.V = float(lyapunov_v)
        self.default_duration = float(default_duration)
        self.ema_alpha = float(ema_alpha)
        self.use_duration_penalty = bool(use_duration_penalty)

        # 论文中的虚拟公平队列 Z_t,n
        self.Z = np.zeros(total_clients, dtype=np.float64)

        # 当前框架下对 tau_hat 的简化在线估计：历史 duration 的 EMA
        self.tau_hat = np.full(total_clients, self.default_duration, dtype=np.float64)
        self.tau_obs_count = np.zeros(total_clients, dtype=np.int64)

        # 调试/可视化用
        self.last_selected = []
        self.last_objective = None

    # =========================================================
    # helper functions
    # =========================================================
    def _get_estimated_duration(
        self,
        client_id: int,
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> float:
        """
        选人阶段使用的时延估计。
        当前框架下优先使用历史 tau_hat；
        若没有观测过，则回退到 default_duration。
        """
        tau = float(self.tau_hat[client_id])

        # 如果不想使用 duration penalty，所有客户端视为相同 duration
        if not self.use_duration_penalty:
            return 1.0

        if not np.isfinite(tau) or tau <= 0:
            tau = self.default_duration

        return float(max(tau, 1e-6))

    def _solve_subproblem(
        self,
        feasible_clients: List[int],
        tau_max: float,
        k: int,
        tau_map: Dict[int, float],
    ):
        """
        对应论文 P4-SUB 的框架内版本：
        在 tau_i <= tau_max 的可行集合中，选 queue 最大的 k 个客户端。
        """
        qualified = [cid for cid in feasible_clients if tau_map[cid] <= tau_max]

        if len(qualified) < k:
            return None, None

        qualified.sort(key=lambda cid: self.Z[cid], reverse=True)
        chosen = qualified[:k]

        # 目标函数：V * tau_max - sum Z_i
        obj = self.V * tau_max - float(np.sum(self.Z[chosen]))
        return chosen, obj

    def _divide_and_conquer_select(
        self,
        available_clients: List[int],
        tau_map: Dict[int, float],
    ) -> List[int]:
        """
        对应论文 Algorithm 1 的当前框架版本。
        遍历可能的 max tau，求每个子问题最优解，再选目标最小者。
        """
        k = min(self.clients_per_round, len(available_clients))
        if k <= 0:
            return []

        best_obj = None
        best_selected = None

        # 所有可能的 max tau 候选
        candidate_nmax = sorted(
            available_clients,
            key=lambda cid: (tau_map[cid], -self.Z[cid], cid)
        )

        for cid_max in candidate_nmax:
            tau_max = tau_map[cid_max]
            chosen, obj = self._solve_subproblem(
                feasible_clients=available_clients,
                tau_max=tau_max,
                k=k,
                tau_map=tau_map,
            )

            if chosen is None:
                continue

            if (best_obj is None) or (obj < best_obj):
                best_obj = obj
                best_selected = chosen

        if best_selected is None:
            # 理论上不应发生；做一个稳妥回退
            fallback = sorted(available_clients, key=lambda cid: self.Z[cid], reverse=True)[:k]
            self.last_objective = None
            return fallback

        self.last_objective = float(best_obj)
        return list(best_selected)

    def _extract_duration_from_result(self, result: Dict[str, Any]) -> Optional[float]:
        """
        从训练结果中提取 observed duration。
        兼容多种字段名，但不要求框架必须提供。
        """
        if not isinstance(result, dict):
            return None

        raw = result.get("completion_time", None)
        if raw is None:
            raw = result.get("duration", None)
        if raw is None:
            raw = result.get("round_time", None)

        if raw is None:
            return None

        try:
            raw = float(raw)
        except Exception:
            return None

        if not np.isfinite(raw) or raw <= 0:
            return None

        return raw

    # =========================================================
    # main API
    # =========================================================
    def select(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> List[int]:
        if not available_clients:
            return []

        if len(available_clients) <= self.clients_per_round:
            selected = list(available_clients)
            self.last_selected = selected
            return selected

        tau_map = {
            cid: self._get_estimated_duration(cid, client_metrics)
            for cid in available_clients
        }

        selected = self._divide_and_conquer_select(
            available_clients=available_clients,
            tau_map=tau_map,
        )

        self.last_selected = list(selected)
        return selected

    def update(
        self,
        selected_clients: List[int],
        results: Dict[int, Dict[str, Any]]
    ):
        """
        1) 更新公平队列:
           Z_{t+1,n} = [Z_{t,n} + beta - x_{t,n}]^+
        2) 用本轮观测到的 duration 更新 tau_hat
        """
        selected_set = set(int(cid) for cid in selected_clients)

        # ---- update fairness queue for all clients ----
        x = np.zeros_like(self.Z)
        for cid in selected_set:
            if 0 <= cid < len(x):
                x[cid] = 1.0

        self.Z = np.maximum(self.Z + self.beta - x, 0.0)

        # ---- update duration estimate for selected clients ----
        for client_id, result in results.items():
            try:
                cid = int(client_id)
            except (ValueError, TypeError):
                continue

            if cid not in selected_set:
                continue

            duration = self._extract_duration_from_result(result)
            if duration is None:
                continue

            old_tau = float(self.tau_hat[cid])
            if (not np.isfinite(old_tau)) or old_tau <= 0:
                old_tau = self.default_duration

            if self.tau_obs_count[cid] == 0:
                new_tau = duration
            else:
                new_tau = self.ema_alpha * duration + (1.0 - self.ema_alpha) * old_tau

            self.tau_hat[cid] = float(max(new_tau, 1e-6))
            self.tau_obs_count[cid] += 1

        try:
            super().update(selected_clients, results)
        except Exception:
            pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "beta": self.beta,
            "V": self.V,
            "avg_queue": float(np.mean(self.Z)) if len(self.Z) > 0 else 0.0,
            "max_queue": float(np.max(self.Z)) if len(self.Z) > 0 else 0.0,
            "avg_tau_hat": float(np.mean(self.tau_hat)) if len(self.tau_hat) > 0 else 0.0,
            "last_objective": self.last_objective,
            "last_selected_count": len(self.last_selected),
        }