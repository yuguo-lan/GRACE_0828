"""
Power-of-Choice 选择算法（按当前实验框架对齐的 pow-d 版本）。

论文：
Cho et al., "Client Selection in Federated Learning: Convergence Analysis and
Power-of-Choice Selection Strategies", 2020.

当前框架下的实现原则：
1. 严格保留论文主干：
   - 从可用客户端中按 p_k 抽取大小为 d 的候选池
   - 使用当前全局模型下的 local loss
   - 从候选池中选择 loss 最高的 m 个客户端
2. 不改 Client / Server / Experiment
3. 直接复用 server 在选人前收集好的 client_metrics["loss"]
4. 若 data_size 不可得，则退化为均匀采样
5. 若部分客户端当前 loss 缺失，则优先使用有当前 loss 的客户端；不足时再随机补齐
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .base import BaseSelector, register_selector


@register_selector("poc")
@register_selector("power_of_choice")
class PowerOfChoiceSelector(BaseSelector):
    """
    Power-of-Choice (pow-d)

    论文版核心：
    1) sample candidate set A of size d without replacement by p_k
    2) estimate current local loss on A
    3) choose top-m highest loss clients from A

    在当前框架中：
    - step 2 已经由 server.collect_pre_selection_metrics() 完成
    - selector 只需要利用 client_metrics 中的当前 loss 即可
    """

    def __init__(
        self,
        total_clients: int,
        clients_per_round: int,
        d: int = 20,
        exploit_frac: float = 1.0,   # 仅为兼容旧配置保留，不实际使用
        **kwargs
    ):
        super().__init__(total_clients, clients_per_round, **kwargs)
        self.d = int(d)
        self.exploit_frac = float(exploit_frac)

    # =========================================================
    # helpers
    # =========================================================
    def _get_sampling_probs(
        self,
        candidate_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Optional[np.ndarray]:
        """
        按论文中的 p_k = D_k / sum_j D_j 构造候选池抽样概率。
        优先从 client_metrics["data_size"] 取；若不可得，则退化为均匀采样。
        """
        data_sizes = []

        for cid in candidate_clients:
            ds = None

            if client_metrics is not None and cid in client_metrics:
                ds = client_metrics[cid].get("data_size", None)

            if ds is None and hasattr(self, "client_info") and cid in getattr(self, "client_info", {}):
                ds = getattr(self.client_info[cid], "data_size", None)

            if ds is None:
                return None

            try:
                ds = float(ds)
            except Exception:
                return None

            if not np.isfinite(ds) or ds <= 0:
                ds = 1.0

            data_sizes.append(ds)

        probs = np.asarray(data_sizes, dtype=np.float64)
        s = probs.sum()

        if s <= 0:
            return None

        probs = probs / s
        return probs

    def _sample_candidate_pool(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        论文 step 1:
        Sample the candidate client set A of size d without replacement by p_k.
        """
        if len(available_clients) <= self.d:
            return list(available_clients)

        probs = self._get_sampling_probs(available_clients, client_metrics)

        if probs is None:
            sampled = np.random.choice(
                available_clients,
                size=self.d,
                replace=False
            ).tolist()
        else:
            sampled = np.random.choice(
                available_clients,
                size=self.d,
                replace=False,
                p=probs
            ).tolist()

        return [int(cid) for cid in sampled]

    def _get_current_loss(
        self,
        client_id: int,
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Optional[float]:
        """
        优先使用本轮 server 刚收集的当前 loss；
        若没有，再退化到缓存 last_loss。
        """
        loss = None

        if client_metrics is not None and client_id in client_metrics:
            loss = client_metrics[client_id].get("loss", None)

        if loss is None and hasattr(self, "client_info") and client_id in getattr(self, "client_info", {}):
            loss = getattr(self.client_info[client_id], "last_loss", None)

        if loss is None:
            return None

        try:
            loss = float(loss)
        except Exception:
            return None

        if not np.isfinite(loss):
            return None

        return loss

    # =========================================================
    # main API
    # =========================================================
    def select(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> List[int]:
        """
        论文对齐版 pow-d：
        1) 按 p_k 从 available_clients 中抽取候选池 A
        2) 读取 A 中客户端当前 local loss
        3) 选择 top-m highest loss clients
        """
        if not available_clients:
            return []

        if len(available_clients) <= self.clients_per_round:
            return list(available_clients)

        # 缓存 server 刚算好的当前 loss，方便其他地方可视化/调试
        if client_metrics is not None and hasattr(self, "client_info"):
            for cid, metrics in client_metrics.items():
                if cid in getattr(self, "client_info", {}) and "loss" in metrics:
                    try:
                        self.client_info[cid].last_loss = metrics["loss"]
                    except Exception:
                        pass

        # Step 1: sample candidate set A by p_k
        candidate_pool = self._sample_candidate_pool(
            available_clients=available_clients,
            client_metrics=client_metrics,
        )

        # Step 2: get current local losses on A
        candidate_losses = []
        missing_loss_clients = []

        for cid in candidate_pool:
            loss = self._get_current_loss(cid, client_metrics)
            if loss is None:
                missing_loss_clients.append(cid)
            else:
                candidate_losses.append((cid, loss))

        # 若一个有效 loss 都没有，直接随机回退
        if not candidate_losses:
            return np.random.choice(
                available_clients,
                size=self.clients_per_round,
                replace=False
            ).tolist()

        # Step 3: choose m highest-loss clients, ties broken randomly
        # 为了更接近论文里的“ties broken at random”，先打乱再排序
        np.random.shuffle(candidate_losses)
        candidate_losses.sort(key=lambda x: x[1], reverse=True)

        selected = [cid for cid, _ in candidate_losses[:self.clients_per_round]]

        # 若因为缺 loss 导致不足 m，则从剩余可用客户端随机补齐
        if len(selected) < self.clients_per_round:
            remaining = [cid for cid in available_clients if cid not in selected]
            if remaining:
                extra = np.random.choice(
                    remaining,
                    size=min(self.clients_per_round - len(selected), len(remaining)),
                    replace=False
                ).tolist()
                selected.extend(extra)

        return selected[:self.clients_per_round]