"""
Cluster-Based 选择算法（按当前实验框架对齐的 MBUT-CS 版本）。

论文：
Zhao et al., "A Cluster-Based Client Selection Model for Federated Learning With Heterogeneous Clients", 2026.

当前框架下的实现原则：
1. 保留论文核心：更新方向特征提取 + 聚类 + balanced/tilted 配额分配 + 簇内 MD 抽样
2. 不强行实现 Load-Ada：
   因为当前框架没有稳定提供资源特征、时间消耗、任务量这些输入
3. 不改 Client / Server / Experiment，只改算法文件
4. 尽量兼容已有参数名，避免 run.py / 配置失效

与论文的对应关系：
- fi = (fi_class, fi_weight)
- fi_class：若框架提供 output_layer_grads，则近似按论文方式构造；否则退化为均匀分布
- fi_weight：使用本轮/最近一次本地模型更新向量，经 PCA 降维
- 聚类：论文用 PAM；这里继续使用 KMeans 近似，避免新增依赖
- 选人：先找 balanced / tilted clusters，再分配配额，再在簇内按样本量 MD 抽样
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Optional

from .base import BaseSelector, register_selector


@register_selector("mbut_cs")
class ClusterBasedSelector(BaseSelector):
    def __init__(
        self,
        total_clients: int,
        clients_per_round: int,

        # ---- 兼容保留：当前框架下不用 Load-Ada ----
        linucb_delta: float = 0.5,
        max_wait_time: float = 10.0,

        # ---- MBUT-CS 核心参数 ----
        pca_components: int = 10,
        num_classes: int = 10,
        n_clusters: int = 10,
        n_balanced: int = 2,
        quota_sigma: float = 0.1,
        tau: float = 1.0,   # 论文中 class distribution 的归一化超参数

        **kwargs
    ):
        super().__init__(total_clients, clients_per_round, **kwargs)

        # 保留参数名，避免外部配置报错；当前框架下不实际使用
        self.linucb_delta = linucb_delta
        self.max_wait_time = max_wait_time

        self.pca_components = pca_components
        self.num_classes = num_classes
        self.n_clusters = n_clusters
        self.n_balanced = n_balanced
        self.quota_sigma = quota_sigma
        self.tau = tau

        # 论文中的 beta = K / n
        self.beta = clients_per_round / max(total_clients, 1)

        # 保存每个客户端最近一次“原始更新向量”
        self.raw_update_cache: Dict[int, np.ndarray] = {}

        # 保存每个客户端最近一次“方向特征”
        self.client_direction: Dict[int, np.ndarray] = {}

        # 是否已初始化所有客户端方向
        self._direction_initialized = False

    # =========================================================
    # feature extraction
    # =========================================================
    def _init_direction_features(self):
        """
        对应论文 Algorithm 2 第 2-5 行：
        t=1 时，fi_class 初始化为 1/c，fi_weight 初始化为 0
        """
        init_class = np.ones(self.num_classes, dtype=np.float32) / max(self.num_classes, 1)
        init_weight = np.zeros(self.pca_components, dtype=np.float32)
        init_feature = np.concatenate([init_class, init_weight], axis=0)

        for cid in range(self.total_clients):
            self.client_direction[cid] = init_feature.copy()
            self.raw_update_cache[cid] = np.zeros(1, dtype=np.float32)

        self._direction_initialized = True

    def _flatten_delta(self, delta: Any) -> Optional[np.ndarray]:
        """
        从 results[cid]["delta"] 中提取一维更新向量
        """
        if isinstance(delta, dict):
            pieces = []
            for _, tensor in delta.items():
                if torch.is_tensor(tensor):
                    pieces.append(tensor.detach().cpu().reshape(-1).float().numpy())
                elif isinstance(tensor, np.ndarray):
                    pieces.append(tensor.reshape(-1).astype(np.float32))
            if pieces:
                vec = np.concatenate(pieces, axis=0).astype(np.float32)
                return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        if torch.is_tensor(delta):
            vec = delta.detach().cpu().reshape(-1).float().numpy()
            return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        if isinstance(delta, np.ndarray):
            vec = delta.reshape(-1).astype(np.float32)
            return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return None

    def _extract_raw_update_vector(self, result: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        优先使用 update_vector，其次用 delta，再次退化到 params 扁平化。
        """
        update_vector = result.get("update_vector")
        if torch.is_tensor(update_vector):
            vec = update_vector.detach().cpu().reshape(-1).float().numpy()
            return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(update_vector, np.ndarray):
            vec = update_vector.reshape(-1).astype(np.float32)
            return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        delta = result.get("delta")
        vec = self._flatten_delta(delta)
        if vec is not None:
            return vec

        params = result.get("params")
        if isinstance(params, dict):
            pieces = []
            for _, tensor in params.items():
                if torch.is_tensor(tensor):
                    pieces.append(tensor.detach().cpu().reshape(-1).float().numpy())
                elif isinstance(tensor, np.ndarray):
                    pieces.append(tensor.reshape(-1).astype(np.float32))
            if pieces:
                vec = np.concatenate(pieces, axis=0).astype(np.float32)
                return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return None

    def _extract_class_feature(self, result_or_metrics: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        论文里 fi_class 来自辅助数据集上的输出层梯度，再近似得到类别分布。
        当前框架默认拿不到这个信息，因此：
        - 若提供 output_layer_grads，则按论文公式思想近似
        - 否则退化为均匀分布
        """
        uniform = np.ones(self.num_classes, dtype=np.float32) / max(self.num_classes, 1)

        if not result_or_metrics:
            return uniform

        grad_out = result_or_metrics.get("output_layer_grads", None)
        if grad_out is None:
            return uniform

        try:
            norms = []
            for g in grad_out:
                if g is None:
                    norms.append(0.0)
                elif torch.is_tensor(g):
                    norms.append(float(torch.norm(g.detach()).cpu().item()))
                else:
                    arr = np.asarray(g, dtype=np.float32)
                    if not np.isfinite(arr).all():
                        norms.append(0.0)
                    else:
                        norms.append(float(np.linalg.norm(arr)))

            if len(norms) != self.num_classes:
                return uniform

            logits = self.tau * (np.asarray(norms, dtype=np.float32) ** 2)
            logits = logits - np.max(logits)
            exp_vals = np.exp(logits)
            dist = exp_vals / (np.sum(exp_vals) + 1e-12)
            dist = np.nan_to_num(dist, nan=1.0 / self.num_classes, posinf=0.0, neginf=0.0)

            if dist.sum() <= 0:
                return uniform
            return dist / dist.sum()
        except Exception:
            return uniform

    def _fit_pca_and_transform(
        self,
        client_ids: List[int],
    ) -> Dict[int, np.ndarray]:
        """
        对已有 raw update cache 做 PCA 降维，返回每个客户端的 fi_weight
        """
        vectors = []
        valid_ids = []

        for cid in client_ids:
            vec = self.raw_update_cache.get(cid, None)
            if vec is None:
                continue
            if not isinstance(vec, np.ndarray):
                continue
            if vec.size == 0:
                continue
            if not np.isfinite(vec).all():
                continue

            valid_ids.append(cid)
            vectors.append(vec)

        if not vectors:
            return {cid: np.zeros(self.pca_components, dtype=np.float32) for cid in client_ids}

        min_len = min(len(v) for v in vectors)
        if min_len <= 1:
            return {cid: np.zeros(self.pca_components, dtype=np.float32) for cid in client_ids}

        X = np.stack([v[:min_len] for v in vectors], axis=0).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 样本太少时，直接取前 pca_components 维 / padding
        if X.shape[0] < 2:
            out = {}
            for cid, row in zip(valid_ids, X):
                feat = row[:self.pca_components]
                if feat.shape[0] < self.pca_components:
                    feat = np.pad(feat, (0, self.pca_components - feat.shape[0]), mode="constant")
                out[cid] = feat.astype(np.float32)
            for cid in client_ids:
                out.setdefault(cid, np.zeros(self.pca_components, dtype=np.float32))
            return out

        n_comp = min(self.pca_components, X.shape[0], X.shape[1])
        if n_comp < 1:
            return {cid: np.zeros(self.pca_components, dtype=np.float32) for cid in client_ids}

        pca = PCA(n_components=n_comp)
        X_red = pca.fit_transform(X)
        X_red = np.nan_to_num(X_red, nan=0.0, posinf=0.0, neginf=0.0)

        out = {}
        for cid, row in zip(valid_ids, X_red):
            feat = row.astype(np.float32)
            if feat.shape[0] < self.pca_components:
                feat = np.pad(feat, (0, self.pca_components - feat.shape[0]), mode="constant")
            out[cid] = feat

        for cid in client_ids:
            out.setdefault(cid, np.zeros(self.pca_components, dtype=np.float32))

        return out

    def _rebuild_direction_features(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        """
        每轮 select 前重建可用客户端的方向特征：
        fi = (fi_class, fi_weight)
        """
        if not self._direction_initialized:
            self._init_direction_features()

        weight_features = self._fit_pca_and_transform(available_clients)

        for cid in available_clients:
            metrics = client_metrics.get(cid, {}) if client_metrics else {}
            class_feature = self._extract_class_feature(metrics)
            weight_feature = weight_features.get(cid, np.zeros(self.pca_components, dtype=np.float32))
            fused = np.concatenate([class_feature, weight_feature], axis=0)
            fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
            self.client_direction[cid] = fused.astype(np.float32)

    # =========================================================
    # clustering + quota allocation
    # =========================================================
    def _cluster_features(
        self,
        cids: List[int],
        feat_matrix: np.ndarray,
        n_clusters: int,
    ):
        """
        论文用 PAM；这里用 KMeans 近似，保持现有依赖不变。
        """
        if feat_matrix.shape[0] == 0:
            return np.array([], dtype=int), np.empty((0, 0), dtype=np.float32), []

        mask = np.isfinite(feat_matrix).all(axis=1)
        filtered_feats = feat_matrix[mask]
        filtered_cids = [cid for cid, ok in zip(cids, mask) if ok]

        if filtered_feats.shape[0] == 0:
            return np.array([], dtype=int), np.empty((0, 0), dtype=np.float32), []

        if filtered_feats.shape[0] == 1:
            return np.array([0], dtype=int), filtered_feats.copy(), filtered_cids

        unique_rows = np.unique(filtered_feats, axis=0)
        effective_clusters = min(n_clusters, filtered_feats.shape[0], unique_rows.shape[0])
        if effective_clusters < 1:
            effective_clusters = 1

        kmeans = KMeans(n_clusters=effective_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(filtered_feats)
        centers = kmeans.cluster_centers_.astype(np.float32)

        return labels, centers, filtered_cids

    def _allocate_cluster_quotas(
        self,
        cluster_sizes: np.ndarray,
        balanced_labels: List[int],
        tilted_labels: List[int],
        total_quota: int,
        total_clients: int,
    ) -> Dict[int, int]:
        """
        按论文公式 (15)(16)(17)(18) 的思想做整数配额分配。
        """
        quotas = {j: 0 for j in range(len(cluster_sizes))}
        if total_quota <= 0 or total_clients <= 0:
            return quotas

        sum_balanced = int(sum(cluster_sizes[j] for j in balanced_labels))
        sum_tilted = total_clients - sum_balanced

        # 论文：Ybal = min(floor(beta*sigma*n + beta*(1-sigma)*sum_balanced), sum_balanced)
        # 这里 beta*n = total_quota
        y_bal_float = total_quota * self.quota_sigma + total_quota * (1.0 - self.quota_sigma) * (
            sum_balanced / max(total_clients, 1)
        )
        Y_bal = min(int(np.floor(y_bal_float)), sum_balanced)
        Y_tilt = total_quota - Y_bal

        def allocate_subset(label_list: List[int], subset_total: int, subset_size: int):
            if subset_total <= 0 or subset_size <= 0 or not label_list:
                return {}, 0

            exact = []
            for j in label_list:
                q = subset_total * (cluster_sizes[j] / max(subset_size, 1))
                exact.append((j, q))

            alloc = {j: int(np.floor(q)) for j, q in exact}
            used = sum(alloc.values())
            remain = subset_total - used

            # 按小数部分从大到小补齐
            residues = sorted(
                [(j, q - np.floor(q)) for j, q in exact],
                key=lambda x: x[1],
                reverse=True
            )

            idx = 0
            while remain > 0 and residues:
                j = residues[idx % len(residues)][0]
                if alloc[j] < cluster_sizes[j]:
                    alloc[j] += 1
                    remain -= 1
                idx += 1
                if idx > 10 * len(residues) and remain > 0:
                    break

            # 截断到 cluster size
            for j in alloc:
                alloc[j] = min(alloc[j], int(cluster_sizes[j]))

            return alloc, sum(alloc.values())

        bal_alloc, bal_used = allocate_subset(balanced_labels, Y_bal, sum_balanced)
        tilt_alloc, tilt_used = allocate_subset(tilted_labels, Y_tilt, sum_tilted)

        quotas.update(bal_alloc)
        quotas.update(tilt_alloc)

        allocated = bal_used + tilt_used

        # 若因 floor / 容量限制仍未达到 total_quota，则全局补齐
        if allocated < total_quota:
            order = balanced_labels + tilted_labels
            ptr = 0
            guard = 0
            while allocated < total_quota and order and guard < 10000:
                j = order[ptr % len(order)]
                if quotas[j] < cluster_sizes[j]:
                    quotas[j] += 1
                    allocated += 1
                ptr += 1
                guard += 1

        return quotas

    def _sample_clients_from_cluster(
        self,
        cluster_cids: List[int],
        q: int,
    ) -> List[int]:
        if q <= 0 or not cluster_cids:
            return []

        if len(cluster_cids) <= q:
            return list(cluster_cids)

        data_sizes = []
        for cid in cluster_cids:
            if cid in getattr(self, "client_info", {}):
                ds = getattr(self.client_info[cid], "data_size", 1)
            else:
                ds = 1
            ds = max(float(ds), 1.0)
            data_sizes.append(ds)

        probs = np.asarray(data_sizes, dtype=np.float64)
        probs = probs / probs.sum()

        chosen = np.random.choice(
            cluster_cids,
            size=q,
            replace=False,
            p=probs,
        ).tolist()
        return chosen

    def _select_clients_mbut(
        self,
        available_clients: List[int],
    ) -> List[int]:
        if not available_clients:
            return []

        valid_cids = []
        valid_feats = []
        for cid in available_clients:
            feat = self.client_direction.get(cid, None)
            if feat is None:
                continue
            feat = np.asarray(feat, dtype=np.float32)
            if not np.isfinite(feat).all():
                continue
            valid_cids.append(cid)
            valid_feats.append(feat)

        total_quota = min(self.clients_per_round, len(available_clients))
        if len(valid_cids) <= total_quota:
            return list(valid_cids)

        feat_matrix = np.stack(valid_feats, axis=0)
        labels, centers, clustered_cids = self._cluster_features(
            valid_cids,
            feat_matrix,
            min(self.n_clusters, len(valid_cids))
        )

        if len(clustered_cids) == 0:
            return np.random.choice(
                available_clients,
                size=total_quota,
                replace=False
            ).tolist()

        n_clu = int(labels.max()) + 1 if labels.size > 0 else 1
        cluster_sizes = np.bincount(labels, minlength=n_clu)

        # 论文公式 (14)：balance point = sum_j (m_j / n) * mu_j
        total_clients = len(clustered_cids)
        balance_point = np.zeros(centers.shape[1], dtype=np.float32)
        for j in range(n_clu):
            if cluster_sizes[j] > 0:
                balance_point += (cluster_sizes[j] / total_clients) * centers[j]

        # offset = squared Euclidean distance to balance point
        offsets = []
        for j in range(n_clu):
            if cluster_sizes[j] > 0:
                off = float(np.sum((centers[j] - balance_point) ** 2))
            else:
                off = float("inf")
            offsets.append((j, off))

        offsets.sort(key=lambda x: x[1])

        h = min(self.n_balanced, n_clu)
        balanced_labels = [lab for lab, _ in offsets[:h]]
        tilted_labels = [lab for lab, _ in offsets[h:]]

        quotas = self._allocate_cluster_quotas(
            cluster_sizes=cluster_sizes,
            balanced_labels=balanced_labels,
            tilted_labels=tilted_labels,
            total_quota=total_quota,
            total_clients=total_clients,
        )

        selected = []
        for j in range(n_clu):
            q = quotas.get(j, 0)
            if q <= 0:
                continue
            idx = np.where(labels == j)[0]
            cluster_cids = [clustered_cids[i] for i in idx]
            selected.extend(self._sample_clients_from_cluster(cluster_cids, q))

        # 去重并补齐
        selected = list(dict.fromkeys(selected))
        if len(selected) < total_quota:
            remain = [cid for cid in available_clients if cid not in selected]
            if remain:
                extra = np.random.choice(
                    remain,
                    size=min(total_quota - len(selected), len(remain)),
                    replace=False,
                ).tolist()
                selected.extend(extra)

        return selected[:total_quota]

    # =========================================================
    # main API
    # =========================================================
    def select(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> List[int]:
        if not self._direction_initialized:
            self._init_direction_features()

        if len(available_clients) <= self.clients_per_round:
            return list(available_clients)

        # 每轮基于“历史缓存的 raw updates + 当前可选客户端集合”重建方向特征
        self._rebuild_direction_features(
            available_clients=available_clients,
            client_metrics=client_metrics,
        )

        selected = self._select_clients_mbut(available_clients)
        if not selected:
            selected = np.random.choice(
                available_clients,
                size=self.clients_per_round,
                replace=False
            ).tolist()

        return selected

    def update(
        self,
        selected_clients: List[int],
        results: Dict[int, Dict[str, Any]],
    ):
        """
        当前框架下，update 负责：
        1) 从 selected clients 的训练结果里缓存 raw update vector
        2) 可选缓存 output_layer_grads（如果以后框架支持）
        3) 调用 BaseSelector.update 维护通用状态
        """
        if not self._direction_initialized:
            self._init_direction_features()

        for cid in selected_clients:
            res = results.get(cid, {})
            raw_vec = self._extract_raw_update_vector(res)
            if raw_vec is not None and raw_vec.size > 0:
                self.raw_update_cache[cid] = raw_vec.astype(np.float32)

        try:
            super().update(selected_clients, results)
        except Exception:
            pass