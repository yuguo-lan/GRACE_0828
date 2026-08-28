import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict, Any, Optional, Tuple

from .base import BaseSelector, register_selector


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class PPOAgent:
    """
    单动作 categorical PPO。
    FedPPO 论文是 loop-based：对同一个 state 多次采样，直到拿到 M 个不同客户端。
    因此 buffer 里按“每个被选客户端一条 transition”存。
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.8,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        update_epochs: int = 4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

    def get_probs(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state_t).squeeze(0).cpu().numpy()

        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if probs.sum() <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()
        return probs

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_states: np.ndarray,
    ):
        if len(states) == 0:
            return

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            values = self.critic(states_t).squeeze(-1)
            next_values = self.critic(next_states_t).squeeze(-1)

            deltas = rewards_t + self.gamma * (1.0 - dones_t) * next_values - values
            advantages = torch.zeros_like(deltas)

            gae = 0.0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones_t[t]) * gae
                advantages[t] = gae

            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            probs = self.actor(states_t)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_t)

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon
            ) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            values_pred = self.critic(states_t).squeeze(-1)
            critic_loss = nn.functional.mse_loss(values_pred, returns)

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()


@register_selector("fedppo")
class FedPPOSelector(BaseSelector):
    """
    FedPPO aligned to our framework.

    保留的论文核心：
    1) PCA + hierarchical clustering 做 noisy filtering / action space reduction
    2) state = [global_acc, acc_1, ..., acc_N]
    3) action dim = |s_clean|
    4) loop-based PPO sampling
    5) reward = alpha * local_acc + beta * global_acc * normalized_loss_drop

    框架化处理：
    1) 不实现 GMM 样本噪声率 refine（当前框架不稳定支持 sample-level losses）
    2) 没有独立 preprocessing stage，因此先 warmup 若干轮随机选客户端，
       利用这些轮次上传的本地模型做 clustering
    """

    def __init__(
        self,
        total_clients: int,
        clients_per_round: int,

        # -------- filtering --------
        pca_components: int = 10,
        n_clusters: int = 5,
        prefilter_min_seen_ratio: float = 1.0,
        prefilter_max_warmup_rounds: int = 10,

        # -------- PPO --------
        hidden_dim: int = 64,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.8,          # 论文实验设置 gamma = 0.8
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        update_interval: int = 5,    # T_step
        update_epochs: int = 4,

        # -------- reward --------
        reward_alpha: float = 0.6,
        reward_beta: float = 0.4,

        **kwargs
    ):
        super().__init__(total_clients, clients_per_round, **kwargs)

        self.pca_components = pca_components
        self.n_clusters = n_clusters
        self.prefilter_min_seen_ratio = prefilter_min_seen_ratio
        self.prefilter_max_warmup_rounds = prefilter_max_warmup_rounds

        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.update_interval = update_interval
        self.update_epochs = update_epochs

        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta

        # state = [Acc_g, Acc_1, ..., Acc_N]
        self.state_dim = total_clients + 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.ppo: Optional[PPOAgent] = None

        # ---------- state history ----------
        self.global_accuracy = 0.0
        self.client_acc_history: Dict[int, float] = {}
        self.client_loss_history: Dict[int, float] = {}

        # ---------- prefilter bootstrap ----------
        self.prefilter_done = False
        self.warmup_rounds = 0
        self.bootstrap_features: Dict[int, np.ndarray] = {}
        self.bootstrap_acc: Dict[int, float] = {}

        self.noisy_client_set = set()
        self.clean_client_list = list(range(total_clients))
        self.clean_id_to_local: Dict[int, int] = {}
        self.local_to_clean_id: Dict[int, int] = {}

        # ---------- PPO buffer ----------
        self.buffer_states: List[np.ndarray] = []
        self.buffer_actions: List[int] = []
        self.buffer_log_probs: List[float] = []
        self.buffer_rewards: List[float] = []
        self.buffer_dones: List[bool] = []
        self.buffer_next_states: List[np.ndarray] = []

        # 上一轮 select() 产生的信息，等 update() 时回填 reward
        self.last_state: Optional[np.ndarray] = None
        self.last_selected_clients: Optional[List[int]] = None
        self.last_selected_action_idx: Optional[List[int]] = None
        self.last_log_probs: Optional[np.ndarray] = None
        self.last_used_ppo: bool = False

        self.finished_rounds = 0
        self.ppo_rounds = 0

    # =========================================================
    # utilities
    # =========================================================
    def _build_state(self) -> np.ndarray:
        state = np.zeros(self.total_clients + 1, dtype=np.float32)
        state[0] = float(self.global_accuracy)
        for cid, acc in self.client_acc_history.items():
            if 0 <= cid < self.total_clients:
                state[cid + 1] = float(acc)
        return state

    def _extract_feature_vector(self, result: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        尽量贴论文：
        优先使用 local model params 中的卷积层参数；
        若不是 CNN，则退化到较高维模型参数；
        再不行退化到 update_vector。
        """
        params = result.get("params")
        if isinstance(params, dict):
            pieces = []

            # 优先取卷积层 / 4D tensor
            for name, tensor in params.items():
                if not torch.is_tensor(tensor):
                    continue
                if tensor.ndim == 4 or "conv" in name.lower():
                    pieces.append(tensor.detach().cpu().reshape(-1).numpy())

            # 没有卷积层时，退化到二维及以上参数
            if not pieces:
                for _, tensor in params.items():
                    if not torch.is_tensor(tensor):
                        continue
                    if tensor.ndim >= 2:
                        pieces.append(tensor.detach().cpu().reshape(-1).numpy())

            # 再退化到全部参数
            if not pieces:
                for _, tensor in params.items():
                    if torch.is_tensor(tensor):
                        pieces.append(tensor.detach().cpu().reshape(-1).numpy())

            if pieces:
                vec = np.concatenate(pieces, axis=0).astype(np.float32)
                return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        update_vector = result.get("update_vector")
        if torch.is_tensor(update_vector):
            vec = update_vector.detach().cpu().numpy().astype(np.float32)
            return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(update_vector, np.ndarray):
            vec = update_vector.astype(np.float32)
            return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return None

    def _prepare_feature_matrix(
        self,
        client_ids: List[int],
        feature_dict: Dict[int, np.ndarray],
    ) -> Tuple[List[int], np.ndarray]:
        valid_ids = [cid for cid in client_ids if cid in feature_dict]
        if len(valid_ids) < 2:
            return valid_ids, np.empty((0, 0), dtype=np.float32)

        vecs = [feature_dict[cid] for cid in valid_ids]
        min_len = min(len(v) for v in vecs)
        if min_len <= 1:
            return valid_ids, np.empty((0, 0), dtype=np.float32)

        X = np.stack([v[:min_len] for v in vecs], axis=0)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return valid_ids, X

    def _run_hierarchical_filter(self):
        """
        论文 noisy filtering 的框架内版本：
        PCA + hierarchical clustering + lowest-average-accuracy cluster
        """
        observed_ids = list(self.bootstrap_features.keys())
        valid_ids, X = self._prepare_feature_matrix(observed_ids, self.bootstrap_features)

        if len(valid_ids) < 2:
            self.noisy_client_set = set()
            self.clean_client_list = list(range(self.total_clients))
            return

        n_comp = min(self.pca_components, X.shape[0], X.shape[1])
        if n_comp < 2:
            reduced = X
        else:
            reduced = PCA(n_components=n_comp).fit_transform(X)
            reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)

        n_clusters = min(self.n_clusters, len(valid_ids))
        if n_clusters < 2:
            self.noisy_client_set = set()
            self.clean_client_list = list(range(self.total_clients))
            return

        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(reduced)

        cluster_acc = {}
        for cid, label in zip(valid_ids, labels):
            acc = self.bootstrap_acc.get(cid, 0.0)
            cluster_acc.setdefault(label, []).append(acc)

        avg_cluster_acc = {
            label: float(np.mean(accs))
            for label, accs in cluster_acc.items()
        }
        noisy_label = min(avg_cluster_acc, key=avg_cluster_acc.get)

        self.noisy_client_set = {
            cid for cid, label in zip(valid_ids, labels)
            if label == noisy_label
        }

        self.clean_client_list = [
            cid for cid in range(self.total_clients)
            if cid not in self.noisy_client_set
        ]

        if len(self.clean_client_list) == 0:
            self.clean_client_list = list(range(self.total_clients))
            self.noisy_client_set = set()

    def _finalize_prefilter(self):
        self._run_hierarchical_filter()
        self.prefilter_done = True

        self.clean_id_to_local = {cid: i for i, cid in enumerate(self.clean_client_list)}
        self.local_to_clean_id = {i: cid for i, cid in enumerate(self.clean_client_list)}

        self.ppo = PPOAgent(
            state_dim=self.state_dim,
            action_dim=len(self.clean_client_list),
            hidden_dim=self.hidden_dim,
            lr_actor=self.lr_actor,
            lr_critic=self.lr_critic,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            update_epochs=self.update_epochs,
            device=self.device,
        )

    def _should_finalize_prefilter(self) -> bool:
        seen_ratio = len(self.bootstrap_features) / max(self.total_clients, 1)
        enough_seen = seen_ratio >= self.prefilter_min_seen_ratio
        hit_max_warmup = self.warmup_rounds >= self.prefilter_max_warmup_rounds
        return enough_seen or hit_max_warmup

    def _sample_without_replacement_from_policy(
        self,
        state: np.ndarray,
        available_clients: List[int],
    ) -> Tuple[List[int], List[int], np.ndarray]:
        """
        在 clean action space 上做 loop-based sampling。
        返回：
        - selected client ids
        - selected local action indices
        - selected log_probs
        """
        assert self.ppo is not None
        probs = self.ppo.get_probs(state)

        masked = probs.copy()
        available_clean = {cid for cid in available_clients if cid in self.clean_id_to_local}

        for cid, local_idx in self.clean_id_to_local.items():
            if cid not in available_clean:
                masked[local_idx] = 0.0

        selected_clients = []
        selected_actions = []
        selected_log_probs = []

        num_need = min(self.clients_per_round, len(available_clients))

        for _ in range(min(num_need, len(available_clean))):
            tmp = masked.copy()
            for a in selected_actions:
                tmp[a] = 0.0

            if tmp.sum() <= 1e-12:
                break

            tmp = tmp / tmp.sum()
            a = int(np.random.choice(len(tmp), p=tmp))
            cid = self.local_to_clean_id[a]

            selected_clients.append(cid)
            selected_actions.append(a)
            selected_log_probs.append(float(np.log(tmp[a] + 1e-12)))

        # clean available 不足 K 时，随机补齐
        if len(selected_clients) < num_need:
            remain = [cid for cid in available_clients if cid not in selected_clients]
            if remain:
                extra = np.random.choice(
                    remain,
                    size=min(num_need - len(selected_clients), len(remain)),
                    replace=False,
                ).tolist()
                selected_clients.extend(extra)
                selected_actions.extend([-1] * len(extra))
                selected_log_probs.extend([0.0] * len(extra))

        return selected_clients, selected_actions, np.asarray(selected_log_probs, dtype=np.float32)

    def _compute_round_rewards(
        self,
        selected_clients: List[int],
        results: Dict[int, Dict[str, Any]],
        global_acc: float,
        prev_loss_map: Dict[int, float],
    ) -> List[float]:
        """
        论文奖励：
        r_t^m = alpha * Acc_t^m + beta * Acc_t^g * (|Δloss_t^m| / sum_m |Δloss_t^m|)

        这里 Δloss 用“正向 loss 降幅率”，loss 上升记为 0。
        """
        local_accs = []
        loss_drop_rates = []

        for cid in selected_clients:
            res = results.get(cid, {})
            local_acc = float(res.get("accuracy", self.client_acc_history.get(cid, 0.0)))
            cur_loss = res.get("loss", None)
            prev_loss = prev_loss_map.get(cid, None)

            if cur_loss is None or prev_loss is None or prev_loss <= 1e-12:
                drop_rate = 0.0
            else:
                drop_rate = max(float(prev_loss) - float(cur_loss), 0.0) / max(float(prev_loss), 1e-12)

            local_accs.append(local_acc)
            loss_drop_rates.append(drop_rate)

        denom = float(np.sum(loss_drop_rates))
        rewards = []

        for local_acc, drop_rate in zip(local_accs, loss_drop_rates):
            contrib = (drop_rate / denom) if denom > 1e-12 else 0.0
            reward = self.reward_alpha * local_acc + self.reward_beta * global_acc * contrib
            rewards.append(float(reward))

        return rewards

    # =========================================================
    # main API
    # =========================================================
    def select(
        self,
        available_clients: List[int],
        client_metrics: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[int]:
        # prefilter 阶段：随机采样，收集本地模型
        if not self.prefilter_done:
            selected = np.random.choice(
                available_clients,
                size=min(self.clients_per_round, len(available_clients)),
                replace=False,
            ).tolist()

            self.last_state = self._build_state()
            self.last_selected_clients = selected
            self.last_selected_action_idx = [-1] * len(selected)
            self.last_log_probs = np.zeros(len(selected), dtype=np.float32)
            self.last_used_ppo = False
            return selected

        # PPO 阶段
        state = self._build_state()
        selected, selected_action_idx, log_probs = self._sample_without_replacement_from_policy(
            state=state,
            available_clients=available_clients,
        )

        self.last_state = state
        self.last_selected_clients = selected
        self.last_selected_action_idx = selected_action_idx
        self.last_log_probs = log_probs
        self.last_used_ppo = True
        return selected

    def update(
        self,
        selected_clients: List[int],
        results: Dict[int, Dict[str, Any]],
    ):
        """
        results 约定：
        {
            cid: {
                "params": ...,
                "loss": ...,
                "accuracy": ...   # 服务器统一评估本地模型得到
            },
            "__meta__": {
                "global_accuracy": ...,
                "global_loss": ...
            }
        }
        """
        meta = results.get("__meta__", {})
        meta_global_acc = meta.get("global_accuracy", None)
        current_global_acc = self.global_accuracy if meta_global_acc is None else float(meta_global_acc)

        # 先保存“更新前”的 loss 历史，reward 要用它
        prev_loss_snapshot = self.client_loss_history.copy()

        # ---------- preprocessing cache ----------
        if not self.prefilter_done:
            for cid in selected_clients:
                res = results.get(cid, {})
                feat = self._extract_feature_vector(res)
                if feat is not None:
                    self.bootstrap_features[cid] = feat

                if "accuracy" in res and res["accuracy"] is not None:
                    self.bootstrap_acc[cid] = float(res["accuracy"])

            self.warmup_rounds += 1
            if self._should_finalize_prefilter():
                self._finalize_prefilter()

        # ---------- 先构造 next_state preview，用于 PPO transition ----------
        next_state_preview = self._build_state().copy()
        next_state_preview[0] = current_global_acc
        for cid in selected_clients:
            res = results.get(cid, {})
            if "accuracy" in res and res["accuracy"] is not None and 0 <= cid < self.total_clients:
                next_state_preview[cid + 1] = float(res["accuracy"])

        # ---------- PPO buffer ----------
        if self.prefilter_done and self.last_used_ppo and self.last_selected_clients is not None:
            round_rewards = self._compute_round_rewards(
                selected_clients=self.last_selected_clients,
                results=results,
                global_acc=current_global_acc,
                prev_loss_map=prev_loss_snapshot,
            )

            for cid, a_idx, logp, reward in zip(
                self.last_selected_clients,
                self.last_selected_action_idx,
                self.last_log_probs,
                round_rewards,
            ):
                # 只有真正来自 actor 的 clean-space 动作才进入 buffer
                if a_idx is None or a_idx < 0:
                    continue

                self.buffer_states.append(self.last_state.copy())
                self.buffer_actions.append(int(a_idx))
                self.buffer_log_probs.append(float(logp))
                self.buffer_rewards.append(float(reward))
                self.buffer_dones.append(False)
                self.buffer_next_states.append(next_state_preview.copy())

            if self.ppo is not None and len(self.buffer_states) > 0:
                self.ppo_rounds += 1
                if self.ppo_rounds % self.update_interval == 0:
                    self.ppo.update(
                        states=np.asarray(self.buffer_states, dtype=np.float32),
                        actions=np.asarray(self.buffer_actions, dtype=np.int64),
                        old_log_probs=np.asarray(self.buffer_log_probs, dtype=np.float32),
                        rewards=np.asarray(self.buffer_rewards, dtype=np.float32),
                        dones=np.asarray(self.buffer_dones, dtype=np.float32),
                        next_states=np.asarray(self.buffer_next_states, dtype=np.float32),
                    )
                    self.buffer_states.clear()
                    self.buffer_actions.clear()
                    self.buffer_log_probs.clear()
                    self.buffer_rewards.clear()
                    self.buffer_dones.clear()
                    self.buffer_next_states.clear()

        # ---------- 再更新 state history ----------
        for cid in selected_clients:
            res = results.get(cid, {})

            if "accuracy" in res and res["accuracy"] is not None:
                self.client_acc_history[cid] = float(res["accuracy"])

            if "loss" in res and res["loss"] is not None:
                self.client_loss_history[cid] = float(res["loss"])

        self.global_accuracy = current_global_acc
        self.finished_rounds += 1

        try:
            super().update(selected_clients, results)
        except Exception:
            pass