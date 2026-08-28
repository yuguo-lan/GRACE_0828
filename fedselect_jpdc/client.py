import copy
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Client:
    """
    selection-only 版本客户端

    返回给选择器/服务器的核心统计量：
    - loss
    - prev_loss
    - loss_drop
    - gradient_norm
    - data_size
    - delta
    - update_vector
    """

    def __init__(
        self,
        client_id: int,
        dataset,
        model_fn,
        device,
    ):
        self.id = client_id
        self.dataset = dataset
        self.model_fn = model_fn
        self.device = device

        self.num_samples = len(dataset)

        # 上一轮缓存
        self.last_loss: Optional[float] = None
        self.last_accuracy: Optional[float] = None
        self.last_grad_norm: Optional[float] = None

    def train(
        self,
        global_model_params: Dict[str, torch.Tensor],
        lr: float,
        epochs: int,
        batch_size: int,
        criterion=nn.CrossEntropyLoss(),
        return_update_vector: bool = True,
    ) -> Dict[str, Any]:
        if self.num_samples == 0:
            zero_delta = {
                name: torch.zeros_like(param).detach().cpu()
                for name, param in global_model_params.items()
            }
            result = {
                "params": {
                    name: param.detach().cpu().clone()
                    for name, param in global_model_params.items()
                },
                "delta": zero_delta,
                "loss": 0.0,
                "prev_loss": self.last_loss,
                "loss_drop": 0.0 if self.last_loss is not None else None,
                "gradient_norm": 0.0,
                "data_size": 0,
            }
            if return_update_vector:
                result["update_vector"] = self._flatten_state_dict(zero_delta)
            return result

        model = self.model_fn().to(self.device)
        model.load_state_dict(global_model_params)

        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        model.train()
        total_loss = 0.0
        num_batches = 0

        grad_norm_sum = 0.0
        grad_norm_steps = 0

        for _ in range(epochs):
            for data, target in loader:
                data = data.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                batch_grad_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach().data.norm(2).item()
                        batch_grad_sq += g ** 2
                batch_grad_norm = batch_grad_sq ** 0.5
                grad_norm_sum += batch_grad_norm
                grad_norm_steps += 1

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_grad_norm = grad_norm_sum / max(grad_norm_steps, 1)

        local_state = model.state_dict()

        delta = {}
        for name in local_state:
            delta[name] = (
                local_state[name].detach().cpu() - global_model_params[name].detach().cpu()
            )

        result = {
            "params": {
                name: tensor.detach().cpu().clone()
                for name, tensor in local_state.items()
            },
            "delta": delta,
            "loss": avg_loss,
            "prev_loss": self.last_loss,
            "loss_drop": (self.last_loss - avg_loss) if self.last_loss is not None else None,
            "gradient_norm": avg_grad_norm,
            "data_size": self.num_samples,
        }

        if return_update_vector:
            result["update_vector"] = self._flatten_state_dict(delta)

        self.last_loss = avg_loss
        self.last_grad_norm = avg_grad_norm

        return result

    def compute_loss(self, model, batch_size: int = 128) -> float:
        if self.num_samples == 0:
            return 0.0

        model = model.to(self.device)
        model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
            for data, target in loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = model(data)
                loss = F.cross_entropy(output, target, reduction="sum")
                total_loss += loss.item()
                total_samples += target.size(0)

        return total_loss / max(total_samples, 1)

    def compute_accuracy(self, model, batch_size: int = 128) -> float:
        if self.num_samples == 0:
            return 0.0

        model = model.to(self.device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
            for data, target in loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        acc = correct / max(total, 1)
        self.last_accuracy = acc
        return acc

    @staticmethod
    def _flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        flat_tensors = []
        for _, tensor in state_dict.items():
            flat_tensors.append(tensor.reshape(-1).float())
        return torch.cat(flat_tensors, dim=0)

    def get_update_vector(
        self,
        local_params: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        delta = {}
        for name in local_params:
            delta[name] = local_params[name].detach().cpu() - global_params[name].detach().cpu()
        return self._flatten_state_dict(delta)