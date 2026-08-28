import time
import random
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Server:
    """Federated server with pluggable client selection and timing instrumentation."""

    def __init__(self, model_fn, selector_manager, device):
        self.model_fn = model_fn
        self.global_model = model_fn().to(device)
        self.selector_mgr = selector_manager
        self.device = device
        self.round = 0

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in self.global_model.state_dict().items()
        }

    def model_size_bytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for p in self.global_model.parameters()))

    def _build_model_from_params(
        self,
        params: Dict[str, torch.Tensor],
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = self.device
        model = self.model_fn().to(device)
        model.load_state_dict(params)
        return model

    def aggregate(self, updates: List[Dict[str, Any]], weights=None):
        if not updates:
            return

        if weights is None:
            total_data = sum(upd.get("data_size", 0) for upd in updates)
            if total_data <= 0:
                n = len(updates)
                weights = [1.0 / n for _ in updates]
            else:
                weights = [upd.get("data_size", 0) / total_data for upd in updates]

        new_params = {}
        param_keys = updates[0]["params"].keys()
        for key in param_keys:
            agg = None
            for weight, upd in zip(weights, updates):
                tensor = upd["params"][key].float().to(self.device)
                agg = weight * tensor if agg is None else agg + weight * tensor
            new_params[key] = agg
        self.global_model.load_state_dict(new_params)

    def evaluate_model(
        self,
        model,
        data_loader: DataLoader,
        criterion=nn.CrossEntropyLoss(),
    ):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                total += target.size(0)
                correct += (pred == target).sum().item()
        if total == 0:
            return 0.0, 0.0
        return correct / total, total_loss / total

    def evaluate(self, test_loader: DataLoader, criterion=nn.CrossEntropyLoss()):
        return self.evaluate_model(self.global_model, test_loader, criterion)

    def evaluate_state_dict(
        self,
        params: Dict[str, torch.Tensor],
        data_loader: DataLoader,
        criterion=nn.CrossEntropyLoss(),
    ):
        model = self._build_model_from_params(params, device=self.device)
        return self.evaluate_model(model, data_loader, criterion)

    def collect_pre_selection_metrics(
        self,
        clients: List[Any],
        available_client_ids: Optional[List[int]] = None,
        metric_batch_size: int = 128,
        include_accuracy: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        if available_client_ids is None:
            available_client_ids = list(range(len(clients)))

        global_params = self.get_global_params()
        client_metrics = {}
        for cid in available_client_ids:
            client = clients[cid]
            eval_model = self._build_model_from_params(global_params, device=client.device)
            current_loss = client.compute_loss(eval_model, batch_size=metric_batch_size)
            prev_loss = client.last_loss
            loss_drop = (prev_loss - current_loss) if prev_loss is not None else None
            metrics = {
                "loss": current_loss,
                "prev_loss": prev_loss,
                "loss_drop": loss_drop,
                "gradient_norm": client.last_grad_norm,
                "data_size": client.num_samples,
            }
            if include_accuracy:
                metrics["accuracy"] = client.compute_accuracy(
                    eval_model, batch_size=metric_batch_size
                )
            client_metrics[cid] = metrics
        return client_metrics

    def attach_server_eval_to_updates(
        self,
        selected_ids: List[int],
        updates: List[Dict[str, Any]],
        eval_loader: Optional[DataLoader] = None,
        criterion=nn.CrossEntropyLoss(),
    ) -> List[Dict[str, Any]]:
        for cid, upd in zip(selected_ids, updates):
            upd["client_id"] = cid
            if eval_loader is not None:
                acc, loss = self.evaluate_state_dict(upd["params"], eval_loader, criterion)
                upd["accuracy"] = acc
                upd["eval_loss"] = loss
        return updates

    def select_clients(
        self,
        available_client_ids: List[int],
        client_metrics: Dict[int, Dict[str, Any]],
    ) -> List[int]:
        return self.selector_mgr.select(available_client_ids, client_metrics)

    def update_selector(
        self,
        selected_ids: List[int],
        updates: List[Dict[str, Any]],
        global_accuracy: Optional[float] = None,
        global_loss: Optional[float] = None,
    ):
        results = {cid: upd for cid, upd in zip(selected_ids, updates)}
        results["__meta__"] = {
            "global_accuracy": global_accuracy,
            "global_loss": global_loss,
        }
        self.selector_mgr.update(selected_ids, results)

    def run_round(
        self,
        clients: List[Any],
        lr: float,
        epochs: int,
        batch_size: int,
        available_client_ids: Optional[List[int]] = None,
        test_loader: Optional[DataLoader] = None,
        public_eval_loader: Optional[DataLoader] = None,
        metric_batch_size: int = 128,
        include_pre_accuracy: bool = False,
        client_dropout_prob: float = 0.0,
        dropout_rng: Optional[random.Random] = None,
        criterion=nn.CrossEntropyLoss(),
    ) -> Dict[str, Any]:
        round_start = time.perf_counter()
        if available_client_ids is None:
            available_client_ids = list(range(len(clients)))

        # 1) Collect only the pre-selection metrics actually required by the selector.
        t0 = time.perf_counter()
        if self.selector_mgr.requires_preselection_metrics:
            client_metrics = self.collect_pre_selection_metrics(
                clients=clients,
                available_client_ids=available_client_ids,
                metric_batch_size=metric_batch_size,
                include_accuracy=include_pre_accuracy,
            )
        else:
            client_metrics = {}
        metric_time = time.perf_counter() - t0

        # 2) Client selection.
        t0 = time.perf_counter()
        selected_ids = self.select_clients(
            available_client_ids=available_client_ids,
            client_metrics=client_metrics,
        )
        selected_ids = [int(cid) for cid in selected_ids]
        selection_time = time.perf_counter() - t0

        # 3) Optional post-selection client dropout. The dedicated RNG keeps the
        # availability/dropout trace independent of selector-internal randomness.
        if dropout_rng is None:
            dropout_rng = random
        client_dropout_prob = min(max(float(client_dropout_prob), 0.0), 1.0)
        successful_ids = [
            cid for cid in selected_ids
            if client_dropout_prob <= 0.0 or dropout_rng.random() >= client_dropout_prob
        ]
        dropped_ids = [cid for cid in selected_ids if cid not in set(successful_ids)]

        # 4) Local training for clients that successfully return an update.
        global_params = self.get_global_params()
        updates = []
        t0 = time.perf_counter()
        for cid in successful_ids:
            upd = clients[cid].train(
                global_model_params=global_params,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                criterion=criterion,
                # All bundled selectors that need an update representation can
                # reconstruct it from the returned delta/params. Avoid storing
                # a third full flattened copy, which is costly for ResNet-18.
                return_update_vector=False,
            )
            upd["client_id"] = cid
            updates.append(upd)
        train_time = time.perf_counter() - t0

        # 5) Optional unified server evaluation of each local model for baselines
        # that require it (currently FedPPO).
        t0 = time.perf_counter()
        if public_eval_loader is not None and updates:
            updates = self.attach_server_eval_to_updates(
                selected_ids=successful_ids,
                updates=updates,
                eval_loader=public_eval_loader,
                criterion=criterion,
            )
        local_eval_time = time.perf_counter() - t0

        # 6) Aggregation.
        t0 = time.perf_counter()
        self.aggregate(updates)
        aggregation_time = time.perf_counter() - t0

        # 7) Global evaluation.
        t0 = time.perf_counter()
        global_accuracy, global_test_loss = None, None
        if test_loader is not None:
            global_accuracy, global_test_loss = self.evaluate(
                test_loader=test_loader,
                criterion=criterion,
            )
        evaluation_time = time.perf_counter() - t0

        # 8) Update selector state only from successfully received updates.
        self.update_selector(
            selected_ids=successful_ids,
            updates=updates,
            global_accuracy=global_accuracy,
            global_loss=global_test_loss,
        )

        round_time = time.perf_counter() - round_start
        result = {
            "round": self.round,
            "available_ids": list(available_client_ids),
            "selected_ids": selected_ids,
            "successful_ids": successful_ids,
            "dropped_ids": dropped_ids,
            "updates": updates,
            "client_metrics": client_metrics,
            "global_accuracy": global_accuracy,
            "global_test_loss": global_test_loss,
            "timing": {
                "metric_time": metric_time,
                "selection_time": selection_time,
                "train_time": train_time,
                "local_eval_time": local_eval_time,
                "aggregation_time": aggregation_time,
                "evaluation_time": evaluation_time,
                "round_time": round_time,
            },
        }
        self.round += 1
        return result
