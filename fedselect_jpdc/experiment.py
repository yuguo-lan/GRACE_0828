import json
import math
import os
import random
import time
from typing import Dict, Any, List

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data_manager import DataManager
from .client import Client
from .server import Server
from .selector_manager import SelectorManager
from .utils import set_seed
from .models import get_model_fn


class Experiment:
    """End-to-end FL experiment with reproducible system traces and rich logging."""

    ROUND_COLUMNS = [
        "round",
        "test_acc",
        "test_loss",
        "available_count",
        "selected_clients",
        "successful_clients",
        "dropped_clients",
        "selected_count",
        "successful_count",
        "dropped_count",
        "consumed_data",
        "elapsed_time",
        "metric_time",
        "selection_time",
        "train_time",
        "local_eval_time",
        "aggregation_time",
        "evaluation_time",
        "round_time",
        "unique_successful_clients",
        "coverage_ratio",
        "jain_fairness",
        "model_size_mb",
        "round_download_mb",
        "round_upload_mb",
        "cumulative_communication_mb",
        "simulated_round_latency_s",
        "cumulative_simulated_latency_s",
    ]

    EVAL_COLUMNS = [
        "round",
        "test_acc",
        "test_loss",
        "coverage_ratio",
        "jain_fairness",
        "consumed_data",
        "elapsed_time",
        "cumulative_communication_mb",
        "mean_selection_time_ms",
        "cumulative_simulated_latency_s",
    ]

    def __init__(self, config: dict):
        self.config = dict(config)
        self.exp_name = str(config.get("exp_name", "default"))
        self.seed = int(config.get("seed", 42))
        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        set_seed(self.seed)

        # Dedicated RNG streams keep the availability/dropout traces independent
        # of selector-internal random choices.
        self.availability_rng = random.Random(self.seed + 100_003)
        self.dropout_rng = random.Random(self.seed + 200_003)

        dataset_name = str(config["dataset"]).lower()
        self.data_mgr = DataManager(
            dataset_name=dataset_name,
            num_clients=int(config["num_clients"]),
            non_iid_alpha=None if config.get("iid", False) else config.get("non_iid_alpha", 0.3),
            seed=self.seed,
            data_dir=config.get("data_dir", "./data"),
            partition_dir=config.get("partition_dir", None),
            force_generate=bool(config.get("force_generate_partition", False)),
        )

        # Data partition generation consumes NumPy RNG state. Reset all global
        # ML RNGs so selector/training randomness is identical whether a saved
        # partition is loaded or generated in this run.
        set_seed(self.seed)

        self.model_fn = get_model_fn(config["model"], dataset_name=dataset_name)
        self.total_rounds = int(config["total_rounds"])
        self.clients_per_round = int(config["clients_per_round"])
        self.local_epochs = int(config["local_epochs"])
        self.batch_size = int(config["batch_size"])
        self.lr = float(config["lr"])
        self.test_interval = int(config.get("test_interval", 5))
        self.metric_batch_size = int(config.get("selection_metric_batch_size", 128))

        selector_name = str(config["selector"])
        self.selector_name = selector_name
        self.include_pre_accuracy = bool(
            config.get("include_pre_accuracy", selector_name == "fedppo")
        )

        self.avail_prob = float(config.get("availability_prob", 1.0))
        self.avail_prob = min(max(self.avail_prob, 0.0), 1.0)
        self.client_dropout_prob = float(config.get("client_dropout_prob", 0.0))
        self.client_dropout_prob = min(max(self.client_dropout_prob, 0.0), 1.0)

        self.system_heterogeneity = str(config.get("system_heterogeneity", "none")).lower()
        if self.system_heterogeneity not in {"none", "mild", "strong"}:
            raise ValueError("system_heterogeneity must be one of: none, mild, strong")
        profile_rng = random.Random(self.seed + 300_003)
        self.system_profiles = {}
        for cid in range(int(config["num_clients"])):
            if self.system_heterogeneity == "none":
                throughput = 5000.0
                bandwidth = 100.0
            elif self.system_heterogeneity == "mild":
                throughput = min(max(profile_rng.lognormvariate(math.log(5000.0), 0.35), 1500.0), 12000.0)
                bandwidth = min(max(profile_rng.lognormvariate(math.log(100.0), 0.35), 20.0), 250.0)
            else:
                throughput = min(max(profile_rng.lognormvariate(math.log(3500.0), 0.75), 500.0), 15000.0)
                bandwidth = min(max(profile_rng.lognormvariate(math.log(60.0), 0.75), 5.0), 300.0)
            self.system_profiles[cid] = {
                "throughput_samples_per_s": float(throughput),
                "bandwidth_mbps": float(bandwidth),
            }

        self.clients: List[Client] = []
        for cid in range(int(config["num_clients"])):
            dataset = self.data_mgr.client_datasets[cid]
            client = Client(
                client_id=cid,
                dataset=dataset,
                model_fn=self.model_fn,
                device=self.device,
            )
            client.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.clients.append(client)

        self.selector_mgr = SelectorManager(
            selector_name=selector_name,
            total_clients=int(config["num_clients"]),
            clients_per_round=self.clients_per_round,
            selector_params=config.get("selector_params", {}),
        )
        client_ids = list(range(int(config["num_clients"])))
        self.selector_mgr.init_clients(client_ids)
        for cid in client_ids:
            info = self.data_mgr.get_client_info(cid)
            if (
                hasattr(self.selector_mgr, "selector")
                and hasattr(self.selector_mgr.selector, "client_info")
                and cid in self.selector_mgr.selector.client_info
            ):
                client_info = self.selector_mgr.selector.client_info[cid]
                client_info.data_size = info["data_size"]
                client_info.capability = self.system_profiles[cid]["throughput_samples_per_s"]
                client_info.bandwidth = self.system_profiles[cid]["bandwidth_mbps"]

        self.server = Server(
            model_fn=self.model_fn,
            selector_manager=self.selector_mgr,
            device=self.device,
        )

        test_batch_size = int(config.get("test_batch_size", 256))
        self.test_loader = self.data_mgr.get_test_dataloader(batch_size=test_batch_size)
        self.public_eval_loader = self.test_loader if selector_name == "fedppo" else None

        self.total_consumed_data = 0
        self.cumulative_communication_mb = 0.0
        self.cumulative_simulated_latency_s = 0.0
        self.participation_counts = [0 for _ in range(int(config["num_clients"]))]
        self.selection_times: List[float] = []

        alpha_str = (
            f"{config.get('non_iid_alpha', 0.3)}"
            if not config.get("iid", False)
            else "IID"
        )
        stem = (
            f"{dataset_name}_{selector_name}_"
            f"nc{config['num_clients']}_K{self.clients_per_round}_"
            f"alpha{alpha_str}_seed{self.seed}"
        )
        result_base = config.get("result_dir", "./results")
        self.csv_dir = os.path.join(result_base, selector_name, self.exp_name)
        os.makedirs(self.csv_dir, exist_ok=True)
        self.round_csv_path = os.path.join(self.csv_dir, f"rounds_{stem}.csv")
        self.eval_csv_path = os.path.join(self.csv_dir, f"eval_{stem}.csv")
        self.summary_json_path = os.path.join(self.csv_dir, f"summary_{stem}.json")
        self.config_json_path = os.path.join(self.csv_dir, f"config_{stem}.json")
        self.csv_path = self.eval_csv_path  # backward-compatible alias
        self._init_output_files()

    @staticmethod
    def _join_ids(ids: List[int]) -> str:
        return ",".join(map(str, ids))

    def _init_output_files(self):
        pd.DataFrame(columns=self.ROUND_COLUMNS).to_csv(self.round_csv_path, index=False)
        pd.DataFrame(columns=self.EVAL_COLUMNS).to_csv(self.eval_csv_path, index=False)
        with open(self.config_json_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False, default=str)

    def _sample_available_clients(self) -> List[int]:
        if self.avail_prob >= 1.0:
            return list(range(len(self.clients)))
        return [
            cid
            for cid in range(len(self.clients))
            if self.availability_rng.random() < self.avail_prob
        ]

    def _jain_fairness(self) -> float:
        counts = self.participation_counts
        denom = len(counts) * sum(x * x for x in counts)
        if denom <= 0:
            return 0.0
        return float(sum(counts) ** 2 / denom)

    def _coverage_ratio(self) -> float:
        if not self.participation_counts:
            return 0.0
        return sum(x > 0 for x in self.participation_counts) / len(self.participation_counts)

    def _append_round_row(self, row: Dict[str, Any]):
        pd.DataFrame([{col: row.get(col, None) for col in self.ROUND_COLUMNS}]).to_csv(
            self.round_csv_path, mode="a", header=False, index=False
        )

    def _append_eval_row(self, row: Dict[str, Any]):
        pd.DataFrame([{col: row.get(col, None) for col in self.EVAL_COLUMNS}]).to_csv(
            self.eval_csv_path, mode="a", header=False, index=False
        )

    def run(self):
        start_time = time.perf_counter()
        model_size_mb = self.server.model_size_bytes() / (1024.0 ** 2)
        last_eval = None

        for round_idx in range(self.total_rounds):
            available_ids = self._sample_available_clients()
            do_evaluate = (
                self.test_interval <= 0
                or ((round_idx + 1) % self.test_interval == 0)
                or round_idx == self.total_rounds - 1
            )
            current_test_loader = self.test_loader if do_evaluate else None

            round_result = self.server.run_round(
                clients=self.clients,
                lr=self.lr,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
                available_client_ids=available_ids,
                test_loader=current_test_loader,
                public_eval_loader=self.public_eval_loader,
                metric_batch_size=self.metric_batch_size,
                include_pre_accuracy=self.include_pre_accuracy,
                client_dropout_prob=self.client_dropout_prob,
                dropout_rng=self.dropout_rng,
            )

            selected_ids = round_result["selected_ids"]
            successful_ids = round_result["successful_ids"]
            dropped_ids = round_result["dropped_ids"]
            test_acc = round_result["global_accuracy"]
            test_loss = round_result["global_test_loss"]
            timing = round_result["timing"]
            self.selection_times.append(float(timing["selection_time"]))

            round_consumed = sum(self.clients[cid].num_samples for cid in successful_ids)
            self.total_consumed_data += round_consumed
            for cid in successful_ids:
                self.participation_counts[cid] += 1

            # One model is downloaded per invited client; one model is uploaded
            # per successful participant. This is a protocol-level estimate.
            round_download_mb = model_size_mb * len(selected_ids)
            round_upload_mb = model_size_mb * len(successful_ids)
            self.cumulative_communication_mb += round_download_mb + round_upload_mb

            simulated_round_latency_s = 0.0
            for cid in successful_ids:
                profile = self.system_profiles[cid]
                compute_s = (
                    self.clients[cid].num_samples * self.local_epochs
                    / max(profile["throughput_samples_per_s"], 1e-9)
                )
                # Analytical critical-path latency includes one model download,
                # local computation, and one update upload for a successful client.
                transfer_s = (model_size_mb * 8.0) / max(profile["bandwidth_mbps"], 1e-9)
                simulated_round_latency_s = max(
                    simulated_round_latency_s,
                    transfer_s + compute_s + transfer_s,
                )
            self.cumulative_simulated_latency_s += simulated_round_latency_s
            elapsed_time = time.perf_counter() - start_time
            coverage = self._coverage_ratio()
            fairness = self._jain_fairness()

            round_row = {
                "round": round_idx + 1,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "available_count": len(available_ids),
                "selected_clients": self._join_ids(selected_ids),
                "successful_clients": self._join_ids(successful_ids),
                "dropped_clients": self._join_ids(dropped_ids),
                "selected_count": len(selected_ids),
                "successful_count": len(successful_ids),
                "dropped_count": len(dropped_ids),
                "consumed_data": self.total_consumed_data,
                "elapsed_time": elapsed_time,
                "metric_time": timing["metric_time"],
                "selection_time": timing["selection_time"],
                "train_time": timing["train_time"],
                "local_eval_time": timing["local_eval_time"],
                "aggregation_time": timing["aggregation_time"],
                "evaluation_time": timing["evaluation_time"],
                "round_time": timing["round_time"],
                "unique_successful_clients": sum(x > 0 for x in self.participation_counts),
                "coverage_ratio": coverage,
                "jain_fairness": fairness,
                "model_size_mb": model_size_mb,
                "round_download_mb": round_download_mb,
                "round_upload_mb": round_upload_mb,
                "cumulative_communication_mb": self.cumulative_communication_mb,
                "simulated_round_latency_s": simulated_round_latency_s,
                "cumulative_simulated_latency_s": self.cumulative_simulated_latency_s,
            }
            self._append_round_row(round_row)

            if test_acc is not None and test_loss is not None:
                mean_select_ms = 1000.0 * sum(self.selection_times) / len(self.selection_times)
                eval_row = {
                    "round": round_idx + 1,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "coverage_ratio": coverage,
                    "jain_fairness": fairness,
                    "consumed_data": self.total_consumed_data,
                    "elapsed_time": elapsed_time,
                    "cumulative_communication_mb": self.cumulative_communication_mb,
                    "mean_selection_time_ms": mean_select_ms,
                    "cumulative_simulated_latency_s": self.cumulative_simulated_latency_s,
                }
                self._append_eval_row(eval_row)
                last_eval = eval_row
                print(
                    f"Round {round_idx + 1}/{self.total_rounds} | "
                    f"Acc {test_acc:.4f} | Loss {test_loss:.4f} | "
                    f"avail {len(available_ids)} | selected {len(selected_ids)} | "
                    f"success {len(successful_ids)} | coverage {coverage:.3f} | "
                    f"Jain {fairness:.3f} | select {timing['selection_time']*1000:.2f} ms"
                )
            else:
                print(
                    f"Round {round_idx + 1}/{self.total_rounds} | test skipped | "
                    f"avail {len(available_ids)} | selected {len(selected_ids)} | "
                    f"success {len(successful_ids)} | select {timing['selection_time']*1000:.2f} ms"
                )

        summary = {
            "selector": self.selector_name,
            "dataset": self.config["dataset"],
            "seed": self.seed,
            "num_clients": len(self.clients),
            "clients_per_round": self.clients_per_round,
            "non_iid_alpha": None if self.config.get("iid", False) else self.config.get("non_iid_alpha"),
            "availability_prob": self.avail_prob,
            "client_dropout_prob": self.client_dropout_prob,
            "system_heterogeneity": self.system_heterogeneity,
            "final_eval": last_eval,
            "coverage_ratio": self._coverage_ratio(),
            "jain_fairness": self._jain_fairness(),
            "total_consumed_data": self.total_consumed_data,
            "cumulative_communication_mb": self.cumulative_communication_mb,
            "cumulative_simulated_latency_s": self.cumulative_simulated_latency_s,
            "mean_selection_time_ms": (
                1000.0 * sum(self.selection_times) / len(self.selection_times)
                if self.selection_times else 0.0
            ),
            "round_csv": self.round_csv_path,
            "eval_csv": self.eval_csv_path,
        }
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    def save_results(self):
        # Results are streamed to CSV each round; kept for API compatibility.
        return self.summary_json_path
