import argparse
import json
from typing import Any, Dict

from .experiment import Experiment


def _parse_scalar(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _selector_params(args) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if args.selector_params_json:
        params.update(json.loads(args.selector_params_json))
    for item in args.selector_param:
        if "=" not in item:
            raise ValueError(f"Invalid --selector-param '{item}', expected key=value")
        key, value = item.split("=", 1)
        params[key.strip()] = _parse_scalar(value.strip())
    return params


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a JPDC-ready federated client-selection experiment")
    p.add_argument("--dataset", choices=["mnist", "fashionmnist", "cifar10", "cifar100"], required=True)
    p.add_argument("--selector", required=True)
    p.add_argument("--model", choices=["mlp", "simplecnn", "resnet18"], default="simplecnn")
    p.add_argument("--num-clients", type=int, default=100)
    p.add_argument("--clients-per-round", type=int, default=None)
    p.add_argument("--participation-rate", type=float, default=None,
                   help="Alternative to --clients-per-round; e.g. 0.1 for 10%%")
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--iid", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--test-interval", type=int, default=5)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--selection-metric-batch-size", type=int, default=128)
    p.add_argument("--availability", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--system-heterogeneity", choices=["none", "mild", "strong"], default="none")
    p.add_argument("--gpu", default="0", help="GPU id, e.g. 0; use 'cpu' for CPU")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--result-dir", default="./results_jpdc")
    p.add_argument("--exp-name", default="jpdc")
    p.add_argument("--partition-dir", default=None)
    p.add_argument("--force-generate-partition", action="store_true")
    p.add_argument("--selector-params-json", default=None)
    p.add_argument("--selector-param", action="append", default=[], help="Repeatable key=value")
    p.add_argument("--print-config", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.clients_per_round is None:
        rate = 0.1 if args.participation_rate is None else args.participation_rate
        if not (0 < rate <= 1):
            raise ValueError("--participation-rate must be in (0, 1]")
        clients_per_round = max(1, int(round(args.num_clients * rate)))
    else:
        clients_per_round = args.clients_per_round
    if clients_per_round > args.num_clients:
        raise ValueError("clients_per_round cannot exceed num_clients")

    device = "cpu" if str(args.gpu).lower() == "cpu" else f"cuda:{args.gpu}"
    config = {
        "exp_name": args.exp_name,
        "seed": args.seed,
        "device": device,
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "clients_per_round": clients_per_round,
        "non_iid_alpha": args.alpha,
        "iid": args.iid,
        "model": args.model,
        "total_rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "test_interval": args.test_interval,
        "test_batch_size": args.test_batch_size,
        "selection_metric_batch_size": args.selection_metric_batch_size,
        "availability_prob": args.availability,
        "client_dropout_prob": args.dropout,
        "system_heterogeneity": args.system_heterogeneity,
        "data_dir": args.data_dir,
        "result_dir": args.result_dir,
        "partition_dir": args.partition_dir,
        "force_generate_partition": args.force_generate_partition,
        "selector": args.selector,
        "selector_params": _selector_params(args),
    }
    print(json.dumps(config, indent=2, ensure_ascii=False))
    if args.print_config:
        return
    summary = Experiment(config).run()
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
