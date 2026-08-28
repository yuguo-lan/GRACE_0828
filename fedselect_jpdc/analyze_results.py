import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_TARGETS = {
    "mnist": [0.80, 0.85, 0.90, 0.95],
    "fashionmnist": [0.70, 0.75, 0.80, 0.83],
    "cifar10": [0.40, 0.45, 0.50, 0.55],
    "cifar100": [0.20, 0.30, 0.40, 0.50],
}


def _load_runs(root: Path):
    runs = []
    for cfg_path in sorted(root.rglob("config_*.json")):
        stem = cfg_path.name[len("config_"):-len(".json")]
        eval_path = cfg_path.with_name(f"eval_{stem}.csv")
        round_path = cfg_path.with_name(f"rounds_{stem}.csv")
        if not eval_path.exists() or not round_path.exists():
            continue
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        try:
            eval_df = pd.read_csv(eval_path)
            round_df = pd.read_csv(round_path)
        except Exception as exc:
            print(f"skip {cfg_path}: {exc}")
            continue
        if eval_df.empty or round_df.empty:
            continue
        runs.append((cfg, eval_df, round_df, cfg_path))
    return runs


def _meta(cfg: Dict) -> Dict:
    return {
        "exp_name": cfg.get("exp_name", ""),
        "dataset": cfg.get("dataset"),
        "selector": cfg.get("selector"),
        "model": cfg.get("model"),
        "num_clients": int(cfg.get("num_clients")),
        "clients_per_round": int(cfg.get("clients_per_round")),
        "alpha": "IID" if cfg.get("iid", False) else cfg.get("non_iid_alpha"),
        "availability": float(cfg.get("availability_prob", 1.0)),
        "dropout": float(cfg.get("client_dropout_prob", 0.0)),
        "system_heterogeneity": cfg.get("system_heterogeneity", "none"),
        "seed": int(cfg.get("seed", 0)),
        "selector_params": json.dumps(cfg.get("selector_params", {}), sort_keys=True, separators=(",", ":")),
    }


def summarize(root: Path, out: Path, budgets: List[int]):
    runs = _load_runs(root)
    if not runs:
        raise SystemExit(f"No completed runs found under {root}")
    out.mkdir(parents=True, exist_ok=True)

    final_rows = []
    checkpoint_rows = []
    target_rows = []

    for cfg, eval_df, round_df, cfg_path in runs:
        meta = _meta(cfg)
        last = eval_df.iloc[-1]
        final_rows.append({
            **meta,
            "final_round": int(last["round"]),
            "final_acc": float(last["test_acc"]),
            "final_loss": float(last["test_loss"]),
            "coverage_ratio": float(last.get("coverage_ratio", np.nan)),
            "jain_fairness": float(last.get("jain_fairness", np.nan)),
            "communication_mb": float(last.get("cumulative_communication_mb", np.nan)),
            "elapsed_time_s": float(last.get("elapsed_time", np.nan)),
            "simulated_latency_s": float(last.get("cumulative_simulated_latency_s", np.nan)),
            "mean_selection_time_ms": 1000.0 * float(round_df["selection_time"].mean()),
            "p95_selection_time_ms": 1000.0 * float(round_df["selection_time"].quantile(0.95)),
            "mean_metric_time_ms": 1000.0 * float(round_df["metric_time"].mean()),
            "source": str(cfg_path),
        })

        for budget in budgets:
            part = eval_df[eval_df["round"] <= budget]
            checkpoint_rows.append({
                **meta,
                "budget_rounds": budget,
                "best_acc": float(part["test_acc"].max()) if not part.empty else np.nan,
            })

        targets = DEFAULT_TARGETS.get(str(cfg.get("dataset")).lower(), [])
        for target in targets:
            hit = eval_df[eval_df["test_acc"] >= target]
            target_rows.append({
                **meta,
                "target_acc": target,
                "rounds_to_target": int(hit.iloc[0]["round"]) if not hit.empty else np.nan,
            })

    final_df = pd.DataFrame(final_rows)
    checkpoint_df = pd.DataFrame(checkpoint_rows)
    target_df = pd.DataFrame(target_rows)
    final_df.to_csv(out / "runs_final.csv", index=False)
    checkpoint_df.to_csv(out / "runs_checkpoint_accuracy.csv", index=False)
    target_df.to_csv(out / "runs_rounds_to_target.csv", index=False)

    group_cols = [
        "exp_name", "dataset", "selector", "model", "num_clients",
        "clients_per_round", "alpha", "availability", "dropout", "system_heterogeneity",
        "selector_params",
    ]
    agg = final_df.groupby(group_cols, dropna=False).agg(
        seeds=("seed", "nunique"),
        final_acc_mean=("final_acc", "mean"),
        final_acc_std=("final_acc", "std"),
        coverage_mean=("coverage_ratio", "mean"),
        coverage_std=("coverage_ratio", "std"),
        jain_mean=("jain_fairness", "mean"),
        jain_std=("jain_fairness", "std"),
        selection_ms_mean=("mean_selection_time_ms", "mean"),
        selection_ms_std=("mean_selection_time_ms", "std"),
        metric_ms_mean=("mean_metric_time_ms", "mean"),
        elapsed_s_mean=("elapsed_time_s", "mean"),
        communication_mb_mean=("communication_mb", "mean"),
        simulated_latency_s_mean=("simulated_latency_s", "mean"),
        simulated_latency_s_std=("simulated_latency_s", "std"),
    ).reset_index()
    agg.to_csv(out / "aggregate_final_mean_std.csv", index=False)

    cp_group = group_cols + ["budget_rounds"]
    cp_agg = checkpoint_df.groupby(cp_group, dropna=False).agg(
        seeds=("seed", "nunique"),
        best_acc_mean=("best_acc", "mean"),
        best_acc_std=("best_acc", "std"),
    ).reset_index()
    cp_agg.to_csv(out / "aggregate_checkpoint_mean_std.csv", index=False)

    rt_group = group_cols + ["target_acc"]
    rt_agg = target_df.groupby(rt_group, dropna=False).agg(
        seeds=("seed", "nunique"),
        reached=("rounds_to_target", lambda s: int(s.notna().sum())),
        rounds_mean=("rounds_to_target", "mean"),
        rounds_std=("rounds_to_target", "std"),
    ).reset_index()
    rt_agg.to_csv(out / "aggregate_rounds_to_target.csv", index=False)

    print(f"Analyzed {len(runs)} completed runs -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", default="./results_jpdc")
    p.add_argument("--out-dir", default="./results_jpdc/analysis")
    p.add_argument("--budgets", default="500,1000,2000")
    args = p.parse_args()
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    summarize(Path(args.result_dir), Path(args.out_dir), budgets)


if __name__ == "__main__":
    main()
