import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def _safe_wilcoxon(x, y):
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if len(diff) < 2 or np.allclose(diff, 0.0):
        return np.nan
    try:
        return float(wilcoxon(x, y, alternative="two-sided").pvalue)
    except ValueError:
        return np.nan


def main():
    p = argparse.ArgumentParser(description="Paired multi-seed significance tests for JPDC experiments")
    p.add_argument("--runs-final", default="./results_jpdc/analysis/runs_final.csv")
    p.add_argument("--reference", default="graph_diversity")
    p.add_argument("--exp-name", default="main")
    p.add_argument("--metric", default="final_acc")
    p.add_argument("--out", default="./results_jpdc/analysis/paired_significance.csv")
    args = p.parse_args()

    df = pd.read_csv(args.runs_final)
    if args.exp_name:
        df = df[df["exp_name"] == args.exp_name].copy()
    if df.empty:
        raise SystemExit("No rows match the requested experiment name")
    if args.metric not in df.columns:
        raise SystemExit(f"Unknown metric: {args.metric}")

    scenario_cols = [
        "exp_name", "dataset", "model", "num_clients", "clients_per_round",
        "alpha", "availability", "dropout", "system_heterogeneity",
    ]
    rows = []
    for scenario, sdf in df.groupby(scenario_cols, dropna=False):
        ref = sdf[sdf["selector"] == args.reference][["seed", args.metric]].dropna()
        if ref.empty:
            continue
        ref = ref.rename(columns={args.metric: "reference_value"})
        for selector, cdf in sdf[sdf["selector"] != args.reference].groupby("selector"):
            comp = cdf[["seed", args.metric]].dropna().rename(columns={args.metric: "comparator_value"})
            paired = ref.merge(comp, on="seed", how="inner").sort_values("seed")
            if len(paired) < 2:
                continue
            x = paired["reference_value"].to_numpy(float)
            y = paired["comparator_value"].to_numpy(float)
            diff = x - y
            t_p = float(ttest_rel(x, y).pvalue) if len(paired) >= 2 and not np.allclose(diff, 0.0) else np.nan
            w_p = _safe_wilcoxon(x, y)
            row = dict(zip(scenario_cols, scenario if isinstance(scenario, tuple) else (scenario,)))
            row.update({
                "reference": args.reference,
                "comparator": selector,
                "metric": args.metric,
                "paired_seeds": ",".join(map(str, paired["seed"].tolist())),
                "n_pairs": len(paired),
                "reference_mean": float(np.mean(x)),
                "comparator_mean": float(np.mean(y)),
                "mean_difference": float(np.mean(diff)),
                "std_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
            })
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} paired comparisons -> {out}")


if __name__ == "__main__":
    main()
