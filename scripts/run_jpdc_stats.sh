#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
RESULT_DIR="${1:-$ROOT/results_jpdc}"
OUT_DIR="${2:-${RESULT_DIR}/analysis}"
EXP_NAME="${EXP_NAME:-main_multiseed}"

python -m fedselect_jpdc.analyze_results --result-dir "$RESULT_DIR" --out-dir "$OUT_DIR"
python -m fedselect_jpdc.statistical_tests \
  --runs-final "$OUT_DIR/runs_final.csv" \
  --reference graph_diversity --exp-name "$EXP_NAME" \
  --out "$OUT_DIR/paired_significance.csv"
