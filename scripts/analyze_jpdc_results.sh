#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python -m fedselect_jpdc.analyze_results \
  --result-dir "$ROOT/results_jpdc" \
  --out-dir "$ROOT/results_jpdc/analysis" \
  --budgets "${BUDGETS:-500,1000,2000}"
