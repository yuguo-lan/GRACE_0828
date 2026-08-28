#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
METHOD="${2:-graph_diversity}"
DATASET="${3:-fashionmnist}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
RATES_STR="${RATES:-0.05 0.10 0.20}"
SEEDS_STR="${SEEDS:-41 42 43}"
ALPHA="${ALPHA:-0.3}"
ROUNDS="${ROUNDS:-1200}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
read -r -a RATES_ARR <<< "$RATES_STR"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"

for rate in "${RATES_ARR[@]}"; do
  for seed in "${SEEDS_ARR[@]}"; do
    python -m fedselect_jpdc.run_experiment \
      --gpu "$GPU" --dataset "$DATASET" --selector "$METHOD" --model simplecnn \
      --num-clients "$NUM_CLIENTS" --participation-rate "$rate" \
      --alpha "$ALPHA" --seed "$seed" --rounds "$ROUNDS" \
      --local-epochs 2 --batch-size 32 --lr 0.01 --test-interval 5 \
      --exp-name participation_rate --result-dir "$ROOT/results_jpdc"
  done
done
