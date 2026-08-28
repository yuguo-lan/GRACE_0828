#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
METHOD="${2:-graph_diversity}"
DATASET="${3:-fashionmnist}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
SEEDS_STR="${SEEDS:-41 42 43}"
ALPHA="${ALPHA:-0.3}"
ROUNDS="${ROUNDS:-1500}"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"

for hetero in none mild strong; do
  for seed in "${SEEDS_ARR[@]}"; do
    python -m fedselect_jpdc.run_experiment \
      --gpu "$GPU" --dataset "$DATASET" --selector "$METHOD" --model simplecnn \
      --num-clients 100 --clients-per-round 10 \
      --alpha "$ALPHA" --seed "$seed" --rounds "$ROUNDS" \
      --local-epochs 2 --batch-size 32 --lr 0.01 --test-interval 5 \
      --system-heterogeneity "$hetero" \
      --exp-name system_heterogeneity --result-dir "$ROOT/results_jpdc"
  done
done
