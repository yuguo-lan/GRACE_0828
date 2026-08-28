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
SCENARIOS=("1.0 0.0" "0.8 0.0" "0.6 0.0" "0.8 0.1" "0.6 0.2")

for scenario in "${SCENARIOS[@]}"; do
  read -r avail dropout <<< "$scenario"
  for seed in "${SEEDS_ARR[@]}"; do
    python -m fedselect_jpdc.run_experiment \
      --gpu "$GPU" --dataset "$DATASET" --selector "$METHOD" --model simplecnn \
      --num-clients 100 --clients-per-round 10 \
      --alpha "$ALPHA" --seed "$seed" --rounds "$ROUNDS" \
      --local-epochs 2 --batch-size 32 --lr 0.01 --test-interval 5 \
      --availability "$avail" --dropout "$dropout" \
      --exp-name availability_dropout --result-dir "$ROOT/results_jpdc"
  done
done
