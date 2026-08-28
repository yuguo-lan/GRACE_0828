#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
METHOD="${2:-graph_diversity}"
DATASET="${3:-fashionmnist}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
CLIENTS_STR="${CLIENTS:-100 200 500}"
SEEDS_STR="${SEEDS:-42}"
ALPHA="${ALPHA:-0.3}"
ROUNDS="${ROUNDS:-1000}"
read -r -a CLIENTS_ARR <<< "$CLIENTS_STR"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"

for n in "${CLIENTS_ARR[@]}"; do
  k=$(( (n + 9) / 10 ))
  for seed in "${SEEDS_ARR[@]}"; do
    python -m fedselect_jpdc.run_experiment \
      --gpu "$GPU" --dataset "$DATASET" --selector "$METHOD" --model simplecnn \
      --num-clients "$n" --clients-per-round "$k" \
      --alpha "$ALPHA" --seed "$seed" --rounds "$ROUNDS" \
      --local-epochs 2 --batch-size 32 --lr 0.01 --test-interval 10 \
      --exp-name scalability --result-dir "$ROOT/results_jpdc"
  done
done
