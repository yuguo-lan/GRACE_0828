#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
METHOD="${2:-graph_diversity}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
SEEDS_STR="${SEEDS:-41 42 43}"
ALPHAS_STR="${ALPHAS:-0.3 0.7}"
ROUNDS="${ROUNDS:-1000}"
LR="${LR:-0.01}"
read -r -a SEEDS_ARR <<< "$SEEDS_STR"
read -r -a ALPHAS_ARR <<< "$ALPHAS_STR"

for alpha in "${ALPHAS_ARR[@]}"; do
  for seed in "${SEEDS_ARR[@]}"; do
    python -m fedselect_jpdc.run_experiment \
      --gpu "$GPU" --dataset cifar100 --selector "$METHOD" --model resnet18 \
      --num-clients 100 --clients-per-round 10 \
      --alpha "$alpha" --seed "$seed" --rounds "$ROUNDS" \
      --local-epochs 1 --batch-size 64 --lr "$LR" --test-interval 10 \
      --exp-name cifar100_resnet18 --result-dir "$ROOT/results_jpdc"
  done
done
