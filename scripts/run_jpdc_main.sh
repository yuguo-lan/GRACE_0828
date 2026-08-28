#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
METHOD="${2:-graph_diversity}"
DATASET="${3:-fashionmnist}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

ALL_METHODS=(poc oort rbcs_f mbut_cs divfl fedcor fedppo graph_diversity)
ALL_DATASETS=(mnist fashionmnist cifar10)
SEEDS_STR="${SEEDS:-41 42 43}"
ALPHAS_STR="${ALPHAS:-0.3 0.5 0.7}"
ROUNDS="${ROUNDS:-2000}"

if [[ "$METHOD" == "all" ]]; then METHODS=("${ALL_METHODS[@]}"); else METHODS=("$METHOD"); fi
if [[ "$DATASET" == "all" ]]; then DATASETS=("${ALL_DATASETS[@]}"); else DATASETS=("$DATASET"); fi
read -r -a SEEDS_ARR <<< "$SEEDS_STR"
read -r -a ALPHAS_ARR <<< "$ALPHAS_STR"

for ds in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    for alpha in "${ALPHAS_ARR[@]}"; do
      for seed in "${SEEDS_ARR[@]}"; do
        python -m fedselect_jpdc.run_experiment \
          --gpu "$GPU" --dataset "$ds" --selector "$method" --model simplecnn \
          --num-clients 100 --clients-per-round 10 \
          --alpha "$alpha" --seed "$seed" --rounds "$ROUNDS" \
          --local-epochs 2 --batch-size 32 --lr 0.01 --test-interval 5 \
          --exp-name main_multiseed --result-dir "$ROOT/results_jpdc"
      done
    done
  done
done
