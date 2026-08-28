#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EXP="${1:-help}"
shift || true

case "$EXP" in
  main)          exec ./scripts/run_jpdc_main.sh "$@" ;;
  noniid)        exec ./scripts/run_jpdc_main.sh "$@" ;;
  scalability)   exec ./scripts/run_jpdc_scalability.sh "$@" ;;
  participation) exec ./scripts/run_jpdc_participation_rate.sh "$@" ;;
  availability)  exec ./scripts/run_jpdc_availability.sh "$@" ;;
  system)        exec ./scripts/run_jpdc_system_heterogeneity.sh "$@" ;;
  cifar100)      exec ./scripts/run_jpdc_cifar100.sh "$@" ;;
  analyze)       exec ./scripts/analyze_jpdc_results.sh "$@" ;;
  stats)         exec ./scripts/run_jpdc_stats.sh "$@" ;;
  help|-h|--help)
    cat <<'USAGE'
Usage: ./run.sh <experiment> [script arguments]

Final methods:
  poc             Power-of-Choice
  oort            Oort
  rbcs_f          RBCS-F
  mbut_cs         MBUT-CS
  divfl           DivFL
  fedcor          FedCor
  fedppo          FedPPO
  graph_diversity GRACE (ours)

Experiments:
  main            Main 3-seed experiments
  noniid          Alias of main (alpha=0.3/0.5/0.7)
  scalability     Client-population scalability
  participation   Participation-rate sensitivity
  availability    Availability/dropout robustness
  system          System-heterogeneity robustness
  cifar100        CIFAR-100 + ResNet-18 generalization
  analyze         Aggregate result CSVs
  stats           Paired significance tests

Examples:
  ./run.sh main 0 graph_diversity fashionmnist
  ./run.sh main 1 poc cifar10
  ./run.sh scalability 2 divfl cifar10
  ./run.sh availability 3 fedcor fashionmnist
  ./run.sh cifar100 0 fedppo
USAGE
    ;;
  *)
    echo "Unknown experiment: $EXP" >&2
    echo "Run './run.sh help' for available experiments." >&2
    exit 2
    ;;
esac
