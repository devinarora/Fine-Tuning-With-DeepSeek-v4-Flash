#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ENV_PY=""

stage="${1:-all}"

run_py() {
  "$ENV_PY" "$@"
}

case "$stage" in
  train)
    run_py -m src.train
    ;;
  evaluate)
    run_py -m src.evaluate
    ;;
  plots)
    run_py -c "from src.evaluate import plot_training; plot_training('outputs/metrics','outputs/figures')"
    ;;
  all)
    run_py -m src.train
    run_py -m src.evaluate
    run_py -c "from src.evaluate import plot_training; plot_training('outputs/metrics','outputs/figures')"
    ;;
  *)
    echo "usage: $0 {train|evaluate|plots|all}"
    exit 1
    ;;
esac