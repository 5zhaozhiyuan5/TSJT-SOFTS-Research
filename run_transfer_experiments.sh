#!/usr/bin/env bash
# Run F1-F4 sequentially and preserve each run's Stage-I/Stage-II checkpoints.
# Usage:
#   PMME_SEEDS="1" bash run_transfer_experiments.sh
#   PMME_SEEDS="1 3 5" bash run_transfer_experiments.sh
#   PMME_VARIANTS="gradient reliable" PMME_SEEDS="1 3 5" bash run_transfer_experiments.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_RUN_ROOT:-$ROOT/experiments/transfer-utility}"
read -r -a VARIANTS <<< "${PMME_VARIANTS:-uniform gradient horizon reliable}"
read -r -a SEEDS <<< "${PMME_SEEDS:-1 3 5}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
mkdir -p "$RUN_ROOT"

if [[ ! -d "$ROOT/data" ]]; then
  echo "Missing $ROOT/data: first supply the original official PMME datasets." >&2
  exit 2
fi

for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    case "$variant" in
      uniform|gradient|horizon|reliable) ;;
      *) echo "Unknown variant: $variant" >&2; exit 2 ;;
    esac
    WORK="$RUN_ROOT/$variant/seed$seed"
    mkdir -p "$WORK"
    for d in data configs model; do
      if [[ ! -e "$WORK/$d" ]]; then
        ln -s "$ROOT/$d" "$WORK/$d"
      fi
    done
    if [[ -f "$WORK/COMPLETED" ]]; then
      echo "[SKIP] F-variant=$variant seed=$seed already finished"
      continue
    fi
    echo "[START] variant=$variant seed=$seed: $WORK"
    rm -f "$WORK/FAILED"
    if (
      cd "$WORK"
      export PMME_TRANSFER_VARIANT="$variant"
      export CUDA_VISIBLE_DEVICES="$GPU"
      export OMP_NUM_THREADS="$THREADS"
      python -u "$ROOT/train_DA_OT.py" \
        --train_epochs 3000 \
        --finetune_epochs 6000 \
        --data_list chengdu_shenzhen_metr \
        --test_dataset pems-bay \
        --target_days 3 \
        --config_filename ./configs/config.yaml \
        --seed "$seed" \
        --classifier 0.5 \
        --xySquaredRatio 100 \
        --DAcoef 1 \
        --DAepoch 1500 \
        --MemSize 10000 \
        --projection 1 \
        --DA OT \
        --model_name SOFTS > train.out 2>&1
    ); then
      if grep -q "2ndStage Source training Ending!!!" "$WORK/train.out"; then
        touch "$WORK/COMPLETED"
        echo "[DONE] variant=$variant seed=$seed"
      else
        touch "$WORK/FAILED"
        echo "[ERROR] Python exited, but no Stage-II completion marker: $WORK/train.out" >&2
        exit 1
      fi
    else
      touch "$WORK/FAILED"
      echo "[ERROR] variant=$variant seed=$seed. Check $WORK/train.out" >&2
      exit 1
    fi
  done
done
echo "All requested F1-F4 runs finished."
