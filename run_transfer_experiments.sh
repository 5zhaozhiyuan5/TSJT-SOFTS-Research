#!/usr/bin/env bash
# F1-F4 sequential runner, with isolated workdirs for EVERY scale and seed.
# First pass: PMME_SEEDS="1" bash run_transfer_experiments.sh
# Strength sanity: PMME_VARIANTS="uniform gradient" PMME_SEEDS="1" PMME_OT_SCALES="0.01 1.0" bash run_transfer_experiments.sh
# Full: bash run_transfer_experiments.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_RUN_ROOT:-$ROOT/experiments/transfer-utility}"
read -r -a VARIANTS <<< "${PMME_VARIANTS:-uniform gradient horizon reliable}"
read -r -a SEEDS <<< "${PMME_SEEDS:-1 3 5}"
read -r -a SCALES <<< "${PMME_OT_SCALES:-0.01}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
TEMP="${PMME_TRANSFER_TEMPERATURE:-8.0}"
MIN_ESS="${PMME_TRANSFER_MIN_ESS:-0.5}"
mkdir -p "$RUN_ROOT"

if [[ ! -d "$ROOT/data" ]]; then
  echo "Missing $ROOT/data: supply the official datasets first." >&2
  exit 2
fi

for scale in "${SCALES[@]}"; do
  # Avoid accepting arbitrary strings before constructing log folders.
  if ! python - "$scale" <<'PY'
import math
import sys
try:
    v = float(sys.argv[1])
except ValueError:
    sys.exit(1)
sys.exit(0 if math.isfinite(v) and v > 0 else 1)
PY
  then
    echo "Invalid positive PMME_OT_SCALES entry: $scale" >&2
    exit 2
  fi
  scale_tag="${scale//./p}"
  for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      case "$variant" in
        uniform|gradient|horizon|reliable) ;;
        *) echo "Unknown variant: $variant" >&2; exit 2 ;;
      esac
      WORK="$RUN_ROOT/$variant/scale$scale_tag/seed$seed"
      mkdir -p "$WORK"
      for d in data configs model; do
        if [[ ! -e "$WORK/$d" ]]; then
          ln -s "$ROOT/$d" "$WORK/$d"
        fi
      done
      if [[ -f "$WORK/COMPLETED" ]]; then
        echo "[SKIP] $variant seed=$seed scale=$scale"
        continue
      fi
      printf 'variant=%s\nseed=%s\not_scale=%s\ntemperature=%s\nmin_ess=%s\ngit_commit=%s\n' \
        "$variant" "$seed" "$scale" "$TEMP" "$MIN_ESS" \
        "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" > "$WORK/run_config.txt"
      echo "[START] $variant seed=$seed scale=$scale, logs=$WORK"
      rm -f "$WORK/FAILED"
      if (
        cd "$WORK"
        export PMME_TRANSFER_VARIANT="$variant"
        export PMME_TRANSFER_OT_SCALE="$scale"
        export PMME_TRANSFER_TEMPERATURE="$TEMP"
        export PMME_TRANSFER_MIN_ESS="$MIN_ESS"
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
          echo "[DONE] $variant seed=$seed scale=$scale"
        else
          touch "$WORK/FAILED"
          echo "[ERROR] No Stage-II completion marker: $WORK/train.out" >&2
          exit 1
        fi
      else
        touch "$WORK/FAILED"
        echo "[ERROR] $variant seed=$seed scale=$scale. See $WORK/train.out" >&2
        exit 1
      fi
    done
  done
done
echo "All selected F1-F4 jobs completed."
