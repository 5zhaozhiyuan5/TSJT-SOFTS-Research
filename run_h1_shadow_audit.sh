#!/usr/bin/env bash
# H1 diagnosis only: F2 Gradient weights are NOT changed by shadow replay.
# Usage: PMME_SEEDS="1" bash run_h1_shadow_audit.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_RUN_ROOT:-$ROOT/experiments/h1-shadow-audit}"
read -r -a SEEDS <<< "${PMME_SEEDS:-1}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
SCALE="${PMME_OT_SCALE:-0.01}"
INTERVAL="${PMME_H1_INTERVAL:-200}"
mkdir -p "$RUN_ROOT"
[[ -d "$ROOT/data" ]] || { echo "[ERROR] Expected original datasets at $ROOT/data" >&2; exit 2; }
python - "$SCALE" "$INTERVAL" <<'PY'
import math, sys
scale=float(sys.argv[1])
interval=int(sys.argv[2])
assert math.isfinite(scale) and scale>0, "Positive OT scale required"
assert interval>=1, "Positive interval required"
PY

SCALE_TAG="${SCALE//./p}"
CURRENT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
for seed in "${SEEDS[@]}"; do
  WORK="$RUN_ROOT/scale$SCALE_TAG/interval$INTERVAL/seed$seed"
  mkdir -p "$WORK"
  for folder in data configs model; do
    [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
  done
  if [[ -f "$WORK/COMPLETED" ]]; then
    if [[ -f "$WORK/run_config.txt" ]] && grep -Fxq "git_commit=$CURRENT_COMMIT" "$WORK/run_config.txt"; then
      echo "[SKIP] H1 seed=$seed (already completed with same code)"
      continue
    fi
    echo "[ERROR] Existing completed experiment uses different code: $WORK" >&2
    exit 3
  fi
  if [[ -f "$WORK/run_config.txt" ]] &&
     ! grep -Fxq "git_commit=$CURRENT_COMMIT" "$WORK/run_config.txt"; then
    echo "[ERROR] Interrupted experiment used different commit: $WORK" >&2
    exit 3
  fi
  printf 'variant=gradient\nfeedback_mode=h1\nseed=%s\not_scale=%s\ninterval=%s\ngit_commit=%s\n' \
    "$seed" "$SCALE" "$INTERVAL" "$CURRENT_COMMIT" > "$WORK/run_config.txt"
  echo "[START] H1 seed=$seed scale=$SCALE interval=$INTERVAL"
  rm -f "$WORK/FAILED"
  if (
    cd "$WORK"
    export PMME_TRANSFER_VARIANT=gradient
    export PMME_FEEDBACK_MODE=h1
    export PMME_TRANSFER_OT_SCALE="$SCALE"
    export PMME_FEEDBACK_INTERVAL="$INTERVAL"
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
    if ! grep -q "2ndStage Source training Ending!!!" "$WORK/train.out"; then
      echo "[ERROR] Stage-II completion marker missing, check $WORK/train.out" >&2
      touch "$WORK/FAILED"
      exit 1
    fi
    if ! grep -q "h1_shadow_audit" "$WORK/train.out"; then
      echo "[ERROR] No H1 shadow audit records, check $WORK/train.out" >&2
      touch "$WORK/FAILED"
      exit 1
    fi
    python "$ROOT/summarize_h1_shadow.py" "$WORK/train.out" \
      --csv "$WORK/h1_shadow_records.csv" \
      --summary "$WORK/h1_shadow_summary.txt"
    touch "$WORK/COMPLETED"
    echo "[DONE] H1 seed=$seed logs=$WORK"
  else
    touch "$WORK/FAILED"
    echo "[ERROR] H1 seed=$seed failed, see $WORK/train.out" >&2
    exit 1
  fi
done
echo "H1 shadow-audit experiments finished."
