#!/usr/bin/env bash
# H2-A1/A2/A3: one F2 training, three non-intervening diagnostics.
# PMME_SEEDS="1" bash run_h2_candidate_audit.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_RUN_ROOT:-$ROOT/experiments/h2-candidate-audit}"
read -r -a SEEDS <<< "${PMME_SEEDS:-1}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
SCALE="${PMME_OT_SCALE:-0.01}"
INTERVAL="${PMME_H2_INTERVAL:-200}"
CITYWISE="${PMME_H2_CITYWISE:-1}"
STAGE2_EPOCHS="${PMME_H2_STAGE2_EPOCHS:-0}"
GAIN_TOL="${PMME_H2_GAIN_TOL:-1e-6}"

[[ -d "$ROOT/data" ]] || { echo "[ERROR] Missing $ROOT/data" >&2; exit 2; }
python - "$SCALE" "$INTERVAL" "$STAGE2_EPOCHS" "$GAIN_TOL" <<'PY'
import math,sys
scale=float(sys.argv[1])
interval=int(sys.argv[2])
stage2=int(sys.argv[3])
tolerance=float(sys.argv[4])
assert math.isfinite(scale) and scale>0
assert interval>0 and stage2>=0
assert math.isfinite(tolerance) and tolerance>=0
PY
case "$CITYWISE" in 0|1) ;; *) echo "[ERROR] PMME_H2_CITYWISE must be 0 or 1" >&2; exit 2 ;; esac
SCALE_TAG="${SCALE//./p}"
CURRENT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
mkdir -p "$RUN_ROOT"
for seed in "${SEEDS[@]}"; do
  WORK="$RUN_ROOT/scale$SCALE_TAG/interval$INTERVAL/citywise$CITYWISE/stage2$STAGE2_EPOCHS/seed$seed"
  mkdir -p "$WORK"
  for folder in data configs model; do
    [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
  done
  if [[ -f "$WORK/run_config.txt" ]]; then
    if ! grep -Fxq "git_commit=$CURRENT_COMMIT" "$WORK/run_config.txt"; then
      echo "[ERROR] Existing output uses a different commit: $WORK" >&2
      exit 3
    fi
  fi
  if [[ -f "$WORK/COMPLETED" ]]; then
    echo "[SKIP] Already complete: $WORK"
    continue
  fi
  printf 'feedback_mode=h2a\nseed=%s\not_scale=%s\ninterval=%s\ncitywise=%s\nstage2_epochs=%s\ngain_tolerance=%s\ngit_commit=%s\n' \
    "$seed" "$SCALE" "$INTERVAL" "$CITYWISE" "$STAGE2_EPOCHS" "$GAIN_TOL" "$CURRENT_COMMIT" > "$WORK/run_config.txt"
  echo "[START] H2-A1/A2/A3 seed=$seed citywise=$CITYWISE stage2=$STAGE2_EPOCHS"
  rm -f "$WORK/FAILED"
  if (
    cd "$WORK"
    export PMME_TRANSFER_VARIANT=gradient
    export PMME_FEEDBACK_MODE=h2a
    export PMME_TRANSFER_OT_SCALE="$SCALE"
    export PMME_FEEDBACK_INTERVAL="$INTERVAL"
    export PMME_H2_CITYWISE="$CITYWISE"
    export PMME_H2_GAIN_TOL="$GAIN_TOL"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export OMP_NUM_THREADS="$THREADS"
    python -u "$ROOT/train_DA_OT.py" \
      --train_epochs 3000 \
      --finetune_epochs "$STAGE2_EPOCHS" \
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
    if ! grep -q "Source training Ending!!!" "$WORK/train.out"; then
      echo "[ERROR] Stage-I completion marker missing" >&2
      touch "$WORK/FAILED"; exit 1
    fi
    if [[ "$STAGE2_EPOCHS" -gt 0 ]] &&
       ! grep -q "2ndStage Source training Ending!!!" "$WORK/train.out"; then
      echo "[ERROR] Stage-II completion marker missing" >&2
      touch "$WORK/FAILED"; exit 1
    fi
    if ! grep -q "h2_candidate_audit" "$WORK/train.out"; then
      echo "[ERROR] No H2 audit logs" >&2
      touch "$WORK/FAILED"; exit 1
    fi
    if ! python "$ROOT/summarize_h2_candidates.py" "$WORK/train.out" \
      --output-dir "$WORK"; then
      echo "[ERROR] Failed to summarize audit" >&2
      touch "$WORK/FAILED"; exit 1
    fi
    touch "$WORK/COMPLETED"
    echo "[DONE] $WORK"
  else
    touch "$WORK/FAILED"
    echo "[ERROR] Training failure, see $WORK/train.out" >&2
    exit 1
  fi
done
echo "H2 diagnostic set finished."
