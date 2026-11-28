#!/usr/bin/env bash
# One independent experiment: original F2 + source Prediction Reliability Calibration.
# The historical original-F2 result is the control; this script does NOT rerun it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_BRANCH="experiment/pmme-f2-prc-reliability-20261128"
CURRENT_BRANCH="$(git -C "$ROOT" branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "[ERROR] Expected branch $EXPECTED_BRANCH, got $CURRENT_BRANCH" >&2
  exit 2
fi
[[ -d "$ROOT/data" ]] || {
  echo "[ERROR] Missing dataset directory $ROOT/data" >&2
  exit 2
}

RUN_ROOT="${PMME_F2_PRC_RUN_ROOT:-$ROOT/experiments/f2-prc-reliability-20261128}"
read -r -a SEEDS <<< "${PMME_F2_PRC_SEEDS:-1}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
SCALE="${PMME_F2_OT_SCALE:-0.01}"
STAGE1="${PMME_F2_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_F2_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_F2_DAEPOCH:-1500}"
BLEND="${PMME_F2_PRC_BLEND:-0.5}"
SENSITIVITY="${PMME_F2_PRC_SENSITIVITY:-0.5}"
Z_CLIP="${PMME_F2_PRC_Z_CLIP:-3.0}"
MULT_MIN="${PMME_F2_PRC_MULTIPLIER_MIN:-0.5}"
MULT_MAX="${PMME_F2_PRC_MULTIPLIER_MAX:-2.0}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

python - "$SCALE" "$STAGE1" "$STAGE2" "$DAEPOCH" "$BLEND" "$SENSITIVITY" "$Z_CLIP" "$MULT_MIN" "$MULT_MAX" <<'PY'
import math, sys
scale, blend, sensitivity, z_clip, mult_min, mult_max = map(float, (
    sys.argv[1], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9]
))
stage1, stage2, daepoch = map(int, sys.argv[2:5])
assert math.isfinite(scale) and scale > 0
assert stage1 > 0 and stage2 >= 0 and 0 <= daepoch < stage1
assert math.isfinite(blend) and 0 <= blend <= 1
assert math.isfinite(sensitivity) and sensitivity >= 0
assert math.isfinite(z_clip) and z_clip > 0
assert math.isfinite(mult_min) and math.isfinite(mult_max)
assert 0 < mult_min <= mult_max
PY

mkdir -p "$RUN_ROOT"
for seed in "${SEEDS[@]}"; do
  WORK="$RUN_ROOT/seed$seed"
  mkdir -p "$WORK"
  for folder in data configs model; do
    [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
  done

  EXPECTED="$(printf 'branch=%s\ngit_commit=%s\nseed=%s\nf2_method=original\nf2_prc=1\ntransfer_variant=gradient\npair_variant=f2\nfeedback_mode=g0\not_scale=%s\nstage1_epochs=%s\nstage2_epochs=%s\ndaepoch=%s\nprc_blend=%s\nprc_sensitivity=%s\nprc_z_clip=%s\nprc_multiplier_min=%s\nprc_multiplier_max=%s\n' \
    "$EXPECTED_BRANCH" "$COMMIT" "$seed" "$SCALE" "$STAGE1" "$STAGE2" "$DAEPOCH" \
    "$BLEND" "$SENSITIVITY" "$Z_CLIP" "$MULT_MIN" "$MULT_MAX")"
  if [[ -f "$WORK/run_config.txt" ]]; then
    [[ "$(cat "$WORK/run_config.txt")" == "$EXPECTED" ]] || {
      echo "[ERROR] Output directory belongs to different code/settings: $WORK" >&2
      exit 3
    }
  else
    printf '%s\n' "$EXPECTED" > "$WORK/run_config.txt"
  fi

  [[ -f "$WORK/COMPLETED" ]] && { echo "[SKIP] $WORK"; continue; }
  rm -f "$WORK/FAILED"
  echo "[START] F2 + PRC seed=$seed"
  if (
    cd "$WORK"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export OMP_NUM_THREADS="$THREADS"
    export PMME_TRANSFER_VARIANT=gradient
    export PMME_FEEDBACK_MODE=g0
    export PMME_PAIR_VARIANT=f2
    export PMME_PAIR_STRENGTH=0
    export PMME_TRANSFER_OT_SCALE="$SCALE"
    export PMME_F2_METHOD=original
    export PMME_F2_PRC=1
    export PMME_F2_PRC_BLEND="$BLEND"
    export PMME_F2_PRC_SENSITIVITY="$SENSITIVITY"
    export PMME_F2_PRC_Z_CLIP="$Z_CLIP"
    export PMME_F2_PRC_MULTIPLIER_MIN="$MULT_MIN"
    export PMME_F2_PRC_MULTIPLIER_MAX="$MULT_MAX"
    export PMME_F2_DIAGNOSTIC=off
    python -u "$ROOT/train_DA_OT.py" \
      --train_epochs "$STAGE1" \
      --finetune_epochs "$STAGE2" \
      --data_list chengdu_shenzhen_metr \
      --test_dataset pems-bay \
      --target_days 3 \
      --config_filename ./configs/config.yaml \
      --seed "$seed" \
      --classifier 0.5 \
      --xySquaredRatio 100 \
      --DAcoef 1 \
      --DAepoch "$DAEPOCH" \
      --MemSize 10000 \
      --projection 1 \
      --DA OT \
      --model_name SOFTS > train.out 2>&1
  ); then
    grep -q 'Source training Ending!!!' "$WORK/train.out" || {
      echo "[ERROR] Stage-I completion marker missing: $WORK" >&2
      touch "$WORK/FAILED"; exit 1
    }
    if [[ "$STAGE2" -gt 0 ]]; then
      grep -q '2ndStage Source training Ending!!!' "$WORK/train.out" || {
        echo "[ERROR] Stage-II completion marker missing: $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      }
    fi
    grep -q 'PMME_F2_PRC True' "$WORK/train.out" || {
      echo "[ERROR] PRC was not enabled: $WORK" >&2
      touch "$WORK/FAILED"; exit 1
    }
    grep -q "f2_prc.*'method': 'f2_prc'" "$WORK/train.out" || {
      echo "[ERROR] PRC marginal calibration was not logged: $WORK" >&2
      touch "$WORK/FAILED"; exit 1
    }
    touch "$WORK/COMPLETED"
    echo "[DONE] $WORK"
  else
    touch "$WORK/FAILED"
    echo "[ERROR] Failed: $WORK; inspect train.out" >&2
    tail -n 80 "$WORK/train.out" >&2
    exit 1
  fi
done

echo "Completed F2 + source prediction reliability calibration. Original F2 was not rerun."
