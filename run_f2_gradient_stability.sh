#!/usr/bin/env bash
# Original F2 + full-horizon target-gradient stability calibration.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_BRANCH="experiment/pmme-f2-gradient-bootstrap-stability-20261128"
CURRENT_BRANCH="$(git -C "$ROOT" branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "[ERROR] Expected branch $EXPECTED_BRANCH, got $CURRENT_BRANCH" >&2
  exit 2
fi
[[ -d "$ROOT/data" ]] || { echo "[ERROR] Missing data directory $ROOT/data" >&2; exit 2; }

RUN_ROOT="${PMME_F2_RUN_ROOT:-$ROOT/experiments/f2-gradient-stability-20261128}"
SEEDS_TEXT="${PMME_F2_SEEDS:-1}"
METHOD="${PMME_F2_METHOD:-stability}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
SCALE="${PMME_F2_OT_SCALE:-0.01}"
STAGE1="${PMME_F2_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_F2_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_F2_DAEPOCH:-1500}"
LAMBDA="${PMME_F2_STABILITY_LAMBDA:-0.5}"
SUBSAMPLES="${PMME_F2_STABILITY_SUBSAMPLES:-6}"
RATIO="${PMME_F2_STABILITY_SAMPLE_RATIO:-0.8}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
read -r -a SEEDS <<< "$SEEDS_TEXT"
[[ "$METHOD" == "original" || "$METHOD" == "stability" ]] || { echo "[ERROR] PMME_F2_METHOD must be original or stability" >&2; exit 2; }

python - "$SCALE" "$STAGE1" "$STAGE2" "$DAEPOCH" "$LAMBDA" "$SUBSAMPLES" "$RATIO" "${SEEDS[@]}" <<'PY'
import math, sys
scale, lam, ratio = map(float, (sys.argv[1], sys.argv[5], sys.argv[7]))
stage1, stage2, daepoch, k = map(int, (sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[6]))
seeds = list(map(int, sys.argv[8:]))
assert math.isfinite(scale) and scale > 0
assert stage1 > 0 and stage2 >= 0 and 0 <= daepoch < stage1
assert math.isfinite(lam) and lam >= 0
assert k >= 2 and math.isfinite(ratio) and 0.5 <= ratio <= 1.0
assert seeds and all(s > 0 for s in seeds)
PY

mkdir -p "$RUN_ROOT"
for seed in "${SEEDS[@]}"; do
  WORK="$RUN_ROOT/seed$seed"
  mkdir -p "$WORK"
  for folder in data configs model; do
    [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
  done
  EXPECTED="$(printf 'branch=%s\ngit_commit=%s\nseed=%s\nf2_method=%s\ntransfer_variant=gradient\npair_variant=f2\nfeedback_mode=g0\not_scale=%s\nstage1_epochs=%s\nstage2_epochs=%s\ndaepoch=%s\nlambda=%s\nsubsamples=%s\nsample_ratio=%s\n' \
    "$EXPECTED_BRANCH" "$COMMIT" "$seed" "$METHOD" "$SCALE" "$STAGE1" "$STAGE2" "$DAEPOCH" "$LAMBDA" "$SUBSAMPLES" "$RATIO")"
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
  echo "[START] F2 method=$METHOD seed=$seed"
  if (
    cd "$WORK"
    export CUDA_VISIBLE_DEVICES="$GPU"
    export OMP_NUM_THREADS="$THREADS"
    export PMME_TRANSFER_VARIANT=gradient
    export PMME_FEEDBACK_MODE=g0
    export PMME_PAIR_VARIANT=f2
    export PMME_PAIR_STRENGTH=0
    export PMME_TRANSFER_OT_SCALE="$SCALE"
    export PMME_F2_METHOD="$METHOD"
    export PMME_F2_STABILITY_LAMBDA="$LAMBDA"
    export PMME_F2_STABILITY_SUBSAMPLES="$SUBSAMPLES"
    export PMME_F2_STABILITY_SAMPLE_RATIO="$RATIO"
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
      echo "[ERROR] Stage-I completion marker missing" >&2
      touch "$WORK/FAILED"; exit 1
    }
    if [[ "$STAGE2" -gt 0 ]]; then
      grep -q '2ndStage Source training Ending!!!' "$WORK/train.out" || {
        echo "[ERROR] Stage-II completion marker missing" >&2
        touch "$WORK/FAILED"; exit 1
      }
    fi
    if [[ "$METHOD" == "stability" ]]; then
      grep -q "f2_extension.*'method': 'stability'" "$WORK/train.out" || {
        echo "[ERROR] Gradient-stability mechanism did not activate" >&2
        touch "$WORK/FAILED"; exit 1
      }
    fi
    touch "$WORK/COMPLETED"
    echo "[DONE] $WORK"
  else
    touch "$WORK/FAILED"
    echo "[ERROR] Failed: $WORK; inspect train.out" >&2
    tail -n 50 "$WORK/train.out" >&2
    exit 1
  fi
done
echo "Completed F2 gradient-stability experiments."
