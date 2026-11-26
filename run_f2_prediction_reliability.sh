#!/usr/bin/env bash
# Only ONE new module: source prediction-reliability calibration of F2 alpha.
# f2:  strength=0 (exact original F2)
# prc: strength=0.5 (one pre-registered pilot, no parameter sweep)
# Stage-I SOFTS / PEMS-BAY 3-day support; Stage-II official untouched.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_PR_RUN_ROOT:-$ROOT/experiments/f2-prediction-reliability}"
read -r -a MODES <<< "${PMME_PR_MODES:-f2 prc}"
read -r -a SEEDS <<< "${PMME_PR_SEEDS:-1}"
STAGE1="${PMME_PR_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_PR_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_PR_DAEPOCH:-1500}"
STRENGTH="${PMME_PR_STRENGTH:-0.5}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
BRANCH="$(git -C "$ROOT" branch --show-current)"
[[ -d "$ROOT/data" ]] || { echo "[ERROR] Missing source/target data directory $ROOT/data" >&2; exit 2; }

python - "$STAGE1" "$STAGE2" "$DAEPOCH" "$STRENGTH" <<'PY'
import math,sys
stage1,stage2,daepoch=map(int,sys.argv[1:4])
strength=float(sys.argv[4])
assert stage1>0 and stage2>=0 and 0<=daepoch<stage1
assert math.isfinite(strength) and 0<=strength<=1
PY
for mode in "${MODES[@]}"; do
  case "$mode" in f2|prc) ;; *)
    echo "[ERROR] PMME_PR_MODES must contain f2 and/or prc" >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    if [[ "$mode" == f2 ]]; then
      CURRENT_STRENGTH=0
    else
      CURRENT_STRENGTH="$STRENGTH"
    fi
    WORK="$RUN_ROOT/$mode/seed$seed"
    mkdir -p "$WORK"
    for folder in data configs model; do
      [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
    done
    CFG="$(printf 'branch=%s\ngit_commit=%s\nseed=%s\nmode=%s\naccuracy_strength=%s\ntransfer_variant=gradient\nf2_method=original\npair_variant=f2\nfeedback_mode=g0\not_schedule=t0\not_scale=0.01\nstage1_epochs=%s\nstage2_epochs=%s\ndaepoch=%s\n' \
      "$BRANCH" "$COMMIT" "$seed" "$mode" "$CURRENT_STRENGTH" "$STAGE1" "$STAGE2" "$DAEPOCH")"
    if [[ -f "$WORK/run_config.txt" ]]; then
      [[ "$(cat "$WORK/run_config.txt")" == "$CFG" ]] || {
        echo "[ERROR] This directory belongs to a different experiment: $WORK" >&2; exit 3;
      }
    else
      printf '%s\n' "$CFG" > "$WORK/run_config.txt"
    fi
    [[ -f "$WORK/COMPLETED" ]] && { echo "[SKIP] $mode seed$seed"; continue; }
    rm -f "$WORK/FAILED"
    echo "[RUN] $mode seed$seed prediction-reliability-strength=$CURRENT_STRENGTH"
    if (
      cd "$WORK"
      export CUDA_VISIBLE_DEVICES="$GPU"
      export OMP_NUM_THREADS="$THREADS"
      export PMME_TRANSFER_VARIANT=gradient
      export PMME_F2_METHOD=original
      export PMME_PAIR_VARIANT=f2
      export PMME_PAIR_STRENGTH=0
      export PMME_FEEDBACK_MODE=g0
      export PMME_F2_DIAGNOSTIC=off
      export PMME_F2_DIAG_READONLY=0
      export PMME_F2_OT_SCHEDULE=t0
      export PMME_TRANSFER_OT_SCALE=0.01
      export PMME_TRANSFER_TEMPERATURE=8
      export PMME_F2_ACCURACY_STRENGTH="$CURRENT_STRENGTH"
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
      if ! grep -Fq 'Source training Ending!!!' "$WORK/train.out" ||
         { [[ "$STAGE2" -gt 0 ]] &&
           ! grep -Fq '2ndStage Source training Ending!!!' "$WORK/train.out"; }; then
        echo "[ERROR] Stage completion marker missing: $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      if ! grep -Fq "PMME_F2_ACCURACY_CONFIG strength $CURRENT_STRENGTH" "$WORK/train.out"; then
        echo "[ERROR] Expected accuracy mode not activated: $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      if [[ "$CURRENT_STRENGTH" != 0 ]] &&
         ! grep -Fq "f2_accuracy_reliability epoch $DAEPOCH" "$WORK/train.out"; then
        echo "[ERROR] Missing accuracy-calibration OT log: $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      touch "$WORK/COMPLETED"
      echo "[DONE] $mode seed$seed"
    else
      touch "$WORK/FAILED"
      echo "[ERROR] $mode seed$seed failed; inspect $WORK/train.out" >&2
      tail -n 30 "$WORK/train.out" >&2
      exit 1
    fi
  done
done
python "$ROOT/summarize_f2_prediction_reliability.py" --run-root "$RUN_ROOT"
echo "[OK] F2/PRC comparison finished."
