#!/usr/bin/env bash
# Four F2-only variants: SAM-GS / GradDrop / GradVac / bilateral UOT.
# Do not re-run historical original F2. Requires dataset and GPU environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_F2_RUN_ROOT:-$ROOT/experiments/f2-four-marginal-20261126}"
read -r -a METHODS <<< "${PMME_F2_METHODS:-mag sign ema biot}"
read -r -a SEEDS <<< "${PMME_F2_SEEDS:-1}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
SCALE="${PMME_F2_OT_SCALE:-0.01}"
STAGE1="${PMME_F2_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_F2_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_F2_DAEPOCH:-1500}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

[[ -d "$ROOT/data" ]] || { echo "[ERROR] Missing dataset directory $ROOT/data" >&2; exit 2; }
python - "$SCALE" "$STAGE1" "$STAGE2" "$DAEPOCH" <<'PY'
import sys, math
scale = float(sys.argv[1])
stage1, stage2, daepoch = map(int, sys.argv[2:])
assert math.isfinite(scale) and scale > 0
assert stage1 > 0 and stage2 >= 0 and 0 <= daepoch < stage1
PY

mkdir -p "$RUN_ROOT"
for method in "${METHODS[@]}"; do
  case "$method" in mag|sign|ema|biot) ;; *)
    echo "[ERROR] Expected one of mag/sign/ema/biot: $method" >&2
    exit 2 ;; esac
  for seed in "${SEEDS[@]}"; do
    WORK="$RUN_ROOT/$method/seed$seed"
    mkdir -p "$WORK"
    for folder in data configs model; do
      [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
    done
    EXPECTED="$(printf 'branch=%s\ngit_commit=%s\nseed=%s\nf2_method=%s\npair_variant=f2\ntransfer_variant=gradient\nfeedback_mode=g0\not_scale=%s\nstage1_epochs=%s\nstage2_epochs=%s\ndaepoch=%s\n' \
      "$(git -C "$ROOT" branch --show-current)" "$COMMIT" "$seed" "$method" "$SCALE" "$STAGE1" "$STAGE2" "$DAEPOCH")"
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
    echo "[START] method=$method seed=$seed"
    if (
      cd "$WORK"
      export CUDA_VISIBLE_DEVICES="$GPU"
      export OMP_NUM_THREADS="$THREADS"
      export PMME_TRANSFER_VARIANT=gradient
      export PMME_FEEDBACK_MODE=g0
      export PMME_PAIR_VARIANT=f2
      export PMME_PAIR_STRENGTH=0
      export PMME_TRANSFER_OT_SCALE="$SCALE"
      export PMME_F2_METHOD="$method"
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
      if ! grep -q 'Source training Ending!!!' "$WORK/train.out"; then
        echo "[ERROR] Stage I completion marker missing for $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      if [[ "$STAGE2" -gt 0 ]] &&
         ! grep -q '2ndStage Source training Ending!!!' "$WORK/train.out"; then
        echo "[ERROR] Stage II completion marker missing for $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      if ! grep -q "f2_extension.*'method': '$method'" "$WORK/train.out"; then
        echo "[ERROR] F2 mechanism did not activate: $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      touch "$WORK/COMPLETED"
      echo "[DONE] $WORK"
    else
      touch "$WORK/FAILED"
      echo "[ERROR] Failed: $WORK; inspect train.out" >&2
      exit 1
    fi
  done
done
echo "Completed all requested F2 extension runs."
