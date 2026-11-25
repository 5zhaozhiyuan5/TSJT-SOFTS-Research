#!/usr/bin/env bash
# Isolated comparison: unchanged F2 vs F2 with internal bidirectional marginals.
# PMME_F2_VARIANTS="gradient gradient_coupled" PMME_F2_SEEDS="1" bash run_f2_internal_bilateral.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_F2_RUN_ROOT:-$ROOT/experiments/f2-internal-bilateral}"
read -r -a MODES <<< "${PMME_F2_VARIANTS:-gradient gradient_coupled}"
read -r -a SEEDS <<< "${PMME_F2_SEEDS:-1}"
GPU="${PMME_GPU:-0}"
SCALE="${PMME_F2_OT_SCALE:-0.01}"
BLEND="${PMME_F2_COUPLED_BLEND:-0.2}"
STEPS="${PMME_F2_COUPLED_STEPS:-2}"
STAGE1="${PMME_F2_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_F2_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_F2_DAEPOCH:-1500}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

[[ -d "$ROOT/data" ]] || { echo "Missing $ROOT/data" >&2; exit 2; }
python - "$SCALE" "$BLEND" "$STEPS" "$STAGE1" "$STAGE2" "$DAEPOCH" <<'PY'
import math, sys
scale, blend = map(float, sys.argv[1:3])
steps, stage1, stage2, daepoch = map(int, sys.argv[3:7])
assert math.isfinite(scale) and scale > 0
assert math.isfinite(blend) and 0 <= blend <= 1
assert steps >= 0 and stage1 > 0 and stage2 >= 0 and 0 <= daepoch < stage1
PY
for mode in "${MODES[@]}"; do
  case "$mode" in gradient|gradient_coupled) ;; *)
    echo "Unknown F2 internal variant $mode" >&2; exit 2 ;; esac
  for seed in "${SEEDS[@]}"; do
    OUT="$RUN_ROOT/$mode/seed$seed"
    mkdir -p "$OUT"
    for folder in data configs model; do
      [[ -e "$OUT/$folder" ]] || ln -s "$ROOT/$folder" "$OUT/$folder"
    done
    CONFIG="$(printf 'commit=%s\nmode=%s\nseed=%s\nscale=%s\nblend=%s\nsteps=%s\nstage1=%s\nstage2=%s\ndaepoch=%s\n' "$COMMIT" "$mode" "$seed" "$SCALE" "$BLEND" "$STEPS" "$STAGE1" "$STAGE2" "$DAEPOCH")"
    if [[ -f "$OUT/run_config.txt" ]] && [[ "$(cat "$OUT/run_config.txt")" != "$CONFIG" ]]; then
      echo "Refusing to mix experiments with different configs at $OUT" >&2
      exit 3
    fi
    printf '%s\n' "$CONFIG" > "$OUT/run_config.txt"
    if [[ -f "$OUT/COMPLETED" ]]; then
      echo "[SKIP] $OUT"; continue
    fi
    rm -f "$OUT/FAILED"
    echo "[START] $mode seed=$seed"
    if (
      cd "$OUT"
      export PMME_TRANSFER_VARIANT="$mode"
      export PMME_FEEDBACK_MODE=g0
      export PMME_PAIR_VARIANT=f2
      export PMME_PAIR_STRENGTH=0
      export PMME_TRANSFER_OT_SCALE="$SCALE"
      export PMME_F2_COUPLED_BLEND="$BLEND"
      export PMME_F2_COUPLED_STEPS="$STEPS"
      export CUDA_VISIBLE_DEVICES="$GPU"
      python -u "$ROOT/train_DA_OT.py" \
        --train_epochs "$STAGE1" --finetune_epochs "$STAGE2" \
        --data_list chengdu_shenzhen_metr --test_dataset pems-bay \
        --target_days 3 --config_filename ./configs/config.yaml \
        --seed "$seed" --classifier 0.5 --xySquaredRatio 100 \
        --DAcoef 1 --DAepoch "$DAEPOCH" --MemSize 10000 \
        --projection 1 --DA OT --model_name SOFTS > train.out 2>&1
    ); then
      if ! grep -q 'Source training Ending!!!' "$OUT/train.out" ||
         { [[ "$STAGE2" -gt 0 ]] && ! grep -q '2ndStage Source training Ending!!!' "$OUT/train.out"; }; then
        touch "$OUT/FAILED"; echo "Missing completion marker $OUT" >&2; exit 1
      fi
      if [[ "$mode" == "gradient_coupled" ]] &&
         ! grep -q 'task_utility gradient_coupled' "$OUT/train.out"; then
        touch "$OUT/FAILED"; echo "Missing coupled diagnostic $OUT" >&2; exit 1
      fi
      touch "$OUT/COMPLETED"
      echo "[DONE] $OUT"
    else
      touch "$OUT/FAILED"; echo "[FAILED] $OUT/train.out" >&2; exit 1
    fi
  done
done
