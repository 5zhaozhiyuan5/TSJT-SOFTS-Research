#!/usr/bin/env bash
# Four independent F2 upgrades. Run TWO at a time:
# PMME_PAIR_VARIANTS="p2 p4" PMME_PAIR_SEEDS="1" bash run_f2_pair_ot_experiments.sh
# PMME_PAIR_VARIANTS="p1 p3" PMME_PAIR_SEEDS="1" bash run_f2_pair_ot_experiments.sh
# Exact regression: PMME_PAIR_VARIANTS="f2" bash run_f2_pair_ot_experiments.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_PAIR_RUN_ROOT:-$ROOT/experiments/f2-gradient-pair-ot}"
read -r -a VARIANTS <<< "${PMME_PAIR_VARIANTS:-p2 p4}"
read -r -a SEEDS <<< "${PMME_PAIR_SEEDS:-1}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
SCALE="${PMME_PAIR_OT_SCALE:-0.01}"
STRENGTH="${PMME_PAIR_STRENGTH:-0.02}"
STAGE2="${PMME_PAIR_STAGE2_EPOCHS:-6000}"
STAGE1="${PMME_PAIR_STAGE1_EPOCHS:-3000}"
DAEPOCH="${PMME_PAIR_DAEPOCH:-1500}"
ANCHORS="${PMME_PAIR_ANCHORS:-8}"
BLOCK_SIZE="${PMME_PAIR_BLOCK_SIZE:-256}"
P4_MOMENTUM="${PMME_P4_MOMENTUM:-0.8}"
P4_REPEAT="${PMME_P4_REPEAT_PENALTY:-0.08}"
CURRENT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

[[ -d "$ROOT/data" ]] || { echo "[ERROR] Missing $ROOT/data" >&2; exit 2; }
python - "$SCALE" "$STRENGTH" "$STAGE2" "$ANCHORS" "$BLOCK_SIZE" "$P4_MOMENTUM" "$P4_REPEAT" "$STAGE1" "$DAEPOCH" <<'PY'
import math, sys
scale, strength = map(float, sys.argv[1:3])
stage2, anchors, block = map(int, sys.argv[3:6])
momentum, repeat = map(float, sys.argv[6:8])
stage1, daepoch = map(int, sys.argv[8:10])
assert 0 < scale and math.isfinite(scale)
assert 0 <= strength <= 1 and math.isfinite(strength)
assert stage2 >= 0 and anchors > 0 and block > 0
assert stage1 > 0 and 0 <= daepoch < stage1
assert 0 <= momentum < 1 and repeat >= 0 and math.isfinite(repeat)
PY
EFFECT_ON="$(python -c 'import sys; print(int(float(sys.argv[1]) > 0))' "$STRENGTH")"
mkdir -p "$RUN_ROOT"
for variant in "${VARIANTS[@]}"; do
  case "$variant" in f2|p1|p2|p3|p4) ;; *)
    echo "[ERROR] Variant must be f2/p1/p2/p3/p4: $variant" >&2; exit 2 ;; esac
  for seed in "${SEEDS[@]}"; do
    WORK="$RUN_ROOT/$variant/seed$seed"
    mkdir -p "$WORK"
    for folder in data configs model; do
      [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
    done
    if [[ -f "$WORK/run_config.txt" ]]; then
      if ! grep -Fxq "git_commit=$CURRENT_COMMIT" "$WORK/run_config.txt" ||
         ! grep -Fxq "pair_strength=$STRENGTH" "$WORK/run_config.txt" ||
         ! grep -Fxq "stage2_epochs=$STAGE2" "$WORK/run_config.txt" ||
         ! grep -Fxq "stage1_epochs=$STAGE1" "$WORK/run_config.txt" ||
         ! grep -Fxq "daepoch=$DAEPOCH" "$WORK/run_config.txt" ||
         ! grep -Fxq "ot_scale=$SCALE" "$WORK/run_config.txt" ||
         ! grep -Fxq "anchors=$ANCHORS" "$WORK/run_config.txt" ||
         ! grep -Fxq "block_size=$BLOCK_SIZE" "$WORK/run_config.txt" ||
         ! grep -Fxq "p4_momentum=$P4_MOMENTUM" "$WORK/run_config.txt" ||
         ! grep -Fxq "p4_repeat=$P4_REPEAT" "$WORK/run_config.txt"; then
        echo "[ERROR] This output path was created with different settings: $WORK" >&2
        exit 3
      fi
    fi
    if [[ -f "$WORK/COMPLETED" ]]; then
      echo "[SKIP] Already complete: $WORK"
      continue
    fi
    printf 'branch=%s\ngit_commit=%s\nseed=%s\npair_variant=%s\npair_strength=%s\not_scale=%s\nstage2_epochs=%s\nstage1_epochs=%s\ndaepoch=%s\nanchors=%s\nblock_size=%s\np4_momentum=%s\np4_repeat=%s\n' \
      "$(git -C "$ROOT" branch --show-current)" "$CURRENT_COMMIT" "$seed" "$variant" \
      "$STRENGTH" "$SCALE" "$STAGE2" "$STAGE1" "$DAEPOCH" "$ANCHORS" "$BLOCK_SIZE" \
      "$P4_MOMENTUM" "$P4_REPEAT" > "$WORK/run_config.txt"
    rm -f "$WORK/FAILED"
    echo "[START] F2+$variant seed=$seed, stage2=$STAGE2"
    if (
      cd "$WORK"
      export PMME_TRANSFER_VARIANT=gradient
      export PMME_FEEDBACK_MODE=g0
      export PMME_TRANSFER_OT_SCALE="$SCALE"
      export PMME_PAIR_VARIANT="$variant"
      export PMME_PAIR_STRENGTH="$STRENGTH"
      export PMME_PAIR_ANCHORS="$ANCHORS"
      export PMME_PAIR_BLOCK_SIZE="$BLOCK_SIZE"
      export PMME_P4_MOMENTUM="$P4_MOMENTUM"
      export PMME_P4_REPEAT_PENALTY="$P4_REPEAT"
      export CUDA_VISIBLE_DEVICES="$GPU"
      export OMP_NUM_THREADS="$THREADS"
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
        echo "[ERROR] Stage-I completion marker missing: $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      if [[ "$STAGE2" -gt 0 ]] &&
         ! grep -q '2ndStage Source training Ending!!!' "$WORK/train.out"; then
        echo "[ERROR] Stage-II completion marker missing: $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      if [[ "$variant" == "p4" && "$EFFECT_ON" == "1" ]] &&
         ! grep -q 'pair_ot_assignment' "$WORK/train.out"; then
        echo "[ERROR] P4 assignment marker missing: $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      if [[ "$EFFECT_ON" == "1" ]] &&
         [[ "$variant" == "p1" || "$variant" == "p2" || "$variant" == "p3" ]] &&
         ! grep -q 'pair_ot_cost' "$WORK/train.out"; then
        echo "[ERROR] Pairwise-cost marker missing: $WORK" >&2
        touch "$WORK/FAILED"; exit 1
      fi
      touch "$WORK/COMPLETED"
      echo "[DONE] $WORK"
    else
      touch "$WORK/FAILED"
      echo "[ERROR] See $WORK/train.out" >&2
      exit 1
    fi
  done
done
echo "All requested F2 pair-OT variants finished."
