#!/usr/bin/env bash
# F2 ablation: original + geometric median + Fisher influence proxy + CVaR + triple.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_BRANCH="experiment/pmme-f2-robust-fisher-cvar-20261128"
CURRENT_BRANCH="$(git -C "$ROOT" branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "[ERROR] Expected branch $EXPECTED_BRANCH, got $CURRENT_BRANCH" >&2
  exit 2
fi
[[ -d "$ROOT/data" ]] || { echo "[ERROR] Missing dataset directory $ROOT/data" >&2; exit 2; }

RUN_ROOT="${PMME_F2_RUN_ROOT:-$ROOT/experiments/f2-robust-fisher-cvar-20261128}"
read -r -a METHODS <<< "${PMME_F2_METHODS:-triple}"
read -r -a SEEDS <<< "${PMME_F2_SEEDS:-1}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
SCALE="${PMME_F2_OT_SCALE:-0.01}"
STAGE1="${PMME_F2_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_F2_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_F2_DAEPOCH:-1500}"
GEOMED_ITERS="${PMME_F2_GEOMED_ITERS:-6}"
FISHER_DAMPING="${PMME_F2_FISHER_DAMPING:-0.1}"
CVAR_BLOCKS="${PMME_F2_CVAR_BLOCKS:-5}"
CVAR_TAIL="${PMME_F2_CVAR_TAIL:-0.4}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

python - "$SCALE" "$STAGE1" "$STAGE2" "$DAEPOCH" "$GEOMED_ITERS" "$FISHER_DAMPING" "$CVAR_BLOCKS" "$CVAR_TAIL" <<'PY'
import math, sys
scale, damping, tail = map(float, (sys.argv[1], sys.argv[6], sys.argv[8]))
stage1, stage2, daepoch, iters, blocks = map(int, (sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[7]))
assert math.isfinite(scale) and scale > 0
assert stage1 > 0 and stage2 >= 0 and 0 <= daepoch < stage1
assert iters >= 1 and math.isfinite(damping) and damping > 0
assert blocks >= 2 and math.isfinite(tail) and 0 < tail <= 1
PY

for method in "${METHODS[@]}"; do
  case "$method" in original|geomed|influence|cvar|triple) ;; *)
    echo "[ERROR] Method must be original/geomed/influence/cvar/triple: $method" >&2
    exit 2 ;; esac
done
mkdir -p "$RUN_ROOT"

for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    WORK="$RUN_ROOT/$method/seed$seed"
    mkdir -p "$WORK"
    for folder in data configs model; do
      [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
    done
    EXPECTED="$(printf 'branch=%s\ngit_commit=%s\nseed=%s\nf2_method=%s\ntransfer_variant=gradient\npair_variant=f2\nfeedback_mode=g0\not_scale=%s\nstage1_epochs=%s\nstage2_epochs=%s\ndaepoch=%s\ngeomed_iters=%s\nfisher_damping=%s\ncvar_blocks=%s\ncvar_tail=%s\n' \
      "$EXPECTED_BRANCH" "$COMMIT" "$seed" "$method" "$SCALE" "$STAGE1" "$STAGE2" "$DAEPOCH" "$GEOMED_ITERS" "$FISHER_DAMPING" "$CVAR_BLOCKS" "$CVAR_TAIL")"
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
    echo "[START] F2 method=$method seed=$seed"
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
      export PMME_F2_GEOMED_ITERS="$GEOMED_ITERS"
      export PMME_F2_FISHER_DAMPING="$FISHER_DAMPING"
      export PMME_F2_CVAR_BLOCKS="$CVAR_BLOCKS"
      export PMME_F2_CVAR_TAIL="$CVAR_TAIL"
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
      if [[ "$method" != "original" ]]; then
        grep -q "f2_extension.*'method': '$method'" "$WORK/train.out" || {
          echo "[ERROR] F2 method did not activate: $method" >&2
          touch "$WORK/FAILED"; exit 1
        }
      fi
      touch "$WORK/COMPLETED"
      echo "[DONE] $WORK"
    else
      touch "$WORK/FAILED"
      echo "[ERROR] Failed: $WORK; inspect train.out" >&2
      tail -n 60 "$WORK/train.out" >&2
      exit 1
    fi
  done
done
echo "Completed requested F2 robust/Fisher/CVaR experiments."
