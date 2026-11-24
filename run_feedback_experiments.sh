#!/usr/bin/env bash
# G0-G3: sequential PEMS-BAY/SOFTS seed1/3/5 feedback ablation.
# First pass: PMME_SEEDS="1" bash run_feedback_experiments.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_RUN_ROOT:-$ROOT/experiments/f2-feedback}"
read -r -a MODES <<< "${PMME_MODES:-g0 g1 g2 g3}"
read -r -a SEEDS <<< "${PMME_SEEDS:-1 3 5}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
SCALE="${PMME_OT_SCALE:-0.01}"
INTERVAL="${PMME_FEEDBACK_INTERVAL:-200}"
mkdir -p "$RUN_ROOT"
[[ -d "$ROOT/data" ]] || { echo "Missing official $ROOT/data" >&2; exit 2; }
for mode in "${MODES[@]}"; do
  case "$mode" in g0|g1|g2|g3) ;; *) echo "Invalid $mode" >&2; exit 2;; esac
done
python - "$SCALE" "$INTERVAL" <<'PY'
import math,sys
v=float(sys.argv[1])
i=int(sys.argv[2])
assert math.isfinite(v) and v > 0 and i > 0
PY
SCALE_TAG="${SCALE//./p}"
for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    WORK="$RUN_ROOT/$mode/scale$SCALE_TAG/seed$seed"
    mkdir -p "$WORK"
    for d in data configs model; do
      [[ -e "$WORK/$d" ]] || ln -s "$ROOT/$d" "$WORK/$d"
    done
    CURRENT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [[ -f "$WORK/COMPLETED" ]]; then
      if [[ ! -f "$WORK/run_config.txt" ]] || ! grep -Fxq "git_commit=$CURRENT_COMMIT" "$WORK/run_config.txt"; then
        echo "[ERROR] Existing completed log used a different code commit: $WORK" >&2
        echo "Move or explicitly archive that run before executing new code." >&2
        exit 3
      fi
      echo "[SKIP] $mode seed=$seed scale=$SCALE"
      continue
    fi
    if [[ -f "$WORK/run_config.txt" ]] &&
       ! grep -Fxq "git_commit=$CURRENT_COMMIT" "$WORK/run_config.txt"; then
      echo "[ERROR] Existing interrupted log uses a different code commit: $WORK" >&2
      exit 3
    fi
    printf 'mode=%s\nvariant=gradient\nseed=%s\not_scale=%s\ninterval=%s\ngit_commit=%s\n' \
      "$mode" "$seed" "$SCALE" "$INTERVAL" \
      "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" > "$WORK/run_config.txt"
    echo "[START] $mode seed=$seed scale=$SCALE"
    rm -f "$WORK/FAILED"
    if (
      cd "$WORK"
      export PMME_TRANSFER_VARIANT=gradient
      export PMME_FEEDBACK_MODE="$mode"
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
      if grep -q "2ndStage Source training Ending!!!" "$WORK/train.out"; then
        touch "$WORK/COMPLETED"
        echo "[DONE] $mode seed=$seed scale=$SCALE"
      else
        touch "$WORK/FAILED"
        echo "[ERROR] No Stage-II completion marker in $WORK/train.out" >&2
        exit 1
      fi
    else
      touch "$WORK/FAILED"
      echo "[ERROR] See $WORK/train.out" >&2
      exit 1
    fi
  done
done
echo "All G0-G3 runs completed."
