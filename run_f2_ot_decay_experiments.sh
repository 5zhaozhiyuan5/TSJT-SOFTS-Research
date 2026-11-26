#!/usr/bin/env bash
# T0: exact fixed F2, T1: late OT decay to 0, T2: decay to 0.25x.
# Stage I: SOFTS, PEMS-BAY target 3 days, Chengdu/Shenzhen/METR-LA.
# Stage II: official untouched PMME_2ndStage through existing train_DA_OT.py.
# Default: three matched full runs; first use a separate smoke root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${PMME_T_RUN_ROOT:-$ROOT/experiments/f2-late-ot-decay}"
read -r -a VARIANTS <<< "${PMME_T_VARIANTS:-t0 t1 t2}"
read -r -a SEEDS <<< "${PMME_T_SEEDS:-1}"
GPU="${PMME_GPU:-0}"
THREADS="${PMME_THREADS:-4}"
STAGE1="${PMME_T_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_T_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_T_DAEPOCH:-1500}"
START="${PMME_T_DECAY_START:-2000}"
END="${PMME_T_DECAY_END:-2999}"
BASE="${PMME_T_OT_SCALE:-0.01}"
FLOOR="${PMME_T_TAIL_FRACTION:-0.25}"
GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
GIT_BRANCH="$(git -C "$ROOT" branch --show-current)"
[[ -d "$ROOT/data" ]] || { echo "[ERROR] Missing $ROOT/data" >&2; exit 2; }

python - "$STAGE1" "$STAGE2" "$DAEPOCH" "$START" "$END" "$BASE" "$FLOOR" <<'PY'
import math,sys
stage1, stage2, daepoch, start, end=map(int,sys.argv[1:6])
base, floor=map(float,sys.argv[6:8])
assert stage1>0 and stage2>=0 and 0<=daepoch<=start<end<stage1
assert base>0 and math.isfinite(base)
assert 0<=floor<=1 and math.isfinite(floor)
PY
[[ "${#SEEDS[@]}" -gt 0 && "${#VARIANTS[@]}" -gt 0 ]] || {
  echo "[ERROR] Empty seeds or variants" >&2; exit 2;
}
for m in "${VARIANTS[@]}"; do
  case "$m" in t0|t1|t2) ;; *)
    echo "[ERROR] Choose PMME_T_VARIANTS='t0 t1 t2' or a subset" >&2; exit 2 ;; esac
done

mkdir -p "$RUN_ROOT"
for seed in "${SEEDS[@]}"; do
  for mode in "${VARIANTS[@]}"; do
    WORK="$RUN_ROOT/$mode/seed$seed"
    mkdir -p "$WORK"
    for dir in data configs model; do
      [[ -e "$WORK/$dir" ]] || ln -s "$ROOT/$dir" "$WORK/$dir"
    done
    CONFIG="$(printf 'branch=%s\ngit_commit=%s\nseed=%s\nmode=%s\nf2_method=original\ntransfer_variant=gradient\nfeedback_mode=g0\npair_variant=f2\nstage1_epochs=%s\nstage2_epochs=%s\ndaepoch=%s\ndecay_start=%s\ndecay_end=%s\not_base_scale=%s\nt2_tail_fraction=%s\ngpu=%s\n' \
      "$GIT_BRANCH" "$GIT_COMMIT" "$seed" "$mode" "$STAGE1" "$STAGE2" \
      "$DAEPOCH" "$START" "$END" "$BASE" "$FLOOR" "$GPU")"
    if [[ -f "$WORK/run_config.txt" ]]; then
      if [[ "$(cat "$WORK/run_config.txt")" != "$CONFIG" ]]; then
        echo "[ERROR] Existing run dir has different code/config: $WORK" >&2
        exit 3
      fi
    else
      printf '%s\n' "$CONFIG" > "$WORK/run_config.txt"
    fi
    if [[ -f "$WORK/COMPLETED" ]]; then
      echo "[SKIP] Completed $mode seed=$seed at commit=$GIT_COMMIT"
      continue
    fi
    rm -f "$WORK/FAILED"
    echo "[RUN] mode=$mode seed=$seed Stage-I=$STAGE1 Stage-II=$STAGE2"
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
      export PMME_TRANSFER_OT_SCALE="$BASE"
      export PMME_TRANSFER_TEMPERATURE=8.0
      export PMME_F2_OT_SCHEDULE="$mode"
      export PMME_F2_OT_DECAY_START="$START"
      export PMME_F2_OT_DECAY_END="$END"
      export PMME_F2_OT_TAIL_FRACTION="$FLOOR"
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
      if ! grep -Fq 'Source training Ending!!!' "$WORK/train.out"; then
        touch "$WORK/FAILED"
        echo "[ERROR] Stage-I completion marker missing: $WORK" >&2
        exit 1
      fi
      if [[ "$STAGE2" -gt 0 ]] &&
         ! grep -Fq '2ndStage Source training Ending!!!' "$WORK/train.out"; then
        touch "$WORK/FAILED"
        echo "[ERROR] Stage-II completion marker missing: $WORK" >&2
        exit 1
      fi
      if ! python "$ROOT/verify_f2_ot_schedule_log.py" \
          --log "$WORK/train.out" --mode "$mode" \
          --stage1 "$STAGE1" --daepoch "$DAEPOCH" \
          --decay-start "$START" --decay-end "$END" \
          --base-scale "$BASE" --tail-fraction "$FLOOR" \
          > "$WORK/schedule_check.txt" 2>&1; then
        touch "$WORK/FAILED"
        echo "[ERROR] OT decay numerical/log verification failed: $WORK" >&2
        cat "$WORK/schedule_check.txt" >&2
        exit 1
      fi
      touch "$WORK/COMPLETED"
      echo "[DONE] mode=$mode seed=$seed logs=$WORK"
    else
      touch "$WORK/FAILED"
      echo "[ERROR] Run failed: $WORK" >&2
      tail -n 40 "$WORK/train.out" >&2
      exit 1
    fi
  done
done
python "$ROOT/summarize_f2_ot_decay.py" --run-root "$RUN_ROOT" || {
  echo "[WARN] Could not summarize all runs; check individual train.out" >&2
}
echo "T0/T1/T2 selected variants completed."
