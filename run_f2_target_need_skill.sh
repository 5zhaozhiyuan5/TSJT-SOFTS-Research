#!/usr/bin/env bash
# NEW Stage-I module: target demand x source skill conditional OT matching.
# Full 3000-epoch Stage-I plus UNCHANGED official Stage-II (6000 epochs).
# This is a real model-training experiment, NOT a post-checkpoint H11 audit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="experiment/pmme-f2-target-need-skill-matching-20261127"
if [[ "$(git -C "$ROOT" branch --show-current)" != "$BRANCH" ]];then
  echo "[ERROR] Wrong branch, expected $BRANCH" >&2
  exit 2
fi
if [[ ! -d "$ROOT/data" ]];then
  echo "[ERROR] Missing existing local PMME data directory" >&2
  exit 2
fi
RUN_ROOT="${PMME_TDSM_RUN_ROOT:-$ROOT/experiments/f2-tdsm}"
RUN_ROOT="$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$RUN_ROOT")"
STAGE1="${PMME_TDSM_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_TDSM_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_TDSM_DAEPOCH:-1500}"
STRENGTH="${PMME_TDSM_STRENGTH:-0.35}"
PROTOTYPES="${PMME_TDSM_PROTOTYPES:-32}"
TEMPERATURE="${PMME_TDSM_TEMPERATURE:-1.0}"
GPU="${PMME_TDSM_GPU:-0}"
SEED="${PMME_TDSM_SEED:-1}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
python - "$STAGE1" "$STAGE2" "$DAEPOCH" "$STRENGTH" "$PROTOTYPES" "$TEMPERATURE" "$SEED" <<'PY'
import math,sys
stage1,stage2,da,seed_points=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[5])
strength,temp=float(sys.argv[4]),float(sys.argv[6])
seed=int(sys.argv[7])
assert 0<stage1 and 0<=stage2 and 0<=da<stage1
assert math.isfinite(strength) and 0<strength<=1
assert seed_points>=2 and math.isfinite(temp) and temp>0 and seed>=1
PY
mkdir -p "$RUN_ROOT"
WORK="$RUN_ROOT/tdsm/seed$SEED"
mkdir -p "$WORK"
for folder in data configs model;do
  [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
done
CONFIG="$(printf 'branch=%s\ncommit=%s\nseed=%s\nmode=tdsm\ntdsm_strength=%s\ntdsm_prototypes=%s\ntdsm_temperature=%s\nprc_strength=0\ntransfer_variant=gradient\nf2_method=original\npair_variant=f2\nfeedback=g0\not_schedule=t0\not_scale=0.01\nprojection=1\nstage1_epochs=%s\nstage2_epochs=%s\ndaepoch=%s\n' \
  "$BRANCH" "$COMMIT" "$SEED" "$STRENGTH" "$PROTOTYPES" "$TEMPERATURE" "$STAGE1" "$STAGE2" "$DAEPOCH")"
if [[ -f "$WORK/run_config.txt" ]];then
  if [[ "$(cat "$WORK/run_config.txt")" != "$CONFIG" ]];then
    echo "[ERROR] Existing run uses another code commit/settings; change RUN_ROOT" >&2
    exit 3
  fi
else
  printf '%s\n' "$CONFIG" >"$WORK/run_config.txt"
fi
if [[ -f "$WORK/COMPLETED" ]];then
  echo "[SKIP] TDSM training completed: $WORK"
  exit 0
fi
rm -f "$WORK/FAILED"
echo "[START] F2+TDSM actual training; target PEMS-BAY, seed$SEED"
if (
  cd "$WORK"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export OMP_NUM_THREADS="${PMME_THREADS:-4}"
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
  export PMME_F2_ACCURACY_STRENGTH=0
  export PMME_F2_TDSM_STRENGTH="$STRENGTH"
  export PMME_F2_TDSM_PROTOTYPES="$PROTOTYPES"
  export PMME_F2_TDSM_TEMPERATURE="$TEMPERATURE"
  python -u "$ROOT/train_DA_OT.py" \
    --train_epochs "$STAGE1" --finetune_epochs "$STAGE2" \
    --data_list chengdu_shenzhen_metr --test_dataset pems-bay \
    --target_days 3 --config_filename ./configs/config.yaml \
    --seed "$SEED" --classifier 0.5 --xySquaredRatio 100 \
    --DAcoef 1 --DAepoch "$DAEPOCH" --MemSize 10000 \
    --projection 1 --DA OT --model_name SOFTS
) >"$WORK/train.out" 2>&1;then
  if ! grep -Fq "Source training Ending!!!" "$WORK/train.out" ||
     { [[ "$STAGE2" -gt 0 ]] &&
       ! grep -Fq "2ndStage Source training Ending!!!" "$WORK/train.out";};then
    echo "[ERROR] Stage completion marker missing" >&2
    touch "$WORK/FAILED";tail -n 35 "$WORK/train.out" >&2;exit 1
  fi
  if ! grep -Fq "PMME_F2_TDSM_CONFIG strength $STRENGTH" "$WORK/train.out" ||
     ! grep -Fq "f2_tdsm_matching epoch $DAEPOCH" "$WORK/train.out";then
    echo "[ERROR] TDSM branch not active in real OT training" >&2
    touch "$WORK/FAILED";exit 1
  fi
  touch "$WORK/COMPLETED"
  python "$ROOT/summarize_f2_tdsm.py" --run-root "$RUN_ROOT" \
    >"$RUN_ROOT/summary.txt" 2>"$RUN_ROOT/summary.err"
  echo "[DONE] F2+TDSM Stage-I and Stage-II; see $RUN_ROOT/summary.txt"
  cat "$RUN_ROOT/summary.txt"
else
  echo "[ERROR] F2+TDSM actual training failed; see $WORK/train.out" >&2
  touch "$WORK/FAILED"
  tail -n 40 "$WORK/train.out" >&2
  exit 1
fi
