#!/usr/bin/env bash
# F2 + DPM source-pattern bank attention-guided OT *pair cost*.
# Actual Stage-I training; official Stage-II remains untouched.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_BRANCH="experiment/pmme-f2-dpm-pattern-attention-ot-20261127"
CURRENT_BRANCH="$(git -C "$ROOT" branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "[ERROR] Expected branch $EXPECTED_BRANCH, got $CURRENT_BRANCH" >&2; exit 2
fi
[[ -d "$ROOT/data" ]] || { echo "[ERROR] Missing $ROOT/data" >&2; exit 2; }
RUN_ROOT="${PMME_DPM_RUN_ROOT:-$ROOT/experiments/dpm-pattern-attention-ot}"
RUN_ROOT="$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$RUN_ROOT")"
STAGE1="${PMME_DPM_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_DPM_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_DPM_DAEPOCH:-1500}"
STRENGTH="${PMME_DPM_STRENGTH:-0.02}"
PATTERNS="${PMME_DPM_K:-32}"
TEMPERATURE="${PMME_DPM_TEMP:-0.75}"
BANK_INTERVAL="${PMME_DPM_INTERVAL:-50}"
BANK_PER_CITY="${PMME_DPM_PER_CITY:-128}"
COST_BLOCK="${PMME_DPM_BLOCK:-128}"
MODE="${PMME_DPM_MODE:-full}"
GPU="${PMME_DPM_GPU:-0}"
SEED="${PMME_DPM_SEED:-1}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
python - "$STAGE1" "$STAGE2" "$DAEPOCH" "$STRENGTH" "$PATTERNS" \
         "$TEMPERATURE" "$BANK_INTERVAL" "$BANK_PER_CITY" "$COST_BLOCK" "$SEED" "$MODE" <<'PY'
import math,sys
a,b,d,k,interval,per_city,block,seed=map(int,(
    sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[5],
    sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10]))
strength=float(sys.argv[4]);tau=float(sys.argv[6]);mode=sys.argv[11]
assert a>0 and b>=0 and 0<d<a
assert 0<strength<=1 and math.isfinite(strength)
assert k>=2 and tau>0 and math.isfinite(tau)
assert interval>=1 and per_city>=1 and block>=1 and seed>=1
assert mode in ('full','pattern')
PY
WORK="$RUN_ROOT/$MODE/seed$SEED"
mkdir -p "$WORK"
for folder in data configs model; do
  [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
done
CONFIG="$(printf 'branch=%s\ncommit=%s\nseed=%s\nmode=%s\ndpm_strength=%s\ndpm_patterns=%s\ndpm_temperature=%s\ndpm_interval=%s\ndpm_per_city=%s\ndpm_block=%s\noriginal_f2_alpha=1\noriginal_f2_beta_uniform=1\ntdsm_strength=0\not_schedule=t0\not_scale=0.01\nstage1_epochs=%s\nstage2_epochs=%s\ndaepoch=%s\n' \
  "$EXPECTED_BRANCH" "$COMMIT" "$SEED" "$MODE" "$STRENGTH" "$PATTERNS" "$TEMPERATURE" \
  "$BANK_INTERVAL" "$BANK_PER_CITY" "$COST_BLOCK" "$STAGE1" "$STAGE2" "$DAEPOCH")"
if [[ -f "$WORK/run_config.txt" ]]; then
  [[ "$(cat "$WORK/run_config.txt")" == "$CONFIG" ]] || {
    echo "[ERROR] Run directory already contains another config/commit: $WORK" >&2; exit 3
  }
else
  printf '%s\n' "$CONFIG" > "$WORK/run_config.txt"
fi
if [[ -f "$WORK/COMPLETED" ]]; then
  echo "[SKIP] DPM already completed: $WORK"; exit 0
fi
rm -f "$WORK/FAILED"
echo "[START] F2 + DPM pattern attention; seed=$SEED; mode=$MODE; OT lambda=$STRENGTH"
if (
  cd "$WORK"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export OMP_NUM_THREADS="${PMME_THREADS:-4}"
  export PMME_TRANSFER_VARIANT=gradient
  export PMME_TRANSFER_OT_SCALE=0.01
  export PMME_TRANSFER_TEMPERATURE=8
  export PMME_F2_METHOD=original
  export PMME_PAIR_VARIANT=f2
  export PMME_PAIR_STRENGTH=0
  export PMME_FEEDBACK_MODE=g0
  export PMME_F2_DIAGNOSTIC=off
  export PMME_F2_DIAG_READONLY=0
  export PMME_F2_OT_SCHEDULE=t0
  export PMME_F2_ACCURACY_STRENGTH=0
  export PMME_F2_TDSM_STRENGTH=0
  export PMME_DPM_COST_STRENGTH="$STRENGTH"
  export PMME_DPM_BANK_K="$PATTERNS"
  export PMME_DPM_ATTENTION_TEMP="$TEMPERATURE"
  export PMME_DPM_BANK_INTERVAL="$BANK_INTERVAL"
  export PMME_DPM_BANK_PER_CITY="$BANK_PER_CITY"
  export PMME_DPM_COST_BLOCK="$COST_BLOCK"
  export PMME_DPM_MODE="$MODE"
  python -u "$ROOT/train_DA_OT.py" \
    --train_epochs "$STAGE1" --finetune_epochs "$STAGE2" \
    --data_list chengdu_shenzhen_metr --test_dataset pems-bay \
    --target_days 3 --config_filename ./configs/config.yaml \
    --seed "$SEED" --classifier 0.5 --xySquaredRatio 100 \
    --DAcoef 1 --DAepoch "$DAEPOCH" --MemSize 10000 \
    --projection 1 --DA OT --model_name SOFTS
) >"$WORK/train.out" 2>&1; then
  if ! grep -Fq "Source training Ending!!!" "$WORK/train.out" ||
     { [[ "$STAGE2" -gt 0 ]] &&
       ! grep -Fq "2ndStage Source training Ending!!!" "$WORK/train.out"; }; then
    echo "[ERROR] Stage training completion marker missing" >&2
    touch "$WORK/FAILED"; tail -n 45 "$WORK/train.out" >&2; exit 1
  fi
  if ! grep -Fq "PMME_DPM_CONFIG strength $STRENGTH" "$WORK/train.out" ||
     ! grep -Fq "dpm_bank_fitted epoch $DAEPOCH" "$WORK/train.out" ||
     ! grep -Fq "dpm_pair_cost epoch $DAEPOCH" "$WORK/train.out"; then
    echo "[ERROR] Pattern bank or attention OT not active in real training" >&2
    touch "$WORK/FAILED"; tail -n 45 "$WORK/train.out" >&2; exit 1
  fi
  touch "$WORK/COMPLETED"
  if ! python "$ROOT/summarize_dpm_pattern_attention_ot.py" --run-root "$RUN_ROOT" \
       >"$RUN_ROOT/summary.txt" 2>"$RUN_ROOT/summary.err"; then
    rm -f "$WORK/COMPLETED"; touch "$WORK/FAILED"
    echo "[ERROR] Checkpoint-selected metrics summary failed" >&2
    tail -n 45 "$RUN_ROOT/summary.err" >&2; exit 1
  fi
  echo "[DONE] F2+DPM real Stage-I; untouched official Stage-II."
  cat "$RUN_ROOT/summary.txt"
else
  touch "$WORK/FAILED"
  echo "[ERROR] Training failed; see $WORK/train.out" >&2
  tail -n 45 "$WORK/train.out" >&2; exit 1
fi
