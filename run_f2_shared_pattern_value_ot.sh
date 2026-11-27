#!/usr/bin/env bash
# Original F2 + Shared Pattern-Value OT; official Stage-II is unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_BRANCH="experiment/pmme-f2-shared-pattern-value-ot-20261127"
CURRENT_BRANCH="$(git -C "$ROOT" branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "[ERROR] Expected branch $EXPECTED_BRANCH, got $CURRENT_BRANCH" >&2
  exit 2
fi
[[ -d "$ROOT/data" ]] || { echo "[ERROR] Missing data directory" >&2; exit 2; }

RUN_ROOT="${PMME_SPV_RUN_ROOT:-$ROOT/experiments/f2-shared-pattern-value-ot}"
STAGE1="${PMME_SPV_STAGE1_EPOCHS:-3000}"
STAGE2="${PMME_SPV_STAGE2_EPOCHS:-6000}"
DAEPOCH="${PMME_SPV_DAEPOCH:-1500}"
COST_STRENGTH="${PMME_SPV_COST_STRENGTH:-0.05}"
MASS_STRENGTH="${PMME_SPV_MASS_STRENGTH:-0.35}"
PATTERNS="${PMME_SPV_PATTERNS:-32}"
TEMPERATURE="${PMME_SPV_TEMPERATURE:-0.75}"
BANK_INTERVAL="${PMME_SPV_BANK_INTERVAL:-50}"
BANK_PER_CITY="${PMME_SPV_BANK_PER_CITY:-128}"
VALUE_EMA="${PMME_SPV_VALUE_EMA:-0.08}"
BLOCK_SIZE="${PMME_SPV_BLOCK_SIZE:-128}"
LOG_INTERVAL="${PMME_SPV_LOG_INTERVAL:-25}"
SEED="${PMME_SPV_SEED:-1}"
GPU="${PMME_SPV_GPU:-0}"
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

python - "$STAGE1" "$STAGE2" "$DAEPOCH" "$COST_STRENGTH" "$MASS_STRENGTH" \
  "$PATTERNS" "$TEMPERATURE" "$BANK_INTERVAL" "$BANK_PER_CITY" \
  "$VALUE_EMA" "$BLOCK_SIZE" "$LOG_INTERVAL" "$SEED" <<'PY'
import math, sys
s1, s2, da = map(int, sys.argv[1:4])
cost, mass, temp, ema = map(float, (sys.argv[4], sys.argv[5], sys.argv[7], sys.argv[10]))
k, interval, per_city, block, log_interval, seed = map(
    int, (sys.argv[6], sys.argv[8], sys.argv[9], sys.argv[11],
          sys.argv[12], sys.argv[13])
)
assert s1 > 0 and s2 >= 0 and 0 <= da < s1
assert 0 <= cost <= 1 and 0 <= mass <= 1
assert k >= 2 and temp > 0 and math.isfinite(temp)
assert interval >= 1 and per_city >= 1 and block >= 1 and log_interval >= 1 and seed >= 1
assert 0 < ema <= 1 and math.isfinite(ema)
assert cost > 0 or mass > 0
PY

WORK="$RUN_ROOT/seed$SEED"
mkdir -p "$WORK"
for folder in data configs model; do
  [[ -e "$WORK/$folder" ]] || ln -s "$ROOT/$folder" "$WORK/$folder"
done
CONFIG="$(printf 'branch=%s\ncommit=%s\nseed=%s\ncost_strength=%s\nmass_strength=%s\npatterns=%s\ntemperature=%s\nbank_interval=%s\nbank_per_city=%s\nvalue_ema=%s\nblock_size=%s\nlog_interval=%s\nstage1_epochs=%s\nstage2_epochs=%s\ndaepoch=%s\n' \
  "$EXPECTED_BRANCH" "$COMMIT" "$SEED" "$COST_STRENGTH" "$MASS_STRENGTH" \
  "$PATTERNS" "$TEMPERATURE" "$BANK_INTERVAL" "$BANK_PER_CITY" \
  "$VALUE_EMA" "$BLOCK_SIZE" "$LOG_INTERVAL" "$STAGE1" "$STAGE2" "$DAEPOCH")"
if [[ -f "$WORK/run_config.txt" ]]; then
  [[ "$(cat "$WORK/run_config.txt")" == "$CONFIG" ]] || {
    echo "[ERROR] Run directory has a different code/config identity: $WORK" >&2
    exit 3
  }
else
  printf '%s\n' "$CONFIG" > "$WORK/run_config.txt"
fi
if [[ -f "$WORK/COMPLETED" ]]; then
  echo "[SKIP] Run already completed: $WORK"
  exit 0
fi
rm -f "$WORK/FAILED"

echo "[START] F2 + Shared Pattern-Value OT; seed=$SEED"
if (
  cd "$WORK"
  export CUDA_VISIBLE_DEVICES="$GPU"
  export OMP_NUM_THREADS="${PMME_THREADS:-4}"
  export PMME_TRANSFER_VARIANT=gradient
  export PMME_TRANSFER_OT_SCALE=0.01
  export PMME_TRANSFER_TEMPERATURE=8
  export PMME_F2_METHOD=original
  export PMME_PAIR_VARIANT=f2
  export PMME_FEEDBACK_MODE=g0
  export PMME_F2_DIAGNOSTIC=off
  export PMME_F2_DIAG_READONLY=0
  export PMME_SPV_COST_STRENGTH="$COST_STRENGTH"
  export PMME_SPV_MASS_STRENGTH="$MASS_STRENGTH"
  export PMME_SPV_PATTERNS="$PATTERNS"
  export PMME_SPV_TEMPERATURE="$TEMPERATURE"
  export PMME_SPV_BANK_INTERVAL="$BANK_INTERVAL"
  export PMME_SPV_BANK_PER_CITY="$BANK_PER_CITY"
  export PMME_SPV_VALUE_EMA="$VALUE_EMA"
  export PMME_SPV_BLOCK_SIZE="$BLOCK_SIZE"
  export PMME_SPV_LOG_INTERVAL="$LOG_INTERVAL"
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
    echo "[ERROR] Training completion marker missing" >&2
    touch "$WORK/FAILED"
    tail -n 50 "$WORK/train.out" >&2
    exit 1
  fi
  for marker in "PMME_SPV_CONFIG" "SPV_BANK_FITTED" "SPV_OT_DIAG" "SPV_BANK_UPDATE"; do
    if ! grep -Fq "$marker" "$WORK/train.out"; then
      echo "[ERROR] Missing activation marker: $marker" >&2
      touch "$WORK/FAILED"
      tail -n 50 "$WORK/train.out" >&2
      exit 1
    fi
  done
  touch "$WORK/COMPLETED"
  echo "[DONE] F2 + Shared Pattern-Value OT; official Stage-II retained."
  echo "Log: $WORK/train.out"
else
  touch "$WORK/FAILED"
  echo "[ERROR] Training failed; see $WORK/train.out" >&2
  tail -n 50 "$WORK/train.out" >&2
  exit 1
fi
