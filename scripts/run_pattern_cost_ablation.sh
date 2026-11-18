#!/usr/bin/env bash
set -euo pipefail

# Run from repository root. Both arms use the SAME seed, source data and
# checkpoint policy. Only the Stage-I pattern matching cost weight changes.
if [[ ! -f train_DA_OT.py ]]; then
  echo "Run this script from the repository root" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-./configs/config.yaml}"
TEST_DATASET="${TEST_DATASET:-metr-la}"
DATA_LIST="${DATA_LIST:-chengdu_shenzhen_pems}"
SEED="${SEED:-1}"
TARGET_DAYS="${TARGET_DAYS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3000}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-6000}"
DA_EPOCH="${DA_EPOCH:-1500}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logged_files/${TEST_DATASET}/${DATA_LIST}/pattern_cost_ablation/${RUN_ID}}"
DA_COEF="${DA_COEF:-0.01}"
XY_SQUARED_RATIO="${XY_SQUARED_RATIO:-100}"
PROJECTION="${PROJECTION:-1}"
MEM_SIZE="${MEM_SIZE:-10000}"

if (( TRAIN_EPOCHS <= DA_EPOCH || FINETUNE_EPOCHS <= 0 )); then
  echo "Need TRAIN_EPOCHS > DA_EPOCH and FINETUNE_EPOCHS > 0" >&2
  exit 2
fi

# For resume/debug, override ARMS="no_pattern" or ARMS="uniform".
read -r -a ARMS_ARRAY <<< "${ARMS:-no_pattern uniform}"
for arm in "${ARMS_ARRAY[@]}"; do
  case "${arm}" in no_pattern|uniform) ;; *)
    echo "Unknown arm: ${arm} (allowed: no_pattern uniform)" >&2
    exit 2 ;;
  esac
done

echo "Pattern-cost causal ablation; arms: ${ARMS_ARRAY[*]}"
echo "Source: ${DATA_LIST}, target: ${TEST_DATASET}, days: ${TARGET_DAYS}, seed: ${SEED}"
echo "Stage-I: ${TRAIN_EPOCHS}, Stage-II: ${FINETUNE_EPOCHS}, OT starts: ${DA_EPOCH}"
echo "Output: ${OUTPUT_ROOT}"

for arm in "${ARMS_ARRAY[@]}"; do
  if [[ "${arm}" == no_pattern ]]; then
    weight=0.0
  else
    weight=0.5
  fi

  run_dir="${OUTPUT_ROOT}/${arm}_seed${SEED}"
  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to append to an existing run: ${run_dir}" >&2
    exit 3
  fi

  echo "=== Running ${arm}: pattern_weight=${weight}, router=uniform ==="
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train_DA_OT.py \
    --config_filename "${CONFIG}" \
    --test_dataset "${TEST_DATASET}" \
    --data_list "${DATA_LIST}" \
    --seed "${SEED}" \
    --target_days "${TARGET_DAYS}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --finetune_epochs "${FINETUNE_EPOCHS}" \
    --DAepoch "${DA_EPOCH}" \
    --DAcoef "${DA_COEF}" \
    --xySquaredRatio "${XY_SQUARED_RATIO}" \
    --projection "${PROJECTION}" \
    --MemSize "${MEM_SIZE}" \
    --gpu 0 \
    --STmodel SOFTS \
    --model_name SOFTS \
    --his_num 288 \
    --DA OT \
    --pattern_moe_router_mode uniform \
    --pattern_moe_pattern_weight "${weight}" \
    --fixed_final_checkpoint \
    --description "controlled_pattern_cost_${arm}" \
    --save_dir "${run_dir}"
done

if [[ " ${ARMS_ARRAY[*]} " == *" no_pattern "* && " ${ARMS_ARRAY[*]} " == *" uniform "* ]]; then
  "${PYTHON_BIN}" scripts/summarize_pattern_cost_ablation.py "${OUTPUT_ROOT}" --seed "${SEED}"
else
  echo "Partial run complete; run summarizer after both arms exist."
fi
echo "Results: ${OUTPUT_ROOT}"
