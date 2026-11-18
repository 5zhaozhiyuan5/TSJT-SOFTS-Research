#!/usr/bin/env bash
set -euo pipefail

# By default run only the four NEW fixed-gate ablations.
# To obtain a strictly matched learned control under the corrected normalization:
# MODES="learned uniform expert1 expert2 expert3" bash scripts/run_moe_expert_ablation.sh
# Defaults reproduce the METR-LA target / PEMS-BAY+Chengdu+Shenzhen source setup.
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-./configs/config.yaml}"
TEST_DATASET="${TEST_DATASET:-metr-la}"
DATA_LIST="${DATA_LIST:-chengdu_shenzhen_pems}"
SEED="${SEED:-1}"
TARGET_DAYS="${TARGET_DAYS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3000}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-6000}"
DA_EPOCH="${DA_EPOCH:-1500}"
DA_COEF="${DA_COEF:-0.01}"
XY_SQUARED_RATIO="${XY_SQUARED_RATIO:-100}"
PROJECTION="${PROJECTION:-1}"
MEM_SIZE="${MEM_SIZE:-10000}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logged_files/${TEST_DATASET}/${DATA_LIST}/moe_expert_ablation/${RUN_ID}}"

read -r -a MODES_ARRAY <<< "${MODES:-uniform expert1 expert2 expert3}"
if ((${#MODES_ARRAY[@]} == 0)); then
  echo "ERROR: MODES cannot be empty" >&2
  exit 2
fi
for mode in "${MODES_ARRAY[@]}"; do
  case "$mode" in
    learned|uniform|expert1|expert2|expert3) ;;
    *) echo "ERROR: invalid router mode: $mode" >&2; exit 2 ;;
  esac
done

echo "MoE expert ablation"
echo "target=${TEST_DATASET}, sources=${DATA_LIST}, seed=${SEED}, target_days=${TARGET_DAYS}"
echo "Stage-I epochs=${TRAIN_EPOCHS}, Stage-II epochs=${FINETUNE_EPOCHS}, DAepoch=${DA_EPOCH}"
echo "DAcoef=${DA_COEF}, xySquaredRatio=${XY_SQUARED_RATIO}, projection=${PROJECTION}, MemSize=${MEM_SIZE}"
echo "modes=${MODES_ARRAY[*]}"
echo "outputs=${OUTPUT_ROOT}"

for mode in "${MODES_ARRAY[@]}"; do
  run_dir="${OUTPUT_ROOT}/${mode}_seed${SEED}"
  echo
  echo "============================================================"
  echo "Starting router mode: ${mode}"
  echo "Run directory: ${run_dir}"
  echo "============================================================"

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
    --description "moe_expert_ablation_${mode}" \
    --pattern_moe_router_mode "${mode}" \
    --save_dir "${run_dir}"
done

echo
echo "Selected MoE routing arms completed."
echo "Results are under: ${OUTPUT_ROOT}"
