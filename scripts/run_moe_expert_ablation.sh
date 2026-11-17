#!/usr/bin/env bash
set -euo pipefail

# Run all five controlled MoE routing arms sequentially.
# Defaults reproduce the METR-LA target / PEMS-BAY+Chengdu+Shenzhen source setup.
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-./configs/config.yaml}"
TEST_DATASET="${TEST_DATASET:-metr-la}"
DATA_LIST="${DATA_LIST:-pems_chengdu_shenzhen}"
SEED="${SEED:-1}"
TARGET_DAYS="${TARGET_DAYS:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3000}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-6000}"
DA_EPOCH="${DA_EPOCH:-1500}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./logged_files/${TEST_DATASET}/${DATA_LIST}/moe_expert_ablation/${RUN_ID}}"

MODES=(learned uniform expert1 expert2 expert3)

echo "MoE expert ablation"
echo "target=${TEST_DATASET}, sources=${DATA_LIST}, seed=${SEED}, target_days=${TARGET_DAYS}"
echo "Stage-I epochs=${TRAIN_EPOCHS}, Stage-II epochs=${FINETUNE_EPOCHS}, DAepoch=${DA_EPOCH}"
echo "outputs=${OUTPUT_ROOT}"

for mode in "${MODES[@]}"; do
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
echo "All five MoE routing arms completed."
echo "Results are under: ${OUTPUT_ROOT}"
