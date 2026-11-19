#!/usr/bin/env bash
set -euo pipefail
# Single new-architecture run; compare its final_metrics.json with archived Uniform.
cd "$(dirname "$0")/.."
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
OUTPUT_DIR="${OUTPUT_DIR:-./logged_files/${TEST_DATASET}/${DATA_LIST}/horizon_response_pattern/${RUN_ID}_seed${SEED}}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite or append to ${OUTPUT_DIR}" >&2
  exit 2
fi
if (( TRAIN_EPOCHS <= DA_EPOCH || FINETUNE_EPOCHS <= 0 )); then
  echo "Require TRAIN_EPOCHS > DA_EPOCH and FINETUNE_EPOCHS > 0" >&2
  exit 2
fi
echo "Running horizon-response Stage-I against fixed Uniform baseline"
echo "Results directory: ${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" train_DA_OT.py \
  --config_filename "${CONFIG}" \
  --test_dataset "${TEST_DATASET}" --data_list "${DATA_LIST}" \
  --seed "${SEED}" --target_days "${TARGET_DAYS}" \
  --train_epochs "${TRAIN_EPOCHS}" --finetune_epochs "${FINETUNE_EPOCHS}" \
  --DAepoch "${DA_EPOCH}" --DAcoef 0.01 \
  --xySquaredRatio 100 --projection 1 --MemSize 10000 \
  --gpu 0 --STmodel SOFTS --model_name SOFTS --his_num 288 --DA OT \
  --pattern_moe_router_mode response --pattern_moe_pattern_weight 0.5 \
  --fixed_final_checkpoint \
  --description horizon_response_pattern \
  --save_dir "${OUTPUT_DIR}"
echo "Read final_metrics.json, response_transfer.csv, alignment_gradient.csv and run_metadata.json in ${OUTPUT_DIR}"
