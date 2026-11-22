#!/usr/bin/env bash
# Controlled ONE-variant Stage-I comparison, no Stage-II fine-tuning.
# Example: bash scripts/run_mlcea_ablation_batch8.sh official 1 0
#          bash scripts/run_mlcea_ablation_batch8.sh stable 1 0
set -euo pipefail
if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 {official|legacy|stable|stable_aux_only|strict} seed gpu" >&2
  exit 2
fi
VARIANT="$1"
SEED="$2"
GPU="$3"
if [[ ! "$VARIANT" =~ ^(official|legacy|stable|stable_aux_only|strict)$ ]]; then
  echo "Unknown variant: $VARIANT" >&2
  exit 2
fi
EXTRA=()
case "$VARIANT" in
  official)
    EXTRA=(--mlcea_enable 0)
    ;;
  legacy)
    EXTRA=(--mlcea_enable 1 --mlcea_neighbor_uniform_mix 0
           --mlcea_adaptive_ridge 0 --mlcea_time_block_mode diagnostic)
    ;;
  stable)
    EXTRA=(--mlcea_enable 1 --mlcea_neighbor_uniform_mix 0.35
           --mlcea_adaptive_ridge 1 --mlcea_time_block_mode diagnostic)
    ;;
  stable_aux_only)
    EXTRA=(--mlcea_enable 1 --mlcea_neighbor_uniform_mix 0.35
           --mlcea_adaptive_ridge 1 --mlcea_time_block_mode diagnostic
           --mlcea_rel_coef 0)
    ;;
  strict)
    EXTRA=(--mlcea_enable 1 --mlcea_neighbor_uniform_mix 0.35
           --mlcea_adaptive_ridge 1 --mlcea_time_block_mode strict)
    ;;
esac

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="$GPU" python -u train_DA_OT.py \
  --config_filename ./configs/config_mlcea_batch8.yaml \
  --test_dataset pems-bay --data_list chengdu_shenzhen_metr \
  --target_days 3 --train_epochs 3000 --finetune_epochs 0 \
  --seed "$SEED" --gpu 0 --STmodel SOFTS --model_name SOFTS \
  --classifier 0.5 --xySquaredRatio 100 --DAcoef 1 --DAepoch 1500 \
  --MemSize 10000 --projection 1 --DA OT \
  --mlcea_anchors 128 --mlcea_neighbors 16 --mlcea_ridge 2 \
  --mlcea_rel_coef 0.5 --mlcea_aux_coef 0.02 --mlcea_lr 0.0002 \
  --mlcea_log_every 1 \
  ${EXTRA[@]}
