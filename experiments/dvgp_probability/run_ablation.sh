#!/usr/bin/env bash
# Reproducible PMME Stage-I probability ablation from repository root.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:=0}"
: "${SEEDS:=1 2 3}"
: "${EPOCHS:=3000}"
: "${TARGET:=metr-la}"
: "${SOURCES:=chengdu_shenzhen_pems}"
: "${RUN_ROOT:=runs/dvgp_probability_20261123}"
mkdir -p "$RUN_ROOT"
for seed in $SEEDS; do
  for spec in "E0:dvgp:mlp" "E1:neural:mlp" "E2:neural:multiscale" "E3:neural:dynamics" "E4:neural:horizon"; do
    IFS=: read -r name estimator variant <<<"$spec"
    save="$RUN_ROOT/seed_${seed}/${name}"
    mkdir -p "$save"
    echo "Running $name seed=$seed estimator=$estimator variant=$variant"
    python run_stage1_experiment.py \
      --version "dvgp-prob-${name}-seed${seed}" \
      --save_dir "$save" --stage1_method pmme \
      --test_dataset "$TARGET" --data_list "$SOURCES" \
      --train_epochs "$EPOCHS" --target_days 3 \
      --seed "$seed" --gpu 0 --model_name SOFTS \
      --classifier 0.3 --DA OT --DAepoch 1500 \
      --DAcoef 0.01 --xySquaredRatio 100 --projection 1 \
      --stage1_selection_metric source_mae \
      --probability-estimator "$estimator" \
      --probability-variant "$variant" \
      --probability-lr 0.001 \
      2>&1 | tee "$save/console.log"
  done
done
