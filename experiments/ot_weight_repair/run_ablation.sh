#!/usr/bin/env bash
# Stage-I OT weight/scale causal diagnosis, seeded and independent.
set -euo pipefail
: "${SEEDS:=1}"
: "${EPOCHS:=3000}"
: "${GPU:=0}"
: "${RUN_ROOT:=runs/ot_weight_repair_20261123}"
mkdir -p "$RUN_ROOT"
for seed in $SEEDS; do
  # F0 and F1 differ only in weighting, F2 in selection, F3 only scaling.
  # All five use the SAME E1 MLP probability estimator and original SOFTS.
  for spec in \
    "F0:legacy:legacy" \
    "F1:uniform:fixed" \
    "F2:smooth:legacy" \
    "F3:legacy:fixed" \
    "F4:smooth:fixed"; do
    IFS=: read -r name weights scale <<<"$spec"
    dir="$RUN_ROOT/seed_${seed}/${name}"
    mkdir -p "$dir"
    echo "[RUN] $name seed=$seed weights=$weights scale=$scale"
    python run_stage1_experiment.py \
      --version "weight-repair-${name}-seed${seed}" \
      --save_dir "$dir" --stage1_method pmme \
      --test_dataset metr-la --data_list chengdu_shenzhen_pems \
      --train_epochs "$EPOCHS" --target_days 3 \
      --seed "$seed" --gpu "$GPU" --model_name SOFTS \
      --classifier .3 --DA OT --DAepoch 1500 \
      --DAcoef .01 --xySquaredRatio 100 --projection 1 \
      --stage1_selection_metric source_mae \
      --probability-estimator neural --probability-variant mlp \
      --probability-lr 0.001 \
      --ot-weight-strategy "$weights" \
      --ot-scale-strategy "$scale" \
      --ot-weight-floor 0.05 --ot-weight-power 1.0 \
      2>&1 | tee "$dir/console.log"
  done
done
