# MLCEA Stage-I Experiment Guide

## Provenance and invariant scope

Upstream author repo: https://github.com/serre20/PMME
Official subtree: PMME_and_Others/
Local official mirror: pmme-stage1-research
New research branch: stage-1-mlcea-official-20261122

Before branching, model/Meta_Models/PMME.py in pmme-stage1-research
was checked to match the original author file byte for byte.
The new branch does NOT inherit the earlier predictive-overlap experiment.

UNCHANGED:
- SOFTS forecasting backbone
- All three original GP-based DVGP classifiers
- Marginal masses alpha/beta and their masking behavior
- The official Stage-II source file (PMME_2ndStage.py)
- Original training and forecasting losses

CHANGED ONLY IN STAGE I:
- New mlcea.py module for history-future evolution compatibility.
- PMME.py uses its alpha/beta-interaction-centered pairwise relation
  to adjust the existing GeomLoss p=1 Sinkhorn cost.
- train_DA_OT.py accepts MLCEA options; --mlcea_enable 0 disables the new path.

## Exact processing order

1. The OFFICIAL SOFTS model extracts representations and provides true
   FUTURE TRAINING labels in its own RevIN normalization.
2. MLCEA derives six history conditions from 5-minute-grid trends and
   six future responses at horizons [6,12,18,24,30,36] steps, corresponding
   to [30,60,90,120,150,180] physical minutes. Chengdu and Shenzhen are
   interpolated onto the same 5-minute grid by the official dataset loader.
3. Six learned positive history-feature weights define a distance. Current
   batch anchors and soft KNN weights determine a sample-specific ridge fit.
4. Each local ridge operator W_i is fitted from OTHER road nodes' training
   history/future pairs, regularized toward a SOURCE-ONLY shared W0.
5. W_source applies to the TARGET sample's history and W_target applies to
   the SOURCE sample's history. Both outputs are compared with observed
   train-window future responses. Three learnable logits weight the
   short/mid/long response phases; a uniform component remains.
6. Compare the two directional errors against W0, combine with a softmin
   (the worse direction matters), and temper by local history support.
7. Produce A[i,j] (a continuous RELATION, not a transport plan).
8. Center R=A-0.5 relative to the original DVGP alpha/beta:
   R_int = R - R*beta (row-wise) - alpha*R (column-wise)
             + alpha*R*beta (global scalar).
   Under balanced OT this removes arbitrary row/column-only scores.
9. Feed C_new=C_official - effective_lambda*R_int into the original weighted
   GeomLoss Sinkhorn. The effective lambda is calibrated to the sampled
   original cost spread and the original Sinkhorn blur (0.001).
10. Only the official SOFTS backbone weights are transferred to Stage II.

The additional trainable parameters are six history metric logits and
three future-phase logits, optimized using a dedicated Adam optimizer.
All local W_i are calculated by differentiable regularized ridge solves.

## Run instructions

First run CPU unit tests (does not require the traffic datasets):

    python -m pytest -q tests/test_mlcea.py

Static syntax checks:

    python -m py_compile model/Meta_Models/mlcea.py model/Meta_Models/PMME.py train_DA_OT.py

When the original repository datasets are available, start a short smoke
run (IMPORTANT: this is NOT a research benchmark run):

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python -u train_DA_OT.py \
      --config_filename ./configs/config.yaml \
      --test_dataset pems-bay --data_list chengdu_shenzhen_metr \
      --model_name SOFTS --STmodel SOFTS \
      --seed 1 --target_days 3 \
      --train_epochs 3 --finetune_epochs 0 --DA OT --DAepoch 0 \
      --DAcoef 1 --classifier 0.5 --xySquaredRatio 100 \
      --projection 0 --mlcea_enable 1 --mlcea_log_every 1

For the real experiment, retain the official script's other parameters
and only compare --mlcea_enable 0 vs 1 under identical seeds. The author's
pems script uses --DAepoch 1500. MLCEA only starts when OT starts, so
there is NO MLCEA data before that epoch. Use seeds 1, 3, 5.

CLI controls:
  --mlcea_enable 1        new architecture; 0 restores official OT cost
  --mlcea_rel_coef 0.5    calibrated relation OT correction strength
  --mlcea_aux_coef 0.02   ridge self-explanation auxiliary fit loss
  --mlcea_lr 0.0002       tiny MLCEA parameter optimizer
  --mlcea_anchors 128     anchors sampled per training batch
  --mlcea_neighbors 16   local ridge neighbour count
  --mlcea_ridge 2.0      regularizer toward common source reference W0
  --mlcea_pair_chunk 96  source rows per relation computation chunk
  --mlcea_log_every 1    metrics CSV frequency (epochs)

## Diagnostics (schema v2)

All run outputs are isolated by target, sources, timestamp, seed,
MLCEA flag and process ID. Directory example:

  logged_files/<target>/<sources>/<timestamp>_seed1_mlcea1_pid1234/
    log.txt
    config.yaml
    mlcea_run_manifest.json
    stage1_epoch_metrics.csv
    stage1_eval_metrics.csv
    mlcea_metrics.csv
    stage2_eval_metrics.csv          (only when Stage-II evaluation runs)

1. mlcea_run_manifest.json:
   Full command line and options, Python/Torch/CUDA environment, seed,
   stage and source repo provenance.

2. stage1_epoch_metrics.csv:
   One row for EVERY Stage-I epoch, including epochs before the OT
   activation threshold and runs with --mlcea_enable 0.
   Fields: original source task loss, target task loss (diagnostic only),
   domain alignment loss, MLCEA auxiliary loss, weighted auxiliary loss,
   total loss, source MAE/RMSE/MSE/MAPE, Adam learning rates,
   training wall time, peak allocated GPU memory, MLCEA gradient norms,
   actual parameter step norms, number of active/skipped source pairs,
   and explicit waiting_for_DAepoch / active / official_OT_active status.

3. mlcea_metrics.csv:
   One row per active SOURCE CITY vs target-city pair, per requested
   logging epoch. This file is absent before DAepoch, when disabled,
   or when MLCEA never enters the OT path.
   - Identity: epoch, seed, source_pair, source_city, target_city, status
   - Directional: err_src_to_tgt, err_tgt_to_src,
     err_shared_to_tgt, err_shared_to_src, gain_src_to_tgt,
     gain_tgt_to_src, gain_positive_src_to_tgt,
     gain_positive_tgt_to_src, gain_direction_disagree,
     gain_direction_abs_gap, src_better_than_shared_rate,
     tgt_better_than_shared_rate
   - Relation: a_mean/std/min/max, rates above/below 0.5,
     s_mean, s_neg_rate, q_mean/std/min/low_rate,
     r_uncentered_std, r_interaction_std/max_abs,
     interaction_suppression_ratio (before/after centering)
   - Local operators: source/target fit mean, p90, max;
     operator norms; effective neighbor average/min; support radii
   - DVGP: kept sample count, keep ratios, probability means,
     margin entropy and actual scaled OT loss
   - OT correction: probe base min/mean/max/std, correction mean/std/
     max abs, relation_vs_blur, relation_vs_base_spread,
     relation_scale and unscaled OT loss
   - Learning: six learned metric weights, three future phase weights,
     gradient norms, grad-is-missing flags, optimizer actual step norms,
     epoch decomposed losses.
   - Every 25 epochs: 48x48 subsample plan L1 shift, entropy before/after,
     and largest sampled mass-shift cell.

   The 48x48 plan is a detached approximate probe, NOT the exact full
   weighted GeomLoss transport plan. Never claim full P changed by
   exactly this much. In addition, a subsample of the already-centered
   full relation can have different submatrix marginal centering.

4. stage1_eval_metrics.csv:
   36 rows for EACH Stage-I evaluation epoch (every 100 original epochs
   and at epoch 0). Each row logs physical horizon minutes, target
   few-shot TRAIN MAE/RMSE/MAPE and target HELD-OUT TEST MAE/RMSE/MAPE.
   The code historically called the former validation_score, but it is
   NOT a separate target validation split. Checkpoint-selection status
   is marked according to the actual source-code criterion.

5. stage2_eval_metrics.csv:
   If Stage-II is executed, append its original test MAE at all 36
   horizons for both WITHOUT_MEMORY and WITH_MEMORY checks.
   The Stage-II model implementation is not changed.

Run the CPU tests:

  python -m pytest -q tests/test_mlcea.py tests/test_mlcea_diagnostics.py

These tests verify tensor dimensions, gradients, alpha/beta interaction
centering, example transport-plan changes, CSV schema, run manifests,
and evaluation horizons. Full PMME GPU training with actual traffic
datasets is still required before interpreting real model performance.

## How to diagnose failures

metric_grad_norm or future_grad_norm consistently zero:
  Check gradient path and whether DVGP masked OT.

a_std / r_interaction_std near zero:
  Relations are neutral or only additive row/column biases exist.
  MLCEA cannot produce a meaningful transport reallocation.

q_mean near zero:
  Cross-city history states fall outside local operator support.
  Relation is forcibly neutralized.

effective_neighbors near one:
  KNN weights are overly concentrated; ridge becomes brittle.

operator_norm huge / fit error large:
  Bad history scaling or unstable local regression, adjust ridge.

relation_vs_blur > 1 for long periods:
  Relation may overpower the official Sinkhorn geometry.

probe_plan_l1 near zero but relation stats nonzero:
  Interaction centering removed trivial row/column signal or correction
  is too small relative to the entropic regularizer.

Probe plan changed but final MAE worse:
  Changing transport is not proof of useful source-to-target transfer.
  Check short/mid/long target performance and multiple seeds.

## Crucial limitations

- Current local neighbors are leave-ROAD-NODE-out; their windows on
  different nodes can still cover overlapping clock intervals.
  This is NOT guaranteed independent chronological cross-fitting.
  Add a time-blocked cross-fit comparison before paper claims.
- No target TEST future is passed to MLCEA. Only allowed Stage-I
  target few-shot TRAIN futures are used.
- Relationship construction uses O(number_of_source_nodes *
  number_of_target_nodes) memory, as does the tensorized official
  Sinkhorn cost. Watch GPU memory on large city pairs. The implementation
  never silently skips large batches.
- Stage-II was not changed and should be verified separately against
  the author's exact Stage-II source.
- Synthetic CPU tests are provided but full GPU training has NOT
  been executed or validated in this conversation.

Suggested controlled ablations:
  (A) --mlcea_enable 0: official PMME
  (B) --mlcea_enable 1 --mlcea_rel_coef 0: auxiliary only
  (C) --mlcea_enable 1 --mlcea_aux_coef 0: relation cost only
  (D) MLCEA with both features enabled
