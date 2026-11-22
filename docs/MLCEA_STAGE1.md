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

## Diagnostics

The official training script creates a unique folder per run:

  logged_files/<target>/<sources>/<timestamp>/
    log.txt
    config.yaml
    mlcea_metrics.csv

CSV: one row per source city / target city pair per recorded epoch.

Identifiers:
  epoch, seed, source_pair, source_city, target_city, status

Compatibility:
  a_mean, a_std, a_gt_half, s_mean, s_neg_rate, q_mean, q_low_rate

Cost and alignment:
  r_interaction_std, r_interaction_max_abs, base_probe_std,
  relation_scale, relation_probe_std, relation_vs_blur, ot_loss

Local ridge:
  operator_fit_src, operator_fit_tgt, operator_norm_src,
  operator_norm_tgt, effective_neighbors_src, effective_neighbors_tgt

Optimization:
  metric_weights, future_weights, metric_grad_norm,
  future_grad_norm, aux_loss

DVGP:
  source_mass_nonzero, target_mass_nonzero,
  source_mass_entropy, target_mass_entropy

Every 25 epochs, additionally:
  probe_plan_l1, probe_entropy_base, probe_entropy_new,
  probe_mass_shift_abs

The probe uses ONLY 48 x 48 selected rows, with re-normalized
subset marginals. It is NOT the exact full GeomLoss transport plan.
Never use it to claim the full-plan transport mass changed.

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
