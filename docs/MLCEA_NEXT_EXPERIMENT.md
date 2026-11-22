# MLCEA Stage-I: next diagnostic/stability experiment

Branch: stage-1-mlcea-official-20261122.
Unchanged: SOFTS backbone, GP/DVGP, official Stage-II model, default
weighted Sinkhorn when MLCEA is disabled. Read-only dataset metadata
and opt-in MLCEA local ridge settings were added.

## Motivation

The uploaded seed1 batch8 run exhibited (i) local conditional evolution
often worse than source-only W0, (ii) some effective neighbor counts near
one although K=16, (iii) MLCEA relation OT cost correction std only
0.0005, and (iv) extreme OT costs sometimes in the millions.

The current patch deliberately DOES NOT boost the OT relation term.
That would amplify unstable evolution scores before proving usefulness.

## Three-day time-support limit

Official target train has 46 windows, stride 12 time steps. One labelled
window covers 288 history plus 36 future steps. Strict time-disjoint
cross-fit requires differences of at least 324/12 = 27 window indices.
Many target queries have NO valid candidate with just 46 windows.

The dataset loader now attaches read-only window-index and stride metadata.
MLCEA always logs the number/fraction of strictly time-separated candidates.
--mlcea_time_block_mode diagnostic measures temporal support but retains
leave-node-out fitting.
--mlcea_time_block_mode strict enforces disjoint windows and uses source
W0 as an explicit fallback when local fitting has no data.

## Stability controls

--mlcea_neighbor_uniform_mix 0.35
  Mix learned local KNN weights with uniform weights over eligible nodes.
--mlcea_min_effective_k 8
  Target effective sample size for regularization.
--mlcea_adaptive_ridge 1
  Increase shrinkage toward common source W0 for weak local neighborhoods.

Old operator behavior is recoverable with:
  --mlcea_neighbor_uniform_mix 0 --mlcea_adaptive_ridge 0
  --mlcea_time_block_mode diagnostic

These controls are proposed, not yet validated improvements.

## Shared OT numeric audit (on official AND MLCEA)

ot_numeric_health.csv logs one row per source city every 25 active OT
epochs. Fields include source/target DVGP keep rates, actual OT loss,
48x48 original OT cost probe p50/p95/p99/max, source/target RevIN
future absolute max/p99, offending road node and absolute window ID,
history standard deviation p01/min, fractions with std<0.001, and
fractions of RevIN future values over abs(100).

The 48x48 probe is NOT the complete OT plan. No SOFTS or official OT
normalization was changed. The point is to identify whether extreme costs
were already present in official PMME.

mlcea_metrics.csv separately logs strict-temporal support, fallback
fractions, actual ridge shrinkage, weights, response clipping and
source/target normalization diagnostics.

## Standalone local operator validation BEFORE 3000 epochs

Run with original training data installed on server:

  python scripts/diagnose_mlcea_operator.py \
    --config configs/config_mlcea_batch8.yaml \
    --target pems-bay --sources chengdu_m,metr-la,shenzhen \
    --seed 1 --device cuda --mode strict \
    --output mlcea_operator_validation_strict.csv

Ablate neighbor smoothing:

  python scripts/diagnose_mlcea_operator.py \
    --config configs/config_mlcea_batch8.yaml \
    --target pems-bay --sources chengdu_m,metr-la,shenzhen \
    --seed 1 --device cuda --mode strict \
    --neighbor-uniform-mix 0 \
    --output mlcea_operator_validation_unsmoothed.csv

Outputs compare local source operators vs global source W0 when applied
to target TRAIN samples selected solely by target historical condition.
Target train FUTURE labels are used only to score predictions, never to
select source neighbors. The target TEST split is never used here.
These scores use initial logits: they are cheap diagnostics, not final
evaluation of the trained MLCEA weights.

## Matched Stage-I runs (one command per variant)

  bash scripts/run_mlcea_ablation_batch8.sh official 1 0
  bash scripts/run_mlcea_ablation_batch8.sh legacy 1 0
  bash scripts/run_mlcea_ablation_batch8.sh stable 1 0
  bash scripts/run_mlcea_ablation_batch8.sh stable_aux_only 1 0

All use seed=1, source batch=8, 3 target train days, official GP threshold
0.5, DA start 1500, 3000 Stage-I epochs, SOFTS and same source cities.
Full Stage-II finetune is deliberately disabled during these diagnosis
runs. Different run IDs isolate logs. Repeat viable comparison with
seeds 3 and 5, then rerun the standard Stage-II protocol.

The author's reported runs use batch size 16; do not call this batch8
comparison a full official hyperparameter reproduction.

## Validation

  python -m pytest -q tests/test_mlcea.py \
    tests/test_mlcea_diagnostics.py tests/test_mlcea_temporal.py
  python -m py_compile scripts/diagnose_mlcea_operator.py
  bash -n scripts/run_mlcea_ablation_batch8.sh

GitHub Actions checks CPU math and logging behavior. The data-dependent
GPU training path remains unverified until the actual traffic run.

If the local operator is still weaker than W0 under strict time-blocked
source-to-target evaluation, do not make MLCEA OT influence larger.
Consider changing the research direction to source-only pattern reuse
or applicability-gated source operators, rather than fitting a target
operator that lacks independent temporal support.
