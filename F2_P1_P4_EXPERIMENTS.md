# F2 + P1/P2/P3/P4: four independently trained Stage-I OT variants

**Branch:** `experiment/pmme-f2-p1-p4-gradient-ot-20261125`

## Scope and baseline guarantees

- Target: PEMS-BAY, first 3 days support only; source: Chengdu, Shenzhen, METR-LA.
- SOFTS backbone, shared F2 marginal `alpha=softmax(8*cos(g_source,g_target))`.
- Original Stage-I source forecast loss, one-sided PCGrad, AdamW, and GeomLoss Sinkhorn `p=1, blur=0.001, debias=False` remain.
- The original Stage-II `PMME_2ndStage.py` is **not edited**. The standard runner may still run it for 6000 epochs.
- All experiments use F2 source marginals; the target marginal is uniform.
- `PMME_PAIR_VARIANT=f2` is original F2. Every `p1/p2/p3/p4` with `PMME_PAIR_STRENGTH=0` also follows original F2, including original target-group assignments.
- `PMME_FEEDBACK_MODE=g0` is required for all four training variants. H1/H2 are kept in the source tree for separate diagnostics, never used as a controller here.

## Four variants (NOT stacked)

| Variant | Borrowed research inspiration (not copied architecture) | Where it acts | What changes |
|---|---|---|---|
| P1: gradient-selected relational anchors | Keypoint-Guided Optimal Transport, NeurIPS 2022 | OT ground cost | High F2-mass sources and strong target gradients form candidate pools; mutual high gradient-head cosine identifies few anchor pairs; pair-to-anchor relation consistency penalizes mismatch |
| P2: gradient-conditioned temporal responses | EDEN NeurIPS 2025 / Match-And-Deform ECML PKDD 2023 | OT ground cost | Forecast-head gradient pair cosine controls confidence of each of three history-relative, future 12-step response distances |
| P3: horizon-conflict-sensitive pair matching | Conflict-Averse Gradient Descent, NeurIPS 2021 | OT ground cost | Three forecast-gradient cosine scores are combined using reliability-weighted soft minimum; a strong negative horizon score penalizes matching |
| P4: task-synergy balanced city-to-group assignment | TAROT ICML 2025 | Group assignment before F2 | An EMA of whole-head gradient cosine, plus historical coverage penalty, chooses a one-to-one permutation of three source cities and 15/15/16 target-window blocks |

All four are **proposed research variants, not guaranteed novel inventions**. Matching, OT metric learning, time-series temporal alignment, and multiobjective gradient handling exist in prior literature. Reproduction and same-protocol comparisons with closest ideas are needed before any originality claim.

### P1: reliable gradient anchors

Source points are prioritized by the unmodified F2 marginal; targets are prioritized by task-gradient magnitude. A small candidate pool is matched using rank-one exact projection2-weight gradient cosine and representation proximity. Mutual top matches are used first; greedy one-to-one filling supplies the remaining anchors.

For each node/window, compute its normalized distances to paired anchors on **its own side**, then penalize differences in relational distance vectors when comparing source and target points.

Anchor **selection** is detached, but point-to-anchor relational features remain differentiable w.r.t. predictor representations. The anchor count is configurable (default 8). This version uses current-batch reliability, **not** a cross-epoch persistent anchor memory.

### P2: gradient-conditioned multiscale response

Use city-standardized history and future labels from the original support data only. For future subwindows [0:12], [12:24], [24:36], extract:
1. Mean change relative to last 12 historical steps.
2. Early-to-late response slope.
3. Response volatility.

Use the pairwise cosine of rank-one projection2-weight gradients (without explicitly materializing the large weight-gradient vectors) to decide how strongly to penalize unmatched response and incompatible task direction. The P2 regularization cost is computed as a dimensionless bounded term using a chunked pairwise kernel.

### P3: horizon conflict

For each source/target node/window pair, compute three separate 12-step projection2 gradient cosine values. Compute positive target-support split-half reliability, with 0.15 prior per band. Combine three cosine scores via a reliability-weighted smooth minimum, and penalize if that value is negative. The parameter soft-min temperature is fixed at 0.15 in this pilot.

This is one single standard GeomLoss Sinkhorn per city, **not three separate OT plans**.

### P4: city/group pairing

The existing 46 target support windows remain split into 15/15/16. Build a 3x3 score matrix of cosine between each source-city aggregate full projection2 gradient and each target group's aggregate gradient. An EMA (momentum 0.8) and a penalty based on how frequently the pair was used (0.08 by default) choose one of six possible one-to-one permutations each active OT epoch. Source alpha F2 is recomputed using its **assigned** target group; actual target representation/label group in OT is changed accordingly.

For P4, `PMME_PAIR_STRENGTH` is a binary *enable switch* (positive enables matching); use `PMME_P4_MOMENTUM` and `PMME_P4_REPEAT_PENALTY` for its continuous hyperparameters.

## OT implementation, scaling, and limitations

For P1-P3 the native unmodified OT joint points are `q=[128-dim SOFTS representation, sqrt(xySquaredRatio)*36-dim target-normalized future]`. Custom tensorized GeomLoss costs have the form:

`C_new(i,j) = ||q_source(i)-q_target(j)||_2 + strength * reference_scale * pair_penalty(i,j)`

The reference scale is the median of a detached small, deterministic `torch.cdist` cross-sample subset of the same OT points. `pair_penalty` for P2/P3 uses detached forecast-gradient scores: it modifies the OT coupling and therefore the learning gradient through the original pair distance, but is not itself differentiated through the forecasting task (no second-order gradients). P1's pair-to-anchor distances are differentiable through current SOFTS features while selected anchors are detached.

The current original project uses `backend="tensorized"`, which still constructs dense N x M OT matrices. P1-P3 row-chunk their *cost calculations* to limit intermediate memory, but this **does not remove the dense final matrix**; the PyTorch/GeomLoss CUDA environment must be stress-tested. This branch does not claim scalable sparse OT or store an explicit transport plan. A future transport-plan diagnostic should use small subsamples and avoid treating cost changes as direct evidence of plan mass changes.

The teacher future data is only the training-support future; evaluation and target test labels are not used to form P1-P4 costs.

## How to run

```bash
git fetch origin
git switch experiment/pmme-f2-p1-p4-gradient-ot-20261125

python -m unittest discover -s tests -p 'test_gradient_pair_ot.py'
python -m unittest discover -s tests -p 'test_transfer_utility.py'

# Original F2 reference, same runner and seed
PMME_PAIR_VARIANTS="f2" PMME_PAIR_SEEDS="1" bash run_f2_pair_ot_experiments.sh

# Round 1: lower-cost P2 and P4
PMME_PAIR_VARIANTS="p2 p4" PMME_PAIR_SEEDS="1" bash run_f2_pair_ot_experiments.sh

# Round 2: pairwise gradient P1 and P3
PMME_PAIR_VARIANTS="p1 p3" PMME_PAIR_SEEDS="1" bash run_f2_pair_ot_experiments.sh
```

Default: Stage-I 3000 epochs (OT begins at 1500), original Stage-II 6000 epochs (per variant).

For a **short real-forward/backward smoke test** that reaches the OT branch immediately, use separate output roots (never reuse them for full experiments):

```bash
PMME_PAIR_RUN_ROOT=./experiments/pair-smoke-round1 \
PMME_PAIR_STAGE1_EPOCHS=2 PMME_PAIR_DAEPOCH=1 \
PMME_PAIR_STAGE2_EPOCHS=0 PMME_PAIR_VARIANTS="p2 p4" \
bash run_f2_pair_ot_experiments.sh

PMME_PAIR_RUN_ROOT=./experiments/pair-smoke-round2 \
PMME_PAIR_STAGE1_EPOCHS=2 PMME_PAIR_DAEPOCH=1 \
PMME_PAIR_STAGE2_EPOCHS=0 PMME_PAIR_VARIANTS="p1 p3" \
bash run_f2_pair_ot_experiments.sh
```

The smoke test performs **one active OT iteration**, so it can catch graph/GeomLoss failures. It is not evidence of forecast improvement.

For an intermediate, Stage-I-only 3000-epoch check:

```bash
PMME_PAIR_STAGE2_EPOCHS=0 PMME_PAIR_VARIANTS="p2 p4" bash run_f2_pair_ot_experiments.sh
```

For exact zero-effect regression, set the strength 0 and compare Stage-I logs/parameters against a run of `f2` with the same seed:

```bash
PMME_PAIR_STRENGTH=0 PMME_PAIR_VARIANTS="p2" PMME_PAIR_STAGE2_EPOCHS=0 bash run_f2_pair_ot_experiments.sh
```

**Warning:** Run roots default to `experiments/f2-gradient-pair-ot/$variant/seed$seed`. Do not reuse the same output directory with different settings. `run_config.txt` records the commit, OT scaling, strength, and Stage-II epochs. For changed anchor/block/P4 settings, set `PMME_PAIR_RUN_ROOT` to a different path, rather than append to existing logs.

Save `train.out`, `run_config.txt`, selected checkpoints and full metric traces. The runner checks Stage-I and (if enabled) Stage-II completion. It DOES NOT claim that a successful run has better metrics.

## Ablation / acceptance rules

- F2 vs P1: compare with fixed random anchors, gradient-only pair cosine, and plain representation-distance OT. Check source/target score consistency and overhead.
- F2 vs P2: compare with dynamic response distance *without gradient* and gradient agreement *without dynamics*.
- F2 vs P3: compare 36-step pooled gradients, equally weighted mean of three 12-step gradients, and simple worst-band gradient.
- F2 vs P4: compare existing random/source ordering, cyclic pairing, and cosine-only greedy assignment without coverage balancing.

First run seed1 to debug; then additional seeds (e.g. 3,5) with the exact training/evaluation conditions. Evaluate 10/60/120/180-minute MAE and RMSE **after original Stage-II**. Keep a separately checked F2 baseline and track OT loss/gradient scaling, cost deltas, memory, and timing. Never tune on target test-set labels.

This work is a code experiment; no full GPU traffic experiments are reported in this branch yet.
