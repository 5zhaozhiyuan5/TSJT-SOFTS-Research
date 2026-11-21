# Stage-I: history-conditioned predictive OT (DVGP replacement)

## Scope and preservation contract
Branch: `stage-1-predictive-overlap-ot-20261122`, based on `pmme-stage1-research`.
Modified code is confined to Stage-I (`PMME.py`, `predictive_overlap.py`,
`train_DA_OT.py`). Neither SOFTS nor `PMME_2ndStage.py` is changed.

Official-style PMME uses three source-vs-target variational GP domain heads to
compute nonnegative marginal weights alpha/beta for weighted Sinkhorn. Original
ground cost is Euclidean p=1 over the joint support
`[SOFTS_rep, sqrt(xySquaredRatio) * future_label]`.
We keep **three pairs, the alpha/beta interface, the original joint support,
the GeomLoss Sinkhorn solver, and the forecast gradient path**.

## New Stage-I training pathway
1. Input: historical flow `H` and known training future `F` per node.
   Recompute RevIN history using exactly the same historical window, mean,
   variance and epsilon as in PatchFSL, so it is comparable to `y_unscale`.
2. Conditional-evolution teacher (training only) separately encodes:
   - 12 downsampled history positions (historical state and approach trend);
   - 12 downsampled future positions anchored to **the final history value**,
     not to the first future value.
   It compares the two halves with independent Soft-DTW paths so neither
   crosses the historical/future boundary. An additional recent-history
   velocity penalty distinguishes approaching a common future from opposite
   historical trends. Teacher score = exp(-distance / temperature).
3. Three independent 2-logit heads trained from **history-only SOFTS rep**
   estimate domain confusion and cross-domain future compatibility.
   Their floored positive predictions form alpha/beta. Marginal weights are
   detached as in the DVGP mechanism.
4. Three independent small 32D pair heads map source and target **historical
   representation** into a common unit feature space. The pair head is
   supervised with random source-target training pairs and a continuous
   history-conditioned teacher score (SmoothL1). Each pair loss is
   `SmoothL1(0.5*(1+cos(projected_source,projected_target)), teacher_score)`.
5. For each source-target transport problem, the p=1 ground cost is:
   `cost_new(i,j) = ||[rep_i,scaled_future_i] - [rep_j,scaled_future_j]||_2
                      + predictive_cost_coef*(1-cos(pair_embed_i,pair_embed_j))`.
   This is **the original OT cost plus a learned pairwise supplement**, not
   a replacement Sinkhorn implementation. GeomLoss `cost=` callback has
   tensorized shape `[B,N,M]`. The 32D pair embeddings are detached during
   transport, so the newly trained pair head cannot erase the backbone's
   original forecast/OT gradient path.
6. The total Stage-I predictor training objective still combines original
   source forecasting and weighted OT; auxiliary heads use a separate Adam
   optimiser, without changing Stage-II.

## Operational switches
`--predictive_cost_coef 1.0` enables learned pairwise cost, `0.0`
disables it while retaining the learned alpha/beta (recommended ablation).
`--classifier 0` disables the new weighting and pairwise cost entirely
(uniform alpha/beta), preserving the original plain Sinkhorn fallback.

## Why the two outputs differ
- alpha/beta: per-node source/target *transport mass* (domain overlap ×
  typical conditional future compatibility).
- predictive pair cost: per-source-target correspondence score. It changes
  **who should match whom**, not total transport mass.

They are trained with different targets. The output is not a calibrated
Gaussian-process predictive distribution, and preserving the conditional
weighted OT does not imply mathematical equivalence to a GP classifier.

## Limitations and required experiment
- The teacher observes real future **only in Stage-I training batches**.
  During target test, only the SOFTS forecasting path is used; no future
  supervision or source test labels are accessed.
- Conditional future outcome can remain stochastic for identical histories.
  The teacher is a proxy, not a guarantee of correct future retrieval.
- Current training data are mixed 5-minute and 10-minute sources. The
  reduced-length teacher currently uses relative progress in each forecasting
  horizon; it has **not** yet been physically time-calibrated across cities.
- Tensorized GeomLoss has O(N*M) memory; added pairwise cost uses the same
  full matrix footprint, plus a 32D projected embedding. Inspect peak GPU
  memory during full training.
- Passing unit tests is not equivalent to end-to-end validated performance.

## Controlled ablations
A. original DVGP (branch `pmme-stage1-research`);
B. replacement weighting only (`--predictive_cost_coef 0`);
C. replacement weighting + pairwise cost (`--predictive_cost_coef 1`);
D. plain OT without classification (`--classifier 0`).

With identical target 3-day split, Stage-II, SOFTS and seeds, report per-horizon
MAE/RMSE (10, 60, 120, 180 minutes), mass entropy, OT objective, pair-loss,
source-target teacher/students correlation, peak GPU memory and wall time.

## Testing
`python -m unittest discover -s tests -p 'test_predictive_overlap.py' -v`
requires torch; the GeomLoss custom-cost integration test requires geomloss.
