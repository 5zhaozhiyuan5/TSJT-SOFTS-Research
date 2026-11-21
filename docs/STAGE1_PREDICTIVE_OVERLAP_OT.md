# Stage-I: predictive-overlap weighting instead of DVGP

## Scope
This branch was forked from `pmme-stage1-research`. Only Stage-I
`model/Meta_Models/PMME.py` and a new Stage-I weighter are changed.
`PMME_2ndStage.py` and SOFTS have not been modified.

## Why this is not a generic distance replacement
Official-style PMME uses three separate variational GP domain classifiers
(one for each source-vs-target pair). The model classifies the **future label
representation** `y` as source/target and converts domain confusion to the
two nonnegative marginal mass vectors passed to weighted GeomLoss Sinkhorn.
The transport support itself is still `[historical_forecasting_rep,
sqrt(xySquaredRatio)*future_label]`.

The alternative keeps **the same pairwise source-vs-target loops, marginal
vector interface, weighted Sinkhorn, joint support, DA loss coefficient,
and forecasting model**. It replaces the three GP classifiers with three
lightweight prediction-aware heads trained on historical backbone features.

## Exact data flow
1. Backbone produces `rep` (history-only) and `y_unscale` (training labels
   normalized exactly as before).
2. Each domain head takes a detached per-node `rep` and predicts:
   - domain logit trained to identify source (0) versus target (1);
   - continuous future-overlap score trained from a DILATE-style teacher.
3. Teacher reads **only labeled Stage-I training futures**: future samples
   are pooled to 12 positions and encoded as level-aware relative changes.
   Pick three quantile-slope representatives from the opposite domain;
   calculate soft-DTW shape distortion plus expected squared time displacement.
   Nearest-prototype compatibility supplies a continuous supervision score.
4. For each source sample, use a floored geometric combination of probability
   of being target-like and predicted future overlap. For each target sample,
   use probability of being source-like and predicted future overlap.
   Normalise each vector separately to sum to one.
5. Pass `alpha, x1|scaled_y1, beta, x2|scaled_y2` to the **original**
   `geomloss.SamplesLoss(loss="sinkhorn", p=1, debias=False, blur=0.001)`.
   Its transport and prediction gradient path remain unchanged.

## Objective
`L_aux = BCEWithLogits(domain, source_or_target) +
0.5*MSE(sigmoid(overlap), DILATE_teacher_score)`.

The teacher score is `exp(-min_prototype_distance/temperature)`, clamped
to [0.05, 1]. This estimates *training-domain conditional future overlap*;
it is **not** a calibrated posterior probability. The loss does not claim
to learn future values at inference when history is ambiguous.

Marginal weights are detached as with the GP baseline. The auxiliary head
is trained with a separate Adam optimiser; the alignment gradients reach
the forecasting backbone through the unchanged transported features.

## Safeguards
- Never use target test labels in the teacher or weight network.
- Keep small positive weights instead of the original hard threshold. This
  prevents undefined/uninitialised alpha/beta in empty selections.
- The `classifier_args` switch retains its on/off function, but when on
  it no longer acts as a hard probability threshold.
- This experiment uses no target validation examples.
- The teacher is reduced-length for computational practicality. It is
  DILATE-style rather than a byte-for-byte reproduction of the original loss.

## Experiments required before claiming improvement
1. Original three-DVGP baseline on identical splits/seed.
2. New weighting with auxiliary future supervision disabled (ablation).
3. New weighting with auxiliary future supervision enabled (this branch).
4. Compare downstream MAE/RMSE at 10/60/120/180 minutes, source/target mass
   distributions, transport loss, training time, and source/target domain
   accuracy. Include several seeds.
5. Compare with GP retained plus auxiliary supervision to identify whether
   the GP provides complementary uncertainty information.

This branch has not been trained end-to-end; smoke tests only check the
new component's basic contract.
