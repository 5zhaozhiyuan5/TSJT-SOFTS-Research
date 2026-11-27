# F2-GS: Full-Horizon Gradient Stability Calibration

## Scope

This branch starts from the original F2 gradient-marginal implementation. It
does not change the prediction horizon, split the 36 output dimensions into
bands, add a prediction network, change SOFTS, modify the Sinkhorn cost, or
change the official Stage-II. The target marginal remains uniform.

## Motivation and literature grounding

- Peng et al., *Robust Multitask Learning With Sample Gradient Similarity*,
  IEEE Transactions on Systems, Man, and Cybernetics: Systems, 54(1), 2024.
  The paper motivates fine-grained sample-gradient agreement and assigning
  greater weight to more consistent samples. This is the closest conceptual
  basis for preserving F2's sample-level gradient scoring.
- Cao et al., *Towards Calibrated Gradient-based Multi-Task Learning*,
  CVPR 2026 Findings. VarGrad reports that gradient variance, especially
  under small batches, can make measured conflicts unstable and harm
  gradient-based optimization. We borrow the variance-awareness motivation,
  not VarGrad's iterative gradient correction or task scheduler.
- Achituve et al., *Bayesian Uncertainty for Gradient Aggregation in
  Multi-Task Learning*, ICML 2024, PMLR 235:117-134. This supports the broader
  principle that gradient estimates have uncertainty and that the uncertainty
  should influence aggregation.

These papers do not propose this exact PMME/F2 estimator. F2-GS is a
task-specific adaptation to calibrate sample-to-target gradient compatibility.

## Formula

Let the original F2 per-source-sample gradient be
\[
g_i^s=\nabla_W\ell_i^s
=\delta_i^s\otimes z_i^s,
\]
and the original full-support target gradient be
\[
G_t=\sum_{j\in\mathcal T}\nabla_W\ell_j^t.
\]
The original F2 score is
\[
s_i^{F2}=\frac{\langle g_i^s,G_t\rangle}
{\|g_i^s\|_2\|G_t\|_2+\epsilon}.
\]

We repeatedly subsample target support windows (not sensors and not prediction
steps). For subsample k, retain every sensor in each selected window and form
\[
G_t^{(k)}=\sum_{j\in\mathcal T_k}\nabla_W\ell_j^t,
\qquad
s_i^{(k)}=\cos(g_i^s,G_t^{(k)}).
\]
The per-sample target-reference sensitivity is
\[
\sigma_i=
\sqrt{\frac{1}{K}\sum_{k=1}^{K}
(s_i^{(k)}-\bar{s}_i)^2+\epsilon},
\qquad
\bar{s}_i=\frac{1}{K}\sum_{k=1}^{K}s_i^{(k)}.
\]
We then use a conservative lower-confidence alignment score:
\[
\boxed{\quad
s_i^{GS}=s_i^{F2}-\lambda_{GS}\sigma_i,
\qquad
\alpha_i^{GS}=\operatorname{Softmax}_i(\tau s_i^{GS}).
\quad}
\]
This leaves F2's central criterion (gradient cosine similarity) intact and
penalizes a source sample only when its apparent alignment changes markedly
across target-support subsets. A sample that aligns strongly with one
particular target subset but conflicts with other subsets will not dominate
the source marginal. The target marginal remains F2's original uniform beta.

## Initial settings

- K=6 repeated random subsets without replacement.
- Each subset uses 80% of the target support windows.
- lambda_GS=0.5.
- Seed is deterministic per epoch and target-support group, shared across all
  source cities assigned to the same target group.
- All 36 prediction-head dimensions are used together; no short/mid/long split.

The resampling unit is the target window (batch dimension B). All sensors within
a selected window remain together to avoid treating spatially correlated
sensor gradients as independent target samples. The gradients are obtained
from already-computed projection2 output deltas and features; no additional
autograd backward pass or trainable parameter is introduced.

## Diagnostics

Each F2 extension log reports full-support cosine, repeated target-window subsampling cosine standard
deviation (mean/median/p90), uncertainty penalty, robust-score distribution,
repeated target-window subsampling sample count/size, source ESS, and L1 shift from original F2 alpha.
Beta is unchanged. Small target groups (<3 windows) disable the variance
penalty and fall back to original F2 for that update.

## Experiment plan

Run one seed first with the same batch size, data, training epochs, OT scale,
and Stage-II as the F2 baseline. Compare:
1. original F2;
2. F2-GS with lambda=0.5;
3. lambda ablation 0.25 and 1.0 (only if the first run is stable).

Then repeat the shortlisted setting with seeds 3 and 5. Report 10/60/120/180
minute and all-step MAE/RMSE, Stage-I and Stage-II separately, plus ESS and
gradient-stability diagnostics. Do not tune on target test labels.

The original F2 score is the exact reference; the new formula does not claim
to reproduce VarGrad or the cited sample-gradient-similarity paper.
