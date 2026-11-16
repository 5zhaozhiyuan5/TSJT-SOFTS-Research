# Historical Stage-I HCLM design (superseded on stage-1-chta)

This file records the preceding HCLM experiment for reproducibility. The active
design on branch `stage-1-chta` is documented in
[STAGE1_CHTA_DESIGN.md](STAGE1_CHTA_DESIGN.md). HCLM is not instantiated by the
active Stage-I code on this branch.

This branch family starts from the authors' official PMME implementation mirrored at
commit `53a44670db06b7f756e84954e8334ef24cec1335` (upstream source commit
`038d8d6958c13384c34b2e706e43fc0f421ea03c`).

## Scope

- Modify Stage-I only.
- Keep the original SOFTS/iTransformer forecasting backbone and original DVGP.
- Add a horizon-conditioned low-rank metric embedding to the Stage-I OT ground cost.
- Keep `model/Meta_Models/PMME_2ndStage.py` byte-for-byte identical to the official source.

## Representation and conditioning

The official SOFTS backbone returns `rep: [B, N, 128]` and
`out: [B, N, 36]`. Chengdu and Shenzhen are linearly interpolated to the
5-minute grid in `datasets.py`, so the configured 36-step prediction window
covers 180 physical minutes for all four datasets.

The new module consumes `rep` directly; it does not reinterpret the 128 feature
channels as time positions. It uses four physical-time anchors (10, 60, 120,
180 minutes). A shared conditioner produces bounded FiLM parameters and
low-rank projection coefficients for each anchor:

```
u_h = GELU((1 + gamma_h) * LayerNorm(rep) + beta_h)
P_h = normalize_rows(P0 + (A * a_h) @ B.T)
z_h = P_h u_h
R = concat(z_10, z_60, z_120, z_180) / sqrt(4)
```

The low-rank projection has rank 8 and uses shared basis matrices. It does not
create one independent network or one independent OT problem per prediction
step.

## OT integration

The official OT embedding is preserved and extended by one compact block:

```
source_cost = concat(rep_source, sqrt(xy_ratio) * y_source,
                     sqrt(relation_weight) * R_source)
target_cost = concat(rep_target, sqrt(xy_ratio) * y_target,
                     sqrt(relation_weight) * R_target)
```

The existing GeomLoss Sinkhorn call, DVGP-derived marginals, and one-transport-
plan-per-source/target pair remain in place. The relation block is therefore
part of the OT ground metric, while DVGP remains unchanged in this first
implementation.

The OT loss is differentiable through the relation embedding and `rep`, so
it can update both HCLM and the SOFTS/iTransformer backbone. The PCGrad path
also includes HCLM parameters; otherwise its gradients would be discarded
when projection is enabled.

## Initial configuration

- anchors: 10, 60, 120, 180 minutes
- low-rank projection rank: 8
- shared basis dimension: 8
- relation weight: 0.25
- adjacent-anchor smoothness coefficient: 1e-4

These are initial engineering defaults, not claimed optimal values. They need
ablation against the official PMME baseline and a shared-metric control.

## Stage-II compatibility

The Stage-I checkpoint stores HCLM parameters under `hclm.*`. The official
Stage-II loader already filters checkpoint keys to `model.*` before loading
the forecasting model. HCLM keys are therefore ignored by Stage-II, and the
official Stage-II source file is not modified.
