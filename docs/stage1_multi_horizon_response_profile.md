# Stage-I Innovation: Multi-Horizon Traffic Response Profile Supervision

## Scope and invariant boundary

This experiment changes Stage-I training only. The official PMME Stage-II implementation remains byte-for-byte identical to the upstream `serre20/PMME` source. The auxiliary head is defined on `STRep`, outside `PatchFSL`, and is not loaded into the official Stage-II model.

## Motivation

PMME Stage-I already learns shared representations and performs GP-conditioned transport alignment. The added objective asks the shared representation to preserve how traffic evolves over distinct future horizons, not merely to reconstruct the future values through the original forecast loss.

## Response profile

The current forecasting horizon is 36 steps. It is divided into three consecutive 12-step bands: short, middle, and long. For each band, construct three dimensionless response descriptors from the RevIN-normalized future target:

1. Normalized net trend: difference between the mean of the last three and first three values, divided by the band's standard deviation.
2. Normalized volatility: standard deviation of first differences, divided by the band's standard deviation.
3. Relative curvature: mean absolute second difference divided by mean absolute first difference.

The concatenated target is a 9-dimensional profile `Q` with the ordering `[short trend, short volatility, short curvature, middle ..., long ...]`.

## Auxiliary prediction head

SOFTS returns a 128-dimensional node representation `H`. A small shared head predicts the response profile:

`128 -> 32 -> GELU -> 9`

`Q_hat = g_phi(H)`

The profile objective uses SmoothL1 loss. Each city contributes one mean profile loss; the four city losses are averaged so that the auxiliary objective does not scale directly with the number of nodes or samples in a city.

## Stage-I objective and gradient protection

`L_stage1 = L_PM​​ME + lambda_profile * mean_city(L_profile_city)`

The target city's own protected objective is extended from its forecasting loss to its forecasting loss plus its profile loss. The existing one-sided gradient projection is retained. The auxiliary head parameters are included in the Stage-I optimizer and in the projected gradient vector.

The default `--profile_coef` is 0.1. Setting it to 0 disables the auxiliary objective for an ablation.

## Stage-II compatibility

The response-profile head is a top-level member of the Stage-I trainer, not part of `PatchFSL`. The official Stage-I checkpoint loading path in `train_DA_OT.py` extracts only keys prefixed with `model.` before loading the forecasting backbone into the official Stage-II model. Therefore the auxiliary head is discarded after Stage-I training, and Stage-II receives the same official PMME backbone interface as before.

`model/Meta_Models/PMME_2ndStage.py` is verified byte-for-byte against the upstream official file. The Stage-II section of `train_DA_OT.py` is also unchanged; only the Stage-I `--profile_coef` command-line option was added.

## Required validation

- Compare `profile_coef=0` against 0.05, 0.1, and 0.2.
- Ablate trend, volatility, and curvature descriptors independently.
- Report overall and 10/60/120/180-minute MAE/RMSE.
- Report profile prediction loss separately from traffic forecast loss.
- Select hyperparameters using source-city pseudo-target validation, never the held-out target test period.
- Verify that the Stage-II model loads the Stage-I forecasting weights with no missing or unexpected backbone keys.

This is an experimental research hypothesis, not a claim of improved accuracy or proven novelty. The contribution must be supported by ablations and comparison with related derivative/shape-supervision literature.