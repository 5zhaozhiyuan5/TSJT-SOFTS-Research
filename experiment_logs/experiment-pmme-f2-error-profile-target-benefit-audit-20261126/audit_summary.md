# H4 error-profile target-benefit audit — 8 capped probes

- Branch: `experiment/pmme-f2-error-profile-target-benefit-audit-20261126`
- Code commit used for the successful run: `a1f44425249d81c755c63323f6d2e6b63921142a` (fixes the Bash `GROUPS` reserved-variable collision in the launcher).
- Existing checkpoint: original F2 Stage-I checkpoint, epoch 2899; SHA256 `f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf`.
- Target: PEMS-BAY; source cities: Chengdu, METR-LA, Shenzhen; seed 1.
- Budget: 8 probe batches, 256 sampled source points, 128 sampled target points, keep fraction 0.5; capped OT (`full_ot_geometry=false`).
- This is a read-only, one-step, in-support shadow audit. `training_updates=0`; it did not train a new model or measure independent test accuracy.

## Selected signals

Positive suppression gain means halving a source group's OT mass reduced the one-step target support-probe MAE; negative means it increased it.

| Source city | Error profile | Batches | Mean suppression gain | Positive fraction | Short / mid / long gain |
|---|---|---:|---:|---:|---:|
| Chengdu | mid_weak | 8 | +0.00015754 | 6/8 (0.75) | +0.00005288 / +0.00020188 / +0.00021777 |
| Chengdu | accurate | 8 | +0.00009719 | 4/8 (0.50) | −0.00000344 / +0.00008348 / +0.00021157 |
| Chengdu | placebo_accurate | 8 | +0.00010148 | 2/8 (0.25) | −0.00000229 / +0.00008431 / +0.00022221 |
| METR-LA | accurate | 8 | +0.00001395 | 3/8 (0.38) | +0.00000708 / +0.00002724 / +0.00000784 |
| METR-LA | placebo_accurate | 7 | +0.00001904 | 2/7 (0.29) | +0.00003598 / +0.00008297 / −0.00006192 |
| Shenzhen | accurate | 8 | +0.00003570 | 3/8 (0.38) | +0.00003432 / +0.00008255 / −0.00000983 |
| Shenzhen | placebo_accurate | 8 | +0.00001585 | 7/8 (0.88) | +0.00001185 / +0.00002745 / +0.00000840 |

The other error-profile groups vary in direction by city; the complete per-city/per-group results are in `h4_summary.csv` and all eight batch records are in `h4_results.jsonl`.

## Interpretation and stopping decision

Chengdu `mid_weak` is a localized exploratory signal: its mean gain is positive in 6/8 probes and all three horizon-band means are positive. The result does not repeat consistently across cities. For `accurate`, the matched random placebo is similar or more directionally consistent in some cities (notably Shenzhen), so this pilot does not show a general accuracy-specific advantage. The 8 probes reuse one checkpoint and correlated windows from the support period; 6/8 is not statistical significance and this is not an independent holdout.

We stopped after the capped pilot. We did not run the optional full-OT stage or any new Stage-I/Stage-II training: the cross-city pattern and placebo comparison do not establish a robust, preselected profile effect, and choosing a full-OT target after viewing all eight capped probes would be post-hoc.

## Successful artifacts

- `pilot_8_corrected/COMPLETED`
- `pilot_8_corrected/run_config.txt`
- `pilot_8_corrected/train.out`
- `pilot_8_corrected/h4_results.jsonl`
- `pilot_8_corrected/h4_results.metadata.json`
- `pilot_8_corrected/h4_summary.csv`
- `pilot_8_corrected/summary.txt`
- `pilot_8_corrected/launcher_pilot_8_corrected.out`
