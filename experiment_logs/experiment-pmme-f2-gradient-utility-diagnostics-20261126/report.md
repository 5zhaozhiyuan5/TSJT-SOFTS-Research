# F2 gradient utility diagnostics — seed1

## Run status

- Branch: `experiment/pmme-f2-gradient-utility-diagnostics-20261126` at commit `d358a4544f639b131de962a826c2cc66ac4dc18f`.
- The branch is a read-only D1/D2 diagnostic. An original F2 Stage-I checkpoint was missing, so the original F2 Stage-I prerequisite was run on the designated server: seed1, SOFTS, PEMS-BAY target, Chengdu/Shenzhen/METR-LA sources, batch 16, 3000 epochs, OT starting at epoch 1500, no Stage II. The selected support-validation checkpoint is epoch 2899.
- Diagnostic: 16 sampled three-day support batches, source budget 192, target-fit budget 96; 48 city/batch records. The diagnostic reports `training_updates=0` and does not report forecast test MAE/RMSE.
- The one-batch smoke used smaller budgets (64/32) and is kept separate from the main result.

## Aggregate contrasts

Positive values favor the first named policy's raw first-order OT direction on the support probe loss. These are counterfactual gradient contrasts, not forecast improvements.

| City | Records | D1: cosine high − low mean (positive fraction) | D1: F2 − uniform mean (positive fraction) | D2: low-conflict − high-conflict mean (positive fraction) | D2: sign − F2 mean (positive fraction) |
|---|---:|---:|---:|---:|---:|
| All | 48 | −0.002216 (0.417) | −0.001507 (0.312) | −0.001688 (0.354) | +0.000092 (0.562) |
| Chengdu | 16 | −0.008340 (0.250) | −0.004729 (0.000) | −0.004736 (0.000) | −0.000227 (0.500) |
| METR-LA | 16 | +0.011765 (1.000) | +0.002759 (0.688) | +0.004375 (0.875) | +0.000384 (0.750) |
| Shenzhen | 16 | −0.010074 (0.000) | −0.002551 (0.250) | −0.004702 (0.188) | +0.000118 (0.438) |

## Reading the result

Across the 48 records, the mean D1 high-cosine advantage, F2-over-uniform advantage, and D2 low-conflict advantage are all negative. The sign-over-F2 contrast is very small. Results also differ by source city: METR-LA is positive on D1 and D2, while Chengdu and Shenzhen are negative on those contrasts. The sampled evidence therefore does not support a consistent cross-city benefit from these ranking assumptions; sign conflict is also almost perfectly anticorrelated with cosine in this sample (mean Pearson correlation −0.9989).

The 16 batches are resampled from the same three-day target support and are correlated; positive fractions are descriptive, not independent-trial rates or p-values. Probe windows are separated from fit windows within each sampled batch, but the trained checkpoint already saw target support data. This is an in-support diagnostic, not a held-out generalization test, and it does not approximate the complete PMME PCGrad/AdamW update.
