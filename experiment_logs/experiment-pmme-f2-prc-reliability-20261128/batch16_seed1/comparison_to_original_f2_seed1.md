# F2 + PRC vs original F2 — PEMS-BAY seed1

- PRC run: `experiment/pmme-f2-prc-reliability-20261128`, commit `b93d0a1234e43cb6fdb8fd85e110fb2f5312c23d`.
- Test block: final Stage II; SOFTS; target PEMS-BAY; sources Chengdu, Shenzhen, METR-LA; seed 1.
- F2 control: historical `original` run from `experiment/pmme-f2-robust-fisher-cvar-20261128`, commit `e640c78063710810c12310a9642b3203aeb375d1`, same seed and training schedule.
- Horizon indices 1, 11, 23, 35 map to 10, 60, 120, 180 minutes. Δ = PRC − F2; positive means higher error.

| Horizon | F2 MAE | PRC MAE | Δ MAE | F2 RMSE | PRC RMSE | Δ RMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 10 min | 1.37044 | 1.38157 | +0.01113 (+0.81%) | 2.68074 | 2.70416 | +0.02342 (+0.87%) |
| 60 min | 2.30205 | 2.32688 | +0.02483 (+1.08%) | 5.17266 | 5.22590 | +0.05324 (+1.03%) |
| 120 min | 2.67932 | 2.69156 | +0.01224 (+0.46%) | 5.97720 | 6.02065 | +0.04345 (+0.73%) |
| 180 min | 2.92548 | 2.95242 | +0.02694 (+0.92%) | 6.42979 | 6.51183 | +0.08204 (+1.28%) |

## Result

For this seed, PRC has higher MAE and RMSE than original F2 at all four horizons. The differences are small (0.46%–1.28%); one seed is only a screening result, not a significance test.

PRC execution was verified: the run completed Stage II, reported `PMME_F2_PRC True`, and emitted 180 PRC diagnostic records. At epoch 2975, `error_alpha_corr_after` was lower than `error_alpha_corr_before` for all three source cities, and `target_marginal_unchanged=True`. Thus the calibration was active and shifted source weights, but the final forecasting metrics did not improve over F2 in this seed.
