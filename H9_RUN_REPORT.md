# H9 Run Report — Chronological Holdout Adaptation

**Branch:** `experiment/pmme-h9-chronological-holdout-adaptation-20261127`  
**Code commit:** `0a1d71b8568e50395dc36b560adc31992abbed0b`  
**Target:** PEMS-BAY, SOFTS, seed-1 original F2 Stage-I checkpoint (epoch 2899)  
**Checkpoint SHA-256:** `f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf`

## What ran

| Run | OT geometry | Updates per cloned policy | Dev windows | Status |
|---|---|---:|---:|---|
| `smoke` | capped (source 256, target 128) | 1 | 16 | completed |
| `h9-a-full-1` | full GeomLoss OT | 1 | 70 | completed |
| `h9-b-pilot-5` | capped (source 256, target 128) | 5 | 70 | completed |

All five policies were processed sequentially from independent copies of the same F2 model and optimizer state. The original checkpoint received **zero updates**; official Stage II ran **zero times**. No other seeds were run.

## Main results

Lower is better. These development metrics are not official paper test metrics.

### Full-OT, one update, all 70 windows

| Policy | Dev MAE | Dev RMSE |
|---|---:|---:|
| F2 | 2.213930 | 4.455327 |
| Global half OT | 2.214009 | 4.455573 |
| Projected half OT | 2.213969 | 4.455528 |
| Norm-matched raw OT | 2.214011 | 4.455577 |
| All OT off | 2.214085 | 4.455802 |

Projected half OT was better than global-half and norm-matched raw OT by about 0.000041–0.000042 MAE, but was worse than F2 by 0.000038 MAE.

### Capped OT, five updates, all 70 windows

| Policy | Dev MAE | Dev RMSE |
|---|---:|---:|
| F2 | 2.217943 | 4.427830 |
| Global half OT | 2.218140 | 4.429016 |
| Projected half OT | 2.218430 | 4.429068 |
| Norm-matched raw OT | 2.218330 | 4.428927 |
| All OT off | 2.220403 | 4.432384 |

At step 5, projected half OT was worse than F2 by 0.000486 MAE and worse than global-half and norm-matched raw OT by 0.000289 and 0.000100 MAE, respectively. RMSE showed the same ordering. The projection advantage therefore did not persist across the pilot trajectory.

## Decision and limits

The branch's stop rule says to stop if the projected policy loses its advantage at five updates. That condition was met, so the planned 20-update full-OT run was **not** launched, and no formal Stage-II run is justified by these results.

H9 scores the early post-support development slice `[864, 2016)` (four days, 70 overlapping rolling windows). The original PMME training logs historically evaluated post-support data, and the original checkpoint's scaler uses the full target series. This slice is therefore not a never-before-seen independent holdout, the windows are not independent observations, and the results are **not directly comparable to the paper's official test metrics**. Later target data after index 2016 remained reserved and was not evaluated here.

## Logs

Each completed run's `train.out`, `log.txt`, configs, JSON results, policy/daywise CSVs, decision file, and completion status are stored in this branch under:

`experiment_logs/experiment-pmme-h9-chronological-holdout-adaptation-20261127/`
