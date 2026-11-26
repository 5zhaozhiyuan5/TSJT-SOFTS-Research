# H3 City Counterfactual — Seed 1

- Branch: experiment/pmme-f2-h3-city-counterfactual-20261126
- Code commit: 76db05e43b22edf027a4aa8cf356de57fcf98eca
- Server: 10.131.139.83
- Run: seed1, 8 probe batches, full OT geometry, GPU 0
- Target city: PEMS-BAY
- Checkpoint: original F2 Stage-I SOFTS seed1, epoch 2899
- Checkpoint SHA-256: f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf
- Status: COMPLETED; 8 result records; training_updates=0

## Summary

This is a read-only one-step shadow audit using the three-day support split. Positive gain means lower probe loss than the original all-three-city F2 reference. It does not use the target test set and is not a paper-level forecasting result.

| Policy | Mean smooth-L1 gain vs F2 | Positive probes | Mean MAE gain vs F2 |
|---|---:|---:|---:|
| all_off | +0.000623256 | 87.5% | +0.000623137 |
| city_off:shenzhen | +0.000120848 | 62.5% | +0.000120759 |
| city_off:chengdu_m | +0.000070125 | 87.5% | +0.000070035 |
| city_uniform:chengdu_m | +0.000002354 | 62.5% | +0.000002176 |
| F2 reference | 0 | — | 0 |
| city_off:metr-la | -0.000012875 | 62.5% | -0.000012904 |
| city_uniform:metr-la | -0.000034511 | 25.0% | -0.000034451 |
| city_uniform:shenzhen | -0.000128508 | 37.5% | -0.000128478 |

The best average policy in this support-only diagnostic was all_off. Treat these values as diagnostic evidence only; they are not independent test-set metrics or a direct comparison with the paper.

## Artifacts

Full successful run artifacts are stored under logs/seed1_gpu_full locally and in this branch folder. They include the 8-row JSONL, CSV summary, run configuration, metadata, train output, and COMPLETED marker.
