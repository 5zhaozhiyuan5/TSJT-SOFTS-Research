# Stage-I METR-LA experiments

All runs use seed 1, SOFTS, source cities `chengdu_m`, `pems-bay`, and `shenzhen`, target city `metr-la`, `train_epochs=3000`, and `finetune_epochs=0`. The table reports the 18 odd forecast horizons (1, 3, ..., 35) selected by the existing evaluation protocol.

| Run | Stage-I method | Checkpoint selection | MAE sum | RMSE sum | Mean MAE | Mean RMSE |
|---|---|---|---:|---:|---:|---:|
| E0 SOFTS | pure SOFTS | source MAE | 86.325812 | 160.728571 | 4.795878 | 8.929365 |
| E1 original PMME | paper PMME | source MAE | 85.189329 | 160.497018 | 4.732741 | 8.916501 |
| E2 current PMME | stabilized PMME Stage-I | source MAE | 84.899582 | 159.348638 | 4.716643 | 8.852702 |
| E3 RHSR | non-PMME RHSR | source MAE | 88.212270 | 163.723920 | 4.900682 | 9.095773 |
| E3b RHSR balanced | RHSR + uniform routing floor | source MAE | 86.397801 | 160.565946 | 4.799878 | 8.920330 |
| E4 RHSR clipped | RHSR + differentiable logit clipping | source MAE | 86.236758 | 161.213728 | 4.790931 | 8.956318 |
| E5 RHSR clipped + target select | same as E4 | target support MAE | 86.236758 | 161.213728 | 4.790931 | 8.956318 |

E4 improves over E3b, but remains behind E2. E5 selecting on the observed target support prefix chooses the same epoch and produces the same held-out test metrics as E4, so checkpoint selection is not the main remaining bottleneck.

Each run directory contains `config.yaml`, `log.txt`, `stage1_test_metrics_seed1.json`, and, for RHSR runs, `rhsr_diagnostics.jsonl`. The server-side checkpoints are copied into the server `experiment_checkpoints` directory with the corresponding run names.
