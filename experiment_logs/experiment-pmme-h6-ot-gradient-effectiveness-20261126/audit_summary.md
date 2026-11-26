# H6 OT gradient effectiveness audit

## Run record

- Branch: `experiment/pmme-h6-ot-gradient-effectiveness-20261126`
- Code commit used on the server: `3749b2643d9a010071b103a0e90cf88cdd0dc833`
- Scope: read-only audit of the pinned original F2 Stage-I checkpoint; **0 training updates and 0 Stage-II runs**.
- Target support: PEMS-BAY. Source cities: Chengdu, METR-LA, Shenzhen.
- Checkpoint: Stage-I epoch 2899, SHA256 `f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf`.
- Successful runs: 1-probe capped smoke, 8-probe capped pilot, 8-probe full-OT audit. All have `COMPLETED` markers.
- The server used its existing PMME environment; no downloads or package installs were done.

## Full-OT results

`MAE gain = original F2 shadow MAE − candidate shadow MAE`; positive values mean the candidate was better for that one shadow update. Probes use correlated target-support windows already seen by the checkpoint. These are not independent test metrics or final model MAE.

| Policy | Mean gain | Median gain | Positive | First / last 4 mean | Max abs. share | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `all_off` | +0.000404656 | +0.000469089 | 6/8 | +0.000686586 / +0.000122726 | 0.253 | Pass |
| `global_quarter` | +0.000206351 | +0.000325799 | 7/8 | +0.000374019 / +0.000038683 | 0.248 | Pass |
| `global_half` | +0.000223964 | +0.000209451 | 8/8 | +0.000242889 / +0.000205040 | 0.184 | Pass |
| `city_off:shenzhen` | +0.000171036 | +0.000178337 | 7/8 | +0.000177383 / +0.000164688 | 0.237 | Pass |
| `city_half:shenzhen` | +0.000049531 | +0.000075459 | 7/8 | +0.000080764 / +0.000018299 | 0.302 | Pass; median exceeds 0.00005 |
| `city_off:chengdu_m` | +0.000033855 | +0.000103831 | 7/8 | +0.000180483 / −0.000112772 | 0.429 | Fail; last-four mean is negative |
| `city_off:metr-la` | −0.000023007 | +0.000088096 | 7/8 | +0.000136673 / −0.000182688 | 0.548 | Fail; last-four mean and share |
| `city_half:chengdu_m` | −0.000142336 | +0.000059366 | 6/8 | −0.000048816 / −0.000235856 | 0.570 | Fail |
| `city_half:metr-la` | −0.000152349 | +0.000015378 | 5/8 | +0.000051439 / −0.000356138 | 0.796 | Fail |

Identity replay noise was 0. The prespecified median threshold was therefore 0.00005. Global OT removal/scaling and Shenzhen-specific policies pass this local screening rule; Chengdu and METR-LA city policies are mixed or fail. This is not evidence that OT removal improves multi-epoch training.

## Gradient / optimizer trace

- Mean full-OT gradient norm: `0.02913`; mean source forecasting gradient norm: `5.12480`.
- OT-gradient cosine with target-fit gradient: mean `−0.37264` (range `−0.44977` to `−0.29703` across 8 probes).
- One-sided PCGrad left the OT component norm effectively unchanged (mean effective/raw norm ratio about `1.0`).
- AdamW shadow parameter-step difference between F2 and no-OT: mean norm `0.002154`; cosine with target-fit descent was negative on all 8 probes, mean `−0.02944`.

This suggests a locally conflicting OT direction after the optimizer shadow update, but it is small and the probe cannot establish generalization or Stage-II impact. H6 results should not be numerically compared with the paper's Stage-II MAE.

## Local files

- All successful logs: `D:\模型日志\experiment-pmme-h6-ot-gradient-effectiveness-20261126\logs\runs\{smoke,pilot_8,full_8}`
- Branch copy: `D:\模型日志\experiment-pmme-h6-ot-gradient-effectiveness-20261126\source\experiment_logs\experiment-pmme-h6-ot-gradient-effectiveness-20261126\{smoke,pilot_8,full_8}`
- Full-OT policy table: `logs\runs\full_8\h6_summary.csv`
- Gradient traces: `logs\runs\full_8\h6_optimizer_gradient_traces.csv` and `h6_city_gradient_traces.csv`
- Archived transfer SHA256: `f676c451d5115453374f28f2250f4a7d6f378883d7cc28b4c9b4d5d3b5dbe9bf`
