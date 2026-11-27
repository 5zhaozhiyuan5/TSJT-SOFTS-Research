# H10 Run Report — Update Provenance

**Branch:** `experiment/pmme-h10-update-provenance-20261127`  
**Code commit used:** `e6ffa27ef4a57d664f0fcfd0cbb55fdf1d67ca7d`  
**Target:** PEMS-BAY / SOFTS / seed-1 original F2 Stage-I checkpoint, epoch 2899  
**Checkpoint SHA-256:** `f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf`

## Completed runs

| Run | Geometry | Update budget | Development windows | Result |
|---|---|---:|---:|---|
| `h10-a-full` | full OT; official-wrapper and explicit-gradient clones | 1 each | parity gate | passed |
| `h10-b-smoke` | capped OT (256/128) | 1 | 16 | completed |
| `h10-b-full-1` | full OT | 1 | all 70 | completed |
| `h10-b-pilot-5` | capped OT (256/128) | 5 | all 70 | completed |

H10-A compared the original `projection=1.0` wrapper step with the explicit-gradient implementation using the full 46-window support. The hard gate passed: parameter max absolute difference **8.12e-7**, AdamW-state max absolute difference **1.13e-6**, buffer difference **0**, AdamW group hyperparameters equal; all were within the **2e-5** tolerance.

The original checkpoint and optimizer were not changed. Original-checkpoint training updates: **0**. Official Stage-II runs: **0**. All phases ran sequentially on the existing PMME environment; no server downloads or installs were used.

## Main results

Values are development MAE / RMSE; lower is better. T0 is the unchanged checkpoint. The full-OT table uses 70 windows. The five-step table uses capped OT, so it does not establish full-OT multi-step behavior.

| Policy | Full OT, step 1 | Capped OT, step 5 |
|---|---:|---:|
| T0 no update | 2.187754 / 4.443161 | 2.187754 / 4.443161 |
| T0F forward only | 2.187754 / 4.443161 | 2.187754 / 4.443161 |
| T1 zero-gradient AdamW | 2.258169 / 4.470142 | 2.677319 / 4.851186 |
| T2 source only | 2.214085 / 4.455802 | 2.220403 / 4.432384 |
| T3 target only | 2.229893 / 4.431728 | 2.210659 / 4.477342 |
| T4 source + PCGrad | 2.214085 / 4.455802 | 2.220403 / 4.432384 |
| T5 official F2, projection=1 | 2.213961 / 4.455422 | 2.217829 / 4.428672 |
| T6 no PCGrad | 2.213961 / 4.455422 | 2.217829 / 4.428672 |
| T7 project half of OT | 2.213983 / 4.455567 | 2.218219 / 4.428942 |

### Interpretation

- **No update had the best MAE** in both the full-OT one-step comparison and the capped five-step pilot. The official F2 step increased MAE by **0.026207** versus T0 in the full-OT run.
- **Zero-gradient AdamW caused substantial drift:** MAE was **0.070415** worse than T0 after one step and **0.489565** worse after five capped steps. This demonstrates that explicit zero gradients are not equivalent to skipping an optimizer step when loading historical AdamW state.
- **No development-metric change was observed between T5 and T6** at the reported milestones; T4 also matched T2. PCGrad removal/projection did not explain the measured degradation in these comparisons.
- T7 was slightly worse than T5 in the full-OT one-step run (**+0.000022 MAE**) and worse by **+0.000389 MAE** in the capped five-step pilot.
- The five-step run is capped-OT diagnostic evidence; it should not be presented as a full-OT five-step result.

## Evaluation limits

The development interval is PEMS-BAY raw indices `[864, 2016)`: four days and 70 overlapping rolling windows. It is not a never-before-inspected independent test: original PMME logs historically evaluated post-support data, and the original checkpoint reuses full-series target normalization. These metrics are **not directly comparable to the paper's official test metrics**. Later target data after index 2016 was not scored.

## Logs

Successful run files (training logs, run configs, JSON evidence, CSV summaries, decision files, and completion markers) are included under:

`experiment_logs/experiment-pmme-h10-update-provenance-20261127/`

Local copies are under `D:\模型日志\experiment-pmme-h10-update-provenance-20261127\logs\runs\`.
