# F2 + TDSM Run Report

- Branch: `experiment/pmme-f2-target-need-skill-matching-20261127`
- Code commit: `605986dffc0dfaf119106b55a8936a768e9b0c7d`
- Target: PEMS-BAY; sources: Chengdu, Shenzhen, METR-LA; model: SOFTS; seed: 1
- TDSM: strength 0.35, 32 prototypes per side, temperature 1.0; DAepoch 1500
- Full run: Stage-I 3000 epochs, official Stage-II 6000 epochs
- Completion: 2026-11-27 11:46:54 (Asia/Shanghai); runner exit code 0

## Selected-checkpoint metrics

| Stage | Best support epoch | 36-step MAE | 36-step RMSE | MAE at 10 min | 60 min | 120 min | 180 min |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-I | 2199 | 2.447902 | 5.345387 | 1.41854 | 2.38150 | 2.76128 | 2.98858 |
| Stage-II | 5099 | 2.395861 | 5.272318 | 1.37295 | 2.31419 | 2.70377 | 2.95445 |

The 3-epoch smoke run also completed successfully. Its log confirms TDSM was active at epoch 1 and both source and target marginal L1 shifts were nonzero. Smoke metrics are diagnostic only.

The training summary lists historical F2/T0 seed-1 Stage-II MAE 2.372713 from a different code revision, not a paired control. Do not treat that number as a controlled comparison. No paper-table comparison is included in this report.

## Artifacts

Successful logs, run configurations, selected-checkpoint CSV summaries, completion markers, and runner status files are stored under `runs/`. Model checkpoints were left on the server and were not copied into this log bundle.
