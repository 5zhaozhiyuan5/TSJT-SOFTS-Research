# F2 + DPM Pattern Attention OT Run Report

- Branch: `experiment/pmme-f2-dpm-pattern-attention-ot-20261127`
- Initial branch commit inspected: `e50435428ad9f9ed3ab967b3faf5e4e304c296f6`
- Code commit used for successful smoke and formal run: `3c2bc73110d970ab6473d2cfc93e7fe2910dc5d9`
- Target: PEMS-BAY; sources: Chengdu, Shenzhen, METR-LA; model: SOFTS; seed: 1
- DPM: lambda 0.02; 32 patterns per band; attention temperature 0.75; source-bank interval 50; 128 samples per source city; row block 128
- Full training: Stage-I 3000 epochs; official Stage-II 6000 epochs; DAepoch 1500
- Completed: 2026-11-27 13:05 (Asia/Shanghai); runner exit code 0

## Selected-checkpoint metrics

| Stage | Best support epoch | 36-step MAE | 36-step RMSE | MAE at 10 min | 60 min | 120 min | 180 min |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-I | 2899 | 2.461693 | 5.305504 | 1.42057 | 2.38741 | 2.78063 | 3.00535 |
| Stage-II | 5299 | 2.389565 | 5.263803 | 1.38407 | 2.33184 | 2.69435 | 2.93684 |

The successful 3-epoch smoke run verified the source-only pattern bank was fitted and the DPM pair-cost callback executed during active OT. The formal log records a source-only frozen bank with 11,520 training patterns, 90 snapshots, and 32 nonempty clusters in each of the three horizon bands.

The initial smoke on the original branch commit failed because the GeomLoss tensorized Sinkhorn backend first calls its custom cost as `cost(target, source)`. The callback assumed only `cost(source, target)`. A minimal orientation-handling fix was pushed as commit `3c2bc73`; the retry smoke and formal run used that commit. The failed smoke log is not included here or in the local log copy.

The branch summary's F2/T0 seed-1 Stage-II MAE 2.372713 is from a different code commit and is historical context only, not a paired control. These results are not claimed as a controlled comparison to that reference or to the paper.

## Artifacts

Successful smoke and formal logs, run configs, selected-checkpoint CSVs, completion markers, and runner status files are under `runs/`. Model checkpoints and the DPM source bank remain on the server and were not copied into the log bundle.
