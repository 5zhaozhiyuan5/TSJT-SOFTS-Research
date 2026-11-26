# H8 OT gradient direction-vs-norm audit

## Run record

- Branch: `experiment/pmme-h8-direction-vs-norm-20261127`
- Code commit run on the server: `c2b3e638b840cf7d53cbcc5281a399c20fe2eed8`
- Scope: read-only original-F2 Stage-I shadow audit; **0 training updates and 0 Stage-II runs**.
- Target support: PEMS-BAY. Source cities: Chengdu, METR-LA, Shenzhen.
- Checkpoint: Stage-I epoch 2899, SHA256 `f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf`.
- Successful runs: 1-probe capped smoke, 8-probe capped pilot, 8-probe full-OT audit. All have `COMPLETED` markers.
- Existing PMME environment was used; no package or data downloads/installs.
- H7 full-OT replay matched: F2 and `global_half` differences were 0; projected-half difference was `1.43e-6` (tolerance `1e-5`). All 8 AdamW step-norm calibrations were within 1%.

## Full-OT results

Positive gain means lower target-support probe MAE for the first named policy. The primary contrast matches raw OT gradient norm; the secondary contrast matches OT-induced AdamW displacement norm.

| Contrast | Mean gain | Median | Positive | First / last 4 mean | Max abs. share | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `projected_half` vs `norm_matched_raw` | +0.000082314 | +0.000071645 | 8/8 | +0.000077426 / +0.000087202 | 0.234 | Pass |
| `projected_half` vs `adamw_step_norm_matched_raw` | +0.000095934 | +0.000089407 | 8/8 | +0.000091255 / +0.000100613 | 0.180 | Pass; step match within 1% |
| `projected_half` vs `global_half` | +0.000096053 | +0.000090003 | 8/8 | +0.000092089 / +0.000100017 | 0.177 | Pass |

Against the same-gradient-norm raw policy, short-, mid- and long-horizon gains were all positive: `+0.000079155`, `+0.000083417`, and `+0.000084370`. Identity replay noise was zero; the median threshold was `0.00005`.

The result is consistent with a local direction effect beyond merely shrinking the OT gradient or its AdamW step. It remains a one-step shadow audit on correlated target-support windows already seen by the checkpoint. It does not establish generalization, statistical significance, or trained-model improvement, and is not numerically comparable with paper Stage-II MAE.

## Local files

- All successful logs: `D:\模型日志\experiment-pmme-h8-direction-vs-norm-20261127\logs\runs\{smoke,pilot_8,full_8}`
- Branch copy: `D:\模型日志\experiment-pmme-h8-direction-vs-norm-20261127\source\experiment_logs\experiment-pmme-h8-direction-vs-norm-20261127\{smoke,pilot_8,full_8}`
- Main policy table: `logs\runs\full_8\h8_policy_summary.csv`
- Calibration/gradient traces: `logs\runs\full_8\h8_optimizer_direction_traces.csv`
- Decision criteria: `logs\runs\full_8\h8_decision.json`
- Server run root: `/home/zhaozhiyuan/experiment-pmme-h8-direction-vs-norm-20261127`
- Successful log archive SHA256: `848adfd1651fe83f925ca9bb7cd282d8e77d6b7183782360fb0c424e35ef433e`
