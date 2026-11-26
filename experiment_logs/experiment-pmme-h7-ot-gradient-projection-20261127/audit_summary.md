# H7 OT gradient projection audit

## Run record

- Branch: `experiment/pmme-h7-ot-gradient-projection-20261127`
- Code commit run on the server: `8d0ce5454037b3eca8b6c52e56e485dcbe7f79c2`
- Scope: read-only audit of the pinned original F2 Stage-I checkpoint; **0 training updates and 0 Stage-II runs**.
- Target support: PEMS-BAY. Source cities: Chengdu, METR-LA, Shenzhen.
- Checkpoint: Stage-I epoch 2899, SHA256 `f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf`.
- Successful runs: 1-probe capped smoke, 8-probe capped pilot, 8-probe full-OT audit. All have `COMPLETED` markers.
- Existing server PMME environment was used; no package/data downloads or installs.
- Full-OT F2 and `global_half` shadow results matched the archived H6 results exactly (max absolute difference `0.0`).

## Full-OT result

`Gain vs global_half = global_half shadow MAE − candidate shadow MAE`; positive values mean the candidate was better on that one shadow update. These are correlated target-support probes already seen by the checkpoint, not independent test results or final model MAE.

| Policy | Mean gain vs half | Median | Positive | First / last 4 mean | Max abs. share | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `city_projection_half` | +0.000096560 | +0.000090718 | 8/8 | +0.000092745 / +0.000100374 | 0.176 | Pass |
| `global_projection` | −0.000109971 | −0.000069261 | 1/8 | −0.000057042 / −0.000162899 | 0.539 | Fail |
| `city_projection` | −0.000110358 | −0.000069618 | 1/8 | −0.000057995 / −0.000162721 | 0.537 | Fail |
| `city_off:shenzhen` | −0.000052929 | −0.000048757 | 2/8 | −0.000065506 / −0.000040352 | 0.426 | Fail |

The passing `city_projection_half` candidate also improved short-, mid- and long-horizon probe scores versus `global_half` by `+0.000089243`, `+0.000079274`, and `+0.000121295` respectively. Identity replay noise was zero; the median threshold was therefore `0.00005`.

All three city OT gradients conflicted with the target-FIT gradient in 8/8 full-OT probes. Global and city projection were numerically equivalent (maximum relative difference `7.42e-8`). Thus `city_projection_half` is effectively half-strength global projection in these probes; the result is not evidence for a separate citywise benefit. Unscaled projection did not beat H6 `global_half`.

This only supports a local one-step hypothesis. The checkpoint had already seen the target support period, and the rolling probes overlap in time. H7 results cannot be compared numerically with paper Stage-II MAE or claimed as trained-model improvement.

## Local files

- All successful logs: `D:\模型日志\experiment-pmme-h7-ot-gradient-projection-20261127\logs\runs\{smoke,pilot_8,full_8}`
- Branch copy: `D:\模型日志\experiment-pmme-h7-ot-gradient-projection-20261127\source\experiment_logs\experiment-pmme-h7-ot-gradient-projection-20261127\{smoke,pilot_8,full_8}`
- Main policy table: `logs\runs\full_8\h7_policy_summary.csv`
- Gradient traces: `logs\runs\full_8\h7_projection_gradient_traces.csv` and `h7_city_projection_traces.csv`
- Server run root: `/home/zhaozhiyuan/experiment-pmme-h7-ot-gradient-projection-20261127`
- Successful log archive SHA256: `97ac19eaf1618a0e7519c65f61489564d695ace5b2005b36ac453ca4aeb91c40`
