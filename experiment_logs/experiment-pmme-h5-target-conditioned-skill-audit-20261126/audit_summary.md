# H5 target-conditioned skill audit — 8 capped probes

- Branch: `experiment/pmme-h5-target-conditioned-skill-audit-20261126`
- Code commit used: `8df378dfbf57f0209daac6cdbeddf3e81f50a485`.
- Existing original F2 Stage-I checkpoint: epoch 2899, SHA256 `f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf`.
- Target: PEMS-BAY; source cities: Chengdu, METR-LA, Shenzhen; seed=1.
- Budget: 8 probe batches, 256 source points, 128 target points, strength=0.5, OT scale=0.01; capped OT (`full_ot_geometry=false`).
- This is a read-only one-step support-shadow audit: `training_updates=0`, `stage2_runs=0`. The F2 checkpoint has already seen this support period, so the fit/probe split is not independent validation.

Positive gain means a candidate improved one-step target support-probe MAE versus unchanged F2 alpha. Negative gain means it was worse. Identity replay's maximum numerical noise was `7.15e-7`.

## Joint candidate results

| Source city | Mean gain | Median gain | Positive batches | First 4 mean | Last 4 mean | Largest-batch share |
|---|---:|---:|---:|---:|---:|---:|
| Chengdu | −0.00001761 | −0.00000691 | 1/8 | −0.00002396 | −0.00001127 | 0.525 |
| METR-LA | −0.00008303 | +0.00000250 | 5/8 | +0.00004983 | −0.00021589 | 0.787 |
| Shenzhen | +0.00002235 | −0.00000095 | 3/8 | +0.00004792 | −0.00000322 | 0.736 |

## Decision

The preregistered Joint gate was not met in any city: no city reached 6/8 positive batches with both four-batch segments positive and median gain at least 0.00005 above identity noise. The gains are unstable and strongly outlier-dominated. The Joint candidate also did not consistently outperform the controls; for example, in Shenzhen `shuffled_skill` had a larger mean gain than Joint, while the METR-LA Joint-vs-DemandSkill advantage was itself outlier-dominated (largest-batch share 0.656).

Per the branch stopping rules, we stopped after the capped 8-batch audit. We did not run optional full OT, new F2 training, or Stage II. These one-step support-shadow results are not final model test metrics and must not be compared numerically to paper Table 2.

## Successful artifacts

- `pilot_8/COMPLETED`
- `pilot_8/run_config.txt`
- `pilot_8/train.out`
- `pilot_8/h5_results.jsonl`
- `pilot_8/h5_results.metadata.json`
- `pilot_8/h5_summary.csv`
- `pilot_8/h5_pairwise_contrasts.csv`
- `pilot_8/summary.txt`
- `pilot_8/launcher_pilot_8.out`
