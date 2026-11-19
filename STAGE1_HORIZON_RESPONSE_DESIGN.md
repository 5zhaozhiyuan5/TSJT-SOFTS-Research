# Horizon-response Stage-I pattern transfer

Branch: `stage-1-horizon-response-pattern`, based on the fixed final-checkpoint
Uniform/NoPattern controlled branch. Stage-II and SOFTS are unchanged.

## What changed

- Source-city **paired training windows only** build five recent-history prototypes
  at 1, 3, 6, 12 and 24 hours and a corresponding four-anchor future response
  (10, 60, 120 and 180 minutes). Source response estimates include posterior
  shrinkage and spread diagnostics.
- Target-city first three days (**46 overlapping training windows**) calibrate
  per-horizon five-scale weights. Shrinkage toward uniform weights prevents
  overconfident calibration. A persistence-comparison reliability gate shuts off
  poor response teachers.
- Response-conditioned pattern similarity is added to horizon OT without
  reweighting the original history/future base terms.
- A small, reliability-gated forecast-response Huber auxiliary loss delivers
  source-derived signal directly to SOFTS in Stage-I. It uses only target
  TRAIN forecasts and source-only response statistics; the test-time forecast
  remains the original SOFTS + official PMME Stage-II.
- Sinkhorn iterations increased to 60 for the new configuration. The final
  source/target marginal errors remain logged and must be checked after training.

## Run

```bash
python -m unittest discover -s tests -p 'test_horizon_response_pattern.py' -v
bash scripts/run_horizon_response_pattern.sh
```

Defaults: METR-LA target, Chengdu/PEMS-BAY/Shenzhen sources,
seed=1, 3 target days, Stage-I=3000, OT from epoch1500,
Stage-II=6000, final epoch checkpoints only.

Inspect `run_metadata.json` (calibration errors, weights, gates),
`response_transfer.csv`, `ot_diagnostics.csv`, `alignment_gradient.csv`,
and `final_metrics.json`. Only one new-architecture experiment is needed
against the already completed fixed-protocol Uniform run.

No guarantee of improved MAE/RMSE is implied until that full GPU experiment
has completed. Patterns are learned on source training *labels*, but current
test sample future labels are never used for retrieval or calibration.
