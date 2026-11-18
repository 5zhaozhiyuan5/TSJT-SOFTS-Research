# Controlled pattern-cost ablation (PMME Stage-I)

## Scientific question

Does the **multi-scale source-only traffic-pattern matching cost** improve
cross-city few-shot traffic forecasts above the existing history/future
horizon-conditioned OT mechanism?

This is **NOT** a three-expert routing comparison. Both arms use the identical
fixed three-expert uniform gate, same MTPB-inspired source-only bank, DVGP,
historical/future descriptors, horizon-specific FiLM, OT regularization,
SOFTS backbone and **unaltered official PMME Stage-II**. Only the coefficient
of the pattern cost inside the Stage-I conditional OT cost is changed.

## Two controlled arms

| Arm | router_mode | pattern_weight | OT cost |
|---|---|---:|---|
| no_pattern | uniform | 0.0 | (history + future) / 2 |
| uniform | uniform | 0.5 | (history + future + 0.5 × pattern) / 2.5 |

The normalization is performed over *active cost weights*, so the no-pattern
arm does not artificially shrink history/future terms. Both arms construct the
same source-only bank for diagnostic comparability. At zero weight, the bank
cannot influence the conditional transport cost or its gradient.

The three experts are **cost groups**, not predictors. No Stage-II router is added.

## Reproducible run

Use this NEW branch, from the repository root:

```bash
bash scripts/run_pattern_cost_ablation.sh
```

Defaults: target METR-LA, source Chengdu / PEMS-BAY / Shenzhen,
3 target days, SOFTS, seed=1, Stage-I 3000 epochs, DA from epoch 1500,
Stage-II 6000 epochs, DAcoef=0.01, xySquaredRatio=100, projection=1,
MemSize=10000, 5-minute processed grid, horizons 10/60/120/180 minutes.

Example override: `GPU=1 SEED=2 RUN_ID=trial_seed2 bash scripts/run_pattern_cost_ablation.sh`.

To run a single arm, use `ARMS="no_pattern" bash scripts/run_pattern_cost_ablation.sh`.
Use **the same OUTPUT_ROOT and SEED** if subsequently running the second arm;
avoid reusing an existing arm directory. The script refuses to append to
existing run directories.

## Strict checkpoint / test protocol

Run flags: `--fixed_final_checkpoint` together with
`--pattern_moe_pattern_weight {0.0,0.5}` and
`--pattern_moe_router_mode uniform`.

1. The two independent executions use the same seed and identical steps
   before DA starts. Epoch 1499 saves
   `stage1_pre_alignment_backbone.pth` as a reproducibility check.
   If pre-alignment parameters differ by >1e-7, the summary audit FAILS;
   the experiments cannot be treated as paired.
2. Stage-I saves **only its final epoch 2999 checkpoint** for Stage-II
   initialization. The old target-few-shot-loss-based checkpoint selector
   is disabled in strict mode.
3. Stage-II uses that FINAL Stage-I SOFTS backbone and saves
   `stage2_...pth` at **epoch 5999** in a separate file. It does NOT
   overwrite the Stage-I checkpoint.
4. Strict mode skips all periodic Stage-I and Stage-II test evaluations
   and pre-Stage-II test peeks. The final Stage-II model is evaluated once.
   No validation data is used for checkpoint selection, no held-out test
   data is used for hyperparameter or checkpoint selection.
5. Report the fixed end-of-training results from `final_metrics.json`.
   `evaluation_metrics.csv` also has the per-step `stage2_final_test`
   entries. Training metrics and gradient diagnostics are saved separately.

The existing legacy training entrypoint remains available without
`--fixed_final_checkpoint`; this branch changes strict protocol ONLY when
the flag is passed.

## Verification / outputs

Each arm has an independent run directory with:

- `run_metadata.json` and `config.yaml`
- `stage1_pre_alignment_backbone.pth`
- `stage1_metr-la_SOFTS_router-uniform_seed1.pth` (final)
- `stage2_metr-la_SOFTS_router-uniform_seed1.pth` (final)
- `final_metrics.json` (10/60/120/180-minute MAE/RMSE)
- `alignment_gradient.csv` (every 100 epochs after DA begins)
- all existing MoE router, pattern bank, pattern usage, OT, gradient, epoch,
  Stage-II, runtime diagnostics

After both arms, the runner invokes:

```bash
python scripts/summarize_pattern_cost_ablation.py \
  ./logged_files/metr-la/chengdu_shenzhen_pems/pattern_cost_ablation/<RUN_ID> --seed 1
```

This audits pre-alignment SOFTS weights, Stage-I/II checkpoint epochs,
common data protocol, routing and pattern coefficients; exports
`comparison.csv` and `comparison_audit.json`. A negative
`delta_*_uniform_minus_no_pattern` is an improvement.

The additional `alignment_gradient.csv` compares unprojected SOFTS
gradients from **the whole conditional OT objective** against the forecast
objective (norms, ratio, cosine). It is not the isolated gradient of the
pattern cost because changing the cost also changes the Sinkhorn plan.
For a causal pattern-cost effect, interpret the paired A/B test results.

## Local checks

```bash
python -m pytest tests/test_pattern_cost_ablation.py -q
bash -n scripts/run_pattern_cost_ablation.sh
```

Run a **single-seed mechanism check first**; if promising, repeat at
multiple seeds and report dispersion before claiming stable improvement.
