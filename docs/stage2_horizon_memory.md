# PMME Stage-II: target-support initialized horizon-conditioned residual memory

> **Legacy reference only (2026-11-12):** This document describes the former horizon-conditioned memory implementation. It is no longer the default Stage-II path on `main`. The active target-only residual architecture is documented in [stage2_target_residual_graph.md](stage2_target_residual_graph.md); the old `horizon_residual_memory.py` is retained only for historical comparison/ablation.

## Decision

The RAFT-style future-trajectory retrieval experiment is no longer the active Stage-II model. It is replaced by a PMME-native horizon-conditioned residual-memory design, using the user's earlier V1 layout as the empirical starting point.

This change is confined to Stage-II. The Stage-I PMME implementation, SOFTS architecture, Stage-I checkpoint, data preprocessing, and three-day target-support protocol are not changed.

## Why the direction changes

The RAFT experiment retrieved source-city future trajectories and fused them with the SOFTS forecast. The user-reported v2 comparison did not improve MAE at any of the four evaluation horizons, and longer-horizon errors became slightly worse. The reliability-gate revision passed synthetic integration tests but has no real-data evidence of improvement.

The earlier V1 experiment was more promising. The user-reported results were:

| Forecast horizon | V1 MAE | V1 RMSE | PMME baseline MAE | PMME baseline RMSE |
|---|---:|---:|---:|---:|
| 10 min | 2.689 | 5.019 | 2.710 | 5.050 |
| 60 min | 4.024 | 8.075 | 4.050 | 8.160 |
| 120 min | 4.685 | 9.185 | 4.690 | 9.300 |
| 180 min | 5.028 | 9.707 | 5.020 | 9.770 |

These are historical user-reported values; the raw run logs are not stored in this repository. They motivate restoring V1 as the controlled Stage-II architecture, not claiming that the new initialization already improves the benchmark.

## V1 horizon-conditioned retrieval

The 36-step forecast is divided into three chronological bands:

- short: steps 1–12
- middle: steps 13–24
- long: steps 25–36

Let \(h\) be the frozen SOFTS representation and \(M\) the shared learnable memory. Each band has its own learnable horizon embedding \(e_k\), while the query/key/value projections remain shared:

\[
q_k = W_q(h + e_k), \qquad
z_k = \operatorname{Attention}(q_k, W_k(M), W_v(M)).
\]

Each retrieved representation is passed through its own 12-step residual generator:

\[
\Delta y_k = f_k(z_k), \qquad
\hat y = \hat y_{\mathrm{SOFTS}} +
[\Delta y_{\mathrm{short}}\,\|\,\Delta y_{\mathrm{middle}}\,\|\,\Delta y_{\mathrm{long}}].
\]

There is no additional duration embedding at the generation side. This preserves the V1 distinction that performed better than the shared-retrieval and dual-encoding variants in the user's earlier ablation.

## New initialization: target-support representation prototypes

The new experiment changes how the shared memory is initialized, not the V1 retrieval/generation layout.

1. Load the unchanged Stage-I checkpoint.
2. Run the frozen SOFTS backbone over the authorized three-day target support inputs.
3. Collect the resulting 128-dimensional node representations. Target labels are not used to initialize memory.
4. Seed the 10,000 shared memory slots from those support representations. If there are more representations than slots, use a seeded subset; if fewer, repeat prototypes with tiny perturbations.
5. Adapt the three horizon-conditioned queries, shared memory, and three residual generators using the original target-support forecasting loss.

The held-out target test split is still constructed only after Stage-II adaptation. No test labels or test-derived features are used for memory initialization or epoch selection.

## Trainable and frozen parameters

| Component | Stage-II behavior |
|---|---|
| PMME Stage-I / SOFTS forecaster | frozen and kept in eval mode |
| Shared memory matrix | trainable; initialized from target-support representations |
| Horizon embeddings (short/mid/long) | trainable |
| Shared \(W_q,W_k,W_v\) | trainable |
| Three 12-step residual generators | trainable |
| Stage-I preprocessing and checkpoint | unchanged |

The attention computation is chunked and checkpointed to reduce peak activation memory for the 10,000-slot bank.

## Diagnostics

Every 25 epochs the runner records, for short/mid/long bands:

- base SOFTS MAE versus corrected MAE
- mean absolute residual correction
- normalized attention entropy
- Stage-II gradient norm
- memory slot count and support representation count

These diagnostics distinguish a useful correction from a memory branch that receives gradients but worsens a horizon.

## Tests and limits

Run from the repository root:

- python tests/test_horizon_residual_memory.py
- python tests/test_pmme_horizon_stage2_integration.py

The unit tests cover deterministic support-prototype initialization, slot filling, output shapes, gradients for all three horizon branches, and Stage-I freezing. The integration test uses the repository's real dataset preprocessing with small generated arrays, initializes memory from the three-day support split, and performs one Stage-II optimizer step.

Passing these tests proves the synthetic Stage-II path executes; it does not prove real METR-LA / PEMS-BAY / Chengdu / Shenzhen MAE/RMSE improvement. The next required experiment is a real-data comparison against Stage-I-only and the prior V1 checkpoint, using the same Stage-I checkpoint, seed, support split, and evaluation script.
