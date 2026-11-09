# Iteration 01 — Inactive-source masking (2026-11-09)

Base: main @ 822d0ce8416e0656df80baedb23c8277f89b4d78
Branch: iteration/01-valid-source-weight-mask

## Repository findings
The existing repository documents SOFTS+source_interaction_v2 and contains synthetic regression tests and configuration files, but SOURCE_COMPATIBILITY_CHECK.md and the main README explicitly state that no new real four-city training/MAE/RMSE result is available. No performance improvement is claimed here.

## Concrete defect and change
The existing source weighting masks zero-norm city gradients before softmax but then adds a uniform exploration floor and historical weight EMA. Both operations can restore nonzero weight to an inactive city. The final weights sum to one but the effective weighted reference is attenuated because the invalid city's adjusted vector is zero.

This iteration zeros weights for invalid source gradients AFTER uniform mixing and EMA, and renormalizes the remaining valid weights. All-zero sources already have an earlier finite zero-vector fallback. The change affects only source_interaction.py, preserving the SOFTS backbone, PGrad virtual trajectories, rotation, meta controller, and residual fusion.

## Evidence and limitations
Added two regression tests for (a) EMA + uniform floor with one zero source and (b) a single valid source. Full pytest suite and real PEMS-BAY training must be run before merging to main. This branch is intentionally unmerged until verified.

Related peer-reviewed research (inspiration, not an equivalent algorithm): PiKE (NeurIPS 2025), Aligned Multi Objective Optimization (ICML 2025), CAGrad (NeurIPS 2021). These methods concern balancing multi-task gradients; none establishes this repository's traffic prediction accuracy.

Next: run PYTHONPATH=src pytest -q tests/ and python -m compileall -q src tests; then compare target MAE/RMSE and source weight diagnostics against the same-seed original branch.
