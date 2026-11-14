# Stage-I-only innovation protocol

## Fixed components

- Target city: METR-LA.
- Source cities: Chengdu, PEMS-BAY and Shenzhen.
- Backbone: SOFTS.
- Seed: 1.
- Target-visible support: 3 days.
- Official Stage-II: `model/Meta_Models/PMME_2ndStage.py`, unchanged.
- Primary horizons: 10, 60, 120 and 180 minutes (`[1, 11, 23, 35]`).

## Baseline

`stage1_method=pmme` uses the restored PMME Stage-I implementation.  Its
checkpoint is passed to the exact same official Stage-II runner as every
candidate.

## Research candidate: FPP

The current candidate combines ideas from frequency-aware pre-training,
traffic-pattern banks and conflict-aware source update aggregation, but is not
a copy of any one method:

1. Frequency-shape reconstruction supervises the forecast spectrum and local
   temporal trend.
2. A learned multi-scale prototype bank produces a target-conditioned,
   horizon-wise source mixture with a uniform prior for stability.
3. Source and target gradients are separated; conflicting source gradients are
   projected before the target-support gradient is added.

The extra router and prototype parameters are discarded when official
Stage-II loads the `model.*` SOFTS backbone weights.  This keeps the second
stage fixed and makes the comparison a Stage-I comparison.

## Versioning

`v00` is the official PMME+SOFTS baseline.  Candidate experiments use `v01`,
`v02`, ... in the same branch and server directory.  Every version must save
its arguments, Git revision, logs, diagnostics, checkpoint and JSON metrics.
