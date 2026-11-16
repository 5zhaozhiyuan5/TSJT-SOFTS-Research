# Stage-I CHTA design

## Scope and invariants

This branch implements Continuous Horizon-conditioned Transfer Alignment (CHTA)
inside PMME Stage-I.

- The official SOFTS implementation is unchanged.
- The official \`PMME_2ndStage.py\` is unchanged.
- The existing Stage-I training entry point and the \`model.*\` checkpoint namespace
  consumed by Stage-II are preserved.
- The original PMME base OT term over \`[rep, normalized future trajectory]\` is
  retained. CHTA adds a separate conditional transport regularizer.

## Backbone interface

The Stage-I wrapper continues to call the official backbone exactly as before.
For SOFTS it receives:

- \`rep: [B, N, 128]\` — one representation of the complete historical window
  for each sample and node; it is not interpreted as 36 temporal tokens.
- \`out: [B, N, 36]\` — the original 36-step forecast.

No time query, adapter, projection, or prediction head is inserted into SOFTS.
CHTA consumes the existing \`rep\` output from outside the backbone.

## Continuous horizon parameterization

The model uses a single Fourier-feature MLP to encode physical forecast time:

\`\`\`
u(h) = (h * minutes_per_step) / (H * minutes_per_step)
q(h) = MLP([u, sin(2πfu), cos(2πfu)] for f in {1,2,4,8})
\`\`\`

There is no per-step embedding table and no separate model for each horizon.
The current training configuration evaluates this continuous function at four
physical-time anchors: steps \`[2, 12, 24, 36]\`, corresponding to
\`10, 60, 120, 180\` minutes on the common five-minute grid.

These four anchors are a small quadrature set for the conditional transport
regularizer. They do not split the predictor into short/mid/long networks and
do not cause 36 independent OT solves.

## Historical-to-future change descriptors

During Stage-I training only, CHTA builds a four-dimensional descriptor for
each sample/node and each anchor prefix:

1. mean future change from the last observed value;
2. standard deviation of that change;
3. linear trend slope;
4. endpoint change at the anchor.

The last history value and future target are represented under the same
sample-wise RevIN transform used by the existing SOFTS wrapper. Target future
labels are available only for the few-shot training windows; test/future labels
are never passed to CHTA.

## Horizon-aware DVGP marginals

The original per-source GP domain classifiers remain intact. Their overlap
probabilities form the base source/target sample weights. For each horizon
anchor, a bounded compatibility gate is computed from the distance between a
sample's future-change descriptor and the opposite domain's descriptor
centroid. The gated weights are normalized to form horizon-specific marginals.

Thus the GP classifier architecture is not replaced; the marginal weights
used by OT become horizon-conditioned.

## Conditional transport

For each source/target pair:

1. Select at most \`max_samples\` deterministic flattened sample/node entries.
2. Build one base history-feature cost and one base entropic transport plan.
3. For each of the four horizon anchors, build a cost correction from the
   horizon-conditioned historical representation and future-change descriptor.
4. Reweight the shared base plan by \`exp(-delta_cost / epsilon)\` and rescale it
   to the corresponding horizon-specific DVGP marginals.
5. Optimize the conditional transport cost plus a KL penalty that keeps each
   conditional plan near the shared base plan.

The conditional plans are therefore distinct while sharing a base plan and
using only small sampled matrices. The original full PMME GeomLoss OT term is
kept as the base distribution-alignment objective.

## Stage-I objective and gradient path

The source forecasting loss and original PMME OT loss remain. CHTA adds its
conditional transport loss and a mild smoothness regularizer. A configurable,
low-weight few-shot target forecasting loss is also included in the Stage-I
objective (default \`0.1\`). In the projected-gradient path, the source/transfer
gradient is projected against the target gradient and the weighted target
gradient is added back.

The CHTA transport loss is differentiable through \`rep\` and the CHTA projection
parameters. It can therefore update the existing SOFTS parameters without
changing the SOFTS architecture or its forward signature.

## Test-time behavior and Stage-II compatibility

CHTA, DVGP, and OT are training-only. Test-time prediction still calls the
original SOFTS forward path and returns the original 36-step forecast. No
source-city data or target future labels are required at test time.

The Stage-I checkpoint continues to store backbone weights under \`model.*\` and
CHTA weights under \`chta.*\`. The existing Stage-II loader selects only
\`model.*\` keys. \`PMME_2ndStage.py\` is not changed.

## Initial engineering defaults

- anchors: \`[2, 12, 24, 36]\` steps (\`10, 60, 120, 180\` minutes);
- continuous time embedding: 16 dimensions;
- history/future relation projection: 32 dimensions;
- sampled transport size: at most 128 source and 128 target entries per pair;
- Sinkhorn scaling: 20 iterations, epsilon 0.08;
- conditional OT loss weight: 0.1 relative to the configured DA coefficient;
- target Stage-I loss weight: 0.1;
- smoothness weight: 1e-4.

These are starting values for controlled ablation, not claimed optimal values.
