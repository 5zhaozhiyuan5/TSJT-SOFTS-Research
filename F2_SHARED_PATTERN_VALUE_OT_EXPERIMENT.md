# F2 + Shared Pattern-Value OT

## Purpose

This experiment starts from the original F2 Stage-I implementation. It does not
stack DPM-V2, CPRC, TDSM, PRC, P1-P4, or another experimental controller.

The question is whether a source sample should receive OT mass when the traffic
mode it represents has repeatedly produced poor forecasts. A single shared
pattern bank is used across all source cities; there is no separate bank per
city.

## Mechanism

For each sample, history-to-future dynamics are described independently in
short/mid/long forecast bands. The descriptor contains normalized level,
within-band temporal change, and volatility.

A source-only K-means bank stores Keys K_{h,k}. Each source/target sample
queries the same bank:

A_{i,h,k} = softmax_k(-||q_{i,h}-K_{h,k}||_2 / tau).

Each pattern has a scalar Value V_{h,k} in [0,1], initialized to 0.5.
The Value is updated from source training forecast skill using attention-weighted
EMA. Source cities contribute equally:

Vhat_{h,k} = mean over source cities of the attention-weighted forecast skill
for pattern (h,k), considering only cities that cover that pattern.

V_{h,k} <- (1-eta_{h,k}) V_{h,k} + eta_{h,k} Vhat_{h,k}.

The Value readout and current source forecast skill form a sample trust score:

r_{i,h} = 0.5 s_{i,h} + 0.5 sum_k A_{i,h,k} V_{h,k}.

The readout calibrates F2's source marginal, while retaining a nonzero floor:

alpha'_i = Norm[ alpha_i^F2 * (1-lambda_m
              + lambda_m(0.15 + 0.85 mean_h(r_{i,h}))) ].

The attention overlap shapes the pairwise ground cost:

B_{ij} = mean_h[ sum_k sqrt(A^s_{i,h,k} A^t_{j,h,k})
                  * r_{i,h} * (0.25 + 0.75 d_{j,h}) ],

C'_{ij} = C_{ij} + lambda_c median(C) (1-B_{ij}).

The final Sinkhorn objective uses (alpha', beta^F2, C'). F2's original
target marginal remains unchanged. The pairwise attention cost is computed in
blocks; no full attention-pair tensor is retained.

## Data isolation and training order

- The Key bank is fitted only from source-city Stage-I training samples.
- Values are updated only after the optimizer step, once per epoch, using source
  training forecasts and labels.
- Target demand uses only the target's three-day Stage-I training support.
- Validation/test labels are never passed to the component.
- SOFTS, F2 gradient weighting, PCGrad, and the official Stage-II code remain
  unchanged.

## Suggested initial run

Use the runner run_f2_shared_pattern_value_ot.sh with seed 1. The script
defaults to a modest pair-cost strength and a conservative source-mass update.
Compare against the same-commit F2-only control with the same seed and schedule.
Then repeat at least seeds 2 and 3 before drawing conclusions.

## Diagnostics

spv_ot_diagnostics.jsonl records source/target counts, Value/skill/need,
attention concentration, pair-cost changes, nearest-neighbour rank changes,
and F2-alpha shifts/ESS. spv_bank_updates.jsonl records per-band/per-pattern
Value updates and source-city coverage. These are training diagnostics, not a
claim that the full GeomLoss transport plan has been explicitly materialized.
