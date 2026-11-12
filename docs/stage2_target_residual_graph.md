# PMME Stage-II: Target-only graph-temporal residual forecasting

## Scope

This Stage-II implementation uses only the target city's three-day support windows and labels to train the residual predictor. The Stage-I SOFTS checkpoint is loaded and frozen. No source-city residuals, source-city residual bank, retrieval memory, or source-city residual experts are used.

## Previous implementation

The previous Stage-II module used a 10,000-slot learnable memory, initialized from target-support SOFTS representations. Three horizon-conditioned queries retrieved from the shared memory, and three independent linear heads generated the 12-step short/mid/long corrections.

That design generated corrections from the current hidden representation, but did not explicitly model the target-city temporal and spatial structure needed to predict the residual.

## New architecture

The new module is `TargetResidualGraphNet` in `model/Meta_Models/target_residual_graph_net.py`.

Inputs:

- `history`: target-city historical speed sequence after the same RevIN normalization as Stage-I, shape `[B, N, 288]`.
- `representation`: frozen Stage-I SOFTS representation, shape `[B, N, 128]`.
- `base_prediction`: frozen Stage-I forecast in RevIN space, shape `[B, N, 36]`.
- `adjacency`: normalized adjacency matrix of the target city, shape `[N, N]`.

The module contains three structural parts:

1. **Causal temporal encoder.** It encodes the target city's historical level and first-order difference using shared dilated causal convolutions. The causal padding ensures no future observation is used.
2. **Target-graph context.** It propagates the encoded node features through one-hop and two-hop target-city adjacency, capturing spatially correlated residual conditions. The graph is the target city's own graph, not a source-city graph.
3. **Backcast-forecast residual decoder.** Three stacked blocks use 3-, 9-, and 36-knot forecast bases. Each block explains part of the historical input through a backcast and adds an interpolated residual forecast. This follows the hierarchical interpolation and doubly-residual design principles of N-HiTS/N-BEATS, adapted to a frozen-base residual correction task.

The correction is added in the same RevIN-normalized space as the Stage-I forecast:

```text
Y_final = Y_base + Delta_R
```

The Stage-I output is detached and Stage-I parameters are frozen. Only `TargetResidualGraphNet` is optimized on target support labels.

## Training objective

The primary objective remains the existing SmoothL1 forecasting loss on the final corrected prediction. A small correction L2 penalty is added to discourage unnecessarily large corrections in the three-day setting:

```text
L = SmoothL1(Y_base + Delta_R, Y) + 1e-4 * mean(Delta_R ** 2)
```

## Protocol notes

- The held-out target test split is not used for Stage-II training or checkpoint selection.
- The existing target support dataset currently uses `inter_step=12` with `batch_size=46`, yielding 46 support windows for the configured 288-step history and 36-step forecast. This data sampling protocol has not been changed by this Stage-II refactor.
- The old `horizon_residual_memory.py` file is retained for historical comparison, but is no longer imported by the default Stage-II path.
- The default hidden width is 32 to limit overfitting risk with only three days of target support data.

## Literature inspiration and distinction

- ResCAL (CIKM 2022) motivates the use of temporal error correlation and target-graph signals for traffic residual correction.
- The 2026 real-time STGCN residual-correction study trains a second STGCN on baseline residuals. The present implementation also learns a residual target, but uses a compact causal encoder and a hierarchical backcast-forecast decoder rather than cloning the base STGCN.
- N-HiTS (AAAI 2023) motivates multi-rate hierarchical interpolation for multi-horizon output.
- N-BEATS (ICLR 2020) motivates backcast/forecast residual links.
- DARC (Neurocomputing 2026) motivates treating residuals as structured signals rather than unstructured noise. Its temporal-modal decomposition is not copied directly here.

These papers provide architectural inspiration; they do not establish that the new module will improve the current PMME results. The new architecture must be validated with the same seeds and target protocol, and its gains must be reported separately for each forecast horizon.
