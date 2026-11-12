# PMME Stage-II: Target-only temporal residual forecasting

## Scope

Stage-II uses only the target city's three-day support windows and labels. The Stage-I SOFTS checkpoint is loaded and frozen. Stage-II does not use source-city residuals, a source residual bank, retrieval memory, road adjacency, or graph propagation.

## Active architecture

The active module is `TargetResidualTemporalNet` in `model/Meta_Models/target_residual_temporal_net.py`.

Inputs:

- `history`: target-city historical speed sequence after the same per-window RevIN normalization as Stage-I, shape `[B, N, 288]`.
- `representation`: detached Stage-I SOFTS representation, shape `[B, N, 128]`.
- `base_prediction`: detached Stage-I forecast in RevIN space, shape `[B, N, 36]`.

There is no adjacency-matrix input. The network is shared across nodes and processes each node's temporal sequence independently. Cross-node context, if present, is already contained in the frozen Stage-I representation; Stage-II itself performs no graph operation.

## Network components

1. **Causal temporal encoder.** The input contains the historical traffic level and its first-order difference. A causal convolutional stem and four dilated temporal residual blocks (dilations 1, 2, 4, 8) extract temporal features. Per-time-step LayerNorm is used instead of temporal GroupNorm so normalization does not aggregate statistics across future positions.
2. **Condition fusion.** The temporal summary is fused with the frozen Stage-I representation and base forecast. No road topology or adjacency features are concatenated.
3. **Hierarchical backcast-forecast decoder.** Three blocks predict residual trajectories from 3, 9, and 36 interpolation knots. The first two blocks backcast explainable portions of the historical input; the final block has no unused backcast head. Each block contributes a forecast component and the components are summed into the 36-step correction.

The final prediction is:

```text
Y_final = Y_base + Delta_R
```

## Explicit residual supervision

The Stage-I forecast and target are represented in the same per-window RevIN space. The supervised residual is explicitly constructed as:

```text
R_target = Y_target_RevIN - Y_base_RevIN
```

The training objective is:

```text
L = SmoothL1(Delta_R, R_target)
  + 0.1 * SmoothL1(Y_final, Y_target)
  + 1e-4 * mean(Delta_R ** 2)
```

Residual prediction is the primary objective. The final forecast loss is retained as a light auxiliary objective, and the correction penalty discourages unnecessarily large changes.

## Training protocol

- Stage-I SOFTS parameters remain frozen; only `TargetResidualTemporalNet` is optimized.
- Only target-city support windows and labels are used to train Stage-II.
- The held-out target test split is not used for training or checkpoint selection.
- The current support-window sampling protocol is unchanged by this architecture refactor.
- Default hidden width is 32 for the three-day target-support setting.

## Literature inspiration and distinction

- ResCAL (CIKM 2022) motivates modeling temporal correlation in traffic prediction errors. This implementation does not copy its graph-based correction path.
- The 2026 real-time STGCN residual-correction study motivates training a separate model against baseline prediction residuals. This implementation instead uses a compact graph-free temporal encoder.
- N-HiTS (AAAI 2023) motivates multi-rate hierarchical interpolation for multi-horizon output.
- N-BEATS (ICLR 2020) motivates backcast/forecast residual links.

These works provide architectural inspiration; they do not establish that this implementation improves the current PMME results. Performance must be measured against the frozen Stage-I baseline and the previous memory-based Stage-II under the same seeds and target protocol.
