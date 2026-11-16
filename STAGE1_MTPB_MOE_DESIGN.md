# Stage-I MTPB-inspired Pattern MoE-OT

## Scope

This branch experiments with a lightweight fusion of PMME Stage-I and the
multi-scale pattern-bank idea from MTPB.

It is deliberately **not** a full MTPB port:

- SOFTS remains the original forecasting backbone and its forward interface is unchanged.
- No MTPB TSFormer/GNN decoder, graph reconstruction, GWN head, Linear head, or Reptile loop is imported.
- The original PMME Stage-II implementation is not edited.
- The source-only pattern bank is used only while training Stage-I.
- No target-city test window or future target is used to build the pattern bank.

## Data flow

1. The existing PMME wrapper runs SOFTS and returns rep and out.
2. datasets.py declares each raw sampling interval and maps the current
   METR-LA/PEMS-BAY (5-min) and Chengdu/Shenzhen (10-min) inputs onto the
   existing common 5-minute grid before window generation.
3. train_DA_OT.py converts physical pattern durations [60, 180, 360, 720,
   1440] minutes into steps using the processed grid, then builds five
   source-only pattern banks. On the current 5-minute grid these are
   [12, 36, 72, 144, 288] steps. A raw 10-minute grid would resolve them to
   [6, 18, 36, 72, 144] steps instead.
4. The five scale banks are grouped into three lightweight alignment experts:
   - short-scale: 1h + 3h;
   - middle-scale: 6h + 12h;
   - long-scale: 24h.
5. Each patch is normalized independently before prototype construction. Query
   patches use the same normalization, preventing a mismatch between bank
   construction and online batch matching.
6. For each source/target sample pair, each expert converts the corresponding
   historical windows into soft assignments over shared source prototypes.
   Pairwise Hellinger distances form the expert-specific pattern costs.
7. A dense router receives the pair's SOFTS contexts, training-only future-change
   descriptors, and physical horizon code. It produces three expert weights
   separately for every source/target pair and horizon anchor.
8. The weighted pattern cost is combined with horizon-conditioned SOFTS
   representation cost and future-change cost.
9. PMME's DVGP probabilities remain the sample-reliability weights. A shared
   base transport plan is computed from history representations, and each
   horizon plan is KL-regularized toward that base plan.
10. The conditional MoE-OT losses replace the old full-sequence GeomLoss OT
    term when pattern_moe.enabled=true. They are the Stage-I main alignment
    objective, not an extra CHTA auxiliary loss.

## Important implementation details

- Physical horizon anchors are [10, 60, 120, 180] minutes. They resolve to
  [2, 12, 24, 36] steps on the current 5-minute grid.
- Physical pattern scales are [60, 180, 360, 720, 1440] minutes. They resolve
  to [12, 36, 72, 144, 288] steps on the current 5-minute grid.
- The data loader explicitly records raw and processed sampling intervals.
  Chengdu/Shenzhen are interpolated using the repository's existing 2x path;
  METR-LA/PEMS-BAY are already on the 5-minute grid.
- The training entrypoint checks that the largest scale fits the configured
  history window and that horizon anchors fit the prediction horizon.
- Pattern prototypes are deterministic NumPy k-means centers built from a
  bounded sample of source-city training windows.
- Every extracted patch is normalized independently, matching the query-time
  patch normalization and reducing city-specific speed-scale effects.
- Router diagnostics report pair-averaged expert weights, pair-level entropy,
  expert costs, router smoothness, plan diversity, and marginal errors.
- The router is dense softmax over three experts. Top-1/Top-2 sparse routing is
  intentionally not used in this first version because the target domain has
  only three days of support data.
- The conditional Sinkhorn plans use the shared base plan as a KL prior and
  horizon-specific DVGP marginals.
- Source-city losses continue to use the existing horizon-wise compatibility
  weighting derived from DVGP/future-change descriptors.
- Router weights are also regularized for smooth changes between adjacent
  physical horizon anchors.
- The existing one-sided PCGrad and low-weight target Stage-I supervision are
  retained.

## What is intentionally not included yet

- No separate frequency-domain expert. First isolate whether multi-scale
  pattern knowledge and horizon-conditioned MoE improve alignment. Frequency
  features can be added as a later ablation if diagnostics show complementary
  information.
- No unbalanced OT in this first implementation. Balanced conditional Sinkhorn
  is retained to isolate the effect of the pattern bank and MoE router.
- No dynamic source-city gate beyond the existing horizon-wise DVGP
  compatibility weighting.
- No change to the target-sample split protocol in the existing PMME training
  loop.

## Configuration

The active block is model.pattern_moe in configs/config.yaml.
The legacy model.chta block remains in the file for ablation and is disabled
by default. Do not enable both at once.

## Training and tests

The existing PMME launcher can still be used, for example:

    bash train_one_OT-metr.sh pems-bay 4 0

The source-only pattern bank is built automatically from source_dataset.x_list.
Run the focused tests with:

    python -m unittest tests.test_pattern_moe -v
    python -m unittest tests.test_training_integration -v

The first full experiment should compare:
1. official PMME;
2. legacy CHTA;
3. pattern bank + horizon-conditioned OT without MoE (fixed equal expert weights);
4. pattern bank + horizon-conditioned dense MoE-OT (this branch default).

Log and inspect router weights, expert costs, conditional plan diversity,
marginal errors, per-horizon MAE/RMSE, and runtime. A non-uniform router by
itself is not evidence of improvement; per-horizon forecast metrics must
support the change.
