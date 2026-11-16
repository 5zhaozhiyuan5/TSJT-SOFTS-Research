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
2. train_DA_OT.py builds five source-only pattern banks from history windows at
   1/3/6/12/24-hour scales (12/36/72/144/288 steps on the common 5-minute grid).
3. The five scale banks are grouped into three lightweight alignment experts:
   - short-scale: 1h + 3h;
   - middle-scale: 6h + 12h;
   - long-scale: 24h.
4. For each source/target batch pair, each expert converts historical windows
   into soft assignments over shared source prototypes. Pairwise Hellinger
   distances between assignments form the expert-specific pattern costs.
5. A dense horizon-conditioned router receives pooled SOFTS context, the
   training-only future-change descriptors, and the physical horizon code. It
   outputs three weights for each of the four horizon anchors.
6. The weighted pattern cost is combined with horizon-conditioned SOFTS
   representation cost and future-change cost.
7. PMME's DVGP probabilities remain the sample-reliability weights. A shared
   base transport plan is computed from history representations, and each
   horizon plan is KL-regularized toward that base plan.
8. The conditional MoE-OT losses replace the old full-sequence GeomLoss OT
   term when pattern_moe.enabled=true. They are the Stage-I main alignment
   objective, not an extra CHTA auxiliary loss.

## Important implementation details

- Horizon anchors: steps [2, 12, 24, 36], i.e. 10/60/120/180 minutes.
- Pattern bank scale steps: [12, 36, 72, 144, 288], i.e. 1/3/6/12/24 hours.
- Pattern prototypes are deterministic NumPy k-means centers built from a
  bounded sample of source-city training windows.
- Each history window is normalized sample-wise before patch descriptors are
  formed. This reduces city-specific speed-scale effects.
- The router is dense softmax over three experts. Top-1/Top-2 sparse routing is
  intentionally not used in this first version because the target domain has
  only three days of support data.
- The conditional Sinkhorn plans use the shared base plan as a KL prior and
  horizon-specific DVGP marginals.
- Source-city losses continue to use the existing horizon-wise compatibility
  weighting derived from DVGP/future-change descriptors.
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
