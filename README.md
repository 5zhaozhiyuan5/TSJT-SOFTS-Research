# TSJT-SOFTS-Research

This repository contains the PMME-based cross-city few-shot traffic forecasting
experiments.

## Active research branches

- **stage-1-chta**: legacy Continuous Horizon-conditioned Transfer Alignment.
- **stage-1-mtpb-moe**: MTPB-inspired multi-scale pattern bank with
  horizon-conditioned MoE-OT as the Stage-I main alignment.

## Stage-I MTPB + MoE-OT

This branch combines:

1. A source-only multi-scale traffic pattern bank inspired by MTPB (1/3/6/12/24h).
2. Three dense MoE alignment experts (short, middle, long historical scales).
3. A physical-horizon router for 10/60/120/180-minute conditional alignment.
4. PMME DVGP sample-reliability weights and a shared-base conditional OT prior.

The full design and implementation boundary are documented in
[STAGE1_MTPB_MOE_DESIGN.md](STAGE1_MTPB_MOE_DESIGN.md).

The original SOFTS backbone is unchanged. The MTPB graph reconstruction and
prediction heads are not imported. PMME Stage-II is not modified.

## Running

Save datasets under ./data, then use the existing launcher, for example:

    bash train_one_OT-metr.sh pems-bay 4 0

The source-only pattern bank is built automatically from source_dataset.x_list.
The active configuration is model.pattern_moe in configs/config.yaml.

## Tests

    python -m unittest tests.test_pattern_moe -v
    python -m unittest tests.test_training_integration -v
