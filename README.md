# PMME Stage-I CHTA Research

This branch implements **Continuous Horizon-conditioned Transfer Alignment
(CHTA)** in PMME Stage-I.

- Official SOFTS backbone: unchanged.
- Official PMME Stage-II: unchanged.
- Stage-I: continuous horizon-conditioned DVGP marginals and conditional OT.
- Design details: [STAGE1_CHTA_DESIGN.md](STAGE1_CHTA_DESIGN.md).

## Running

Save the datasets under `./data`. Use the existing PMME launcher:

```bash
bash train_one_OT-metr.sh pems-bay 4 0
```

Adjust the target city, thread count, and GPU index as needed. The active CHTA
configuration is in `configs/config.yaml`.

For the original standard baseline, run `normal_baseline.sh`; for the original
domain-adaptation baseline, run `DA_baseline.sh`.
