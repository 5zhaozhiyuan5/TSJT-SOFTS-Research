# TSJT-SOFTS-Research

Cross-city few-shot traffic forecasting research code: shared SOFTS backbone, source interaction v2, and retained target-guided gradient rotation / meta-learning / residual fusion.

**Status:** Initial research snapshot. Local synthetic/regression tests pass (49/49 on 2026-11-09); real four-city 50-epoch training and MAE improvements have **not** been verified here.

**Important dependency:** This package is **not standalone**. It imports the existing `tsjt_c1` training/data engine and expects traffic datasets to be provided separately. No private datasets, checkpoints, or experiment logs are included.

- [Detailed Chinese guide](README_SOFTS_%E6%BA%90%E5%9F%8E%E5%B8%82%E4%BA%A4%E4%BA%92%E6%94%B9%E8%BF%9B.md)
- [Compatibility check](SOURCE_COMPATIBILITY_CHECK.md)
- [Main experiment configuration](configs/source_interaction_v2_pemsbay_full.yaml)
- [Interaction module](src/tsjt_pgrad/pgrad/source_interaction.py)
- [SOFTS backbone](src/tsjt_pgrad/models/softs_backbone.py)

## Local regression tests

```bash
pip install torch numpy pyyaml pytest
PYTHONPATH=src pytest -q tests
```

## Run with existing TSJT engine

```bash
export PYTHONPATH="$(pwd)/src:/path/to/tsjt_c1:${PYTHONPATH}"
python -m tsjt_pgrad.run --config configs/source_interaction_v2_smoke.yaml
python -m tsjt_pgrad.run --config configs/source_interaction_v2_pemsbay_full.yaml
```

This is research code inspired by the publicly released SOFTS structure and is not the official SOFTS repository. See [SOFTS](https://github.com/Secilia-Cxy/SOFTS).
