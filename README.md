# TSJT Graph WaveNet Research

Cross-city few-shot traffic forecasting with a shared **Graph WaveNet-style** backbone, source interaction v2, target-guided gradient rotation, angle meta-learning and residual fusion.

**Default backbone: Graph WaveNet**, using history-derived dynamic graph supports (no fixed topology, POI, or city-specific node embeddings). The historical SOFTS implementation is retained for reproducibility, but is not used by the default TSJTPGrad model.

**Status:** Backbone migration committed as research code. Synthetic tests and real four-city training have not yet been validated for this migration. There is no verified Graph WaveNet MAE/RMSE improvement. Existing SOFTS checkpoints cannot be loaded into the new backbone.

**External dependency:** This package is not standalone: training imports the original `tsjt_c1` engine and expects separate datasets.

- [Migration details](docs/GRAPH_WAVENET_MIGRATION.md)
- [Backbone implementation](src/tsjt_pgrad/models/graph_wavenet_backbone.py)
- [Source interaction](src/tsjt_pgrad/pgrad/source_interaction.py)
- [Full experiment configuration](configs/source_interaction_v2_pemsbay_full.yaml)

```bash
pip install torch numpy pyyaml pytest
PYTHONPATH=src pytest -q tests/test_graph_wavenet_backbone.py
PYTHONPATH=src pytest -q tests
export PYTHONPATH="$(pwd)/src:/path/to/tsjt_c1:${PYTHONPATH}"
python -m tsjt_pgrad.run --config configs/source_interaction_v2_smoke.yaml
```

Graph WaveNet reference: Wu et al., IJCAI 2019, https://github.com/nnzhan/Graph-WaveNet . This node-count-agnostic implementation differs from the original fixed node-embedding adaptive adjacency.

See historical SOFTS notes in [legacy guide](README_SOFTS_%E6%BA%90%E5%9F%8E%E5%B8%82%E4%BA%A4%E4%BA%92%E6%94%B9%E8%BF%9B.md).

## MTPB one-stage guided-transfer experiment (separate branch)

**Experimental**: [MTPB one-stage protocol](docs/MTPB_ONE_STAGE_GUIDED_TRANSFER.md).

The new `scripts/run_mtpb_one_stage.py` uses the original MTPB
Pattern Bank + reconstructed graph + Graph WaveNet + long-term branch
to predict **288→36**. METR-LA/Chengdu/Shenzhen source tasks determine
the parameter update, while **PEMS-BAY's first three training days only guide
source gradient selection, rotation, residual fusion, and angle feedback**.

**There is no target fine-tuning stage.** The source-trained model is
evaluated directly on the held-out target set; target test labels are never
used during training or checkpoint selection. The offline source pattern-bank
pretraining prerequisite is still present, so "one-stage" describes the
**target adaptation protocol**, not a single end-to-end run from raw data.

This research branch is based on `main` at
`a4f72961cb26e8204b9b12587207c0998a0d04c9`.
The default `main` TSJT Graph WaveNet pipeline has not been changed.
No verified four-city MAE/RMSE improvement yet.
