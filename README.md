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


## Experimental MTPB integration (unmerged)

The newer, **experimental** MTPB integration is developed in branch
\`iteration/mtpb-replace-meta-update-288x36\`. This branch keeps the author's
full Pattern Bank + graph reconstruction + Graph WaveNet + long-term forecasting
head and changes **one internal training component**: it replaces
\`STRep.meta_train_revise\` with first-order, target-guided multi-source
meta-gradients. This is NOT a post-hoc correction to native MTPB parameters.

- Historic baseline on main: the previous TSJT 12-step Graph WaveNet workflow.
- Research branch: native MTPB 288-step history -> 36-step horizon.
- Sources: METR-LA, Chengdu, Shenzhen; target: PEMS-BAY with only 3 days of
  training supervision.
- The official MTPB source/data/pattern bank is an external dependency
  and must be obtained separately. No real four-city MAE/RMSE claim yet.
- Original SOFTS code remains historical only.

Read [replacement design and experiment protocol](docs/MTPB_INTEGRATED_META_REPLACEMENT.md).

\`\`\`bash
PYTHONPATH=src pytest -q tests/test_mtpb_integrated_meta.py tests/test_mtpb_train_only_norm.py
PYTHONPATH=src python scripts/run_mtpb_integrated.py --help
\`\`\`
