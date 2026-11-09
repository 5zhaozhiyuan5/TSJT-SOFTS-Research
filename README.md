# MTPB 288->36 + TSJT target-guided gradient research (iteration branch)

**Current research candidate: native MTPB PatchFSL/GWN (ACM TKDD 2025)**, retaining its official traffic pattern bank, graph reconstruction, long-term branch and 288-step input / 36-step prediction protocol. The previously migrated Graph WaveNet 12-step model below is now **historical**, not this branch's default experimental protocol. The original MTPB checkout is required; the author's code has not been copied.

- [MTPB official repository](https://github.com/zhyliu00/MTPB) (pinned SHA `5676a38a34bf3aebbd83cef3cb0d617a9a153b15`)
- [Migration audit, upstream compatibility and run commands](docs/MTPB_GRADIENT_MIGRATION.md)
- [Gradient source interaction bridge](src/tsjt_pgrad/mtpb/adapter.py)
- [Native/GD comparison runner](scripts/run_mtpb_pgrad.py)
- [MTPB 288/36 protocol](configs/mtpb_pemsbay_288_36.yaml)

**Status: code integration candidate, independent iteration branch; no actual four-city MTPB 288->36 run or MAE/RMSE result.** Source-only vs target-guided modes share the same official MTPB model and a weight-preserving repair of upstream 4D convolution usage. MTPB graph data are unchanged baseline inputs, not a new graph innovation.

---

## Historical TSJT Graph WaveNet 12-step research (retained for comparison)


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
