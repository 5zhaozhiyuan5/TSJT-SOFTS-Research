# Graph WaveNet backbone migration (2026-11-09)

Baseline: main at `822d0ce8416e0656df80baedb23c8277f89b4d78`.

## Change
Default prediction backbone changed from SOFTS to Graph WaveNet-style gated dilated temporal convolution + diffusion graph convolution. The adjacency is computed dynamically from historical node speed correlations, not learned city-specific node embeddings. This preserves one parameter set across METR-LA, Chengdu, Shenzhen, and PEMS-BAY with differing node counts. Optional supplied graph is disabled by default (`use_knowledge_graph: false`).

Source interaction, PGrad, gradient rotation, meta-learning, and residual fusion are unchanged. Original SOFTS source is retained for historical ablation only.

## Input/output
Inputs `[B,12,N,3]` or `[B,6,N,3]`; outputs `[B,12,N,1]` or `[B,6,N,1]`. Six-step 10-minute history is interpolated using past-only data to the 12-step 5-minute canonical grid; outputs at 10-minute future times are selected. Do not feed future data into dynamic supports.

## Validation
Run `PYTHONPATH=src pytest -q tests/test_graph_wavenet_backbone.py` and `PYTHONPATH=src pytest -q tests`. Synthetic unit tests are not evidence of real traffic MAE/RMSE improvements. Existing SOFTS checkpoint tensors are incompatible. External `tsjt_c1` engine and real city datasets are required for smoke and 50-epoch evaluations. The repository still needs a full four-city run to establish results.

## Related work
Graph WaveNet: Wu et al., IJCAI 2019, https://arxiv.org/abs/1906.00121 ; https://github.com/nnzhan/Graph-WaveNet . This is a node-count-agnostic adaptation, not the original learned node-embedding adjacency implementation.
