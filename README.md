# TSJT research — Graph WaveNet default backbone

Cross-city few-shot traffic forecasting with **Graph WaveNet** as the default
shared forecasting backbone and retained source interaction v2, multi-order
PGrad, target-guided gradient rotation, angle meta-learning and residual fusion.

**Status (2026-11-09):** Graph WaveNet adapter checked on synthetic tensor
regressions (6/6 local tests); no complete four-city training, 50-epoch
benchmark, MAE/RMSE improvement, or checkpoint is provided. Prior SOFTS
snapshot reported 49/49 synthetic tests, but this is NOT a claim that the
entire suite passed after switching backbones.

**External dependency:** This is not a standalone training package. Training
imports the original tsjt_c1 data/TSB engine, which must be separately
installed. The repository contains no private traffic data or pretrained
weights. Use a NEW output directory: old SOFTS and fixed-node Graph WaveNet
weights are not interchangeable with this adapted backbone.

## Backbone and four-city protocol

- **Sources**: METR-LA (5 min, history/horizon 12/12); Chengdu and Shenzhen
  (10 min, 6/6). **Target**: PEMS-BAY (5 min, 12/12), with only the first
  **three labeled training days** used for target adaptation.
- All cities use the same parameters of
  [GraphWaveNetBackbone](src/tsjt_pgrad/models/graph_wavenet_backbone.py),
  with input [B,T,N,3] and output [B,H,N,1]. Graph WaveNet's gated dilated
  temporal convolutions and K-hop diffusion convolutions are retained.
- **Cross-city adaptation, not identical to official Graph WaveNet**:
  the original architecture learns node-embedding matrices of fixed
  node count. Here the adaptive adjacency uses **shared projections of
  historical node signals**, so cities with different N share one model.
  No POI, external data or target test labels are used. By default
  use_knowledge_graph: false means no supplied road adjacency is required.
  Setting it true includes the supplied adjacency as an ordinary backbone
  input, not as a research innovation.
- Six 10-minute historical values are interpolated to a canonical twelve
  5-minute sequence using history only; six outputs select prediction slots
  +10,+20,...,+60 minutes. This is an engineering time-alignment assumption
  and should be ablated.
- SOFTS implementation remains in
  [softs_backbone.py](src/tsjt_pgrad/models/softs_backbone.py) for **explicit
  historical controls only** (backbone_type: softs).

## Commands

Install Python dependencies (torch, numpy, PyYAML, pytest) and the
external tsjt_c1 package, then:

~~~bash
export PYTHONPATH="$(pwd)/src:/path/to/tsjt_c1:$PYTHONPATH"
PYTHONPATH=src pytest -q tests
python -m compileall -q src tests

# The following two commands REQUIRE private four-city data and tsjt_c1.
python -m tsjt_pgrad.run --config configs/source_interaction_v2_smoke.yaml
python -m tsjt_pgrad.run --config configs/source_interaction_v2_pemsbay_full.yaml
~~~

The full configuration keeps the original **50 epochs** and 3-day target
protocol. Never select hyperparameters with target test labels.

## Comparable Graph WaveNet controls

| Run | Configuration | Comparison |
| --- | --- | --- |
| Original TSB | [graph_wavenet_original_tsb_baseline.yaml](configs/graph_wavenet_original_tsb_baseline.yaml) | Target-skewed backward baseline |
| Legacy PGrad | [graph_wavenet_legacy_pgrad_baseline.yaml](configs/graph_wavenet_legacy_pgrad_baseline.yaml) | Single-order PGrad + existing rotation/fusion |
| Source interaction v2 | [source_interaction_v2_pemsbay_full.yaml](configs/source_interaction_v2_pemsbay_full.yaml) | Current multi-order source relation candidate |

Keep seed, data splits, model size and training budget identical when comparing
gradient methods; do not treat old SOFTS vs. Graph WaveNet differences as
source-interaction gains. Collect target MAE/RMSE, pred std, Persistence,
source-target/pairwise gradient cosines, weight distributions, real parameter
deltas, and actual source-update contribution.

## Research / attribution

- [Graph WaveNet, IJCAI 2019, original source](https://github.com/nnzhan/Graph-WaveNet)
- [Graph WaveNet paper](https://www.ijcai.org/Proceedings/2019/0264.pdf)
- [Source interaction module](src/tsjt_pgrad/pgrad/source_interaction.py)
- [Backbone-switch review](docs/iterations/2026-11-09-graph-wavenet-switch.md)
- [Previous SOFTS guide](README_SOFTS_源城市交互改进.md) (historical, not the active default)
- [Previous compatibility report](SOURCE_COMPATIBILITY_CHECK.md) (historical SOFTS migration)
