# 2026-11-09: SOFTS -> Graph WaveNet (backbone only)

## Baseline and intent

- Source of truth: repository main 822d0ce8416e0656df80baedb23c8277f89b4d78,
  not any historical ZIP.
- The previous inactive-source masking fix was on a separate iteration branch
  and is NOT silently included here.
- Change requested: Graph WaveNet becomes the default forecaster. Retain source
  interaction v2/PGrad, gradient rotation, AngleController, meta updates and
  residual fusion without algorithm edits. Four-city setup: METR-LA, Chengdu,
  Shenzhen -> PEMS-BAY, three labeled target training days; mixed 12/6 steps.

## Technical decision and limitations

Paper: Wu et al., Graph WaveNet for Deep Spatial-Temporal Graph Modeling,
IJCAI 2019, https://www.ijcai.org/Proceedings/2019/0264.pdf
Official source: https://github.com/nnzhan/Graph-WaveNet

Original Graph WaveNet uses dilated gated temporal convolutions, K-hop graph
diffusion and trainable adaptive adjacency, normally with N-dependent node
embeddings. These are incompatible with sharing the exact same gradient
parameter vector across cities having different node counts.

Adaptation implemented: adaptive adjacency is calculated from historical-only
node speed statistics via a shared node Q/K network. This differs from original
Graph WaveNet; its quality is UNVERIFIED. Optional supplied adjacency is
disabled by default to avoid requiring external topology for the experiment.
Use of a road adjacency is allowed as a backbone input if separately configured,
but is not claimed as a contribution. No POI/LLM/target test labels.

A graph block reads the skip path after residual GCN and BatchNorm so the last
graph convolution and normalization parameters receive non-None gradients.
6x10min input is history-only interpolated to 12x5min; 6 output positions
are +10,+20,+30,+40,+50,+60 minutes. This temporal alignment was retained from
SOFTS and is not Graph WaveNet's original design.

## Files / reproducibility

- NEW: src/tsjt_pgrad/models/graph_wavenet_backbone.py
- UPDATED: src/tsjt_pgrad/models/pgrad_model.py defaults to graph_wavenet and
  retains explicit softs for historical controls.
- UPDATED: nine active configs pin graph_wavenet, disable supplied topological
  adjacency, use new output directories and remove inactive SOFTS settings.
- UPDATED: three softs_* historical controls explicitly pin softs.
- NEW: two Graph WaveNet gradient-baseline configs (TSB and legacy PGrad).
- NEW: tests/test_graph_wavenet_backbone.py; UPDATED README.md.
- NOT MODIFIED: source_interaction.py, pgrad.py, rotation.py, fusion.py,
  meta.py, trainer.py, original SOFTS code and historical docs.

Local CPU runtime: Python 3.13, torch 2.10.0+cpu; OMP_NUM_THREADS=2.
Actual command and result:
    cd <repo-root>
    PYTHONPATH=src OMP_NUM_THREADS=2 pytest -q tests/test_graph_wavenet_backbone.py
    6 passed

Also compiled the two changed source files locally. Synthetic tests cover
four city sizes, time alignment and 6/12 readout, differentiable adaptive
adjacency, functional_call virtual parameter state, optional real adjacency
and absence of a required graph input.

Not tested: all previous 49 suite tests with full original repository,
external tsjt_c1 integration, four-city real-data smoke, or 50-epoch training.
No MAE/RMSE is supplied. Do not infer improvement from synthetic tests.

## Next falsifiable checks

1. Pull commit and run whole pytest suite with original tsjt_c1 installed.
   Validate BatchNorm functional_call buffer behaviour and last-layer gradients.
2. Real four-city short smoke: check finite loss, optimizer-visible updates,
   nonzero source contribution, source/target gradient cosines, pred std and
   Persistence; ensure 6/12 predictions cover matching physical horizons.
3. Fix architecture and train budget across Graph WaveNet + original TSB,
   Graph WaveNet + legacy PGrad and full source-interaction method, evaluating
   50 epochs only with approved compute. Report all target-city MAE/RMSE,
   failures and seeds, not only winning runs.

## Known tradeoffs

- Replacing fixed learned node embeddings with historical Q/K may produce
  nearly uniform adjacency or batch-composition-dependent predictions.
- Graph supports scale quadratically in city node count; profiling at N=627
  is necessary before long-run training.
- Model checkpoint keys and dimensions differ from original fixed-node GWN
  and SOFTS; resume is disabled and new directories are mandatory.
- A change of backbone is not evidence of source interaction innovation.
