# PMME Stage-II replacement: RAFT-style cross-city multi-scale retrieval

## Decision

The previous Stage-II residual-memory implementation is replaced by a cross-city adaptation of **RAFT (Retrieval Augmented Time Series Forecasting, ICML 2025)**. The PMME Stage-I implementation and the SOFTS forecasting backbone remain unchanged.

This is an implementation experiment, not a claim that RAFT has already improved the target-city MAE/RMSE. The repository does not contain the four-city data or checkpoints required to run the full benchmark in this environment.

## Why the previous residual-memory module is being removed

The previous Stage-II stored source-city forecasting residuals:

\[
r_s=y_s-\hat y^{\mathrm{SOFTS}}_s
\]

and assumed that a target history similar to a source history would also have a useful source residual. That is a stronger assumption than ordinary traffic-pattern similarity: two cities can have similar historical speed shapes but different forecast errors because their congestion regimes, sensor noise, and source-model calibration differ.

The previous implementation also coupled several mechanisms in one stage: online reservoir updates, stationarity scoring, MMR selection, a residual gate, source/target joint losses, and one-sided PCGrad. This made it difficult to identify why the correction helped or hurt. The replacement removes that chain rather than tuning its thresholds.

## Candidate-paper comparison

### 1. RAFT — selected

- **Paper:** Sungwon Han et al., *Retrieval Augmented Time Series Forecasting*, ICML 2025, PMLR 267.
- **Paper:** https://proceedings.mlr.press/v267/han25d.html
- **Official code:** https://github.com/archon159/RAFT
- **Core method:** for a query history, retrieve training histories with similar temporal patterns, retrieve their paired future trajectories, construct retrieval-based forecasts at multiple temporal granularities, and fuse those forecasts with the model's own prediction.
- **Why it fits:** it is a modular retrieval-and-fusion method, does not require a road graph, and can be placed after the frozen SOFTS predictor without replacing PMME Stage-I.
- **Limitation:** the original RAFT paper is general time-series forecasting, not cross-city traffic transfer. Its single-series retrieval assumptions do not directly solve cross-city scale differences. This implementation therefore adds per-window scale normalization and a balanced multi-source-city bank; those are cross-city adaptations, not claims about the original RAFT paper.

### 2. FEPCross — not selected as a Stage-II plug-in

- **Paper:** *Frequency Enhanced Pre-training for Cross-city Few-shot Traffic Forecasting*, ECML-PKDD 2024.
- **Paper:** https://arxiv.org/abs/2406.02614
- **Official code:** https://github.com/zhyliu00/FEPCross
- **Core method:** pretrain a cross-domain spatial-temporal encoder over time, amplitude, and phase domains with masked reconstruction and contrastive learning; in few-shot tuning, reconstruct masked target samples for data enrichment and momentum-update a target graph.
- **Why not selected:** its fine-tuning modules depend on the FEPCross pretrained encoder and a graph-based forecasting pipeline. Importing only the augmentation or momentum graph would not reproduce the paper's method, and replacing PMME's SOFTS stage would violate the controlled “Stage-II only” comparison.

### 3. MTPB — not selected as the replacement

- **Paper:** *Multi-scale Traffic Pattern Bank for Cross-city Few-shot Traffic Forecasting*, ACM TKDD 2025.
- **Paper:** https://doi.org/10.1145/3727622
- **Official code:** https://github.com/zhyliu00/MTPB
- **Core method:** pretrain a traffic patch encoder, cluster multi-scale traffic patterns into a bank, query target patterns, aggregate meta-knowledge, and use it for graph reconstruction and forecasting.
- **Why not selected:** MTPB is highly relevant to cross-city traffic, but its pattern-bank query and graph reconstruction would overlap with the PMME Stage-I pattern-matching path and would not be an isolated replacement of the current Stage-II correction module.

### 4. CrossSRM — not selected as the primary evidence base

- **Code:** https://github.com/wyeyss/CrossSRM
- Its README describes a three-day target-visible protocol (two adaptation days plus one validation day), 288-step history, and 12-step horizon. However, the repository says the complete citation will be updated after publication, and the code uses a graph-based architecture. It is therefore not treated as a verified peer-reviewed method or a clean SOFTS Stage-II plug-in.

### 5. Residual-memory traffic methods — reason to avoid another residual bank

Recent traffic residual-memory proposals such as DeCyReM and MURECAST also store historical forecasting errors and retrieve them for correction. They make it harder to justify “real residual memory + similarity retrieval + gate” as a new mechanism, and they are primarily single-city/node-local correction settings rather than the multi-source cross-city setup here.

- DeCyReM: https://www.mdpi.com/2227-7390/14/17/3207
- MURECAST: https://www.mdpi.com/2073-8994/18/9/1485

## New Stage-II data flow

The new Stage-II bank stores **future traffic trajectories**, not model residuals.

For a source-city history window \(x_s\), first calculate its window mean and standard deviation, and normalize the history. The paired future is expressed relative to the last observed value and the history scale:

\[
z_s=\frac{x_s-\mu_s}{\sigma_s},\qquad
v_s=\frac{(y_s-\mu_c)/\sigma_c-x_{s,L}}{\sigma_s}.
\]

Here \(\mu_c,\sigma_c\) are the source city's dataset-level normalization statistics. This places source cities with different speed levels into a comparable retrieval space.

For each period \(p\in\{4,2,1\}\), average-pool the normalized history into non-overlapping patches, center and L2-normalize it, and calculate cosine similarity with the corresponding source-bank keys. Retrieve the top-20 candidates with temperature 0.1:

\[
\alpha^{(p)}_{q,j}
=\operatorname{softmax}_{j\in\operatorname{TopK}}
\left(\frac{\operatorname{sim}(k_q^{(p)},k_j^{(p)})}{0.1}\right).
\]

The weighted future patches are projected back to the full 36-step horizon by a learnable linear layer for each period. The three scale forecasts are averaged. The retrieved future delta is then mapped to the target query's own level and scale, and fused with the frozen SOFTS prediction:

\[
\hat y_{\mathrm{RAFT}}
= W_{\mathrm{fusion}}
[\hat y_{\mathrm{SOFTS}}\,\|\,\hat y_{\mathrm{retrieved}}].
\]

The fusion layer is initialized as an identity on the SOFTS half and zero on the retrieval half. Therefore, before target adaptation, the new Stage-II output is exactly the SOFTS base forecast; an untrained retrieval branch cannot arbitrarily damage it.

## Training and leakage controls

1. Build the bank once from the cached source-city windows only.
2. Explicitly reject the target city if it appears in the bank's city list.
3. Balance the bank capacity across source cities and sample across the complete source-window index range.
4. Freeze the SOFTS backbone; train only the RAFT scale projections and forecast-fusion layer.
5. Adapt with the target city's three-day support set. The source bank is not updated with target samples.
6. Remove Stage-II PCGrad and source/target gradient projection. The replacement uses ordinary target supervised optimization with gradient clipping.
7. Keep the PMME Stage-I training code and checkpoint loading path unchanged.

## Current implementation defaults

| Setting | Value |
|---|---:|
| SOFTS input history | 288 |
| Prediction horizon | 36 |
| Source cities | all configured cities except the target |
| Source bank capacity | 10,000 total entries by default |
| Temporal periods | 4, 2, 1 |
| Retrieval candidates per period | 20 |
| Retrieval temperature | 0.1 |
| Trainable Stage-II parameters | scale projections + fusion layer |
| Frozen Stage-II parameters | all SOFTS parameters |
| Target support | existing 3-day target-few-shot split |

## Tests and what they do not prove

Run: **python -m unittest tests.test_raft_cross_city -v**

The tests check source-only bank construction, per-city balancing, target-city exclusion, 36-step and 12-step output shapes, identity initialization, gradients, frozen-backbone parameter selection, and checkpoint buffer restoration.

Passing these tests establishes that the module runs and obeys the intended data-flow constraints. It does **not** establish a traffic-forecasting performance improvement. That requires running the real METR-LA / PEMS-BAY / Chengdu / Shenzhen experiment with the same Stage-I checkpoint, seed, data split, and evaluation script, and comparing Stage-I-only vs. old residual-memory Stage-II vs. RAFT Stage-II.
