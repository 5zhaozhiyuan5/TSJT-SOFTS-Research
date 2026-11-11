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
6. The three target days are all support data; there is no separate target validation split. Save periodic recovery checkpoints, save the final adapter, and evaluate the held-out target test period once after training. Do not select epochs using the test set.
7. Remove Stage-II PCGrad and source/target gradient projection. The replacement uses ordinary target supervised optimization with gradient clipping.
8. Keep the PMME Stage-I training code and checkpoint loading path unchanged.

### Existing normalization caveat

This replacement does not change the PMME dataset pipeline. In the current `datasets.py`, each city's mean and standard deviation are computed before the target/source time split is applied. For the target city, those global statistics can therefore include observations from the held-out period. This is pre-existing behavior shared with the PMME Stage-I/base forecast, not target windows inserted into the new RAFT bank. The current PR preserves Stage-I weights and preprocessing so that the Stage-II comparison remains controlled. A strictly leakage-free normalization protocol requires a separate dataset-pipeline change and a fresh Stage-I retraining; do not describe the current run as eliminating this normalization-statistics issue.

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

Run the two smoke suites from the repository root:

- **python tests/test_raft_cross_city.py**
- **python tests/test_pmme_raft_stage2_integration.py**

The adapter suite checks source-only bank construction, per-city balancing, target-city exclusion, 36-step and 12-step output shapes, identity initialization, gradients, frozen-backbone parameter selection, and checkpoint buffer restoration.

The Stage-II integration suite goes through the repository's actual `traffic_dataset` preprocessing, constructs a `PMME_2ndStage.STRep` with the real SOFTS backbone, builds a source-only bank, executes one Stage-II forward/backward/optimizer step, verifies finite losses and an actual fusion-parameter update, and verifies that SOFTS weights remain unchanged. It uses small generated arrays, not the real four-city data.

### Runtime defects found and fixed during integration

1. `PatchFSL.build_retrieval_bank()` read `self.args.test_dataset`, but `PatchFSL.__init__()` had not stored `self.args`. The constructor now retains the arguments, and the integration test covers this path.
2. Both `PMME.py` and `PMME_2ndStage.py` imported `SamplesLoss` from `geomloss.samples_loss`, which failed with the installed GeomLoss package. Both now use the supported public import `from geomloss import SamplesLoss`.
3. The Stage-II runner evaluated the held-out target test set again immediately after building the retrieval bank but before adaptation. That redundant pre-adaptation test call was removed; the Stage-I-only baseline measurement remains separate, and the Stage-II held-out test is evaluated after adaptation.
4. CI now installs the runtime dependencies and imports both PMME modules through the integration test rather than relying on `py_compile` alone.

**Verified CI run:** [GitHub Actions run](https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research/actions/runs/43031478916), commit `2cae7bdedf635305b220b5131db4a085ebfcf725`. It passed Python compilation, shell/source-split checks, 6 adapter tests, and 2 production-path integration tests. The integration log includes a completed Stage-II step with finite loss and non-zero adapter gradient.

Passing these tests establishes that the adapter and a synthetic end-to-end Stage-II training step run. It does **not** establish that the full launch script completes with the real datasets/checkpoints, or that target-city MAE/RMSE improves. The repository does not contain the real METR-LA / PEMS-BAY / Chengdu / Shenzhen arrays or the Stage-I checkpoint. A real benchmark must still compare Stage-I-only vs. old residual-memory Stage-II vs. RAFT Stage-II using the same Stage-I checkpoint, seed, data split, and evaluation script.
