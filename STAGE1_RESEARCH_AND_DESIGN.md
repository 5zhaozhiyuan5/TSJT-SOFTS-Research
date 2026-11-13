# PMME Stage-I Literature Review and Design Rationale

Date: 2026-11-13

## Scope and constraints

This review targets only PMME Stage-I representation learning and cross-city alignment. The target-city support budget remains approximately three days; no POI, external city semantics, road topology enhancement, or language model is introduced. The Stage-II implementation and the tensor interface consumed by Stage-II are kept unchanged.

## Relevant high-quality journal work

### 1. SERT — Regional Knowledge Transfer for Urban Traffic Flow Prediction via Satellite Imagery Assisted Contrastive Domain Adaptation
- Venue: IEEE Transactions on Intelligent Transportation Systems, 2025, 26(11), 19120–19134.
- DOI: 10.1109/TITS.2025.3589218
- Public code: https://github.com/MobiXg/SERT
- Relevance: contrasts matched regions to promote useful transfer and pushes unmatched regions apart, addressing negative transfer.
- Adaptation to this project: transfer the *relevance-aware positive/negative matching* principle to latent traffic samples. Do not use satellite imagery or external region semantics.

### 2. CRRL — Contrastive Region Relevance Learning Framework for Cross-city Traffic Prediction
- Venue: Information Fusion, 2025, 103215.
- DOI: 10.1016/j.inffus.2025.103215
- Relevance: dual-branch spatiotemporal representation, similarity-based pseudo-labels, and reliable-region selection with contrastive learning.
- Adaptation: use reliability to reweight source-target transport mass rather than importing its graph/region pipeline.

### 3. SFMGTL — Semantic-fused Multi-granularity Cross-city Traffic Prediction
- Venue: Transportation Research Part C, 2024, 162, 104604.
- DOI: 10.1016/j.trc.2024.104604
- Public code: https://github.com/zeonchen/SFMGTL
- Relevance: common/private memories and domain-invariant representation learning provide evidence that forcing all city information into one shared representation can discard domain-specific information.
- Limitation for this project: its semantic fusion and hierarchical graph clustering require additional graph/semantic machinery that is deliberately excluded here.

### 4. Cross-city Transfer Learning for Traffic Forecasting via Incremental Distribution Rectification
- Venue: Knowledge-Based Systems, 2025, 315, 113336.
- DOI: 10.1016/j.knosys.2025.113336
- Relevance: emphasizes correcting source-to-target distribution mismatch rather than treating source knowledge as directly transferable.
- Adaptation: target-conditioned reliability weighting is a lightweight alternative to introducing another full distribution-transport stack. Its incremental update mechanism is not copied.

### 5. Cross-City Traffic Prediction Based on Deep Domain Adaptive Transfer Learning
- Venue: Transportation Research Part C, 2025, 176, 105152.
- DOI: 10.1016/j.trc.2025.105152
- Relevance: combines city-agnostic feature extraction, knowledge distillation, domain adaptation, and evidential uncertainty.
- Adaptation: supports the importance of separating transferable information from city-specific effects and treating uncertainty as relevant to transfer decisions. The auxiliary urban data and imputation pipeline are not used.

### 6. Spatial-temporal Domain Generalization for Cross-city Traffic Prediction (Mix-ST)
- Venue: Expert Systems with Applications, 2026, 329, 132955.
- DOI: 10.1016/j.eswa.2026.132955
- Relevance: multi-granularity contrastive reconstruction and statistical spatiotemporal representation mixing improve source-domain diversity.
- Adaptation: informs future extensions such as source-style perturbation, but the graph-contrastive reconstruction module is not imported into this first implementation.

### 7. Graph-agnostic Temporal Pretraining for Multi-city Traffic Forecasting (GATP)
- Venue: Physica A: Statistical Mechanics and its Applications, 2026.
- Relevance: node-wise spatial disentanglement and masked temporal pretraining aim to isolate reusable temporal dynamics from city-specific graph structure.
- Adaptation: reinforces disentanglement as a useful design principle. Its separate pretraining/frozen encoder plus graph-conditioning stage is not adopted because this project preserves the PMME backbone and its Stage-II interface.

### 8. A Transferable Framework for Cross-city Forecasting via Invariant Structural Knowledge Learning (TSKNET)
- Venue: Engineering Applications of Artificial Intelligence, 2026, 164, 113161.
- DOI: 10.1016/j.engappai.2025.113161
- Relevance: studies cross-city invariance and explicitly models heterogeneity when deciding what to transfer.
- Limitation: its structural graph construction is outside this project's no-extra-topology constraint.

### 9. MC-HSTA — A Multi-source Cross-domain Hybrid Spatio-temporal Attention Network for Traffic Flow Prediction
- Venue: Neural Networks, 2026, 196, 108358.
- DOI: 10.1016/j.neunet.2025.108358
- Relevance: explicitly considers inter-source dependencies and target-domain shift in multi-source traffic transfer.
- Adaptation: motivates source-wise reliability diagnostics and future source interaction experiments. The current change stays lighter and does not add a multi-head cross-domain attention stack.

## Cross-domain methodological grounding

### Domain Adaptation and Entanglement: an Optimal Transport Perspective
- Venue: AISTATS 2025, PMLR 258, 3034–3042.
- Relevance: derives an OT bound containing conditional-distribution entanglement, warning that marginal alignment alone does not guarantee useful transfer.
- Implication: align the component intended to be shared, rather than indiscriminately aligning the complete representation.

## Synthesis and identified gap

The literature points to three recurring issues:
1. **Entangled information:** shared cross-city dynamics and city-specific variation are often mixed in one representation.
2. **Uniform transfer:** all source samples or all apparent matches are not equally useful for a target city.
3. **Alignment without conditional reliability:** reducing a global distribution distance can still align semantically or predictively incompatible samples.

The current PMME Stage-I already has GP-based sample filtering and conditional OT over latent features plus future values. Therefore simply adding another OT loss or another generic domain classifier would be redundant.

## Implemented Stage-I proposal

The first implementation adds a compact **Shared–Private Reliability Transfer (SPRT)** block:

1. **Identity-initialized shared/private decomposition.** A 36-D backbone representation is partitioned into a 24-D shared component and a 12-D private component. The initial projections and decoder exactly reconstruct the input, avoiding a destructive random feature remapping at initialization.
2. **Disentanglement preservation.** Reconstruction loss preserves information, while normalized cross-covariance penalizes leakage between shared and private components.
3. **Shared-only distribution alignment.** OT, CMMD, MMD, and CORAL alignment operate on the shared representation. The prediction backbone and its output contract are unchanged.
4. **Target-guided reliability weighting.** Each source/target sample receives a weight based on its standardized distance to the opposite domain's feature centroid. These weights are fused with the existing GP confidence weights and normalized into valid transport marginals. This is O((Ns+Nt)D) rather than constructing a dense Ns-by-Nt distance matrix.
5. **Multi-source relevance gating.** A detached source-level score estimates standardized source-to-target centroid shift. The three source alignment losses are reweighted by these scores, preserving the original summed-loss scale when all source domains are equally relevant.
6. **Gradient projection coverage.** When the existing one-sided gradient projection is enabled, the new Stage-I parameters are included in the same projection operation and optimizer.

## Important novelty caveat

Shared/private disentanglement, contrastive relevance, and reliability-aware domain adaptation are established ideas individually. The contribution to investigate is their specific integration with PMME's GP-filtered conditional OT, using target-conditioned reliability to control transport over a learned shared subspace while explicitly preserving city-specific information. A broader prior-art search is still required before making a novelty claim.

## Validation status

- Unit tests cover identity initialization, output dimensions, finite normalized reliability marginals, and gradient flow.
- Full training/integration validation requires the project datasets, CUDA/dependency stack, and the user's runtime environment. Do not describe a full PMME training run as passed until it has actually been executed.
- Stage-II files and the Stage-II checkpoint loading interface must remain unchanged. Stage-II receives the unchanged forecasting backbone state under the existing `model.*` checkpoint key prefix.
