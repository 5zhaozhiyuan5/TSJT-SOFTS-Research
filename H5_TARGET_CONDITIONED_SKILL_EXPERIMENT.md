# H5 — Target-Conditioned Forecast Skill × Horizon Gradient OT Read-only Audit

**Date:** 2026-11-26  
**Branch:** `experiment/pmme-h5-target-conditioned-skill-audit-20261126`  
**Scope:** Design and implement H5 scientific *diagnostics*, NOT training a new F2/PRC and NOT a finished new module.  
**All candidates use an EXISTING ORIGINAL F2 Stage-I checkpoint including AdamW optimizer state.**  
**Stage-II is never run or modified.**

## 1. Motivation from verified H4 outcomes

H4 capped 8-batch pilot: Chengdu `mid_weak` showed 6/8 positive one-step effects, yet 89% of its net gain came from one batch; its median gain was only about 0.00000823. Across cities the signs were inconsistent and the placebo often rivaled the true `accurate` group. The already-completed PRC seed1 36-horizon MAE was 2.394411, versus historical original F2/T0 2.372713 (different code runs; NOT paired). These findings do **not** support “lower source prediction error = better cross-city transfer”.

New question (pre-registered): Does source forecasting *skill specifically for the prediction horizons where the target-fit city has difficulty*, conditioned on **within-horizon source–target head-gradient compatibility**, provide more predictive utility than F2, PRC, skill-only, gradient-only, and conditional-information-destroying controls?

We are **testing an idea**, not claiming an original novel contribution or superior performance.

## 2. Literature novelty boundary (verified, published work with code)

| Work | Publication and official code | Already known mechanism; limitation for H5 |
|---|---|---|
| TAROT: Targeted Data Selection via Optimal Transport | ICML 2025, https://proceedings.mlr.press/v267/feng25l.html ; https://github.com/vita-epfl/TAROT | Uses target-aware OT selection and whitened gradient-feature distances. H5 must not claim “target-guided OT” or “gradient-OT data weighting” as new. |
| Data Valuation with Gradient Similarity (DVGS) | 2024, https://pmc.ncbi.nlm.nih.gov/articles/PMC11118599/ ; https://github.com/nathanieljevans/DVGS | Source–target gradient similarity values sources; not specifically horizon-conditioned traffic forecasting. |
| TSJT, Drawing Informative Gradients from Sources | AAAI 2025, https://ojs.aaai.org/index.php/AAAI/article/view/32102 | TSB projects/removes harmful source gradient directions; the use of target gradient information itself is not new. |
| CrossTReS | KDD 2022, https://github.com/KL4805/CrossTReS | Source-city region reweighting for cross-city traffic; source selection is not new. |
| CMSVC | IEEE TITS 2025, https://github.com/pku-smart-city/source_code/tree/main/CMSVC | Multi-source selective cross-city transfer; H5 must not claim multi-source weighting as new. |
| Meta-Weight-Net | NeurIPS 2019, https://github.com/xjtushujun/meta-weight-net | Learns sample weights using meta-validation; ours is a fixed read-only counterfactual test, not a new general bilevel weight learner. |

**Novelty status:** limited targeted search, NOT proof of no identical paper. H5 distinguishes itself experimentally by *forecast-band target-demand × source relative skill × same-band gradient*, with direct real-GeomLoss + all-parameter AdamW-shadow tests. It must demonstrate incremental predictive value; multiplying existing scores is insufficient novelty alone.

## 3. Inputs and exact mathematical protocol

- Source cities: METR-LA / Chengdu / Shenzhen; target: PEMS-BAY 3-day SUPPORT.
- Model: original SOFTS/F2 Stage-I complete checkpoint, default epoch 2899, seed1, source batch16, horizon36 (3 × 12).
- Source pool: window×node sample, 256 subsampled points/city in pilot; target-fit pool: 128 subsampled points/target group.
- Target 46 support windows are split **whole-window** modulo 5: 36 fit, 10 probe. Targets are split into original 15/15/16 windows per source city, with only fit windows in each group used for OT.
- Source true future (y_i) is TRAIN-only; target (y^t) is FIT-only when computing weights. Target probe futures are used **only to evaluate** shadows, after all candidate weights, costs, gradients, optimizer updates have been specified.
- The pretrained F2 checkpoint **has already seen the entire target support period**. The fit/probe split is only a shadow-update isolation, **not independent validation**. Overlapping rolling windows further increase correlation.
- `true target test` and `Stage-II` are not loaded.

### 3.1 Source relative skill, per prediction band

For (b\in\{S,M,L\}) (12 forecasts per band), source i and its city-scale standard deviation (σ_c):

[
e_{i,b}=\frac{\operatorname{mean}_{h\in b}|\hat y_{i,h}-y_{i,h}|}{\sigma_c},\quad
p_{i,b}=\frac{\operatorname{mean}_{h\in b}|x_i^{last}-y_{i,h}|}{\sigma_c}.
]

[
s_{i,b}=\operatorname{clip}\left(\frac{p_{i,b}-e_{i,b}}{\max(p_{i,b},0.1)},-10,10\right),
\qquad c_{i,b}=\operatorname{sigmoid}(2s_{i,b}).
]

This is **relative competence versus persistence**, not target transfer probability. City-scale normalization permits fairer across-city reporting. Raw absolute error is separately recorded by original PRC control.

### 3.2 Target need using FIT labels only

[
r_b = \frac{1}{\sigma_t}
 \operatorname{mean}_{(w,n,h)\in \text{target FIT},h\in b}
|\hat y^t_{w,n,h}-y^t_{w,n,h}|,\quad
d_b=\frac{r_b+10^{-6}}{\sum_k r_k+3\cdot10^{-6}}.
]

(Σ_b d_b=1). If these are nearly uniform (reported `demand_range`), the claimed target *conditional* signal is weak.

### 3.3 Matched within-band source–target gradient

From SOFTS projection2 input features (f_i) and current prediction-loss output derivatives (δ_i), without a cross-band dot product:

[
g_{i,b}=\operatorname{vec}(\delta_{i,b}\otimes f_i),\quad
G^t_b=\sum_{j\in \mathrm{target\ FIT}}\operatorname{vec}(\delta^t_{j,b}\otimes f^t_j),
]

[
a_{i,b}=\frac{1+\operatorname{cos}(g_{i,b},G^t_b)}{2}\in[0,1].
]

Important: **same-band only** because projection2 output coordinates for non-overlapping bands are disjoint. The gradients are a **cheap head-gradient proxy** for transfer relevance; the subsequent real OT gradient uses **all** model parameters.

### 3.4 Fixed candidates, single alpha change, no training

For every source city, let original F2 alpha be (α_i^{F2}), and candidate quality (q_i^k\in[0,1]). All six policies use the identical PRC bounded gate (strength (ρ=0.5)):

[
m_i^k=0.5+0.5\operatorname{clip}\left(\frac{q_i^k}{\operatorname{mean}(q^k)+\epsilon},0,2\right),
\qquad
\alpha_i^k=\frac{\alpha_i^{F2}m_i^k}{\sum_j \alpha_j^{F2}m_j^k}.
]

The six *pre-selected*, separately tested candidates:

| Candidate | Quality score | Question |
|---|---|---|
| **PRC** | Existing exact `source_prediction_quality` implementation | Does original source error × persistence quality work in the same checkpoint shadow setup? |
| **DemandSkill** | (\sum_b d_b c_{i,b}) | Does target difficulty-weighted source relative skill help? |
| **DemandGrad** | (\sum_b d_b a_{i,b}) | Does target-horizon-gradient compatibility alone help? |
| **Joint** | (\sum_b d_b c_{i,b}a_{i,b}) | Do skill and gradient **interact** to reveal useful target-conditioned candidates? |
| **RotatedDemand** (negative control) | (\sum_b d_{b+1} c_{i,b}a_{i,b}) (cyclic) | Is the association to the correct target horizon meaningful? |
| **ShuffledSkill** (negative control) | (\sum_b d_b c_{\pi(i),b}a_{i,b}) | Is the actual source skill/profile pairing meaningful, beyond gradient and F2-alpha rank? Shuffle only within original F2-alpha quartiles. |

The baseline is **original F2 alpha unchanged**. For each candidate, only **one city’s alpha** is changed, with the other two cities fixed at F2. One extra `identity` shadow uses the same F2 gradients to check numerical noise.

Total per batch: **1 baseline + (3 cities × 6 candidates) + 1 identity = 20** shadow scores, with 19 genuine policy choices. No candidate changes target beta, cost geometry, sampled points, OT scale, SOFTS, PCGrad or optimizer state.

### 3.5 Actual optimizer-aware local response, not gradient-only “benefit”

[
g^{OT}_{c,k}=\nabla_\theta\big[0.01\cdot\operatorname{GeomLoss}
(\alpha_{c,k},[rep_s,10y_s^{norm}],\beta,[rep_t,10y_t^{norm}])\big].
]
All source cities contribute to the common OT gradient. The virtual one-step update is

[
g_k=\mathrm{OneSidedPCGrad}(g_{source}+\sum_c g^{OT}_{c,k},g_{target,fit}),
\qquad
\theta_k^+=\mathrm{AdamWShadow}(\theta,\mathrm{saved\ optimizer\ state},g_k).
]

For every changed city c, candidate k:

[
\Delta_{c,k}^{MAE}=\mathrm{MAE}_{probe}(\theta_{F2}^+)
 -\mathrm{MAE}_{probe}(\theta_k^+).
]

Positive means candidate improved the **one-step shadow target SUPPORT-probe MAE**. This is NOT independent generalization or final Stage-II MAE. Evaluate 3 horizon bands, 10/60/120/180-min MAE and RMSE as well.

## 4. Three stages + optional, no new F2 training

### H5-0 — Unit test (no data, no GPU)

```bash
git fetch origin
git switch experiment/pmme-h5-target-conditioned-skill-audit-20261126
python -m unittest discover -s tests -p 'test_h5_target_conditioned_skill.py'
```

Tests: source city scaling, horizon demand, zero/negative gradient, deterministic within-alpha-strata shuffle, preservation of (Σ α=1), bounded multiplier, strength zero bitwise unchanged, invalid inputs.

### H5-1 — Real GPU smoke (one batch, capped OT, no training)

```bash
PMME_H5_PROBES=1 PMME_H5_SOURCE_POINTS=128 PMME_H5_TARGET_POINTS=64 \
  PMME_H5_RUN_ROOT=./experiments/h5-smoke \
  bash run_h5_target_skill.sh
```

### H5-2 — Preregistered 8-batch exploratory shadow test

```bash
PMME_H5_PROBES=8 PMME_H5_SOURCE_POINTS=256 PMME_H5_TARGET_POINTS=128 \
  PMME_H5_RUN_ROOT=./experiments/h5-pilot-8 \
  bash run_h5_target_skill.sh
```

Disclose capped OT and same-checkpoint support leakage prominently in summary; avoid test-time cherry-picking. The first four batches are *descriptive discovery* and the final four *in-support directional confirmation*, NOT independent train/validation experiments.

### H5-3 — Optional exact full-OT limited replication only if decision rules pass

For a **pre-fixed** source city / candidate identified by H5-2, repeat with `PMME_H5_SOURCE_POINTS=0 PMME_H5_TARGET_POINTS=0`. Default H5 runner computes all methods under full OT, so this costs much more than capped pilot and is **not automatically scheduled**. It still is not independent target validation.

**No 3000-epoch F2**, **no 6000-epoch Stage-II**, **no new trained checkpoint** in H5.

## 5. Prespecified decision rules (mechanism screening, not significance)

A new mechanism is worth *further* investigation only if:

1. The **Joint** per-city paired gain relative to F2 is positive in ≥6 of 8 batches, positive in both 0–3 and 4–7 segments, and median gain is at least **0.00005 MAE** above the replay noise floor. The criterion must hold in **at least two of three source cities**, not only Chengdu.
2. Joint must also outperform **DemandSkill**, **DemandGrad**, **PRC**, **RotatedDemand**, and **ShuffledSkill** in at least those two cities on paired median gain. At least the two negative controls must show a coherent advantage, or the information-conditional hypothesis fails.
3. No single batch may account for more than 50% of the sum of absolute paired improvements; otherwise H4's outlier issue repeats.
4. Candidate alpha must change appreciably without ESS collapse; record alpha L1 shift, ESS ratio and per-band error, source prediction skill, fit-demand range, head-gradient cosines. Treat demand nearly uniform as **inconclusive**, not evidence for a target-aware mechanism.
5. Exact `identity` replay must have near-zero difference. A probe label can never influence candidate generation.

If **any** serious negative-control failure, unstable sign, or source-specific anomaly occurs, **stop H5**, do not add a trainable controller. There is no statistical significance from 8 correlated support probes. Even if all rules pass, claim only an exploratory mechanism lead. A truly leakage-free validation requires a checkpoint that did **not** train on the evaluation period (a new *different* chronological holdout or a previously available appropriately trained checkpoint). The current instructions prohibit rerunning F2, so that cannot be manufactured here.

## 6. Logs and reproducibility

- `run_config.txt`: branch, commit, checkpoint path and SHA256, method budget, seed, 0 updates.
- `h5_results.metadata.json`: target-support-only disclosure; full/capped OT; fixed 6 methods; optimizer assumptions.
- `h5_results.jsonl`: 8 complete shadow records; per-city source and target sampling SHA256; F2-alpha statistics; source skill per band; target FIT demand; head-gradient cosines; each candidate α ESS/L1 shift; post-update MAE/RMSE at 10/60/120/180 minutes and the three bands; fit and probe window IDs.
- `h5_summary.csv`: paired per-method, per-city mean/median gain; sign consistency, outlier share, first/last 4 directional means, demand contrast, ESS.
- `h5_pairwise_contrasts.csv`: paired `Joint-minus-* ` negative-control and ablation comparisons.
- `summary.txt`, `train.out`, `COMPLETED` or `FAILED`.

### Known limits

- `GeomLoss` may not materialize transport matrix Π; **do not claim Π is logged**. We log α/β summaries, not the explicit coupling.
- We do not replace F2, do not edit official Stage-II or final inference path.
- The three-way score is an experiment-specific test, **not established superior novelty**.
- Do not compare shadow probe MAE directly against paper Table 2 or Stage-II 36-horizon test MAE as if they were the same evaluation.
