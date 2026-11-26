# H7 — Target-FIT OT Component Gradient Projection: Existing F2 Read-only Test

**Branch:** `experiment/pmme-h7-ot-gradient-projection-20261127`  
**Based on H6 completed full-OT run commit:** `184db4b5c04982b269bee7d29e3bbc5b82d2cd57`  
**Status:** Code committed; server CPU unit tests / GPU smoke / full-OT audit NOT yet executed. No claimed forecast accuracy improvement.

## 1. Motivation from real, previously uploaded logs

In H6's 8 full-OT one-step support-only shadow probes, original F2 OT gradient cosine against target-FIT gradient was negative across all 8 batches, mean `-0.37264`; source-task gradient dominated the sum and the whole-update one-sided PCGrad did not remove the conflicting OT component. On the same **previously-seen target SUPPORT** probes, halving all F2 OT gradients yielded 8/8 lower MAE (mean gain `+0.00022396`). Turning off Shenzhen OT yielded 7/8 gains. This is **neither independent validation nor proof that a trained model with weaker OT performs better**. Historical T0/T1/T2 Stage-II MAEs were 2.37271/2.37636/2.38693 under different OT schedules; local gains may not predict final MAE.

**H7 hypothesis:** does orthogonally removing the *negative target-FIT component* from F2 OT gradients outperform the much simpler H6 **global half** rule under the *identical full-parameter PCGrad+stateful AdamW shadow*, given the exact original OT/gradient?

This is deliberately an **ablative falsification test**, not a claim of a new gradient surgery invention. PCGrad is NeurIPS 2020 and RCGrad 2024 likewise studies gradient-conflict rotation/projection.

Scientific references:
- Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html ; code https://github.com/tianheyu927/PCGrad
- *Enhancing molecular property prediction with auxiliary learning and task-specific adaptation*, 2024 (RCGrad): https://pmc.ncbi.nlm.nih.gov/articles/PMC11270959/
- *Gradient Harmonization in Unsupervised Domain Adaptation*, 2024: https://arxiv.org/abs/2408.00288
- *Gradient Alignment Improves Test-Time Adaptation for Medical Image Segmentation*, AAAI 2025, with released code: https://ojs.aaai.org/index.php/AAAI/article/view/32244 and https://github.com/Chen-Ziyang/GraTa

Do not claim novelty on the basis of OT gradient projection alone.

## 2. Inputs, unchanged baseline, and all seven shadow scores

**Pinned existing original F2 Stage-I SOFTS checkpoint** (model **and AdamW state**, epoch 2899; seed 1; target PEMS-BAY, 3-day support; sources Chengdu, METR-LA, Shenzhen; 36 horizon):
```text
/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth
SHA256 f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf
```

Checkpoint mismatch **fails**. H7 never runs F2 training or Stage-II. Original `model/Meta_Models/PMME.py`, `transfer_utility.py`, prediction backbone and official Stage-II are **not edited**.

Data and exact gradient/OT:
- Target **FIT**: 36 of 46 three-day support windows; **PROBE**: remaining 10 where whole-window local index mod 5 == 0. Window identities differ but rolling windows overlap in time, and checkpoint ALREADY saw the full support period.
- One city-specific original F2 GeomLoss (p=1, blur=0.001, debias=False) per source. OT cost representation: `[SOFTS_rep, 10*RevIN_normalized_true_future]`. Target OT sees **FIT** true futures only; never PROBE futures. Original source alpha (original gradient cosine temperature 8) and uniform beta. OT scale fixed at `0.01`.
- Source forecasting gradient `g_s` from three sources (source TRAIN futures), target FIT forecasting gradient `g_t`, original F2 full-parameter OT gradient `g_c = gradient(L_OT,c)` for city c. Uses target FIT label only.

```text
G = g_chengdu + g_metr-la + g_shenzhen

Project(g; g_t):
    if dot(g,g_t) < 0:
        g - dot(g,g_t)/(||g_t||²+1e-8)*g_t
    else:
        g
```

**Precommitted six comparisons (and one numerical control)**:

| Code | Intervention to the original OT gradient G | Main question |
|---|---|---|
| `f2` | G | Original F2 reference |
| `global_half` | 0.5G | **Primary H6 control: everything else must beat this** |
| `city_off:shenzhen` | G - g_shenzhen | Strong city control from H6 |
| `global_projection` | Project(G;g_t) | Does correcting the global OT direction help? |
| `city_projection` | Sum_c Project(g_c;g_t) | Is city-level projection more valuable than global? |
| `city_projection_half` | 0.5 Sum_c Project(g_c;g_t) | Does projection + reduced strength beat half alone? |
| `identity` | EXACT SAME G as F2 | Noise/determinism control |

All seven share *identical* source and target OT geometry, original sample weights, GeomLoss gradients, source and target FIT forecasting gradients, and original checkpoint AdamW state. No other changes.

For every candidate policy `G_k`, H7 **retains the original one-sided whole-update PCGrad**:
```text
v_k = OriginalOneSidedPCGrad(g_s + G_k, g_t)
theta_k = FullParameterShadowAdamW(theta_F2, saved_optimizer_state, v_k)
gain_vs_F2(k)   = MAE_probe(theta_F2) - MAE_probe(theta_k)
gain_vs_HALF(k) = MAE_probe(theta_HALF) - MAE_probe(theta_k)
```
Positive gain means candidate improved one-step SUPPORT-probe predictions. Predicted physical-speed MAE, RMSE, short/mid/long SmoothL1 and 10/60/120/180min metrics are reported. No `optimizer.step`, no model weights or AdamW moments changed. The shadow forwards use the **same SOFTS random seed as H6** (`681947 + batch_index`) to permit a reference replay check.

### Key analytical constraint: global vs city projection can be IDENTICAL

Since projection along one fixed target gradient is linear whenever all three OT city gradients are conflicting, if **all** `<g_c,g_t><0`, then:

```text
Project(sum_c g_c;g_t) = sum_c Project(g_c;g_t)
```

subject to floating point error and the common `eps`. H6 found negative city cosines across all 8 full-OT probes. Therefore **city_projection and global_projection may give effectively identical outcomes**. This MUST be reported as lack of independent city-level benefit, rather than promoted as a new mechanism. The code records `all_cities_conflicting`, raw/proj city norms, and absolute/relative `global_vs_city_projection_difference`.

If the city-level result is mathematically redundant, the surviving research question is *global projection vs global half*, not inventing a second city innovation.

## 3. Intended execution, no new F2

```bash
git fetch origin
git switch experiment/pmme-h7-ot-gradient-projection-20261127

# CPU-only math/unit tests, no traffic dataset/GeomLoss/GPU:
python -m unittest discover -s tests -p 'test_h7_ot_component_projection.py'

# 1 real GeomLoss smoke with 256/128 sampled OT points:
PMME_H7_PROBES=1 PMME_H7_SOURCE_POINTS=256 PMME_H7_TARGET_POINTS=128 \
PMME_H7_RUN_ROOT=./experiments/h7-smoke \
bash run_h7_ot_projection.sh

# 8 capped exploratory probes:
PMME_H7_PROBES=8 PMME_H7_SOURCE_POINTS=256 PMME_H7_TARGET_POINTS=128 \
PMME_H7_RUN_ROOT=./experiments/h7-pilot-8 \
bash run_h7_ot_projection.sh

# IMPORTANT: H6 directional signal came from FULL OT; replicate full geometry:
PMME_H7_PROBES=8 PMME_H7_SOURCE_POINTS=0 PMME_H7_TARGET_POINTS=0 \
PMME_H7_RUN_ROOT=./experiments/h7-full-8 \
bash run_h7_ot_projection.sh
```

- With `0/0`, all source and target FIT points are used, not a capped proxy.
- Runs always include checkpoint SHA and code commit and refuse to append to another configuration's folder.
- The full-OT runner automatically compares `f2` and `global_half` replay MAEs with the inherited archival `experiment_logs/experiment-pmme-h6-ot-gradient-effectiveness-20261126/full_8/h6_results.jsonl` if available. `h7_vs_h6_reference_check.json` logs matched/unmatched at 1e-5 tolerance. If mismatch, investigate stochasticity/dataset/order before interpreting H7 as an H6 replicate.

No requirement to run full OT if smoke fails.

## 4. Output artifacts

```text
experiments/h7-full-8/
  run_config.txt                 # original F2 SHA, git HEAD and run settings
  train.out                      # complete diagnostics
  h7_results.jsonl              # detailed 7-policy scores, gradients, step traces
  h7_results.metadata.json      # exact checkpoint/cost and leakage warnings
  h7_policy_summary.csv         # PRIMARY: paired gain relative to global HALF
  h7_projection_gradient_traces.csv
  h7_city_projection_traces.csv
  h7_vs_h6_reference_check.json # only if full OT and H6 archival log available
  summary.txt
  summary.err
  COMPLETED / FAILED
```

Each per-batch record tracks 3-city gradient projections, OT alpha ESS/top mass/sample checksums, original beta uniformity/ESS, target FIT/probe window indexes, exact training updates=0. **The explicit transport plan Π is not materialized/logged**, so do not claim otherwise.

## 5. Prespecified decision / stop rules

**Screening only; NEVER a final test or significance test**:

1. First require `identity` F2 replay MAE difference smaller than `1e-4` (expected near 0), checkpoint SHA correct, all 8 full-OT records complete, no train/Stage-II updates.
2. A projection policy must beat `global_half` in **at least 6 of 8** paired full-OT shadow probes, have positive first-four and last-four means, median improvement `>max(5e-5, 10×identity noise)`, and no single batch account for more than 50% of total absolute improvement. At least no conspicuous short/mid/long degradation.
3. Show projection is genuinely correcting the OT direction (report pre/post target cosine and norm) and survives the *actual* optimizer. Do not treat tiny `<1e-6` numerical differences as transfer information.
4. If `city_projection≈global_projection`, reject **citywise projection novelty** even if both beat half.
5. If candidate does not beat global half, **STOP** additional projection variants. Do not retrain official F2 or add complicated trainable branches merely to seek small gains.
6. If all screens pass, still treat as promising **in-support one-step hypothesis**, not proven improvement. Independent chronological validation where the checkpoint never saw the evaluation period would be necessary before formal training/inference claims.

## 6. Files added (no changes to baseline implementation)

- `model/Meta_Models/h7_ot_component_projection.py`
- `audit_h7_ot_component_projection.py`
- `run_h7_ot_projection.sh`
- `summarize_h7_ot_projection.py`
- `tests/test_h7_ot_component_projection.py`
- `H7_OT_COMPONENT_PROJECTION_EXPERIMENT.md`

H7 is an isolated diagnostic code path. The upstream Stage-I F2 and Stage-II official training code remain unmodified.
