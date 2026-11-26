# H6 — F2 OT 梯度有效性、PCGrad/AdamW 优化路径与强度诊断（只读）

> Branch: `experiment/pmme-h6-ot-gradient-effectiveness-20261126`  
> Scope: **Only diagnostics on the already trained original F2 Stage-I checkpoint.**  
> No F2 retraining, no Stage-II retraining, no new trainable controller, and no test-set evaluation.

## 1. Why H6 is necessary

Existing evidence (read the **actual logged** reports, not research-plan speculation):

1. H3 full-batch OT, eight *support-only one-step* probes: `all_off` gave mean MAE gain +0.00062314 against all-city F2, with 7/8 positive probes. `city_off:chengdu_m` +0.00007004, 7/8 positive; `city_off:shenzhen` +0.00012076, 5/8; `city_off:metr-la` −0.00001290.
2. D1/D2 probes found source-specific opposite signs in cosine-based proxy transfer effects and little orthogonal information in sign-conflict scores.
3. H4 source error-profile suppression did not generalize across cities.
4. H5 `Joint` target-difficulty × source skill × gradient score failed prespecified 8-probe rules in all 3 source cities. Source alpha changed nontrivially but did not stabilize transfer improvement.
5. Separate **trained** T0/T1/T2 OT schedule variants: Stage-II 36-step MAE 2.37271 / 2.37636 / 2.38693 (seed 1, matched experiment). Earlier on-support one-step advantages DO NOT predict these multi-epoch results by themselves.

**Hypothesis to TEST, not a result**: F2 OT can be locally unfavorable after the actual PCGrad + AdamW update, because (i) its gradient conflicts with the target fit task, (ii) the original optimizer can transform the raw gradient's effect, and/or (iii) some source-city OT components are locally detrimental. Before implementing another alpha controller, quantify these effects directly.

## 2. H6 scope and controls

Target: PEMS-BAY 3-day support; source cities Chengdu, METR-LA, Shenzhen; SOFTS forecast horizon 36 = 3 × 12; original source batch16; F2 scale .01; F2 temperature 8; same F2 alpha and target beta as existing code.

**Mandatory checkpoint is pinned to the existing F2 Stage-I epoch-2899 checkpoint and full optimizer state**:
- Default: `/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth`
- Required SHA256: `f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf`

If file or checksum is absent/mismatched, **run fails**. It will NOT launch any 3000-epoch F2 preparation.

### H6-1: OT vector before and after real one-sided PCGrad and AdamW

For the **same** source/target-FIT batch:

```text
g_s = grad(sum of 3 source forecasting losses)       [all SOFTS parameters]
g_t = grad(target FIT-only forecasting loss)          [all SOFTS parameters]
g_c = grad(original F2 GeomLoss OT from source city c) [all SOFTS parameters]
g_OT = sum_c g_c

G_0 = PCGrad_one_side(g_s, g_t)
G_1 = PCGrad_one_side(g_s + g_OT, g_t)

theta_0 = AdamWShadow(original_checkpoint, optimizer_state, G_0)
theta_1 = AdamWShadow(original_checkpoint, optimizer_state, G_1)
```

Log (a) norms and cosine of each `g_c`, `g_OT`, `g_s` versus FIT `g_t`; (b) `G_1 - G_0` norm and cosine with FIT `g_t`; (c) optimizer-step displacement `theta_1 - theta_0`, cos with `-g_t`, and size. For each city c also report conditional PCGrad and optimizer-step contribution `F2 - city_off(c)`.

**Do not assert that dot with target-fit gradient is a held-out transfer utility.**

### H6-2: Fixed global OT gradient strength (not alpha surgery)

Single fixed set `lambda in {0, .25, .5, 1}`:

```text
G_lambda = PCGrad_one_side(g_s + lambda*g_OT, g_t)
theta_lambda = AdamWShadow(checkpoint, saved_optimizer, G_lambda)
gain_lambda = MAE_probe(theta_1) - MAE_probe(theta_lambda)
```

Positive `gain_lambda` means **lambda performed better than original F2** for this one shadow update; negative means worse.

Importantly: `lambda*g_OT` is a linear scaling of a **precomputed real OT gradient**. The alpha/beta Sinkhorn objective is calculated once per city at the original F2 weighting. We are NOT computing approximate cost proxies for individual strength factors.

### H6-3: Source-city-specific gradient contributions

For `c in {Chengdu, METR-LA, Shenzhen}`:

```text
G_city_off(c)  = PCGrad(g_s + sum_{j != c} g_j,         g_t)
G_city_half(c) = PCGrad(g_s + sum_{j != c} g_j + .5*g_c,g_t)
```

Compare same shadow checkpoint and probe; record city + associated target group, the per-city F2 alpha ESS and top 24 weights, beta uniformity/ESS, source/target sampled-point checksums, and short/mid/long / 10,60,120,180 minute gains.

**11 evaluations per batch** (one original F2, one all-off, global quarter, global half, 3 city-off, 3 city-half, and an identical F2 replay). The identical F2 replay is a numerical noise control.

Every policy and city uses identical fixed source/target point set, original F2 marginal weights, original GeomLoss cost `concat(SOFTS_repr,10*normalized_future)`, scale .01, PCGrad and AdamW state. Only OT-gradient scaling changes.

## 3. Data and leakage restrictions

- Target sample contains 46 rolling support windows. `masked_support_split` marks 10 whole windows as PROBE (indices % 5 == 0), 36 as FIT.
- Target head gradient, OT weights, Sinkhorn target cost and virtual optimizer are constructed **exclusively using FIT target labels**, not PROBE labels.
- PROBE labels are used only to score predictions **after** all virtual updates have been computed.
- The checkpoint was previously trained on the three-day target support, including the PROBE time period. All 8 probes are correlated; rolling windows may overlap. Thus this is **not truly independent testing, validation or statistical evidence of generalization**.
- Target test dataset and Stage-II are not loaded.
- Sinkhorn via GeomLoss does not expose an explicit transport matrix Pi here; **do not claim Pi or an explicit NxM cost matrix was logged**. Source/target weight summaries and exact cost specification are logged.
- A result in H6 cannot be directly compared numerically to the Stage-II 36-step MAE of 2.37271.

## 4. Run steps (never retrain F2)

```bash
git fetch origin
git switch experiment/pmme-h6-ot-gradient-effectiveness-20261126

# No data/GPU needed; checks gradient combination and read-only AdamW shadow.
python -m unittest discover -s tests -p 'test_h6_ot_gradient_effect.py'

# Fast 1-batch real GeomLoss smoke, *capped* OT:
PMME_H6_PROBES=1 PMME_H6_SOURCE_POINTS=256 PMME_H6_TARGET_POINTS=128 \
PMME_H6_RUN_ROOT=./experiments/h6-smoke \
bash run_h6_ot_gradient.sh

# Pilot 8 batches, capped OT and 11 virtual steps each:
PMME_H6_PROBES=8 PMME_H6_SOURCE_POINTS=256 PMME_H6_TARGET_POINTS=128 \
PMME_H6_RUN_ROOT=./experiments/h6-pilot-8 \
bash run_h6_ot_gradient.sh
```

**Important:** H3 already had an **8-batch full-OT** result; we recommend H6's capped pilot mainly to check gradient/optimizer traces and numerical invariance. For a *methodologically comparable replication of the H3 all-off/city-off signal*, you must do the exact full-OT geometry after smoke:

```bash
PMME_H6_PROBES=8 PMME_H6_SOURCE_POINTS=0 PMME_H6_TARGET_POINTS=0 \
PMME_H6_RUN_ROOT=./experiments/h6-full-8 \
bash run_h6_ot_gradient.sh
```

The last command may be expensive in GPU time/memory, but it never trains the model. Start with smoke first. Runs are isolated in distinct directories; a mismatched commit or checkpoint cannot reuse previous `run_config.txt`.

### Files produced

```text
h6_results.jsonl                # full per-probe/per-policy scores and gradients
h6_results.metadata.json        # SHA, source/target budgets, read-only constraints
h6_summary.csv                  # paired shadow gains, sign and outlier metrics
h6_optimizer_gradient_traces.csv# pre/post PCGrad, AdamW OT-effect directions
h6_city_gradient_traces.csv     # per-city conditional gradient/optimizer effects
summary.txt                     # summary and interpretation guardrails
train.out                       # complete per-batch log
run_config.txt                  # git commit, checkpoint SHA, exact flags
COMPLETED or FAILED             # audit status, no training
```

## 5. Precommitted stop/next-step criteria

This is **mechanism screening**, NOT a significance test:

1. The `identity` replay difference must stay below 1e-4 MAE in the 8 correlated support probes; investigate if much above H5's ~7.15e-7 noise floor.
2. A candidate scaling/city policy must show at least 6/8 *positive* gains, positive first and last 4-batch mean, median gain > max(0.00005, 10× identity-noise), and no single batch >50% of total absolute effect. Otherwise stop that candidate.
3. See whether the OT-induced gradient / PCGrad / actual AdamW-step delta materially changes magnitude or alignment and whether those shifts associate with the *candidate* differences. Large raw gradient effects but tiny effective optimizer differences imply a weak gradient-value controller.
4. If `all_off` remains better than F2 in full OT, do not immediately remove OT from training: T0/T1/T2 full training suggests multi-epoch consequences are different. Need a separately established causal/predictive link before implementing a new controller.
5. If effects are inconsistent, stop adding hand-designed scores. Revisit OT's objective/traffic-pattern correspondence, not more trial-and-error formula coefficients.

## 6. Files changed

New files only; original F2 code, training entrypoints, original `model/Meta_Models/PMME.py`, source/target data, Stage-II implementation all untouched:

- `model/Meta_Models/h6_ot_gradient_effect.py`
- `audit_h6_ot_gradient.py`
- `tests/test_h6_ot_gradient_effect.py`
- `summarize_h6_ot_gradient.py`
- `run_h6_ot_gradient.sh`
- `H6_OT_GRADIENT_EFFECT_EXPERIMENT.md`

**Implementation was committed before execution. The server execution record below was added after the audit completed.**

## 7. Server execution record (2026-11-26/27)

The audit ran on branch commit `3749b2643d9a010071b103a0e90cf88cdd0dc833` using the existing PMME environment on the server. No packages or data were downloaded or installed. The pinned checkpoint was the original F2 Stage-I epoch 2899 checkpoint; SHA256 matched `f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf`.

| Run | Probes | OT points | Status |
| --- | ---: | --- | --- |
| Smoke | 1 | capped 256 source / 128 target | `COMPLETED` |
| Pilot | 8 | capped 256 source / 128 target | `COMPLETED` |
| Full OT | 8 | uncapped (`0` means full geometry) | `COMPLETED` |

All three run directories contain `COMPLETED`. The audit made **0 training updates** and ran **0 Stage-II evaluations**. The target is PEMS-BAY; source cities are Chengdu, METR-LA and Shenzhen. Probe results below are paired one-step shadow-update gains on correlated, previously seen target-support windows. They are not independent test metrics or paper-comparable Stage-II MAE.

### Full-OT paired shadow gains

Positive gain means the listed policy had lower support-probe MAE than the original F2 shadow update. The screening threshold for median gain is `max(0.00005, 10 × identity replay noise)`; observed identity noise was zero. “First/last 4” are the means of batches 0–3 / 4–7.

| Policy | Mean gain | Median | Positive | First / last 4 mean | Max absolute share | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `all_off` | +0.000404656 | +0.000469089 | 6/8 | +0.000686586 / +0.000122726 | 0.253 | Pass |
| `global_quarter` | +0.000206351 | +0.000325799 | 7/8 | +0.000374019 / +0.000038683 | 0.248 | Pass |
| `global_half` | +0.000223964 | +0.000209451 | 8/8 | +0.000242889 / +0.000205040 | 0.184 | Pass |
| `city_off:shenzhen` | +0.000171036 | +0.000178337 | 7/8 | +0.000177383 / +0.000164688 | 0.237 | Pass |
| `city_half:shenzhen` | +0.000049531 | +0.000075459 | 7/8 | +0.000080764 / +0.000018299 | 0.302 | Pass; median clears threshold |
| `city_off:chengdu_m` | +0.000033855 | +0.000103831 | 7/8 | +0.000180483 / −0.000112772 | 0.429 | Fail; last-half mean negative |
| `city_off:metr-la` | −0.000023007 | +0.000088096 | 7/8 | +0.000136673 / −0.000182688 | 0.548 | Fail; last-half mean and share |
| `city_half:chengdu_m` | −0.000142336 | +0.000059366 | 6/8 | −0.000048816 / −0.000235856 | 0.570 | Fail; sign persistence and share |
| `city_half:metr-la` | −0.000152349 | +0.000015378 | 5/8 | +0.000051439 / −0.000356138 | 0.796 | Fail |

### Gradient and optimizer trace

Across the 8 full-OT probes, the original OT gradient’s cosine with target-fit gradient averaged `−0.3726` (range `−0.4498` to `−0.2970`). Its norm averaged `0.02913`, versus `5.12480` for the source forecasting gradient. One-sided PCGrad left the OT component norm effectively unchanged (mean post/pre ratio approximately 1.0). The AdamW parameter-step difference between F2 and no-OT had mean norm `0.002154`; its cosine with target-fit descent averaged `−0.02944` (all 8 were negative). This indicates a locally conflicting direction after the optimizer shadow update, but the direction effect is small and this probe does not establish multi-epoch utility.

### Interpretation

The full-OT screen supports a local one-step benefit for global OT removal/scaling and Shenzhen removal/scaling under the prespecified probe rule. Chengdu and METR-LA policies are inconsistent or fail the rule, so there is no evidence for a uniform city-level controller. This is mechanism-screening evidence only. It does **not** show that removing OT improves trained model performance; the separate Stage-II T0/T1/T2 results remain the relevant multi-epoch comparison.

Successful raw run logs are stored under `experiment_logs/experiment-pmme-h6-ot-gradient-effectiveness-20261126/{smoke,pilot_8,full_8}`. Each folder includes the result JSONL, metadata, summary CSV, gradient traces, `train.out`, `log.txt`, run config and completion marker. No failed-run logs were included.
