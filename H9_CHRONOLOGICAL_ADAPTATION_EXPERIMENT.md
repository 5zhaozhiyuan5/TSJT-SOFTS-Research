# H9 — Chronological Development Validation and Multi-step F2 Clone Adaptation

**Branch:** `experiment/pmme-h9-chronological-holdout-adaptation-20261127`  
**Parent:** completed H8 results at `c565b8b8618e2bdc74e9ca39302f58aaf6712aec`.

> **IMPORTANT STATUS:** Code submitted, but GPU runs and integration tests on the actual PMME server have NOT been executed yet. No H9 performance claims.

## 1. Why H9 is needed

H6/H7/H8 showed locally favorable OT-gradient projection after a cloned one-step AdamW update, but all 8 probes were **within the previously seen 3-day target support** and correlated. H8 full OT `projected_half` exceeded an equally norm-scaled raw OT alternative in 8/8 one-step support probes, with a mean local MAE gain `+0.00008231`. The actual final Stage-II 36-step test MAE has **not** been improved by this evidence.

H9 asks whether the same mechanism works beyond that checkpoint-seen period, and whether any advantage persists across multiple **actual optimizer steps performed on independent model copies**.

## 2. Strict chronological development window, not the paper's official test

Original data loader `datasets.py` uses first `3 × 288 = 864` time steps for target support. Its `stage='test'` loads **all** later time steps. Using that test dataset to choose a new policy would contaminate the reported paper benchmark. H9 deliberately introduces a separately declared **early post-support DEVELOPMENT slice**, and prohibits reading later targets into dev batches:

```text
Raw chronological target time (5min per time step):
  [0, 864)           = original 3-day target SUPPORT (F2 checkpoint saw it)
  [864, 2016)        = H9 DEVELOPMENT only: next four days [day4,day8)
  [2016, end)        = RESERVED FUTURE / never used by H9

  DEV windows (H=288, F=36, stride=12):
  earliest history [864,1152)  forecast [1152,1188)
  latest history  [1692,1980) forecast [1980,2016)

  70 chronologically indexed DEV windows with default `validation-max-windows=0`
  (windows overlap each other; they are NOT 70 independent observations).
```

- Every dev input history and forecast label lies entirely inside `[864,2016)`.
- No DEV label enters adaptation gradients, OT source/target alpha, Sinkhorn cost, optimizer updates, adaptive policy choice, or model forward: model receives `y=zeros`; labels are passed separately to metrics.
- After dev, `[2016,end)` remains reserved and uninspected by this experiment; *do not* quietly evaluate it to tune policies.
- **Crucial historical exposure:** The original `train_DA_OT.py` (around lines 162-183) called `rep_model.evaluation(finetune_dataset, test_dataset,...)` at Stage-I checkpoint intervals. This reported post-support **testing_score** in original training logs, although the checkpoint save decision used the **support validation_score**, not the testing_score. Consequently H9's post-support development time is OUT-OF-SUPPORT FOR GRADIENT UPDATES, but is **not a never-before-inspected independent research holdout**. Avoid claims of entirely fresh testing or formal statistical generalization.
- **Important baseline limitation:** the original `traffic_dataset` computed its normalization mean/std using the **entire target city series** before truncating to 3 support days. H9 must retain this inherited scaler for consistency with the already-trained original F2 checkpoint. Thus the split is chronologically disjoint in X/Y and labels, **but NOT strictly train-only normalized**.
- By allocating this early post-support period to development, H9 changes what can count as untouched test. The remaining time after time index 2016 can be a future held-out reporting split, but its metrics will NOT be numerically comparable to the original paper full post-support test split without an explicitly standardized rerun.

## 3. Same pinned F2 model, no re-training

Required existing original F2 Stage-I checkpoint:
```text
/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth
sha256=f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf
epoch=2899, target PEMS-BAY, sources Chengdu/METR-LA/Shenzhen
SOFTS H=288, F=36
```

The runner REFUSES missing/mismatched checkpoints. Neither `train_DA_OT.py`, original `PMME.py`, original SOFTS, original `datasets.py`, nor official PMME Stage-II is modified.

H9 deep-copies the original PatchFSL model and **the checkpoint's AdamW optimizer state** separately for each of the 5 policies. Updates happen ONLY to these independent ephemeral clones, and the original F2 weights and AdamW state are verified unchanged after every run. No adapted checkpoints overwrite the original.

## 4. Adaptation mechanisms tested (recompute all quantities every step)

For current policy model parameters `theta^(k)`, recompute its own source task gradient `g_s^(k)`, target-FIT gradient `g_t^(k)`, and **true GeomLoss OT gradient** `g_OT^(k)` at every adaptation step. Sinkhorn is original F2 `GeomLoss(p=1,blur=.001,debias=False)`; source alpha = F2 original temperature-8 gradient weight, target beta unchanged, cost `concat(SOFTS_rep, 10×normalized_future)`, coefficient .01.

**DO NOT freeze/replay a Stage-I gradient vector for multiple steps.** The source predictions, F2 OT weights, parameter gradients, PCGrad direction and AdamW moments change during adaptation.

```text
P(g,t) = g - min(0,<g,t>)/(||t||²+1e-8)*t

policy f2:
  g_policy = g_OT
policy global_half:
  g_policy = 0.5*g_OT
policy projected_half:      [H8 winner]
  g_policy = 0.5*P(g_OT,g_t)
policy norm_matched_raw:
  g_policy = 0.5*(||P(g_OT,g_t)||/||g_OT||)*g_OT
policy all_off:
  g_policy = 0

g_final = OriginalOneSidePCGrad(g_s + g_policy, g_t)
AdamW_cloned.step(g_final)
```

- The FIT mask is the same as H8: 36 of 46 target support windows; the remaining 10 are not used in adaptation, and **never used for model selection**.
- Every policy's update k receives the **same source-city batch/window fingerprints**, generated from the same deterministic seed, irrespective of previous policy evaluations.
- The SOFTS train-forward stochastic seed is fixed at each step across policies.
- No evaluation gradients are computed, and all dev scores are **computed only after the fixed optimizer update**.
- For memory conservation, the five policies are processed **sequentially**; all start from identical checkpoint weights/moments.
- If raw OT point caps are used (e.g. 256/128), results are **capped/approximate** and should not be conflated with full GeomLoss OT. For true full geometry, use source_points=0 and target_points=0.

### H9-A / H9-B

H9-A sets `--max-updates 1` and compares dev MAE/RMSE at 0 versus 1 update.

H9-B sets `--max-updates 5`, `10`, or `20` and evaluates the exact same per-policy trajectories at **0,1,5,10,20** as applicable, recording overall dev MAE/RMSE, 10/60/120/180-minute MAE, short/mid/long horizon MAE, and *dev forecast-day* metrics. **H9-B is not Stage-II; it performs cloned Stage-I-like adaptation updates only.**

## 5. Execution

```bash
git fetch origin
git switch experiment/pmme-h9-chronological-holdout-adaptation-20261127

# CPU-only safety tests (no dataset, geomloss, checkpoint or GPU necessary)
python -m unittest discover -s tests -p 'test_h9_chronological.py'

# H9-A smoke: 1 update, 5 policies, capped OT, 16 fixed DEV windows
PMME_H9_MAX_UPDATES=1 \
PMME_H9_SOURCE_POINTS=256 PMME_H9_TARGET_POINTS=128 \
PMME_H9_DEV_MAX_WINDOWS=16 \
PMME_H9_RUN_ROOT=./experiments/h9-smoke \
bash run_h9_chronological.sh

# H9-A real full OT: 1 update, ALL 70 DEV windows
PMME_H9_MAX_UPDATES=1 \
PMME_H9_SOURCE_POINTS=0 PMME_H9_TARGET_POINTS=0 \
PMME_H9_DEV_MAX_WINDOWS=0 \
PMME_H9_RUN_ROOT=./experiments/h9-a-full-1 \
bash run_h9_chronological.sh

# H9-B early viability check: cap OT budget, 5 updates
PMME_H9_MAX_UPDATES=5 \
PMME_H9_SOURCE_POINTS=256 PMME_H9_TARGET_POINTS=128 \
PMME_H9_DEV_MAX_WINDOWS=0 \
PMME_H9_RUN_ROOT=./experiments/h9-b-pilot-5 \
bash run_h9_chronological.sh

# H9-B full OT 20-update validation only if first steps are sensible;
# may require substantial GPU memory/time (5 policies * 20 real OT updates).
PMME_H9_MAX_UPDATES=20 \
PMME_H9_SOURCE_POINTS=0 PMME_H9_TARGET_POINTS=0 \
PMME_H9_DEV_MAX_WINDOWS=0 \
PMME_H9_RUN_ROOT=./experiments/h9-b-full-20 \
bash run_h9_chronological.sh
```

The `run_config.txt` includes branch, git SHA, checkpoint SHA, all budgets and chronological validation setup and refuses reusing the directory with different flags/commits. H9-A and H9-B never need a 3000-epoch F2 retrain.

### Outputs

```text
h9_results.json                      # all policy, day/horizon/milestone scores,
                                     # each step OT stats, source batch hashes
h9_results.metadata.json             # fixed checkpoint, slicing and caveats
h9_policy_development.csv             # per policy and 0/1/5/10/20 update
h9_daywise_development.csv            # per dev forecast-day, per policy/update
h9_decision.json                      # explicit comparisons, no test claim
train.out                             # complete adaptation logs
summary.txt
run_config.txt
COMPLETED / FAILED
```

## 6. Decision rule

The key falsifiable question is:

```text
MAE_dev(norm_matched_raw) - MAE_dev(projected_half) > 0?
MAE_dev(global_half)      - MAE_dev(projected_half) > 0?
```

- Positive at H9-A one step **on post-support time** is a first sign of transfer beyond checkpoint-support overfitting.
- For H9-B, persistence at 5/10/20 steps, similar behavior across 3 forecast days and short/mid/long horizons, and no severe regression is more important than a tiny one-step benefit.
- No significance testing from 70 overlapping rolling windows; analyze separately by dev forecast-day and repeat on new seed/city **only after selecting mechanism**.
- If projection advantage disappears on development or reverses at 5 steps, STOP and revisit OT objective; do not train Stage-II from an unsupported mechanism.
- If benefits persist, develop a **training-time** OT gradient calibration module and run an official Stage-I+Stage-II evaluation using a newly declared, untouched test split (not the consumed dev period). A simple one-sided PCGrad projection is not, by itself, a novel algorithm.

## 7. Files added

1. `model/Meta_Models/h9_temporal_holdout.py`
2. `model/Meta_Models/h9_checkpoint_adaptation.py`
3. `audit_h9_chronological_adaptation.py`
4. `summarize_h9_chronological.py`
5. `run_h9_chronological.sh`
6. `tests/test_h9_chronological.py`
7. `H9_CHRONOLOGICAL_ADAPTATION_EXPERIMENT.md`

**Current status: implementation only; no H9 GPU results claimed.**
