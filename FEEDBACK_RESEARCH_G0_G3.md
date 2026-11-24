# F2 → G0–G3: target-probe feedback for OT weight decisions
Updated: 2026-11-24

**Scope:** first **exploratory** feedback mechanism, not a claim of a unique
paper contribution. Preserves official SOFTS/iTransformer, official
`train_DA_OT.py` stage-I sampling/checkpoint selection, Sinkhorn SamplesLoss
parameters, one-sided gradient projection and `PMME_2ndStage.py`.

## A. Motivating empirical evidence (PEMS-BAY/SOFTS seed1, scale 0.01)

F1 Uniform MAE (10/60/120/180min): 1.37072 / 2.31568 / 2.69209 / 2.93426.
F2 Gradient: 1.37044 / 2.30205 / 2.67932 / 2.92548.
F3 3-band Horizon: 1.37638 / 2.30512 / 2.68591 / 2.93000.
F4 Reliable: 1.39060 / 2.31761 / 2.69094 / 2.95848.
Only one seed: these are exploratory observations, NOT statistical results.
F3 had higher measured shared-layer OT/target gradient cosine than F2 but
worse prediction; a cosine is not itself an estimate of forecast gain.
F4's ESS>=50% shrinkage strongly diluted candidate weighting.

## B. Targeted literature search / similarity audit

Papers were checked by publisher/author abstract, not by reproducing their
full proofs. Related ideas are **not new**:

1. **Ren et al. (ICML 2018)**, *Learning to Reweight Examples for Robust Deep
   Learning*, https://proceedings.mlr.press/v80/ren18a.html:
   one-step meta gradient of clean reference/validation loss for sample
   reweighting. G2's feedback notion has clear precedent.
2. **Koç et al. (ICML 2026)**, *Incorporating Importance Weighting in Optimal
   Transport Based Domain Alignment*,
   https://proceedings.mlr.press/v306/koc26a.html:
   importance-weighted OT for domain alignment and its unbalanced-OT limit.
   OT sample reweighting is not, by itself, a contribution.
3. **Zhang et al. (AAAI 2025)**, *Drawing Informative Gradients from
   Sources: A One-Stage Transfer Learning Framework for Cross-City
   Spatiotemporal Forecasting*,
   https://mlanthology.org/aaai/2025/zhang2025aaai-drawing/:
   target-guided source gradients already explored in EXACT application
   domain; F2's gradient similarity alone is not new.
4. **Amos et al. (ICML 2023)**, *Meta Optimal Transport*,
   https://proceedings.mlr.press/v202/amos23a.html:
   amortized OT map prediction; *different* problem from this fixed-Sinkhorn
   target-loss feedback. Similar title does not imply method identity.
5. **Naram et al. (ICML 2025)**, *Theoretical Performance Guarantees for
   Partial Domain Adaptation via Partial Optimal Transport*,
   https://proceedings.mlr.press/v267/naram25a.html:
   selective/partial mass transfer as a principled related direction.
6. **Grangier et al. (TMLR 2024)**, bilevel data weighting for distribution
   adaptation (author PDF:
   https://david.grangier.info/papers/2024/grangier_ablin_hannun_bilevel_data_weighting_tmlr_2024.pdf):
   predicts which training gradients help target distribution; close to
   G1/G2 and important when evaluating novelty.
7. **Bi-level unbalanced optimal transport for partial domain adaptation**
   (Pattern Recognition, 2026),
   https://www.sciencedirect.com/science/article/pii/S0031320325016619:
   another example of bilevel OT weight estimation, but in a different
   domain/label setting.
8. **Label-guided optimal transport for domain adaptation regression**
   (Pattern Recognition, Nov 2026),
   https://www.sciencedirect.com/science/article/pii/S003132032600662X:
   label-guided transfer cost for regression already investigated;
   availability/relation to this architecture requires more detailed reading.

**Novelty status:** do not claim novelty for meta-weights, OT marginal
selection, gradient cosine, or short/mid/long risk aggregation. The current
work tests whether a low-compute **shared-head local OT counterfactual**
is more informative than F2 cosine for few-shot cross-city forecasting.
If helpful across seeds, NEXT identify a distinct formal objective or
temporal generalization mechanism and conduct further exact-match searches.

## C. Exact four-step ablation

| Experiment | Added mechanism | Changes actual Stage-I OT? |
|---|---|---|
| G0 | Exact F2 gradient marginals, scale 0.01 | Baseline F2 |
| G1 | G0 + candidate local OT probes/logs every 200 steps | NO |
| G2 | G1 + choose blend rho in {0,0.5,1}, or no-OT if harmful | YES |
| G3 | G2 + penalize negative virtual gain in any 12-step band | YES |

### Local counterfactual algorithm

For each of 3 source cities, on the first OT-enabled epoch and then every
200 epochs:

1. Reuse **F2's** full-source marginal `alpha_f2`. For three candidates
   `alpha(rho)=(1-rho)*uniform+rho*alpha_f2`, rho=0/0.5/1.
2. Sample at most 256 source and 128 target node/window pairs *evenly*
   across flattened batches. Exclude every fifth window of the 15-window
   target blocks from the **proxy OT**. Score with the unchanged official
   `SamplesLoss(loss="sinkhorn",p=1,backend="tensorized",debias=False,blur=.001)`
   and identical normalized future scale. This is a SUBSET proxy, not a full
   3-city OT recomputation.
3. Obtain the candidate OT gradients wrt source and target projected
   representations. Use the actual `projection1` inputs to analytically
   recover the **shared projection1.weight** gradient while keeping the
   costly encoder frozen: `gW = grad_rep_source.T @ pre_source +
   grad_rep_target.T @ pre_target`. NO full-backbone per-item backward.
4. Compute approximate marginal change in PMME's one-sided projected task
   gradient after adding this candidate OT gradient. For candidate scoring,
   calculate F2 source weights and the target projection direction using
   ONLY the non-probe target windows (the actual G0/F2 training weights
   still follow the original full-support route). Apply a bounded virtual
   change **only to projection1.weight**, not to actual model weights. The
   effect on `projection2` and the final *unnormalized physical predictions*
   is linear and exactly evaluated with cached encoder activations.
5. Compare probe Huber loss before/after this virtual head-only update on
   every fifth target support window. The counterfactual proxy's OT target
   subset and its probes are disjoint within the batch, but the whole
   three-day target support is still part of original PMME training; **not
   an independent validation set**. Forecast horizons use 12/12/12 split.
6. G1 prints results and uses F2 unchanged. G2 chooses the best overall
   positive virtual gain, falling back to no OT for that source city if
   all candidate gains <=1e-8. G3 instead scores `overall_gain -
   risk_penalty * max(0,-min(band_gains))`. This source-city action
   persists until the next 200-epoch probe.
7. All computations are diagnostics/selection only: no optimizer changes,
   second-order gradients, learned controller, new loss, or additional
   trainable parameters.

### IMPORTANT scope/causality limitations

- A local **projection1-only virtual SGD perturbation** is not the real
  AdamW + all-layer PCGrad update. It is a surrogate, with a max-norm cap;
  the score cannot be described as actual full-model forecast gain.
- Only source-city decisions are optimized, not individual OT entries.
  The final full Sinkhorn on every training iteration is unchanged.
- G1 is designed to be training-equivalent to G0 (modulo computational cost
  and any underlying nondeterministic kernels). Confirm identical checkpoint
  and Stage-II outcomes in a same-seed run.
- The original model already uses all 45 target support windows for
  supervised target projection. The G1/G2 local probe is separated from
  **candidate OT estimation, candidate F2 weighting, and proxy target
  projection gradient**, but NOT from all model training. Windows may
  have overlapping forecast intervals even when distinct in this fixed
  index split, and support labels were seen by earlier model updates.
  This is NOT a truly held-out chronological validation set.
- Source-city feedback is keyed by the distinct source sensor counts,
  **not the per-epoch random source-city order**. A policy persists for
  the intended city across randomly ordered source mini-batches.
- Selecting from three candidate actions plus no-OT is a discrete,
  interval-level procedure; G2 is NOT a differentiable full bilevel solver.
- The candidate OT sample and virtual readout are approximations; the best
  proxy action may not give the best actual target outcome. This is what
  G1-G3 test, not something assumed true.

## D. Controlled execution

Run from repository root with original data and Python dependencies:

```bash
git fetch origin
git switch experiment/pmme-f2-feedback-20261124
python -m unittest discover -s tests -p 'test_feedback_utility.py'
python -m unittest discover -s tests -p 'test_transfer_utility.py'
PMME_SEEDS="1" bash run_feedback_experiments.sh
# After initial checks:
PMME_SEEDS="3 5" bash run_feedback_experiments.sh
```

To select variants:
`PMME_MODES="g0 g1" PMME_SEEDS="1" bash run_feedback_experiments.sh`

Each 3000+6000 epoch run has an isolated working directory:
`experiments/f2-feedback/<mode>/scale0p01/seed<seed>/train.out`.
The runner checks the full Stage-II ending marker and stops on errors.

Common fixed settings: SOFTS, PEMS-BAY target 3 days, source
Chengdu/Shenzhen/METR-LA, DAcoef=1, DAepoch=1500, xyR=100,
projection=1, MemSize=10000, 0.01 fixed OT scaling, 8.0 F2 temperature.
Default G1-G3 interval 200 steps; sample cap 256/128; virtual step 0.05,
max-step norm 0.02; horizon penalty=1.0.

### What to inspect in logs

- `PMME_STAGE1_FEEDBACK` (mode, interval, cap).
- `transfer_feedback` (source city, epoch, candidate rho, virtual gains,
  band gains, selected action).
- `task_utility_gradient_diagnostic` (real full-OT gradient ratio and
  cosine at projection1, from existing F1-F4 instrumentation).
- G0 vs G1 must have same predictions; unexpected differences are an
  implementation red flag.
- G2 vs G0 and G3 vs G2: MAE/RMSE at 10, 60, 120, 180 min.
  REPORT the author-selected best checkpoint by target support score, not
  cherry-picked best test epoch. Three seeds needed to discuss stability.

**Runtime status:** GitHub code and CPU tests committed; a complete
GeomLoss+GPU+dataset Stage-I/II server run has NOT been executed or claimed.
