# F2 + DPM-OT — source-pattern memory attention guides the OT pair cost

Branch: experiment/pmme-f2-dpm-pattern-attention-ot-20261127

This is a REAL new Stage-I module, not another H6-H11 optimizer audit.

**Original F2 retained**: projection2 gradient similarity produces its unchanged
alpha_F2; target beta stays uniform. SOFTS, PCGrad, OT coefficient, original
GeomLoss Sinkhorn and official Stage-II remain in place. DPM adds a new
attention-conditioned cost to the original OT operator.

## 1. Actual F2 -> DPM -> Sinkhorn data flow

- Original SOFTS source/target representations and current forecasts.
- Original F2 compute_weights(..., "gradient") -> alpha_F2 and uniform beta.
- Original PMME OT points: x_i=[rep_source_i, 10*future_revIn_normalized_i]
  and y_j=[rep_target_j, 10*future_revIn_normalized_j].
- Original cost: C_F2(i,j)=Euclidean_distance(x_i,y_j). This is important:
  F2 ALREADY sees future support labels in the original cost!
- New DPM bank pattern matching returns a *pair-dependent* additional cost.
- Sinkhorn is now called with (alpha_F2,x,beta_uniform,y) and its new
  cost callback; marginals remain bitwise unchanged by DPM.

## 2. Frozen source training pattern bank

Each sample has three 12-step forecast bands h in {short,mid,long}.
The existing TDSM descriptors give each band a 3D evolution vector:
(history->future relative change, future local slope, future volatility).

Before DAepoch, sample <=128 descriptors from each of three source cities
every 50 training epochs (not merely current OT batch). Source cities have
equal observation budgets. No target examples enter construction.

At DAepoch, robustly normalize each band using the SOURCE reservoir's
median/IQR, fit 32 K-means centers for each band, and FREEZE. Bank saved at
logged_files/.../dpm_source_evolution_bank.pt on the experiment server.
No additional learnable predictor head or second-stage changes.

## 3. Per-source/target Attention matching (same source-trained bank)

For each band h, source or target SUPPORT sample query r and centers M:
  A[i,h,k] = softmax_k(-distance(norm(r[i,h]), M[h,k])/tau)
where tau defaults to 0.75.

Pattern affinity for source i and target j:
  S_ij^h = sum_k sqrt(A_source[i,h,k] * A_target[j,h,k])
This is the Bhattacharyya coefficient in [0,1].

Current source training forecasts/labels give q_i^h (skill relative to
persistence), and target three-day SUPPORT forecasts/labels give d_j^h
(target need). Normalize each separately per horizon within its domain:
  qrel_i^h = clip(q_i^h / mean_source(q^h), 0, 2)
  drel_j^h = clip(d_j^h / mean_target(d^h), 0, 2)

Joint matching benefit (in [0,4]):
  B_ij = (1/3) sum_h qrel_i^h * drel_j^h * S_ij^h

The attention, skill and need are detached: they guide Sinkhorn pairing.
The SOFTS representation in the original cost remains differentiable.

## 4. EXACT way Attention changes OT

  C_new(i,j) = C_F2(i,j) + lambda * C_ref * (1 - B_ij/4)

lambda defaults to 0.02; C_ref is a DETACHED median of sampled original
pairwise costs. Pattern-similar, reliable source / high-need target pairs
receive LOWER additional penalties.

This is pair-specific, not a sum of source-only and target-only offsets.
Row-only/column-only offsets cannot change an optimal coupling with fixed
balanced OT marginals, so DPM deliberately uses the product q*d*S.

Final operator:
  Pi_new = Sinkhorn(alpha_F2, beta_uniform, C_new)
  L_OT_new = original_pmme_ot_coefficient * <Pi_new,C_new> + entropy term

Actual GeomLoss callback computes the base distance from its xx/yy
inputs, so original SOFTS gradients still flow. It computes attention
costs in row blocks; this does not remove the baseline tensorized
full N_source*N_target OT matrix requirement.

No validation/test labels enter the pattern bank or OT. Target future
labels are used only in the few-shot Stage-I SUPPORT, as they already
are in the original PMME OT.

## 5. Experiment files

- model/Meta_Models/dpm_pattern_attention_ot.py
  Persistent source bank, frozen attention, pairwise cost callback
- model/Meta_Models/PMME.py
  Opt-in Stage-I integration only, with exact default-off F2 path
- tests/test_dpm_pattern_attention_ot.py
  CPU unit checks for attention, source-only bank, pair preference,
  source skill/target need, gradient and zero-strength F2 fallback
- run_f2_dpm_pattern_attention_ot.sh
  One seed1 new-module run; does not repeat old F2 by default
- summarize_dpm_pattern_attention_ot.py
  Support-validation selected Stage-I / official Stage-II metrics

## 6. Execute on the existing PMME server

~~~bash
git fetch origin
git switch experiment/pmme-f2-dpm-pattern-attention-ot-20261127

python -m unittest discover -s tests -p 'test_dpm_pattern_attention_ot.py'

# Smoke only: 0th epoch collect source bank, first active OT at epoch 1.
PMME_DPM_STAGE1_EPOCHS=3 PMME_DPM_STAGE2_EPOCHS=0 \
PMME_DPM_DAEPOCH=1 PMME_DPM_RUN_ROOT=./experiments/dpm-smoke \
bash run_f2_dpm_pattern_attention_ot.sh

# Full novel module: Stage-I 3000 epochs, official Stage-II 6000 epochs
PMME_DPM_RUN_ROOT=./experiments/dpm-full-seed1 \
bash run_f2_dpm_pattern_attention_ot.sh
~~~

Defaults: DPM lambda=0.02, 32 centers/band, attention tau=0.75,
source bank sample interval=50 epochs, source samples/city=128,
cost row block=128, seed=1, F2 unchanged, OT coef=0.01.

Optional pattern-only ablation: PMME_DPM_MODE=pattern (requires a different
run root). Full conditional version: PMME_DPM_MODE=full (default).

## 7. Diagnostics and decision

- dpm_bank_fitted: 3 source-only banks, training sample counts, nonempty
  clusters and dominant-mode fraction; verify no class collapse.
- dpm_pair_cost: each source city, epoch, target group, attention top masses,
  pair bonus mean/std, reference cost, extra penalty size, no alpha/beta edits.
- f2_ot_schedule and task_utility: original F2 weights/OT scale.
- Support-selected MAE/RMSE over all 36 steps and 10/60/120/180 min.
- GeomLoss does NOT expose full Pi: do not mislabel Attention as transport.
  Compare small diagnostic OT subsets if true pair changes need direct audit.
- A real first-stage smoke and full training are required before accuracy
  claims. Original F2/T0 stage-II 36MAE 2.372713 and F2+TDSM 2.395861 are
  historical seed1 references FROM DIFFERENT CODE COMMITS, not paired controls.

Novelty is not established by code alone. Existing MTPB, TPB, attention-cost
OT papers, source-label OT and the prior PMME P2 cost variant require further
mechanism-level comparison before a publishable novelty claim.
