# F2 + DPM-OT V2: online pattern-prediction reliability and adaptive attention→OT

Branch: experiment/pmme-f2-dpm-v2-dynamic-reliability-ot-20261127

This is a direct modification to the previously successful V1 architecture.
**NO additional prediction model or trainable module** was added: original F2,
SOFTS, GeomLoss balanced Sinkhorn, PCGrad and original Stage-II are preserved.
Only the existing pattern bank's stored statistics, the existing Attention
matching function, and its conversion into GeomLoss pairwise cost are changed.

## What changes (and what does not)

Previous F2+V1:
- F2 source alpha = per-node-window projection2 gradient cosine with target
  aggregate gradient; target beta uniform; same OT coefficient 0.01.
- Source-only frozen bank: three horizon bands, K=32 robust K-means prototypes.
- V1 Attention gave pair bonus B, Cnew=C_F2 + lambda*C_ref*(1-B/4).

V2:
- Pattern CENTERS M remain **SOURCE-ONLY and frozen**.
- Each pattern per band gains Q (rolling source prediction skill), R (rolling
  source-target SUPPORT descriptor compatibility), plus observation mass.
  Q and R are statistics in the original bank, not extra model modules.
- Attention queries source and target samples against the SAME bank, as V1.
- Match value uses current source skill, stored Q/R and target need.
- V2A retains global V1-style penalty on that dynamic matching value.
- V2B uses candidate-local matching thresholds, distance margins and
  an absolute source-reliability gate to set original Sinkhorn ground cost.
- All Stage-I label-dependent statistics are detached; target labels are
  the THREE-DAY TRAINING SUPPORT only. NEVER validation/test future labels.
- DO NOT update Q or R inside the OT cost function. Update ONCE after the
  original PCGrad/optimizer step, aggregating feedback from all 3 cities.

## Original F2, unchanged

Given F2 sample-level gradient agreement s_i=cos(g_source_i,g_target),
alpha_F2=softmax(temperature*s_i). beta_target = 1 / N_target.
No per-source-target-pair backprop is introduced. The pattern module's
pairwise lookup uses feature distances and attention, not per-pair gradients.
The original differentiable C_F2 uses [SOFTS_rep, sqrt(xySquaredRatio)*Y].

## Source-trained pattern bank with online accuracy

Before DAepoch collect equal-budget source-only samples (pre-OT 90 snapshots,
11,520 patterns under the original 1500/50x128x3 configuration). Unlike V1,
capture both evolution descriptor r_i^h and source quality q_i^h.
Initialize pattern Q by source soft-membership-weighted q, R=1 initially.

For band h and pattern k:

  Q_observed[k,h] = sum_i A_source[i,h,k]*q_i[h] /
                    (sum_i A_source[i,h,k] + eps)

  rho[i,j,h] = exp(-||z_source[i,h]-z_target[j,h]||/tau_rho)

  R_observed[k,h] =
       sum_(i,j) A_source[i,h,k]*A_target[j,h,k]*rho[i,j,h] /
       (sum_(i,j) A_source[i,h,k]*A_target[j,h,k]+eps)

Evaluate R only for source samples paired with a small number of CLOSE
target SUPPORT candidates selected using the original F2 OT distance.
Across 3 cities, normalize each city's Q/R sufficient statistics BEFORE
combining them; prevent Shenzhen's larger node count from dominating.
Use EMA with per-pattern occupancy-gated update rate eta_k <= eta.
Pattern centers do not move. Record count, occupancy, Q/R per horizon
and per pattern at each logged update.

SOURCE quality is not the same as source-target matching accuracy:
Q measures how well SOFTS predicts SOURCE training samples;
R measures whether source-target evolution descriptors are compatible.
These are diagnostic proxies, NOT a demonstrated transfer benefit.

## Attention: reliability-conditioned without a new layer

A_s=A(source descriptor, source bank), A_t=A(target descriptor, source bank).
For source i, target j, horizon h and pattern k:

  q_mix[i,h,k] = xi*q_current_source[i,h]+(1-xi)*Q[h,k]
  U[i,h,k]     = sqrt(A_s[i,h,k])*q_mix[i,h,k]*R[h,k]
  V[j,h,k]     = sqrt(A_t[j,h,k])*target_need[j,h]

  B[i,j] = (1/3)*sum_h sum_k U[i,h,k]*V[j,h,k].

B is the same source-target pair-specific benefit. Q/R stored from the
PREVIOUS optimizer step; this step updates the bank AFTER all city OT
computations and the original PCGrad step.

Full cost uses B only; sparse nearest-candidate raw descriptor distances
are used to UPDATE R, avoiding 3 additional full Ns x Nt descriptor cdist
tensors and matching only source training/target SUPPORT pairs.

## The two mandatory ablations: V2-A vs V2-B

V2-A: same dynamic Q/R Attention B but preserve V1-style global fusion:

  C_V2A[i,j] = C_F2[i,j] + lambda_global*C_ref*(1-B[i,j]).

Use PMME_DPM_MODE=v2a PMME_DPM_STRENGTH=0.02.

V2-B: dynamic Q/R Attention B with local threshold from the closest
L original-F2-distance candidates among target-support anchor samples:

  theta_i = median_(j in N_i) B[i,j]
  s_i = max(IQR_(j in N_i) B[i,j],0.005)
  p_ij = sigmoid(clip((B[i,j]-theta_i)/s_i,-6,6))

  m_i = min(IQR_(j in N_i) C_F2[i,j],
            0.20*median_(j in N_i) C_F2[i,j])

The absolute confidence of source traffic pattern representation:

  T_i=0.2+0.8*sqrt(
       (1/3)sum_h,k A_source[i,h,k]*q_mix[i,h,k]*R[h,k])

  C_V2B[i,j] = C_F2[i,j]+lambda_local*m_i*T_i*(1-p_ij).

Use PMME_DPM_MODE=v2b PMME_DPM_STRENGTH=0.35.

T_i matters: without it, all R or Q values being scaled uniformly can
cancel out in (B-theta_i)/s_i. With T_i, an absolute degradation in
stored prediction reliability reduces the amount of OT-cost influence.

The cost callback handles both orientations cost(source,target) and
cost(target,source), as required by actual GeomLoss Sinkhorn. Everything
except the original base C_F2 feature distance is detached from autograd.

Exact source alpha_F2 and uniform target beta are passed unchanged to:

  OT = GeomLoss.SamplesLoss(
      loss="sinkhorn", p=1, backend="tensorized",
      debias=False, blur=.001, cost=C_V2)
  L_OT = original_DAcoef * original_OT_scale * OT(alpha_F2,x,beta,y).

The candidate-local matching advantage can change transport preferences;
this is a proposal to test, not a guarantee of MAE improvement.
No full NxM transport Pi is saved by the GeomLoss API.

## Detailed logs: one file per experiment, no arbitrary performance claims

All normal Stage-I/II logs remain; V2 adds in args.save_dir:

- dpm_v2_diagnostics.jsonl, with DPM_V2_DIAG and DPM_V2_BANK_UPDATE
  also PRINTED to train.out, so GitHub log syncing retains these even if
  the separately written JSONL file is not copied.
- dpm_source_evolution_bank.pt containing original source-only centers,
  medians, spreads plus per-pattern Q/R EMA values, masses, feedback count.
- DPM_V2_DIAG per 25 epochs and DAepoch:
  city, target group, source/target count, original F2 alpha/beta ESS,
  per-band Q/R mean/min/max, source/target Attention top mass, current q/d,
  pair B mean/std, local threshold and cost-margin distributions, trust
  gate, cost mean/std and sample nearest-neighbor ranking changes.
- At selected epochs, 8 source/target pattern-matching examples with
  flattened source/target indices, current source skill and target need.
- Selected 64x64 balanced Sinkhorn plans from identical coverage subset,
  SAME F2 subset marginals and an explicitly documented DIAGNOSTIC epsilon:
  Pi L1/Linf change, old/new pattern bonus mass, matched original cost,
  marginal-error and 6 most changed pair masses. This subset Pi is NOT
  the entire training GeomLoss coupling.
- DPM_V2_BANK_UPDATE per 25 epochs:
  before/observed/after Q/R by short/mid/long, L1 update sizes,
  Q/R min/max, complete 3x32 per-pattern Q/R arrays, per-pattern weighted
  Q/R observation counts, per-city descriptor distance/rho and alpha ESS.
- The last Stage-I epoch persists latest Q/R bank.
- Existing full checkpoint-selected MAE/RMSE and all-horizon metrics.

Use summarize_dpm_v2_diagnostics.py on the run root after syncing logs.
It exports CSVs of attention OT statistics and bank-update history, plus
per-city summary JSON. No additional training or network data needed.

## Quick start, actual server

~~~bash
git fetch origin
git switch experiment/pmme-f2-dpm-v2-dynamic-reliability-ot-20261127

# CPU regression V1 + V2, GeomLoss callback and syntax:
python -m unittest discover -s tests -p 'test_dpm_pattern_attention_ot.py' -v
python -m unittest discover -s tests -p 'test_dpm_v2.py' -v

# Do this FIRST to catch data/shape/GeomLoss mismatch.
PMME_DPM_MODE=v2b PMME_DPM_STAGE1_EPOCHS=3 \
PMME_DPM_STAGE2_EPOCHS=0 PMME_DPM_DAEPOCH=1 \
PMME_DPM_RUN_ROOT=./experiments/dpm-v2-smoke \
bash run_f2_dpm_v2_ab.sh

# Full V2-B, one seed 1, Stage-I 3000 and official Stage-II 6000:
PMME_DPM_MODE=v2b PMME_DPM_STRENGTH=0.35 \
PMME_DPM_RUN_ROOT=./experiments/dpm-v2-paired \
bash run_f2_dpm_v2_ab.sh

# Paired V2-A: SAME code commit, source cities/seed/learning schedule,
# diff only V2 local cost vs original global cost.
PMME_DPM_MODE=v2a PMME_DPM_STRENGTH=0.02 \
PMME_DPM_RUN_ROOT=./experiments/dpm-v2-paired \
bash run_f2_dpm_v2_ab.sh

# Analyze already synced logs without running anything:
python summarize_dpm_v2_diagnostics.py \
  --run-root ./experiments/dpm-v2-paired
~~~

The runner explicitly requires the new branch; writes COMMIT, seed, mode,
V2 knobs, run completion guards; rejects mismatched re-use of a run folder.

## Validity caveats

- First smoke must pass with real training data. CPU tests do not establish
  real full-Sinkhorn integration, memory viability or predictive improvement.
- Current target support is reused for computing its need d. To claim
  prediction benefit, use held-out support windows for evaluation/audits,
  or strict control checkpoint comparison, never validation/test future in
  the Stage-I matching computation.
- The frozen bank's stored source training Q can be optimistic due to
  source overfitting. Q/R EMA are heuristic proxies, not unbiased expected
  cross-domain benefit. Multi-seed experiments and V2-A/B comparisons
  with same-commit F2 are required.
- The new top-L distance search is O(Ns*anchor_count), not full NxM;
  GeomLoss tensorized OT STILL materializes full Ns x Nt costs.
- Prior literature uses pattern dictionaries, prediction-error reliability,
  label-guided OT and target data selection. No novelty guarantee or
  performance gain is claimed without evidence and a full literature check.
