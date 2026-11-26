# F2: four literature-derived experiment variants (2026-11-26)

## What this branch does

Four independent upgrades to the previously trained F2 source/target OT marginal estimator. No PFR; no new predictor; no P1/P2/P3/P4 pair-cost changes; no gradient surgery; no DVGP; no official Stage-II changes. Original GeomLoss cost and backbone remain unchanged.

Existing historical F2 result (not necessarily a config-matched comparison): PEMS-BAY, SOFTS, 60min MAE=2.30205 and 180min MAE=2.92548. No baseline retrain is scheduled. All four new methods require real experiment results before effectiveness claims.

## Four mutually exclusive methods

G1, method=mag, SAM-GS (ECML PKDD 2025) inspiration.
  The per-sample readout gradient norm is ||delta_i||*||feature_i||. Due to different loss reductions, compare source/target within-domain log-median-centered norms, not raw cross-domain magnitudes. Use the target norm interquartile interval to compute a penalty for an outlying source norm; reweight original F2 alpha by exp(-penalty). Target beta stays uniform. This is a magnitude mismatch diagnostic, NOT a proof of negative transfer.

G2, method=sign, GradDrop (NeurIPS 2020) inspiration.
  Pooled target weight-gradient G=Sum_j outer(delta_tj,feat_tj).
  Each source has numerator p=<g_i,G>, denominator a=Sum_k |g_i,k*G_k|.
  Conflict fraction d=clip((1-p/(a+eps))/2,0,1).
  Reweight original F2 alpha by exp(-1.5*d). No masking or projection. The rank-one gradient factorization avoids dense [num_source,4608] storage. Beta stays uniform.

G3, method=ema, GradVac (ICLR 2021) inspiration.
  Maintain a detached EMA of the normalized pooled target gradient per target-support group. Blend the current target gradient with the prior EMA, re-score source readout gradients against the blended direction. EMA momentum=0.85, scoring blend=0.40. Beta stays uniform. This is NOT the original GradVac gradient-update rule. It differs from earlier F4's within-batch split-half reliability.

G4, method=biot, importance-weighted / unbalanced OT (ICML 2026) inspiration.
  Partition source and target sample features into <=32 coarse groups each using deterministic feature anchors. Build group mean features and exact prior-weighted mean readout gradients. Form internal group-to-group compatibility cost = normalized representation distance + 0.5*(1 - gradient cosine). Run 35 generalized Sinkhorn iterations with relaxed source and target group masses. Use row/col mass to reweight original F2 alpha and uniform target beta, retaining 60% of original priors. Thus beta_j >= 0.60/N_target (up to float tolerance). This internal GROUP coupling is not the final PMME OT transport plan; unchanged PMME Sinkhorn uses the output marginals and its original cost.

## Running these experiments

First switch to the separate branch:
  git fetch origin
  git switch experiment/pmme-f2-gradient-marginal-four-20261126

CPU unit checks (require torch, no traffic data):
  python -m unittest discover -s tests -p 'test_f2_four_extensions.py'

Real-forward/backward two-epoch smoke (ONE active OT epoch; NOT scientific results):
  PMME_F2_STAGE1_EPOCHS=2 PMME_F2_STAGE2_EPOCHS=0 PMME_F2_DAEPOCH=1 PMME_F2_RUN_ROOT=./experiments/f2-four-smoke bash run_f2_four_methods.sh

Four full GPU experiments with seed1:
  PMME_F2_METHODS="mag sign ema biot" PMME_F2_SEEDS="1" bash run_f2_four_methods.sh

Later, only for shortlisted methods, seed3/5 robustness:
  PMME_F2_METHODS="sign biot" PMME_F2_SEEDS="3 5" PMME_F2_RUN_ROOT=./experiments/f2-four-seeds35 bash run_f2_four_methods.sh

Defaults: source batch=16; target batch=46 split to 15/15/16; Stage-I 3000 epochs, OT begins epoch 1500; original Stage-II 6000; OT scale=0.01; seed1; sources Chengdu, Shenzhen, METR-LA; target PEMS-BAY 3 days.

Logs:
  experiments/f2-four-marginal-20261126/{mag,sign,ema,biot}/seed1/train.out
  experiments/f2-four-marginal-20261126/{mag,sign,ema,biot}/seed1/run_config.txt

Training logs contain original task_utility diagnostics PLUS f2_extension diagnostics every 25 epochs: source/target ESS, L1 changes from F2, and method-specific gradient/coupling diagnostics.

## Acceptance rules

Use best target-support validation checkpoint. Report target test MAE/RMSE at 10/60/120/180 minutes and average over all 36 steps, Stage-I and Stage-II separately. G1/G2/G3 must leave beta uniform; G4 must keep nonzero mass coverage. Compare training stability, ESS collapse, source gradient conflict, EMA direction disagreement, target mass concentration and compute overhead. Follow up promising methods with seeds 3 and 5 before treating them as repeatable results. Do not use target test errors for hyperparameter or checkpoint selection.

The old F2 result was produced in a potentially different code/batch/GeomLoss configuration. Treat it as a historical reference until matched-setting comparability is verified. Do not falsely declare a new method beats F2 on the basis of mismatched configs. The original PMME final transport plan Pi is not exposed by GeomLoss in this implementation; the G4 32x32 group coupling should NOT be described as Pi.
