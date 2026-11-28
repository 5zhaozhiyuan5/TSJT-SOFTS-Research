# Three Independent F2 Gradient Refinement Experiments

## Experimental rule

This branch contains **three separate alternatives**, each added independently
to the existing F2 source-marginal mechanism:

1. F2 + geometric-median target reference (`geomed`);
2. F2 + Fisher-preconditioned influence utility (`influence`);
3. F2 + lower-tail CVaR risk calibration (`cvar`).

They are **not combined**. No `triple` method is implemented or run.

The already completed original F2 result is the control group. The experiment
script deliberately does not accept `original` and never retrains that
baseline. Each of the three new variants still needs its own complete training
run, because each changes the source marginal used by OT during training.

## Shared interface and preserved components

- Source sample gradients are constructed from the existing SOFTS `projection2`
  output deltas and input features.
- The three variants each output a source marginal `alpha_new`.
- The original target marginal `beta` is returned unchanged.
- The original OT cost matrix and Sinkhorn implementation are unchanged.
- SOFTS backbone, prediction horizon, Stage-II implementation, and training
  data protocol are unchanged.
- No prediction-horizon split, pattern bank, new prediction head, or extra
  trainable network is introduced.

For each target window (b), sensors are aggregated into one readout gradient
(G_b^t\in\mathbb R^{36\times D}). Sensors within a window are not treated
as independent target windows.

## Experiment 1 — F2 + geometric median (`geomed`)

**Research source:** robust statistics and robust aggregation in federated
learning, e.g. Pillutla et al., *Robust Aggregation for Federated Learning*,
IEEE Transactions on Signal Processing, 2022.

The original arithmetic mean target reference is replaced by an approximate
geometric median:

[
G_{GM}=\arg\min_G\sum_b\|G-G_b^t\|_F.
]

The source sample score remains ordinary cosine similarity:

[
s_i^{GM}=\cos(g_i^s,G_{GM}).
]

Then:

[
\alpha_i^{GM}=\operatorname{Softmax}_i(\tau s_i^{GM}),
\qquad \beta^{GM}=\beta^{F2}.
]

**Only the target gradient reference estimator changes.** The Fisher and CVaR
scores are not used in this experiment.

## Experiment 2 — F2 + Fisher-preconditioned influence utility (`influence`)

**Research source:** training-data influence and information geometry, motivated
by Koh and Liang, *Understanding Black-box Predictions via Influence
Functions*, ICML 2017.

A lightweight diagonal empirical-Fisher proxy is computed from target-window
gradients:

[
F_{hd}=\frac{1}{B_t}\sum_b(G_{b,hd}^t)^2,
\qquad
P_{hd}=\frac{1}{F_{hd}+\lambda_F\operatorname{mean}(F)+\epsilon}.
]

For a virtual preconditioned source update
[
\theta_i'=\theta-\eta P\odot g_i^s,
]
the first-order target-loss reduction is approximated by
[
U_i=\langle G_t,P\odot g_i^s\rangle_F,
\qquad
L_t(\theta)-L_t(\theta_i')\approx\eta U_i.
]

The raw utilities are median/MAD standardized and bounded with tanh before
constructing the source marginal:

[
\widehat U_i=\tanh\left(
\operatorname{clip}\left(
\frac{U_i-\operatorname{median}(U)}
{1.4826\operatorname{MAD}(U)+\epsilon},-4,4
\right)\right).
]

[
\alpha_i^{IF}=\operatorname{Softmax}_i(\tau\widehat U_i),
\qquad \beta^{IF}=\beta^{F2}.
]

This is a **first-order, diagonal-Fisher influence-inspired proxy**, not an
exact Hessian influence function. It uses no second derivatives and does not
claim to recover the full Hessian curvature.

**Only the sample scoring geometry changes in this experiment.** Geometric
median and CVaR scoring are not used.

## Experiment 3 — F2 + lower-tail CVaR (`cvar`)

**Research source:** risk-sensitive and distributionally robust optimization,
e.g. Levy et al., *Large-Scale Methods for Distributionally Robust
Optimization*, NeurIPS 2020.

The target support windows are kept in chronological order and divided into
contiguous, balanced blocks. For each block (k), calculate source-target
gradient cosine similarity:

[
s_{ik}=\cos(g_i^s,G_k^t).
]

The lower-tail score averages the least compatible fraction of blocks:

[
R_i^{lower}=\frac{1}{\lceil qK\rceil}
\sum_{k\in\operatorname{Bottom}_{\lceil qK\rceil}(s_{ik})}s_{ik}.
]

To preserve the original F2 signal, this experiment uses:

[
s_i^{CVaR}=0.5s_i^{F2}+0.5R_i^{lower}.
]

Then:

[
\alpha_i^{CVaR}=\operatorname{Softmax}_i(\tau s_i^{CVaR}),
\qquad \beta^{CVaR}=\beta^{F2}.
]

**Only the lower-tail risk criterion is added to the original F2 score.** The
geometric-median and Fisher influence scores are not used.

## Ablation plan

| Run | Method | Baseline relationship |
|---|---|---|
| Existing result | Original F2 | Reuse previously completed result; do not rerun |
| Experiment 1 | `geomed` | F2 + geometric median only |
| Experiment 2 | `influence` | F2 + Fisher influence utility only |
| Experiment 3 | `cvar` | F2 + lower-tail CVaR only |

Run the same seed and all other settings for the three new variants. Compare
each against the already available original F2 result. Do not add a combined
three-method experiment.

## Initial settings

- Geometric-median Weiszfeld iterations: 6.
- Fisher relative damping: 0.1.
- Requested temporal blocks: 5; adaptive block construction keeps at least
  five target windows per block where possible.
- CVaR lower-tail fraction: 0.4.
- All 36 prediction-head gradient coordinates remain together.
- Target marginal `beta` is unchanged.

## Diagnostics

The logs record the method name, source ESS, source marginal L1 shift, target
marginal L1 shift, final score statistics, and top source sample indices and
weights. Method-specific diagnostics include:

- `geomed`: geometric-median displacement from the mean and cosine to mean;
- `influence`: Fisher diagonal scale, damping and raw/standardized utility;
- `cvar`: number of temporal blocks, lower-tail count, CVaR score and
  across-block score variation.

For these source-only variants, target marginal L1 shift should be zero up to
floating-point representation.

## Run command

The script defaults to exactly these three independent variants:

```bash
PMME_F2_METHODS="geomed influence cvar" PMME_F2_SEEDS="1" bash run_f2_robust_fisher_cvar.sh
```

The runner rejects `original` and combined methods. Outputs are isolated in:

- `experiments/f2-robust-fisher-cvar-20261128/geomed/seed1`
- `experiments/f2-robust-fisher-cvar-20261128/influence/seed1`
- `experiments/f2-robust-fisher-cvar-20261128/cvar/seed1`

Each folder records the branch, commit, seed and configuration to prevent
accidental reuse of incompatible outputs.

## Evaluation

Use the same already established original-F2 run as the control. For each
independent variant, compare Stage-I and Stage-II MAE/RMSE at 10/60/120/180
minutes, all 36 prediction steps, source ESS, source-alpha shift, and runtime.
One seed is an initial screening result; promising variants should then be
checked with additional matched seeds. Choose settings using validation data,
not test labels.

The methods are hypotheses to test, not guaranteed improvements. In
particular, target windows overlap, so temporal blocks are stress-test slices,
not independent identically distributed samples.
