# F2-RFC: Robust Gradient Reference, Fisher Influence Proxy, and CVaR

## Scope and motivation

This branch starts from `experiment/pmme-f2-gradient-utility-diagnostics-20261126`,
which contains the original F2 gradient-marginal path. It adds three
complementary ideas from robust statistics/federated learning, information
geometry/data influence, and distributionally robust optimization.

No prediction horizon is split. All 36 prediction-head output-gradient
coordinates remain together. No shared pattern bank, additional predictor,
DVGP, gradient surgery, OT-cost modification, target-marginal modification,
or Stage-II change is introduced. The new mechanism changes only the
source-side F2 marginal `alpha`; target `beta` is returned unchanged.

## Three selected methods

### 1. Geometric-median target reference (robust statistics)

Pillutla, Kakade and Harchaoui, *Robust Aggregation for Federated Learning*,
NeurIPS 2022 (original arXiv 2019), motivates replacing an arithmetic mean
of potentially corrupted updates by an approximate geometric median.
Zuo et al., *Byzantine-resilient Federated Learning With Adaptivity to Data
Heterogeneity*, 2024, also applies geometric-median aggregation under
heterogeneous data.

For each target time window (b), first aggregate the sensor-level
projection2 readout gradients while retaining the whole window:
[
G_b^t=\sum_{n=1}^{N_t}\delta_{bn}^t(z_{bn}^t)^\top
\in\mathbb R^{36\times D}.
]
Flatten each (G_b^t) and compute a Weiszfeld approximation to the geometric
median:
[
G_{GM}=\arg\min_G\sum_b\|G-G_b^t\|_F.
]
Then score a source sample (g_i^s=\delta_i^s(z_i^s)^\top) with
[
s_i^{GM}=\cos(g_i^s,G_{GM}).
]
This reduces sensitivity to a small number of unusually large target-window
gradients. It does not imply that rare but legitimate traffic regimes are
outliers; this must be checked in diagnostics.

### 2. Damped diagonal-Fisher influence proxy (information geometry)

Koh and Liang, *Understanding Black-box Predictions via Influence Functions*,
ICML 2017, motivate evaluating how training examples affect a target loss
through inverse-curvature-weighted gradients. Exact influence functions
require Hessian inverse-vector products and rely on local smoothness/
stationarity assumptions; a survey of training-data influence methods notes
that these assumptions are often violated in deep networks.

To keep the F2 extension lightweight, use the target-window gradients to
estimate a diagonal empirical-Fisher proxy:
[
F_{hd}=\frac{1}{B_t}\sum_b (G_{b,hd}^t)^2,
\qquad
P_{hd}=\frac{1}{F_{hd}+\lambda_F\,\operatorname{mean}(F)+\epsilon}.
]
Define the Fisher-metric cosine:
[
s_i^{IF}=\frac{\langle g_i^s,P\odot G_t\rangle_F}
{\sqrt{\langle g_i^s,P\odot g_i^s\rangle_F
\langle G_t,P\odot G_t\rangle_F}+\epsilon}.
]
This is an influence-inspired, diagonal-Fisher preconditioned compatibility
score, **not an exact Hessian influence function** and not a claim of exact
curvature. It requires no second derivatives and no extra autograd pass.
The diagonal estimate is regularized by relative damping to avoid unstable
near-zero coordinates.

### 3. Lower-tail CVaR across temporal support blocks (DRO/risk measures)

Levy et al., *Large-Scale Methods for Distributionally Robust Optimization*,
NeurIPS 2020, analyze CVaR objectives for distributionally robust learning.
Mhammedi et al., *PAC-Bayesian Bound for the Conditional Value at Risk*,
NeurIPS 2020, study CVaR as a coherent risk measure in learning.

Preserve target-window order and split each source city's assigned target
support group into (K) contiguous, balanced temporal blocks. For block (k),
form its mean target readout gradient (G_k^t) and compute compatibility
(s_{ik}=\cos(g_i^s,G_k^t)) (or its Fisher-metric version). The empirical
lower-tail score is
[
\operatorname{CVaR}^{lower}_q(s_i)=
\frac{1}{\lceil qK\rceil}
\sum_{k\in\operatorname{Bottom}_{\lceil qK\rceil}(s_{ik})}s_{ik}.
]
It rewards samples that remain compatible in the less favorable target
temporal blocks, rather than samples that score highly only against the
overall mean. It is a robustness criterion, not a guarantee that every
future regime is covered.

## Combined method (method=triple)

The combined score retains original F2 as an anchor and adds the three
selected ideas:
[
s_i^{triple}=0.20s_i^{F2}
+0.40s_i^{GM-Fisher}
+0.40\operatorname{CVaR}^{lower}_q(s_i^{Fisher,block}).
]
Then:
[
\alpha_i^{triple}=\operatorname{Softmax}_i(\tau s_i^{triple}),
\qquad \beta^{triple}=\beta^{F2}.
]
The combined score is only a hypothesis to test. It is not assumed to
outperform any single component.

## Ablation methods

- `geomed`: geometric-median target reference + ordinary cosine.
- `influence`: diagonal-Fisher preconditioned compatibility using the mean
  target reference (influence-inspired, not exact IF).
- `cvar`: 50% original-F2 / 50% lower-tail block compatibility.
- `triple`: all three ideas with the formula above.
- `original`: original F2 control.

The three individual methods and combined method allow contribution
attribution rather than only comparing a large combined module with F2.

## Initial settings

- Geometric-median Weiszfeld iterations: 6.
- Fisher relative damping: 0.1.
- Requested temporal blocks: 5; adaptive rule keeps at least five windows
  per block where possible. A 15-window target group becomes 3 blocks.
- Lower-tail fraction: 0.4; with 3 blocks the two least-compatible blocks
  are averaged.
- All 36 prediction dimensions are kept together.
- Target beta is returned exactly from the original F2 call.

## Diagnostics

Every method logs:
- geometric-median shift from the mean and cosine to the mean;
- diagonal-Fisher scale, p90 and preconditioner p90/p10;
- original F2, geometric-median, Fisher and robust-Fisher score means;
- number of temporal blocks, lower-tail block count, CVaR mean/p10;
- across-block score standard deviation;
- final score mean/std, source ESS, source alpha L1 shift and target beta
  L1 shift (expected exactly zero for these one-sided methods);
- top source sample indices, weights and scores.

No sample or prediction interval split is introduced.

## Experiment plan

Run matched settings and the same seed for:
1. `original`;
2. `geomed`;
3. `influence`;
4. `cvar`;
5. `triple`.

First use seed 1; only continue promising variants with seeds 3 and 5.
Report target test MAE/RMSE at 10/60/120/180 minutes and all 36 steps,
Stage-I and Stage-II separately, plus source ESS, alpha shift and runtime
overhead. Select checkpoints/hyperparameters using target validation only,
never test labels. A result from one seed is preliminary.

## Literature limitations

Geometric median results in federated learning often assume a bound on the
fraction of corrupted clients; a target traffic window is not a malicious
client, so the transfer is an analogy, not a theorem for traffic data.
Classical influence functions have assumptions that may not hold for a
non-convex SOFTS backbone. The implementation therefore uses a damped
diagonal empirical-Fisher proxy, and names it accordingly. CVaR's tail
estimate is sensitive to the number and dependence of blocks; target
windows overlap, so contiguous blocks are stress-test slices, not i.i.d.
samples. All three mechanisms require empirical validation on PEMS-BAY.
