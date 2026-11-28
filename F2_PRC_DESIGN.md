# F2 + PRC: Source Prediction Reliability Calibration

## 1. Research question

Original F2 estimates whether a source sample's shared forecasting-head
gradient is compatible with the target-city gradient. It does not directly
measure whether that source sample was forecast accurately by the source
forecaster.

PRC adds this second, independent signal:

- **F2 transfer compatibility:** Is the source sample's parameter-update
  direction useful for the target city?
- **Source predictive reliability:** Does the source sample itself have a
  small forecasting error?

The intended rule is that high source-to-target gradient compatibility alone
must not guarantee high OT mass when the source forecast is unreliable.

This is one new calibration module on top of original F2. Geomed, Fisher
influence, CVaR, a shared pattern bank, and target-probe counterfactual
optimization are not used in this experiment.

## 2. Where the source reliability is measured

During the existing PMME forward pass, each source city already produces:

- forecast `out`, in the same physical scale used by the source task loss;
- ground-truth future `y`;
- shared projection2 input features `source_features` with shape [B,N,128];
- task-loss output deltas `source_deltas` with shape [B,N,36].

The F2 marginal has B*N entries. Each entry corresponds to one
**source time-window × sensor-node** item, not an entire city and not an
independent learned traffic-mode prototype.

PRC computes a per-item forecasting error from the already available forward
pass:

[
e_i=\frac{1}{36}\sum_{h=1}^{36}
|\hat y_{i,h}^{s}-y_{i,h}^{s}|.
]

No additional model forward, backward pass, trainable network, or prediction
head is introduced. Prediction and label shapes are checked against the F2
marginal item count before calibration.

## 3. City-local robust normalization

Raw MAE values should not be compared directly between Chengdu, Shenzhen and
METR-LA because their traffic scales and temporal sampling differ. Each
source city's errors are therefore normalized within that city's current
source support:

[
m_e=\operatorname{median}_i(e_i),\qquad
d_e=\operatorname{median}_i|e_i-m_e|,
]

[
z_i=\operatorname{clip}\left(
\frac{e_i-m_e}{1.4826d_e+\epsilon},-z_{max},z_{max}
\right).
]

If MAD is numerically zero, the implementation falls back to the standard
deviation; if that is also zero, the epsilon floor yields a neutral score.
Positive z means worse-than-city-median forecast error; negative z means
better-than-city-median error.

A bounded reliability multiplier is then formed:

[
r_i=\operatorname{clip}
(\exp(-\kappa z_i),r_{min},r_{max}).
]

The initial settings are `z_max=3`, `kappa=0.5`, and
`r_min/r_max=0.5/2.0`. The multiplier is relative within each source city;
it is not interpreted as a calibrated probability.

## 4. Fuse reliability with original F2 without replacing it

Let `alpha_F2` be the source marginal returned by the existing original F2
gradient utility. First form a reliability-tilted marginal:

[
\widetilde\alpha_i=
\frac{\alpha_i^{F2}r_i}
{\sum_j\alpha_j^{F2}r_j}.
]

Then use a conservative convex calibration:

[
\boxed{
\alpha_i^{PRC}=
(1-\rho)\alpha_i^{F2}
+\rho\widetilde\alpha_i
}
]

The default is `rho=0.5`. This retains half of the original F2 distribution
and prevents one batch's forecast errors from fully replacing F2's transfer
compatibility judgement.

The final OT call remains:

[
\Pi^*=\operatorname{Sinkhorn}(C,\alpha^{PRC},\beta^{F2}).
]

**Preserved:** target marginal beta, original cost matrix C, Sinkhorn operator,
SOFTS backbone, task loss, Stage-II implementation and the original F2
gradient calculation.

PRC only calibrates the source marginal. It does not directly change the
total loss coefficient assigned to each source city. Since the current PMME
implementation runs one OT term per source city, this first version adjusts
the relative mass of samples within each city; it does not yet learn a
cross-city global mixture weight.

## 5. Runtime and data-use caveat

The error comes from the source training forward pass already used by PMME.
It is therefore a **training-support predictive-reliability signal**, not an
independent held-out estimate of generalization error. It must not be
described as source validation accuracy. Robust city-local normalization,
bounded multipliers and the 0.5 blend reduce the impact of individual
outliers, but cannot remove in-sample optimism.

The source training labels are permitted supervision for this calibration.
No target test labels or future evaluation data are used. The target-city
marginal remains the original F2 beta.

A later study may compare this inexpensive online training-support signal
against a temporally held-out or cross-fitted source reliability estimate.
That is deliberately not included in this first implementation.

## 6. Diagnostics

Every 25 epochs (and at the first OT epoch), the `f2_prc` log reports:

- source forecast MAE mean, median and p90;
- robust error scale and standardized-error statistics;
- reliability multiplier min/p10/p90/max and fraction above one;
- source marginal ESS before and after PRC;
- source marginal L1 shift;
- error-to-alpha correlation before and after calibration;
- highest-weight source item indices, their reliability multipliers, and
  highest-error source item indices;
- confirmation that target beta was not changed.

Interpretation checks:

1. `error_alpha_corr_after` should generally be lower than
   `error_alpha_corr_before` when high-error samples are penalized.
2. `source_l1_shift` should be nonzero when errors differ, but not explode.
3. `source_ess_frac_after` should remain finite and not collapse.
4. `target_marginal_unchanged=True` must always appear.
5. Compare per-city diagnostics; do not compare raw source MAE across cities.

## 7. Initial experiment settings

| Setting | Initial value |
|---|---:|
| F2 method | original |
| PRC enabled | true |
| reliability blend rho | 0.5 |
| error sensitivity kappa | 0.5 |
| robust z clip | 3.0 |
| reliability multiplier bounds | [0.5, 2.0] |
| source/target OT | original GeomLoss Sinkhorn |
| pair OT | off (pair_variant=f2) |
| feedback policy | g0 |
| Stage-I epochs | 3000 |
| Stage-II epochs | 6000 |
| DA start epoch | 1500 |
| target city | PEMS-BAY, 3-day support |
| source cities | Chengdu, Shenzhen, METR-LA |

## 8. Run

The historical original-F2 result is the control and is **not retrained**.

On the new branch, after checking out the code and ensuring the local dataset
directory exists:

```bash
PMME_F2_PRC_SEEDS="1" bash run_f2_prc.sh
```

Optional sensitivity checks (each uses a distinct output root):

```bash
PMME_F2_PRC_BLEND="0.25" PMME_F2_PRC_RUN_ROOT="experiments/f2-prc-blend025" bash run_f2_prc.sh
PMME_F2_PRC_BLEND="0.75" PMME_F2_PRC_RUN_ROOT="experiments/f2-prc-blend075" bash run_f2_prc.sh
```

The first run is the main PRC experiment. Do not launch the sensitivity runs
until the initial run has been inspected.

## 9. Evaluation protocol

Compare PRC against the already completed original-F2 run using the same
seed, Stage-I/Stage-II settings and data protocol. Report MAE/RMSE at
10/60/120/180 minutes, plus the full 36-step curve. Include all source-city
PRC diagnostics and runtime.

One seed is an initial screening result, not evidence of statistical
significance. If the initial result is promising, repeat with matched seeds
and use validation results for hyperparameter selection. The test set is
used only for final reporting.
