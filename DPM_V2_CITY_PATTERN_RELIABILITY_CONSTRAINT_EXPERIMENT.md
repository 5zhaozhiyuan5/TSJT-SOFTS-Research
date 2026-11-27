# DPM-V2 + City-Pattern Predictive-Reliability Constraint (CPRC)

## 1. Research problem

The current DPM-V2 bank uses shared prototypes for short/mid/long traffic evolution,
but its historical pattern quality table is pooled across source cities:
`Q[h,k]`. A pattern that forecasts well in one source city can
therefore mask poor predictive quality for the same pattern in another city.

The new CPRC component addresses exactly this issue:

> For each source city and each shared traffic pattern/horizon band, estimate
> source forecasting reliability separately, then constrain how much OT mass
> that city-pattern combination is allowed to contribute.

This is an optional component on the DPM-V2 branch. It does not change F2's
`compute_weights()`, the SOFTS backbone, DPM-V2's pairwise
cost callback, Sinkhorn hyperparameters, PCGrad, or official Stage-II.

## 2. City-conditioned pattern reliability

Let A_{i,h,k} be the shared-bank soft attention of source
sample i to pattern k in horizon band
h\in\{short,mid,long\}. Let s_{i,h}\in[0,1]
be the existing source forecast-skill score calculated from the source training
prediction error and persistence baseline.

For source city c, the pattern-specific empirical
skill is:

\[
\hat Q_{c,h,k}=
\frac{\sum_{i\in c} A_{i,h,k}s_{i,h}}
     {\sum_{i\in c} A_{i,h,k}+\epsilon}.
\]

Small clusters are shrunk toward the existing global bank quality
Q^{global}_{h,k}:

\[
Q_{c,h,k}=
\frac{N_{c,h,k}\hat Q_{c,h,k}+
      \kappa Q^{global}_{h,k}}
     {N_{c,h,k}+\kappa},\quad
N_{c,h,k}=\sum_{i\in c}A_{i,h,k}.
\]

This adds a city dimension to reliability while keeping the existing shared
pattern prototypes. It does not create a separate prototype bank per city.

After the bank is fitted, city quality is updated after each optimizer step
using the source-only feedback already produced by DPM-V2:

\[
Q^{t+1}_{c,h,k}=
(1-\eta_{c,h,k})Q^t_{c,h,k}
+\eta_{c,h,k}\hat Q^t_{c,h,k},\quad
\eta_{c,h,k}=\eta_0\frac{N^t_{c,h,k}}{N^t_{c,h,k}+\kappa}.
\]

Low-support patterns therefore update more conservatively.

## 3. Constrain the source rows of the OT matrix

For a current source sample, its city-conditioned pattern reliability is:

\[
r_{c,i}=\frac{1}{3}\sum_h\sum_k A_{i,h,k}Q_{c,h,k}.
\]

F2 still computes the original source marginal \alpha^{F2}_{c}.
CPRC treats it as a prior and applies a reliability gate:

\[
\bar r_c=\sum_i\alpha^{F2}_{c,i}r_{c,i},
\quad
g_{c,i}=(1-\lambda)+
\lambda\operatorname{clip}
\left(\frac{r_{c,i}}{\bar r_c+\epsilon},
r_{\min},r_{\max}\right)^\gamma,
\]

\[
\tilde\alpha_{c,i}=
\frac{\alpha^{F2}_{c,i}g_{c,i}}
     {\sum_j\alpha^{F2}_{c,j}g_{c,j}},
\qquad
\beta_c=\beta^{F2}_c.
\]

Thus, unreliable city-pattern mixtures receive less row mass; reliable mixtures
receive more. The F2 weighting algorithm itself is not changed, and the target
marginal remains the original F2 marginal.

## 4. Constrain the total contribution of each source city

A city can be poor across many patterns, even if its within-city row weights
have been renormalized. CPRC therefore computes an occupancy-weighted city
reliability score:

\[
S_c=\frac{1}{3}\sum_h\sum_k
O_{c,h,k}Q_{c,h,k},
\]

where O_{c,h,k} is the source-city pattern occupancy estimated
from source training snapshots.

The mean-one city weight is:

\[
\omega_c=
\frac{\operatorname{clip}
((S_c/\bar S)^\gamma,\omega_{\min},\omega_{\max})}
{\frac{1}{C}\sum_d
 \operatorname{clip}((S_d/\bar S)^\gamma,
 \omega_{\min},\omega_{\max})}.
\]

The existing DPM-V2 cost remains unchanged. For each city, the OT term uses
\tilde\alpha_c and the original target marginal, then its
contribution to the summed Stage-I OT objective is scaled by \omega_c.
This creates a hierarchical constraint:

- sample/pattern level: adjust source row marginal \tilde\alpha_c;
- source-city level: adjust the city's effective OT contribution \omega_c;
- target side: keep the original F2 \beta_c unchanged;
- pairwise geometry: keep DPM-V2's C^{DPM-V2} unchanged.

## 5. Why this is different from current DPM-V2

| Mechanism | Current DPM-V2 | New CPRC |
|---|---|---|
| Reliability granularity | Global pattern quality pooled across cities, plus current sample skill | Separate quality for each city × pattern × horizon |
| OT object changed | Pairwise ground cost C | Source row marginal and city contribution weight |
| Main question | Which source-target pair should be matched? | How much transport mass should this city-pattern contribute? |
| F2 | Unchanged | Unchanged |
| DPM-V2 cost | Active | Unchanged and retained |

CPRC is therefore complementary to the DPM-V2 attention cost, not a replacement
for F2 or a new version of its gradient weighting.

## 6. Implementation and logging

New module:
- `model/Meta_Models/city_pattern_reliability_constraint.py`

Integration:
- collects per-city source-only reliability snapshots before `DAepoch`;
- initializes Q_{c,h,k} after the shared bank is fitted;
- adjusts \alpha immediately before the existing DPM-V2 Sinkhorn call;
- updates city-specific reliability after the optimizer step;
- leaves Stage-II untouched.

New diagnostics:
- `dpm_city_pattern_constraint.jsonl`: per-city alpha shift,
  effective sample size, pattern mass before/after, low-quality pattern mass,
  city weight and pattern reliability;
- `dpm_city_pattern_reliability_state.pt`: resumable city-specific
  reliability state.

The logged pattern_mass_* values are row-marginal-induced
pattern masses, not the full GeomLoss transport plan. The existing DPM-V2
64×64 subset Sinkhorn diagnostic remains a subset plan and must not be described
as the full training \Pi.

## 7. Experiment

Run on the current branch:

`bash run_f2_dpm_v2_city_pattern_constraint.sh`

Default configuration:
- target: PEMS-BAY, 3-day target support;
- sources: Chengdu, Shenzhen, METR-LA;
- backbone: SOFTS;
- seed: 1;
- DPM mode: V2-B;
- DPM strength: 0.35;
- CPRC strength: 0.5;
- Stage-I / Stage-II: 3000 / 6000 epochs;
- `DAepoch=1500`;
- original F2 method, original OT schedule, original official Stage-II.

The run is stored in a new directory:
`experiments/dpm-v2-city-pattern-reliability-constraint/v2b/seed1`.
It does not overwrite the paired V2-A/V2-B runs.

### Required comparisons

1. Existing V2-B vs V2-B + CPRC: test whether the new constraint improves
   target MAE/RMSE.
2. Per-city pattern quality: verify that low-prediction-quality patterns in
   a specific city are actually assigned less source row mass.
3. City contribution weights: verify that a globally weak source city is
   down-weighted without one city dominating.
4. Short/mid/long: inspect each band separately; do not rely only on aggregate
   MAE.
5. Stage-I and Stage-II: check whether Stage-I matching gains survive official
   Stage-II adaptation.

First run a short smoke test with a fresh run root and reduced epochs.
The runner executes `py_compile` and the CPRC CPU unit tests before training.
Example smoke command:

```bash
PMME_DPM_STAGE1_EPOCHS=3 \
PMME_DPM_STAGE2_EPOCHS=1 \
PMME_DPM_DAEPOCH=2 \
PMME_DPM_INTERVAL=1 \
PMME_DPM_V2_LOG_INTERVAL=1 \
PMME_DPM_CPRC_RUN_ROOT="$PWD/experiments/dpm-v2-cprc-smoke" \
PMME_DPM_CPRC_SKIP_SUMMARY=1 \
bash run_f2_dpm_v2_city_pattern_constraint.sh
```

Only after the code path, logs and checkpoints are verified should the full
3000/6000-epoch run be started. If seed 1 is promising, repeat with seeds 2
and 3.

### Interpretation guardrails

- Source labels are used only for source training forecasts and source-side
  reliability. Target support labels are not used by CPRC.
- No target validation/test labels enter CPRC.
- Current source forecast quality is measured on the source training batches
  used by the existing DPM-V2 online feedback. It is not an independent
  source-validation estimate; report this limitation and, if needed, add a
  chronological source holdout audit before making a generalization claim.
- Do not claim success from a changed alpha or subset plan alone. The primary
  success criterion is improved target-domain forecasting metrics.

## 8. Related methodological inspiration

CPRC is motivated by the transport-or-discard / partial-transport principle,
but it is not a reproduction of these methods. Its specific contribution is
to condition transport budgets on source-city-specific, horizon-specific
traffic-pattern forecast reliability.

- Naram et al., ICML 2025, *Theoretical Performance Guarantees for Partial
  Domain Adaptation via Partial Optimal Transport*:
  https://proceedings.mlr.press/v267/naram25a.html
- Koç et al., ICML 2026, *Incorporating Importance Weighting in Optimal
  Transport Based Domain Alignment*:
  https://proceedings.mlr.press/v306/koc26a.html
- Chen et al., ICML 2026, *Transport or Discard: Robust Unbalanced Optimal
  Transport for Cross-Domain Policy Adaptation*:
  https://proceedings.mlr.press/v306/chen26gd.html
