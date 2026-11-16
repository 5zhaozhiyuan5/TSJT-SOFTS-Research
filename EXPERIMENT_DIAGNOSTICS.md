# Experiment diagnostics for PMME + PatternMoE-OT

Every invocation of `train_DA_OT.py` creates a unique run directory under
`logged_files/<target>/<sources>/<timestamp>/`. The diagnostic logger writes
append-only CSV streams into that directory and records fatal Python exceptions
to `exception_traceback.txt`. Logging is observational only: it does not add
loss terms, change gradients, or alter optimizer updates.

## Output files

| File | Contents | Primary use |
|---|---|---|
| `run_metadata.json` | CLI arguments, seed, datasets, time grid, history/prediction lengths, split sizes, resolved scales and anchors, software/runtime metadata | Reproduce and interpret a run |
| `log.txt` | Existing human-readable training messages and configuration | Quick inspection |
| `epoch_metrics.csv` | Stage-I epoch losses, source/target loss terms, MAE/RMSE/MSE/MAPE, finite-loss flag, source-city weights | Detect convergence, loss imbalance and instability |
| `horizon_metrics.csv` | Per-output-step train metrics for Stage-I and Stage-II | Locate which forecast steps degrade |
| `evaluation_metrics.csv` | Per-output-step few-shot validation, test, and Stage-II final-test metrics, with sample counts | Compare horizon-wise generalization |
| `moe_router.csv` | Per-source-batch and horizon router weights, pairwise router spread/top-1 fraction, entropy, expert costs | Detect expert collapse or unchanging gates |
| `pattern_bank.csv` | Initial prototype-bank statistics for each historical scale | Check bank construction and scale coverage |
| `pattern_usage.csv` | Per-scale combined/source/target prototype usage, source-target usage gap, entropy, top-1 fraction, and mean pairwise cost | Detect unused/collapsed prototypes and source-target pattern coverage mismatch |
| `ot_diagnostics.csv` | Per-horizon conditional OT loss, plan diversity/entropy/concentration, row/column marginal errors, sample counts | Check whether conditional transport differs, collapses to a few pairs, and converges |
| `gradient_diagnostics.csv` | Stage-I/II module gradient norm, missing-gradient fraction, non-finite gradient count, parameter norms and sampled update magnitude | Detect disconnected modules, exploding/invalid gradients, or negligible updates |
| `runtime.csv` | Epoch time, cumulative runtime, learning rate, CUDA allocated/reserved memory | Diagnose speed and memory issues |
| `stage2_metrics.csv` | Official PMME Stage-II losses and source prediction metrics | Diagnose target adaptation training |
| `exception_traceback.txt` | Unhandled Python exception and traceback, if one occurs | Locate runtime failures |

## Logging cadence

- Stage-I and Stage-II scalar epoch metrics, per-step training metrics, gradient
  summaries, and runtime are written once per training epoch.
- Pattern/MoE/OT diagnostics are written for each source-city alignment call and
  each configured horizon anchor (normally 10, 60, 120 and 180 minutes).
- Evaluation metrics are written at each existing evaluation event, separately
  for few-shot validation and test. The existing evaluation schedule is not
  changed by this diagnostics patch.
- The initial pattern bank is recorded once when the Stage-I model is created.

## Interpretation notes

1. Router weights are averages over sampled source-target pairs. Pairwise
   standard deviation and top-1 fractions are also recorded; averages alone
   can hide pair-level collapse.
2. Pattern usage is reported for source and target samples separately as well as
   jointly. The source-target usage gap indicates whether the two domains use
   the prototype bank differently. It is a diagnostic of prototype utilization,
   not a claim that prototypes have human-interpretable semantic labels.
3. OT marginal errors are finite-iteration Sinkhorn residuals. A nonzero value
   is expected; compare trends and configured iteration counts rather than
   treating a single small residual as proof of useful domain transfer.
4. Plan diversity measures how different horizon-conditioned transport plans
   are. Plan entropy and row entropy describe how diffuse the transport weights
   are; maximum weight and row top-1 mass indicate concentration. Neither high
   nor low entropy is inherently best. Compare these values with horizon-wise
   validation/test metrics and marginal errors.
5. Parameter update magnitude is estimated from the first 4096 flattened
   parameter values in each module to keep logging memory overhead small. It
   complements, but does not replace, a full checkpoint comparison.
6. Training metrics are computed per forecast step from the current batch
   outputs; evaluation metrics aggregate all evaluated samples per forecast
   step. MAPE uses a small denominator clamp to avoid division by zero.
7. The existing model-selection and evaluation policy is intentionally
   unchanged. In particular, this patch does not change which split is used
   for checkpoint selection.

## Quick inspection

After a run, inspect `run_metadata.json` first to confirm the target/source
cities, seed, time grid and horizon mapping. Then compare `horizon_metrics.csv`
with `evaluation_metrics.csv`; inspect `moe_router.csv`,
`pattern_usage.csv`, and `ot_diagnostics.csv` to explain any forecast
degradation. Finally use `gradient_diagnostics.csv` and `runtime.csv` to
check whether the relevant modules trained and whether the run was stable.
