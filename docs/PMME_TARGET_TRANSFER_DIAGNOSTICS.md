# PMME Stage-I Target Transfer Audit

This branch is based on the upstream-exact PMME baseline in the
pmme-stage1-research branch, NOT on the modified main or stage-1-creativity
branches. The official Stage-II remains unchanged.

## Research question

Do GP-to-OT target weights underrepresent some target speed-change patterns
that are both hard to predict and supported by source-city examples?

The audit is **descriptive**, not proof of causal transfer benefit.

## Modifications

- datasets.py: record original sliding-window indices as metadata.
- train_DA_OT.py: add optional --diagnostic_every and
  --diagnostic_source_cap arguments; auditing is OFF by default.
- model/Meta_Models/PMME.py: at selected epochs, collect deterministic
  evaluation-mode target prediction errors, raw GP probabilities,
  GP-thresholded beta, source-city identity, and OT losses/activation.
- model/Meta_Models/pmme_diagnostics.py: compressed node-window NPZ
  logs plus pair_summary.jsonl and epoch_summary.jsonl.
- scripts/analyze_pmme_diagnostics.py: per-mode, error-quartile, source,
  and forecast-horizon CSV summaries.
- tests/test_pmme_diagnostics.py: synthetic tests.

No loss, forecast model, GP classifier, optimizer or OT objective was changed.

## Commands

Diagnostic smoke test (changes the official OT start schedule;
NOT comparable to a full upstream run):

    python train_DA_OT.py --test_dataset pems-bay \
      --data_list chengdu_shenzhen_metr \
      --train_epochs 10 --finetune_epochs 0 \
      --DA OT --DAepoch 0 \
      --diagnostic_every 1 --diagnostic_source_cap 256

To preserve the author's delayed OT schedule instead, use
--DAepoch 50000 and --train_epochs higher than 50000; this can take a
very long time. The upstream default --train_epochs=0 runs no Stage-I
updates. When OT has not started, OT_ACTIVE=false in the output; such
records cannot be interpreted as active OT updates.

Find the run's log directory at:

    logged_files/<test_dataset>/<data_list>/<timestamp>/diagnostics

Then summarize it with:

    python scripts/analyze_pmme_diagnostics.py \
      logged_files/pems-bay/chengdu_shenzhen_metr/<timestamp>/diagnostics

Run tests:

    python -m unittest discover -s tests -p 'test_pmme_diagnostics.py'

## Per-target-sample fields

The NPZ file epoch_XXXXXX_pair_K_<source>.npz stores one row per
target node-window within that source/target pair:

- window_idx and node_idx: original window and sensor identifiers,
  recovered after target permutation and its 15/15/16 split.
- gp_target_probability: raw probability assigned by DVGP that the
  target input is source-like, BEFORE applying the threshold.
- beta: thresholded, within-pair normalized OT marginal mass;
  this is NOT the probability of being predicted correctly.
- accepted: whether beta is nonzero.
- eval_mae / train_mae: MAE on known TARGET TRAINING windows, from
  deterministic eval mode / stochastic SOFTS training forward.
- eval_mae_10min/60min/120min/180min: absolute errors at
  prediction positions 2/12/24/36 (5-minute intervals), NOT
  cumulative errors for these horizons.
- future_delta, future_std, history_last, history_recent_change:
  original-unit training traffic statistics.
- mode: -1 falling, 0 stable, +1 rising future speed, with a
  descriptive training-label-derived threshold.
  These are NOT verified congestion or event labels.
- approx_source_distance: distance in the concatenated [128-D
  representation, 36-D future] feature space to the nearest of up to
  --diagnostic_source_cap evenly selected source node-windows.
  Source and target known TRAIN labels are involved. It is a
  diagnostic support proxy, NOT counterfactual transfer benefit.
- approx_nearest_source_window / node: identity of the nearest
  source candidate within that sampled subset.

pair_summary.jsonl additionally records source-city and target
window identities, retained rates, effective beta sample counts,
GP/weight correlations with target train-window errors, OT active
status and losses, and the mass/count ratio for each target trend mode.

epoch_summary.jsonl records task and DA losses, the separately
calculated target task loss, and train/eval mode target errors.

The analysis script creates mode_by_pair.csv,
difficulty_by_pair.csv, horizon_by_pair.csv,
source_summary.csv and diagnostic_readout.json.

## Interpretation and safeguards

1. Never compare raw beta from different source/target pairs without
   accounting for per-pair normalization and different node counts.
2. Overlapping target windows are not independent observations.
3. The near-source distance uses TRAIN future labels, so it is only
   descriptive. It cannot show that increasing OT weight will help.
4. Separate train-mode and eval-mode errors: SOFTS STAR uses stochastic
   pooling during training.
5. A low-weight, high-error mode may be noise or may lack an appropriate
   source pattern. Do not automatically upweight it.
6. Evidence for a useful change requires a CONTROLLED intervention,
   held-out chronological target evaluation, multiple seeds, and
   comparisons with uniform/random weights and simple target supervision.
7. Do not use target TEST labels to set weights, thresholds or hyperparameters.

## Unmodified upstream protocol caveats

The official datasets.py calculates normalization mean/std BEFORE
time-slicing target train/test segments. Under a strict few-shot
protocol this can leak test-period statistics. The official
evaluation() also reuses the 3-day training windows for validation.
These issues were intentionally NOT changed by this diagnosis-only
branch; any protocol-corrected runs must be reported separately.

## Performance

Every audited update adds one deterministic target forward plus
bounded nearest-neighbor support calculations and compressed NPZ
outputs. This has nontrivial overhead, so use --diagnostic_every 100
or 200 for long runs. With --diagnostic_every 0 (default), no new
diagnostic forward or disk output is performed.
