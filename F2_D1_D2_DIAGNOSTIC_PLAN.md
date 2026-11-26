# D1/D2 diagnostics: Does original F2 gradient similarity predict useful OT updates?

## Task and scope

The previous G1 (mag), G2 (sign), G3 (EMA), and G4 (BiOT) four-run batch16 seed1 experiment did not clearly improve over the *historical* original F2. Before introducing a new F2 weighting rule, we test the underlying ranking assumptions.

This branch adds **two diagnostic experiments**, not four new training variants:
- **D1** — Is high per-node source/target head-gradient cosine predictive of a better real Sinkhorn OT gradient on the *target support probe* loss?
- **D2** — Within cosine-stratified source groups, does GradDrop-inspired weighted-sign disagreement add useful information beyond the original F2 cosine ranking?

No PFR, no graph/forecast backbone changes, no DVGP, no Stage-II changes, no new trainable parameters, no modified original OT cost. Original F2 training weights (alpha) and uniform beta are UNCHANGED if audit instrumentation is enabled during training.

Preferred execution mode: **read-only F2 checkpoint probe** (loads existing trained original F2 Stage-I model and samples 3-day target support; does not retrain or evaluate target test). Only use a genuine original F2 checkpoint, not G1–G4 or P1–P4 checkpoints.

## Mathematical design

For each source node/window, with projection2 input h_i and output loss derivative delta_i:
  g_i = vec(delta_i outer h_i).
For the target **fit-only** support windows:
  G_fit = sum_j (delta_j outer h_j).
D1 ranking score:
  c_i = cosine(g_i, G_fit).
D2 weighted coordinate sign conflict:
  d_i = (1 - <g_i,G_fit>/sum_k |g_ik G_fit,k|) / 2.
The zero-gradient case is neutral (d_i=0.5).
No gradient surgery is performed.

Target support sampling has 46 windows grouped into 15/15/16. For the audit, windows with local index % 5 == 0 are probe windows; other support windows are fit windows. The probe and fit sets share no *window identities* in that diagnostic pass, but sliding windows can overlap in time, and the fitted checkpoint ALREADY SAW target-support windows while training. This is not chronological validation, not a generalization test, and not a leakage-free training pipeline in a strict temporal sense.

On each source-city/target-group comparison, use a single fixed source pool (top/bottom cosine plus evenly spread) and a fixed target-fit pool, e.g. 192 source and 96 target sensor/window points. For all candidate policies the ORIGINAL GeomLoss cost and support are exactly the same. The only manipulated variable is the source alpha vector; target beta remains uniform.

Seven candidate alpha choices:
  uniform: equal source weights
  f2: softmax(8*cosine)
  sign: softmax(8*cosine - 1.5*conflict)
  cos_high: 95% mass among top cosine quartile, 5% uniform floor
  cos_low: 95% mass among bottom cosine quartile, 5% uniform floor
  sign_low_cosmatched: 95% mass among low-conflict samples within EACH cosine quartile
  sign_high_cosmatched: 95% mass among high-conflict samples within EACH cosine quartile

The last two choices are cosine-STRATIFIED, not exact matched pairs. Their mean cosine gap is logged so that mismatch can be identified. A positive mass floor ensures GeomLoss never sees zero Sinkhorn weights.

For each candidate alpha:
  L_OT(alpha) = 0.01 * GeomLoss_original(alpha, [rep_source,10*true_future_source],
                             uniform_beta,[rep_target_fit,10*true_future_fit]).
  g_OT(alpha) = gradient(L_OT(alpha), projection1.weight).
Probe prediction gradient:
  g_probe = gradient(SmoothL1(pred_probe, true_probe), projection1.weight).

First-order loss reduction estimate (per unit infinitesimal positive LR):
  U(alpha) = <g_probe, g_OT(alpha)>.
Positive U means the isolated -g_OT direction is locally downhill for probe loss. The real PMME update uses SOURCE-LOSS + OT, full-parameter one-sided PCGrad, and stateful AdamW; therefore U is NOT realized improvement from the full training step.

Report:
  D1_cos_high_minus_low_gain = U(cos_high)-U(cos_low)
  D1_f2_minus_uniform_gain = U(f2)-U(uniform)
  D2_sign_low_minus_high_gain_given_mid_cos =
      U(sign_low_cosmatched)-U(sign_high_cosmatched)
  D2_sign_minus_f2_gain = U(sign)-U(f2)

Also record source/sample/target indices, target probe/fit windows, per-pool cosine and conflict, gradient dot, norm, cosine, source shifts, and city/source group label. The actual final OT coupling Pi is NOT exposed or claimed. Raw probe gradient is not a separate test set.

## Run on the existing Linux GPU server

Use a path to the already-trained **original F2 Stage-I** checkpoint. This does NOT launch 3000/6000 epochs.

1. Fetch and switch:
   git fetch origin
   git switch experiment/pmme-f2-gradient-utility-diagnostics-20261126

2. CPU unit tests (no GeomLoss import):
   python -m unittest discover -s tests -p 'test_f2_gradient_utility_audit.py'

3. One-batch smoke on the real server (GeomLoss + SOFTS, no training):
   PMME_F2_DIAG_PROBES=1 PMME_F2_DIAG_RUN_ROOT=./experiments/d1d2-smoke \
   bash run_f2_d1d2_diagnostics.sh /absolute/path/to/original-f2-stageI.pth

4. Read-only diagnosis (prefer >=16 sampled batches):
   PMME_F2_DIAG_PROBES=16 \
   bash run_f2_d1d2_diagnostics.sh /absolute/path/to/original-f2-stageI.pth

5. Summarize and inspect:
   cat experiments/f2-d1d2-readonly-seed1/summary.txt
   head -n 1 experiments/f2-d1d2-readonly-seed1/f2_d1d2.jsonl

The runner writes:
  run_config.txt (commit/checkpoint/seed/probe budget)
  train.out (full audit text + JSON records)
  f2_d1d2.jsonl (machine-readable per-city/per-probe record)
  f2_d1d2_summary.csv
  summary.txt
  COMPLETED or FAILED

Avoid reusing the same run directory for different checkpoints, seeds, budgets or probe counts. The runner should reject mismatched configurations. No target test dataset is loaded by the standalone driver. CPU can technically run but real traffic experiment on the original CUDA server is recommended.

## What would count as evidence?

- D1 is supported if high-cosine source strata systematically give better probe OT gradient alignment than low-cosine strata, and F2 is better than uniform on the same fixed sample pool.
- D2 provides NEW information only if low-sign-conflict strata beat high-conflict strata **after cosine stratification**, and sign reweighting beats F2. Merely detecting many negative sign coordinates is not enough.
- Report per-city means, directions and fraction of positive effects, not p-values based on correlated node samples. Repeating 16 resampled support batches does not create 16 independent training datasets.
- Use additional held-out target days ONLY for final offline reporting, NEVER for tuning the F2 sample weights within training.
- If D1 and D2 both fail, stop modifying gradient indicators and investigate the actual OT matching/marginal mechanism. If D2 shows repeatable incremental utility, build a new F2 weight rule and run matched full Stage-I+Stage-II training only then.
- This code tests the original F2's ranking/proxy assumptions; it does not itself report any MAE/RMSE gain.

## Important limitations

1. Existing Stage-I checkpoint is required; the GitHub run-log archives alone do not contain checkpoint tensors.
2. Target-support probe windows may overlap temporally with fit windows; trained checkpoint already saw both. This is exploratory internal validation, not statistical generalization.
3. The model is not updated; read-only driver restores train-mode buffers between probes and checks trainable parameters bitwise.
4. This audit explicitly uses a small matched OT subset, not the exact full-city training coupling. Results should be confirmed on a larger sample or with an exact H2 full-batch audit before claims about final PMME transfer.
5. The diagnostic observes OT gradients on projection1.weight only, not full all-parameter PCGrad + AdamW. Follow up with the existing H1/H2 shadow machinery if signs warrant it.
