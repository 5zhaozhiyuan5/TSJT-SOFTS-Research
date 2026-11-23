# Controlled repair of PMME Stage-I OT weights (2026-11-23)

This is a **diagnostic** branch for the seed-1 E0–E3 problem, not a claim that
smooth weights or neural probability estimation are original contributions.

## Findings that motivate the tests

The neural domain classifier is trained with the same domain labels as DVGP:
source=0, target=1. The OT weights are *reverse-domain probabilities*
`P(target|source)` and `P(source|target)`. High classification accuracy can
make the selected mass vanish behind the hard 0.3 threshold. The original code
also scales the OT loss by `sqrt(mean_source_weight*mean_target_weight)`.

The OT solver, SOFTS architecture, Stage-II and training data splits remain
unchanged. Official DVGP continues to be available via
`--probability-estimator dvgp --ot-weight-strategy legacy --ot-scale-strategy legacy`.

## Ablation matrix (same seed, same E1 MLP, same 3000 epochs)

| Run | Sample weights | OT loss scale | Interpretation |
| --- | --- | --- | --- |
| F0 | legacy hard threshold | legacy dynamic | Replicates E1 control |
| F1 | uniform | fixed=1 | Whether learned selection hurts |
| F2 | smooth positive relative | legacy dynamic | Isolate removing hard threshold |
| F3 | legacy hard threshold | fixed=1 | Isolate dynamic loss attenuation |
| F4 | smooth positive relative | fixed=1 | Combined repair |

Default smooth settings: floor=0.05, power=1.0; these are
hyperparameters to be verified, not pre-optimized. F2's scale uses *new*
smoothed weights, so F2 changes the distribution of scale as well as the
selection mask; use F3 and F4 alongside it rather than attributing F2 to
masking alone. To isolate the threshold with *exactly the same original scale*
would require a dedicated raw-scale option and constitutes a follow-up.

## Usage

```bash
python -m unittest discover -s tests -p 'test_ot_weight_strategies.py' -v
SEEDS="1" EPOCHS=3000 bash experiments/ot_weight_repair/run_ablation.sh
```

Output: `runs/ot_weight_repair_20261123/seed_1/F*/` containing
`stage1_metrics_seed1.json`, `console.log`,
`ot_weight_diagnostics.jsonl`, and checkpoints.

For every source city (identified by source node count 524, 325, 627),
the diagnostics record selected fractions, raw acceptance fractions, ESS,
normalized ESS, OT scale, OT loss, and whether the transport was active.
Sampling is once per 25 epochs. Zero selected mass under legacy is still a
zero OT loss, identical to the original code.

## Interpret the results

1. If F1 > F0 in target quality, current probability selection is harmful.
2. If F3 > F0, original dynamic scaling suppresses useful OT updates.
3. If F4 > F1, soft selection adds value beyond uniform OT.
4. If all are similar, rethink supervision/OT objective before adding response
   CNNs or horizon-aware networks.

All performance claims require **Stage-I and official Stage-II evaluated
with the same settings** and at least three seeds. Train only on target
three-day support and source training split. Do not tune on target test.
The Stage-I checkpoint selection uses source MAE (existing protocol).
