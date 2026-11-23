# DVGP Probability Replacement: Controlled Stage-I Ablations

## Verified original probability semantics

In `model/Meta_Models/PMME.py`, PMME trains three independent GP Bernoulli domain classifiers on the **normalized ground-truth 36-step future** `y_unscale`, one per source city. For each classifier, source label=0, target label=1. The official OT weights are **misclassification probabilities**:

- source weight: `P(domain=target | y_source)`
- target weight: `P(domain=source | y_target)`

These are *not* pairwise source/target alignment probabilities. They are thresholded at `--classifier`, normalized separately to `alpha`/`beta`, detached, and passed to the unchanged `geomloss.SamplesLoss` OT solver. All alternatives preserve this exact event and weight transformation. The new networks do **not** consume test labels.

## Versions

| ID | Estimator | Variant | Incremental change |
|---|---|---|---|
| E0 | dvgp | -- | Official GP classifiers |
| E1 | neural | mlp | Small learned domain probability head |
| E2 | neural | multiscale | + multi-kernel temporal convolution |
| E3 | neural | dynamics | + first/second difference statistics |
| E4 | neural | horizon | + three-window conditional descriptors |

E3/E4 are **temporal-difference signatures, not SOFTS input-perturbation responses**. The proposed Jacobian/finite-difference response mechanism and true future supervision are **not yet implemented**. These controlled alternatives test the probability-estimator replacement first; no claim of novel response matching is justified by these variants.

## Run

From the repository root, with the original data and PMME requirements installed:

```bash
python -m unittest discover -s tests -p 'test_domain_probability.py' -v
SEEDS="1 2 3" EPOCHS=3000 bash experiments/dvgp_probability/run_ablation.sh
```

Use the same target/source cities, timestamp split, target-days=3, SOFTS checkpoint initialization protocol, optimizer settings, training budget and Stage-II for every run. `run_stage1_experiment.py` records source-selected best checkpoint and official evaluation time points; it evaluates test data for reporting only. Test results **must not** drive model selection. Source-selected checkpointing matches the chosen fixed protocol; it is not a target validation metric.

## Acceptance checks before full training

1. `E0` must preserve the original training path without neural allocations.
2. All alternatives must output finite probabilities in [0,1] and preserve target/source orientation.
3. `alpha` and `beta` must remain normalized and detached, and zero-selected-mass OT logic unchanged.
4. Source/target labels for GP and neural variants must be exactly identical.
5. Check that classifier gradients are nonzero and the classifier state persists in `STRep.state_dict()`.
6. Run one full epoch of E0 and E1; check no `nan`, device error, shape mismatch, or extra source target leakage.
7. Record per-source BCE, classification accuracy, accepted weight fraction, OT loss scale, OT loss, wall time and GPU peak memory.
8. Only then launch 3-seed experiments and compare 10/60/120/180 min MAE/RMSE with paired seed variation.

## Limitations

The neural classifier uses `binary_cross_entropy_with_logits` while the official GP uses variational ELBO. This is the intended estimator change. Architectures and hyperparameters are *research candidates* rather than validated improvements. Training and metrics have not been run on traffic data in this commit. Independent source heads remain separate, and Stage-II and SOFTS architecture are unchanged. A future stage can implement genuinely input-perturbed SOFTS response matching, **after** checking whether replacing GP probabilities is useful at all.
