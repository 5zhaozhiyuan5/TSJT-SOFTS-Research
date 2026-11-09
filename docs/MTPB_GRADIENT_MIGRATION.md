# MTPB native backbone + target-guided source gradients (research integration)

Date: 2026-11-09. This is a **new, unvalidated research branch**; it does NOT
establish a new PEMS-BAY MAE/RMSE. The existing TSJT 12-step trainer is retained
only for historical experiments. No existing Graph WaveNet/SOFTS checkpoints
can be loaded into native MTPB.

## Verified upstream facts

Paper: Zhanyu Liu, Guanjie Zheng, Yanwei Yu, *Multi-scale Traffic Pattern Bank
for Cross-city Few-shot Traffic Forecasting*, ACM TKDD 19(4), 2025,
DOI 10.1145/3727622.

Official code: https://github.com/zhyliu00/MTPB
Pinned revision: `5676a38a34bf3aebbd83cef3cb0d617a9a153b15` (master).

Source modules checked:
- `train.py`: GWN default, `target_days=3`, `train_epochs=50`,
  `finetune_epochs=300`.
- `configs/config.yaml`: MAML history **288**, forecast **36**.
- `model/Meta_Models/rep_model_final.py`: full PatchFSL combines pattern
  bank, graph reconstruction, GWN short-term embedding, long-term encoder,
  and a forecast fusion head. `STRep.meta_train_revise` does support/query
  adaptation followed by a native meta-parameter update.
- `model/Meta_Models/reconstruction.py`: `BatchA_patch_gwnet`.
- `datasets.py`: 5-minute interpolation for Chengdu/Shenzhen,
  source training, first-three-day target adaptation, and subsequent test.
- `README.md`: links to official traffic datasets / pretrained artifacts.

**Important protocol difference:** upstream `traffic_dataset(...,
stage='source_train')` defaults to `add_target=True`: its source meta-dataset
can include the *first three days* of PEMS-BAY. Our both-mode comparison uses
`add_target=False` so the three named source cities are truly separate from
the target-gradient guide. This is **not identical to the author's default
training distribution**. Do not compare obtained metrics to paper numbers
without controlling for the difference.

The original MTPB uses supplied adjacency matrices, but the present proposed
novelty is **gradient adjustment**, not graph topology. No POI, LLM, external
features, or target-test-label optimization is added.

## This iteration: single change, gradient correction on native MTPB meta-step

At the common initial weights theta, independently estimate gradients for
METR-LA, Chengdu, and Shenzhen, plus two distinct minibatches from the
PEMS-BAY **target-training** 3-day split. Reuse the repository's order-balanced
PGrad source interaction, target-orthogonal rotation, learned angle controller
(first-order query surrogate), and nonnegative residual matching. Preserve
the upstream `STRep.meta_train_revise` support/query update.

Let `g_raw = mean(g_city)` and `g_fused` be the residual-matched rotated
source direction. The sole modification to the native model step is

    delta_g = g_fused - g_raw
    theta_new = theta_MTPB - alpha * delta_g
    alpha = min(correction_lr, cap * ||theta_MTPB-theta_old|| / ||delta_g||)

when the norms are nonzero, otherwise the update is untouched.

This ensures the correction cannot exceed `cap=0.2` of the actual native
meta-update norm, so a tiny/failing gradient cannot force an arbitrarily large
correction. Target gradient guides source correction; it is not separately
added as an extra target-weight training step. The first-order angle update
uses only target *training* windows, not the target test split.

**Hypothesis, NOT result:** target-guided changes to the source direction may
reduce negative transfer while preserving the author's cross-city pattern
bank. It may also fail if the target 3-day gradient is noisy, all fusion
weights vanish, or its extra computation exceeds available memory.

## Reproduce (no training was run in this iteration)

Use a *separate* local clone of the official MTPB repository, pinned to the
commit above. Download its official traffic data and pattern files per the
upstream README (do not commit data). Install a compatible PyTorch +
PyTorch-Geometric environment; this adapter additionally requires
`torch.func.functional_call` (PyTorch 2.x), so upstream PyTorch 1.10
environment is **not directly sufficient**. Verify dependency compatibility
before attempting expensive training.

    git clone https://github.com/zhyliu00/MTPB /path/to/MTPB
    cd /path/to/MTPB
    git checkout 5676a38a34bf3aebbd83cef3cb0d617a9a153b15
    cd /path/to/TSJT-SOFTS-Research

    PYTHONPATH=src pytest -q tests/test_mtpb_adapter.py
    PYTHONPATH=src python scripts/run_mtpb_pgrad.py --help

    # ONLY after real data, dependency and smoke checks:
    PYTHONPATH=src python scripts/run_mtpb_pgrad.py \
      --upstream /path/to/MTPB --mode native --epochs 1 \
      --skip-finetune --output outputs/mtpb_native_smoke

    PYTHONPATH=src python scripts/run_mtpb_pgrad.py \
      --upstream /path/to/MTPB --mode target_guided --epochs 1 \
      --order-count 1 --skip-finetune --output outputs/mtpb_grad_smoke

Formal benchmark (requires authorization and resources) holds the same
MTPB config, source-only protocol, seed, source epochs=50,
target finetuning=300, and target three-day split; only --mode differs.
Do **not** use the target test set for model selection.

Collect from real runs: per-horizon MAE/RMSE (1..36), overall MAE/RMSE,
pred std, Persistence baseline, source-target and source-source cosine,
source mixing weights, rotation triggers, fusion weight sum, actual native
step and realized correction norms, and proof weights changed.
The runner emits `diagnostics.jsonl`; actual prediction metrics remain
the responsibility of native MTPB finetuning/testing.

## Validation state

- Static compilation and 4 isolated mock adapter tests completed locally.
- GitHub branch tests should be rerun against actual repository PGrad modules;
  upstream MTPB imports/data were unavailable in this runtime.
- No official-four-city smoke, GPU trial, pretrained checkpoint, or MAE/RMSE
  result was produced. Keep on iteration branch until validated.

## Next blocker

Run a real single-batch native MTPB forward/backward under PyTorch 2.x,
especially check whether the original model's in-forward `torch.tensor`
copies, buffer mutations, and `torch.func.functional_call` support
preserve gradients. Compare native baseline and target-guided one-step
parameter delta. Only then start the 50-epoch evaluation.

## Upstream code compatibility finding (not a model novelty)

In official `model/Meta_Models/reconstruction.py`,
`BatchA_patch_gwnet` defines `gate_convs`, `residual_convs`, and
`skip_convs` using `nn.Conv1d` with 2-D kernel tuples while applying them
to [B,C,N,T] tensors. This causes a rank error on modern PyTorch. The
runner applies `repair_mtpb_gwnet_convs()` **in both modes**: replace
these with `nn.Conv2d` while copying original weights/biases identically.
An isolated tensor-shape test passed locally; a full upstream model run has
not been validated. Run `PYTHONPATH=src pytest -q tests/test_mtpb_upstream_compat.py`.

This is a compatibility fix only. Do not attribute forecasting improvement
to it, or compare repaired and unrepaired models as a gradient ablation.
