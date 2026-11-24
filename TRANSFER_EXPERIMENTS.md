# F1-F4: Task-utility OT ablations (no DVGP/domain classifiers)

**New branch:** experiment/pmme-task-utility-20261124

The sole architectural edit is to Stage-I OT marginal/weight generation in
`model/Meta_Models/PMME.py`. The shared SOFTS/iTransformer backbones,
original GeomLoss Sinkhorn OT **operator**, Stage-I optimizer, official
sample selection/checkpoint rule and `PMME_2ndStage.py` are unchanged.
Unlike DVGP and E1-E4, **none of F1-F4 trains a domain classifier**.

All task-utility variants use actual Stage-I target support batches (only
the target's first three days), never target test labels.

## Exact experiments

| Name | PMME_TRANSFER_VARIANT | Source weights | Target weights |
|---|---|---|---|
| F1 | `uniform` | uniform | uniform |
| F2 | `gradient` | cosine of per-node head-gradient with target-support head gradient, softmax | uniform |
| F3 | `horizon` | equal mean of three 12-step head-gradient cosines, softmax | uniform |
| F4 | `reliable` | F3 with target split-half stability weights + ESS >= 50% of source node/windows | uniform |

**Gradient scope:** the shared `projection2` forecasting layer, not the full
SOFTS/iTransformer parameter vector. For one node/window, its final linear
layer gradient is a rank-1 outer product of the layer input and the loss
gradient wrt the layer output. Similarity can be evaluated without
materializing thousands of these matrices or doing one full backward pass
per node. This is a model-specific first-order proxy for useful transfer,
not a guaranteed estimate of true post-OT target improvement.

All four variants keep official `--DAcoef 1` and
`--classifier 0.5` arguments for reproducibility of the author-provided
PEMS-BAY/SOFTS launch configuration; the `classifier` threshold is
**intentionally ignored** in these variants because they do not generate
DVGP probabilities. OT scales are set to **1** for F1-F4; this differs from
the original DVGP-derived `OT_loss_scale`, and F1 is the crucial control
for determining the effect of that change. Do not attribute an F2-vs-E0
change solely to gradient utility.

## Sequential launch

From repo root, with official PEMS-BAY/METR-LA/Chengdu/Shenzhen datasets in
`./data` and Python dependencies installed:

```bash
# Fast first-pass: run the four variants sequentially for seed 1
PMME_SEEDS="1" bash run_transfer_experiments.sh

# Then complete seed 3 and seed 5 sequentially
PMME_SEEDS="3 5" bash run_transfer_experiments.sh

# Or all 12 jobs (F1-F4 x 3 seeds) in one sequential command:
bash run_transfer_experiments.sh

# Specific variants:
PMME_VARIANTS="gradient reliable" PMME_SEEDS="1" bash run_transfer_experiments.sh
```

Log roots: `experiments/transfer-utility/<variant>/seed<seed>/train.out`.
Every run has its **own working directory**, support selection logs,
`logged_files/` directory, and `l1_pems-bay_SOFTS_seed*.pth` checkpoint.
No run may overwrite another variant's Stage-I checkpoint.

A run is considered complete only if Python exits successfully and its log
contains `2ndStage Source training Ending!!!`. The runner writes a
`COMPLETED` file and skips those jobs on rerun; remove `COMPLETED` to rerun.

Optional controls: `PMME_GPU=0`, `PMME_THREADS=4`,
`PMME_RUN_ROOT=/path/to/results` (the destination must be writable).

## Control settings

- Target `pems-bay`, source list `chengdu_shenzhen_metr`, target 3 days.
- SOFTS, 3000 Stage-I epochs, 6000 Stage-II epochs; seeds 1, 3, 5.
- `DA=OT`, `DAepoch=1500`, `DAcoef=1`, `xySquaredRatio=100`,
  `projection=1`, `MemSize=10000`.
- The runner is sequential, **not parallel**. GPU testing/MAE improvement
  has not been claimed.

## Interpretation

First compare F2-F4 to **F1** using identical seeds (same new weight/scale
route), then compare against the pristine E0 official run. Review
`task_utility` log lines (source ESS, mean cosine, positive fraction,
band reliabilities). Select Stage-I and Stage-II best checkpoints based on
the official target support criterion, not the last test score.

A lower test error on one seed is not a verified improvement. F3/F4 are
nested diagnostic ablations and do not establish method novelty on their
own. Avoid claiming superiority or a novel method before repeated runs
and a proper literature review.
