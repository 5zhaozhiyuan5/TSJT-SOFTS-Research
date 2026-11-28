# STUNet official baseline and diagnostic logging

Official upstream: https://github.com/JimmyChen6/STUNet  
Pinned upstream commit: \`11d595c91a1559193f26d516573ec2262b716f6d\`

This branch adds the official implementation as a Git submodule at \`third_party/STUNet\`. Diagnostic hooks live outside the submodule and do not rewrite the official model source.

## Clone this branch

\`\`\`bash
git clone --branch baseline/stunet-official-diagnostics-20261128 --recurse-submodules https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research.git
cd TSJT-SOFTS-Research
\`\`\`

If already cloned without its submodule:

\`\`\`bash
git submodule update --init --recursive
\`\`\`

## Official model workflow

The upstream code uses three training stages: spatial tokenizer pretraining, temporal tokenizer pretraining, and STUNet backbone training.

1. Install upstream dependencies. Its README specifies Python 3.10.16, PyTorch 2.6.0, Transformers 4.46.2, PyG 2.6.1, Lightning 2.x, torchmetrics, Hydra, and the listed logging packages. The upstream snapshot does not contain a requirements.txt.
2. Prepare LargeST data following the upstream LargeST instructions.
3. Edit \`third_party/STUNet/configs/paths/default.yaml\` and \`data/largest.yaml\`.
4. Change \`devices: [6]\` in \`third_party/STUNet/configs/trainer/default.yaml\` to a GPU index available on your machine.
5. Set valid checkpoint paths in \`model/SpatialAE.yaml\`, \`model/TemporalAE.yaml\`, and \`model/STUNet.yaml\`. Stage II loads the two pretrained encoders.

Run from the repository root:

\`\`\`bash
python scripts/run_stunet.py --entry train --config-name=spatial
python scripts/run_stunet.py --entry train --config-name=temporal
python scripts/run_stunet.py --entry train --config-name=train
python scripts/run_stunet.py --entry eval --config-name=eval
\`\`\`

The launcher installs diagnostic hooks before importing the upstream Hydra entry point. It applies an in-memory compatibility alias for the upstream \`STUNet.py\` import of \`GFBackbone.GFBackBone\` versus the supplied \`Backbone.py:BackBone\`. The official submodule files remain unchanged.

**Dataset limitation:** the pinned upstream datamodule is LargeST-specific. It hard-codes LargeST dataset names, node/edge counts, and 35,040 time steps. It is not a ready-to-run PEMS-BAY/METR-LA few-shot transfer loader. This branch preserves that official setup rather than silently changing the experimental task.

## Diagnostic sampling and overhead

Defaults limit overhead while retaining per-batch loss/timing:

\`\`\`bash
export STUNET_DIAG_EVERY=20
export STUNET_DIAG_ATTENTION_EVERY=100
export STUNET_DIAG_GRAD_EVERY=50
export STUNET_DIAG_TOKEN_EVERY=50
export STUNET_DIAG_PRED_EVERY=20
export STUNET_DIAG_DIR=runs/stunet_diagnostics
export STUNET_RUN_NAME=spatial_seed1
\`\`\`

For a short intensive diagnostic run, set the intervals to \`1\`. Attention/token summaries use small batch samples to avoid retaining full large tensors. Diagnostic metrics are sampled, so use identical settings when comparing experiments.

## Log files

Each run writes to \`runs/stunet_diagnostics/<run-name>/rank0\`:

| File | Main contents |
|---|---|
| \`events.jsonl\` | Timestamped run metadata, checkpoint-load events, model/data summaries and warnings |
| \`batch_stats.csv\` | Input/label shapes and distributions, finite/zero ratios, adjacency density/symmetry, degree and isolated-node ratios, target/context node counts |
| \`token_stats.csv\` | Node/graph token scale, variance, norms and cross-token cosine |
| \`attention_stats.csv\` | Query/aggregate attention entropy, effective support, top-1/top-5 mass, head-wise entropy and aggregate diagonal mass |
| \`attention_layer_stats.csv\` | Per-layer attention-block input/output and update magnitudes |
| \`backbone_residual_stats.csv\` | Additional outer residual magnitude and post-residual token scale per backbone layer |
| \`generated_graph_stats.csv\` | Generated adjacency attention symmetry, diagonal, row sums, entropy and top-1 mass (when graph generation is enabled) |
| \`prediction_metrics.csv\` | Sampled MAE/RMSE/MAPE, bias, valid ratio and non-finite counts |
| \`horizon_metrics.csv\` | Error and bias at every prediction horizon |
| \`regime_metrics.csv\` | Error across low/middle/high target-value regimes |
| \`node_metrics.csv\` | Highest-error nodes and their global node IDs |
| \`target_context_metrics\` events | Target versus sampled context-node errors during training |
| \`gradient_stats.csv\` | Gradient/parameter norms, missing/zero/non-finite gradients by module |
| \`parameter_gradients.csv\` | Per-parameter gradient norm and gradient-to-parameter ratio |
| \`step_metrics.csv\` | Loss, batch time, samples/sec, node-timesteps/sec, current/peak GPU memory |
| \`epoch_summary.csv\` | Lightning epoch metrics and learning rates |
| \`spatial_pretrain_stats.csv\` | Adjacency reconstruction MSE/MAE, edge density and thresholded precision/recall |
| \`temporal_pretrain_stats.csv\` | Temporal pretraining loss and per-horizon prediction metrics |

## First-pass diagnosis order

1. Check \`batch_stats.csv\` first: input scale, zero/non-finite values, adjacency density, graph degree, isolated nodes, and target/context ratio.
2. Check \`spatial_pretrain_stats.csv\`: compare reconstruction error with adjacency sparsity. Thresholded precision/recall at 0.5 are diagnostic proxies, not the official objective.
3. Check \`temporal_pretrain_stats.csv\` and \`horizon_metrics.csv\`: identify whether long horizons already fail in temporal pretraining.
4. Check \`token_stats.csv\`: look for token scale imbalance or near-constant representations.
5. Check \`attention_stats.csv\` with \`backbone_residual_stats.csv\`: distinguish diffuse/concentrated attention from residual amplification or representation collapse.
6. Check \`gradient_stats.csv\`: determine whether trainable temporal tokenizer, attention backbone, and prediction head receive finite and non-vanishing gradients. The spatial graph encoder is frozen in the default tokenizer.
7. Check \`node_metrics.csv\`, \`regime_metrics.csv\`, and the masking-mode events: locate sensor-specific, high-flow, and train/evaluation-scope failures.

Do not infer a defect from one metric or one seed. Use repeated runs and targeted ablations before introducing a new module.
