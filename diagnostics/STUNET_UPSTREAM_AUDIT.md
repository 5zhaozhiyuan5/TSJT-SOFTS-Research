# Source-level audit of the pinned official STUNet implementation

Upstream: https://github.com/JimmyChen6/STUNet  
Pinned commit: \`11d595c91a1559193f26d516573ec2262b716f6d\`

These are source-code observations and research hypotheses, not claims that the paper is incorrect.

## Implementation facts worth testing

1. **Import-name mismatch.** \`src/models/STUNet/core/STUNet.py\` imports \`GFBackbone.GFBackBone\`, while the pinned tree supplies \`Backbone.py\` with class \`BackBone\`. The diagnostic launcher registers an in-memory alias. It does not alter upstream files.
2. **LargeST-specific data path.** \`lightning_datamodule.py\` hard-codes dataset node/edge counts for CA, GLA, GBA, SD and cleaned variants, plus \`num_step=35040\`. This is not a plug-in loader for METR-LA or PEMS-BAY.
3. **Stage-I checkpoints are part of the default workflow.** Stage-II Tokenizer loads the temporal encoder and graph encoder. The YAML files contain placeholder checkpoint paths that must be replaced with real pretrained checkpoints.
4. **Spatial encoder capacity.** \`GraphEncoderModel\` splits the adjacency matrix into fixed-size patches and maps each flattened patch through an MLP. A LayerNorm is declared but is not used in its forward path. Test whether this simple patch encoder captures enough topology.
5. **Inactive paths.** \`Tokenizer\` and \`TemporalAE\` construct \`Patcher\` objects but do not call them in their current forward paths. With the supplied \`use_adj: true\` configuration, \`GraphGeneratorAttn\` is bypassed. \`BackBone\` reads \`valid_N\` and defines \`ln\`, but neither is used in its current forward path.
6. **Nested residual path.** Each \`QAAttentionLayer\` applies residual updates internally; \`BackBone.backbone_forward\` then adds the layer input to the returned node tokens again. The added \`backbone_residual_stats.csv\` measures the scale of this outer residual separately.
7. **Training/evaluation node scope differs.** \`STUNet.forward\` applies \`target_node_mask\` only when \`self.training\` is false. Training loss therefore covers the sampled subgraph; validation/test prediction is restricted to target nodes. The logs record masking mode and training target/context errors.
8. **Zero-target masking.** \`MAELossMask\` excludes positions where \`y_true == 0\`. Diagnostic metrics separately report zero-target and non-finite ratios.
9. **Global normalization.** The LargeST datamodule computes one scalar mean and standard deviation from all training times and nodes, rather than per-sensor statistics.
10. **Metric aggregation.** The backbone trainer logs per-batch NumPy metrics. The diagnostic CSVs retain batch-level and horizon-level values so metrics can be recomputed over concatenated valid predictions rather than averaging batch metrics blindly.
11. **Machine-specific config.** The supplied trainer config uses \`devices: [6]\`; set this to an available GPU. The upstream README specifies Python 3.10.16 and PyTorch 2.6.0; check compatibility before installing on a different environment.

## Questions the diagnostics are intended to answer

- Is spatial pretraining learning useful topology or mainly reproducing adjacency sparsity?
- Are graph tokens and node tokens on comparable scales, and do either collapse?
- Does query attention become overly concentrated on a few graph patches or too diffuse to select useful spatial context?
- Does aggregate attention become dominated by diagonal/self-attention?
- Does the extra outer residual amplify representations layer by layer?
- Does temporal pretraining show a consistent long-horizon error increase?
- Are gradients missing, non-finite, or imbalanced across temporal tokenizer, attention backbone and prediction head?
- Are errors concentrated on particular target nodes, context nodes, or high-value traffic regimes?
- Does the training-versus-evaluation node-mask difference materially change the optimization target?

Attention entropy, residual ratios and reconstruction scores are diagnostic signals, not standalone proof. Confirm any suspected issue with multiple seeds and controlled ablations.
