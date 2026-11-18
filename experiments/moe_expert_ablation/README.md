# MoE expert-validity ablation

This experiment compares five Stage-I routing modes while keeping the MTPB pattern
bank, DVGP sample weights, conditional OT, SOFTS backbone, target few-shot protocol,
and official Stage-II implementation unchanged.

## Arms

| Mode | Routing gate | Meaning |
|---|---|---|
| `learned` | learned pair-conditioned softmax | Dynamic MoE control |
| `uniform` | `[1/3, 1/3, 1/3]` | Equal contribution from all three experts |
| `expert1` | `[1, 0, 0]` | Only 1h/3h pattern-cost group |
| `expert2` | `[0, 1, 0]` | Only 6h/12h pattern-cost group |
| `expert3` | `[0, 0, 1]` | Only 24h pattern-cost group |

The fixed modes bypass the learned router. Balance loss is disabled for fixed modes
because it cannot change a fixed gate. FiLM horizon conditioning and its smoothness
term remain active in every arm.

## Run

From the repository root (default: only the four NEW fixed-gate arms, no repeated learned run):

```bash
bash scripts/run_moe_expert_ablation.sh
```

For a strictly matched five-arm study under the corrected normalization protocol:

```bash
MODES="learned uniform expert1 expert2 expert3" bash scripts/run_moe_expert_ablation.sh
```

For a single arm, use e.g. `MODES="expert2" bash scripts/run_moe_expert_ablation.sh`.

Defaults match the current METR-LA target setup:
- target: `metr-la`
- sources (historical order): `chengdu_m`, `shenzhen`, `pems-bay`
- target few-shot period: first 3 days
- seed: 1
- Stage-I: 3000 epochs
- Stage-II: 6000 epochs
- DA start epoch: 1500
- `DAcoef=0.01`, `xySquaredRatio=100`, `projection=1`, `MemSize=10000` (matched to the recorded previous run)

Override defaults through environment variables, for example:

```bash
TEST_DATASET=pems-bay DATA_LIST=metr_chengdu_shenzhen SEED=2 GPU=1 bash scripts/run_moe_expert_ablation.sh
```

Each arm writes diagnostics and its Stage-I checkpoint into a separate run directory.
The Stage-I checkpoint filename includes the router mode, preventing one arm from
overwriting another arm's checkpoint.

## Important protocol change

This branch sets `data.target_normalization_scope: target_train`. For the target
city, normalization mean/std are computed only from the first `target_days` of
observations and reused for target training and test windows. Source-city
normalization remains based on the full source series, as before.

Earlier runs used full-sequence target statistics. To save GPU time, the default
runner skips the already completed learned arm, but the previous learned-MoE
score is **historical context only**, not a statistically matched control for
fixed-gate arms with corrected normalization. For a valid head-to-head
comparison, run the learned arm under this branch's new data protocol.

The shell runner also explicitly passes the historical `DAcoef`,
`xySquaredRatio`, `projection` and `MemSize` flags, rather than relying on
training-script defaults which differ from the previous experiments.

## Diagnostics

Every run records `router_mode` in `run_metadata.json`, `moe_router.csv`,
and `ot_diagnostics.csv`. Compare the four horizon-specific router weights,
conditional OT losses, plan diversity, expert costs, and Stage-II evaluation
metrics. For fixed modes, router weights are the imposed gates, not learned outputs.
