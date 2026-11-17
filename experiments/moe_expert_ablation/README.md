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

From the repository root:

```bash
bash scripts/run_moe_expert_ablation.sh
```

Defaults match the current METR-LA target setup:
- target: `metr-la`
- sources: `pems-bay`, `chengdu_m`, `shenzhen`
- target few-shot period: first 3 days
- seed: 1
- Stage-I: 3000 epochs
- Stage-II: 6000 epochs
- DA start epoch: 1500

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

Because earlier runs used full-sequence target statistics, rerun the `learned`
control on this branch. Do not directly treat the old learned-MoE score as a
strictly matched control for the four fixed-gate arms.

## Diagnostics

Every run records `router_mode` in `run_metadata.json`, `moe_router.csv`,
and `ot_diagnostics.csv`. Compare the four horizon-specific router weights,
conditional OT losses, plan diversity, expert costs, and Stage-II evaluation
metrics. For fixed modes, router weights are the imposed gates, not learned outputs.
