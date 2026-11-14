# PMME Stage-I research records

Only one Git branch is used for this research: `pmme-stage1-research`.
Experiment versions are directories, not branches.

Each server run is stored under:

```text
/root/autodl-tmp/pmme_stage1_research/experiments/<version>/
  stage1/
    config.json
    log.txt
    diagnostics.jsonl
    stage1_best.pth
    stage1_metrics_seed1.json
  stage2/
    log.txt
    stage2_best.pth
    stage2_metrics_seed1.json
  comparison_to_v00.json
```

The primary comparison uses the four paper horizons: indices 1, 11, 23 and
35, corresponding to 10, 60, 120 and 180 minutes.  MAE and RMSE are
compared separately against both the same-protocol official `SOFTS+PMME`
baseline and the four values reported in the paper.  The comparison JSON
records both win counts.

The stopping rule is strict: a candidate must improve both MAE and RMSE on a
majority of the four paper horizons against the paper reference.  The
same-protocol v00 comparison is retained as a reproducibility diagnostic.  A
Stage-I-only result is diagnostic and cannot be called a final win until it
has passed the unchanged official few-shot Stage-II.
