# MTPB Time-Pattern Guided Gradient Transfer Research

Cross-city few-shot traffic forecasting based on the official MTPB architecture, integrating one-stage target-guided source-gradient transfer and five-scale pattern-retrieval confidence.

**Main research entry: MTPB one-stage guided transfer** (`scripts/run_mtpb_one_stage.py`). It uses the official MTPB Pattern Bank, pattern retrieval/aggregation, reconstructed graph, Graph WaveNet and long-term branch (288→36). The older TSJT Graph WaveNet + PGrad runner remains in the repository as a legacy comparison path; it is not the MTPB research entry.

**Status:** MTPB one-stage code and retrieval-entropy guidance are now on `main`. Synthetic tests are not evidence of real traffic improvement. No verified four-city MAE/RMSE gain is available yet.

**External dependency:** This package is not standalone: training imports the original `tsjt_c1` engine and expects separate datasets.

- [MTPB one-stage protocol](docs/MTPB_ONE_STAGE_GUIDED_TRANSFER.md)
- [Pattern retrieval entropy × gradient guidance](docs/MTPB_RETRIEVAL_ENTROPY_GRADIENT.md)
- [MTPB one-stage config](configs/mtpb_one_stage_target_guided.yaml)
- [Source interaction implementation](src/tsjt_pgrad/pgrad/source_interaction.py)
- [Legacy Graph WaveNet migration notes](docs/GRAPH_WAVENET_MIGRATION.md)

```bash
pip install torch numpy pyyaml pytest
PYTHONPATH=src pytest -q tests/test_mtpb_retrieval_confidence.py tests/test_mtpb_one_stage_gradient.py tests/test_mtpb_one_stage_eval.py tests/test_mtpb_train_only_norm.py

# Requires a separate checkout of the official MTPB repository and its data:
PYTHONPATH=src python scripts/run_mtpb_one_stage.py --upstream /path/to/MTPB --mode target_guided --epochs 1 --order-count 1 --retrieval-guidance-strength 1 --skip-eval --output outputs/mtpb_smoke
```

Graph WaveNet reference: Wu et al., IJCAI 2019, https://github.com/nnzhan/Graph-WaveNet . This node-count-agnostic implementation differs from the original fixed node-embedding adaptive adjacency.

See historical SOFTS notes in [legacy guide](README_SOFTS_%E6%BA%90%E5%9F%8E%E5%B8%82%E4%BA%A4%E4%BA%92%E6%94%B9%E8%BF%9B.md).

## Main research method: MTPB one-stage guided transfer

**Experimental**: [MTPB one-stage protocol](docs/MTPB_ONE_STAGE_GUIDED_TRANSFER.md).

The new `scripts/run_mtpb_one_stage.py` uses the original MTPB
Pattern Bank + reconstructed graph + Graph WaveNet + long-term branch
to predict **288→36**. METR-LA/Chengdu/Shenzhen source tasks determine
the parameter update, while **PEMS-BAY's first three training days only guide
source gradient selection, rotation, residual fusion, and angle feedback**.

**There is no target fine-tuning stage.** The source-trained model is
evaluated directly on the held-out target set; target test labels are never
used during training or checkpoint selection. The offline source pattern-bank
pretraining prerequisite is still present, so "one-stage" describes the
**target adaptation protocol**, not a single end-to-end run from raw data.

Promoted to `main` on 2026-11-10 from `iteration/20261110-mtpb-retrieval-entropy-gradient`.
The original TSJT Graph WaveNet + PGrad implementation remains as a legacy comparison path.
No verified four-city MAE/RMSE improvement yet.
