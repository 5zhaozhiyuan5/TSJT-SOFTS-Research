# PMME F2/P1–P4 batch=16 run

- Code branch: `experiment/pmme-f2-p1-p4-gradient-ot-20261125`
- Code commit: `1f6fc5f53d73c296079c858a66312601ce2c6001`
- Run source: server `source_patched`, original `batch_size: 16`
- Dataset: PEMS-BAY; training domains Chengdu, METR-LA, Shenzhen
- Variants: P2, P4, P1, P3; seed1 only; F2 excluded
- Stage I: 3000 epochs; Stage II: 6000 epochs
- Completion: all four variants completed both stages on 2026-11-26 12:49 CST
- Source compatibility patch: `geoloss-symmetric-cost-v1-6312734f4adeb30a`

Each variant folder contains `train.out.gz` (full console log), a timestamped `logged_files/.../log.txt.gz` (training metrics), plaintext `run_config.txt`, `runner.log`, and `batch_override.txt`, plus a `COMPLETED` marker. Decompress a log with `gzip -dk train.out.gz`. `SHA256SUMS.txt` lists hashes of the original uncompressed log files.

The `queue_status.log` and `run_summary.tsv` record the sequential run and completion markers. No model checkpoints or dataset files are included.
