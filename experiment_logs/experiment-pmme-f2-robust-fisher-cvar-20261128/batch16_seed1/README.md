# PMME F2 robust Fisher-CVaR — seed1 run logs

- Requested branch: `experiment/pmme-f2-robust-fisher-cvar-20261128`
- Executed source commit: `e640c78063710810c12310a9642b3203aeb375d1`
- Dataset/model: PEMS-BAY / SOFTS
- Sources: Chengdu, Shenzhen, METR-LA
- Batch size: 16; seed: 1
- Methods in the executed commit: `original`, `geomed`, `influence`, `cvar`, `triple`
- Training: Stage I 3000 epochs, Stage II 6000 epochs, DAepoch 1500, OT scale 0.01
- Status: all five runs completed successfully (`COMPLETED` markers; automation exit code 0)
- Each method/seed folder contains `train.out`, detailed `logged_files/.../log.txt`, config files, and completion marker. Model checkpoints are excluded from this log archive.
- Server training used the pre-existing PMME environment; no packages or data were downloaded on the server.
- Provenance note: the remote branch now points to `0a80ced`; later commits removed the `original` and combined `triple` runs. This archive records the exact earlier source commit `e640c78` that was used for these runs, not the updated branch HEAD.
