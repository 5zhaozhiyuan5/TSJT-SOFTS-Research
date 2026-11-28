# PMME F2 + PRC — seed1 run logs

- Branch: `experiment/pmme-f2-prc-reliability-20261128`
- Executed commit: `b93d0a1234e43cb6fdb8fd85e110fb2f5312c23d`
- Dataset/model: PEMS-BAY / SOFTS; sources Chengdu, Shenzhen, METR-LA
- Seed: 1; batch size: 16
- Settings: Stage I 3000 epochs, Stage II 6000 epochs, DAepoch 1500, PRC blend 0.5, sensitivity 0.5, z clip 3.0, multiplier bounds [0.5, 2.0].
- Status: Stage I and Stage II completed successfully; runner exit code 0.
- PRC was enabled and logged 180 diagnostic records. The historical original-F2 control was not rerun, per the branch design; its same-seed result is used for comparison.
- `prc/seed1` contains `train.out`, detailed `logged_files/.../log.txt`, configs, `run_config.txt`, and `COMPLETED`. The model checkpoint is excluded.
- Server reused the pre-existing PMME environment; no packages or data were downloaded on the server.
- See `comparison_to_original_f2_seed1.md` for MAE/RMSE comparison.
