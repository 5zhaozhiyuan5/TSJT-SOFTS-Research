# PRC seed1 实验结果与论文比较

- 分支：`experiment/pmme-f2-prediction-reliability-20261126`
- 训练代码提交：`e27888bfff1d527b443a533d88fecd4e90148517`
- 方法：原始 F2 权重后增加 Prediction Reliability Calibration（PRC），strength=0.5；不重复训练 F2。
- 模型：SOFTS；源城市：METR-LA、Chengdu、Shenzhen；目标城市：PEMS-BAY；seed1；batch size 16。
- 正式训练：Stage I 3000 轮，Stage II 6000 轮，DAepoch=1500。训练完成标记存在，汇总脚本确认 Stage I/II 完成。
- Stage II checkpoint 按 PEMS-BAY 三天支持集验证分数选取：epoch 5199。
- 论文参照：Table 2 的 `SOFTS +OT+Ycat+DVGP+Proj+RM` 完整 PMME 行。论文指标显示两位小数，差值及百分比为近似值。
- 差值为本次结果减论文值；正数表示本次误差较高，负数表示较低。

| 预测时长 | MAE：本次 / 论文（差值） | RMSE：本次 / 论文（差值） |
|---|---:|---:|
| 10 分钟 | 1.38537 / 1.39（−0.00463，−0.33%） | 2.71076 / 2.70（+0.01076，+0.40%） |
| 60 分钟 | 2.33070 / 2.32（+0.01070，+0.46%） | 5.20854 / 5.18（+0.02854，+0.55%） |
| 120 分钟 | 2.69847 / 2.70（−0.00153，−0.06%） | 6.03388 / 6.02（+0.01388，+0.23%） |
| 180 分钟 | 2.93745 / 2.95（−0.01255，−0.43%） | 6.46195 / 6.50（−0.03805，−0.59%） |

## 结果说明

PRC 本次 Stage II 的 36 步平均 MAE 为 2.39441。分支汇总脚本还与历史 F2/T0 seed1 的 2.372713 作了参考比较，PRC 高 0.021698（约 0.91%）。该历史数值来自不同代码提交和运行，**不是同一提交下配对的因果对照**。本次只有一个 seed，且论文值四舍五入，不能据此判断统计显著性。

## 日志与产物

- 原始训练日志：`formal/prc/seed1/train.out`。
- 训练器日志：`formal/prc/seed1/logged_files/pems-bay/chengdu_shenzhen_metr/2026-11-26-21-44-35/log.txt`。
- 实验配置和完成标记：`formal/prc/seed1/run_config.txt`、`formal/prc/seed1/COMPLETED`。
- 选中 checkpoint 的汇总指标：`formal/selected_checkpoint_metrics.csv`；launcher 汇总：`formal/formal_launcher.out`。
- 论文引用值来源：Table 2；本地既有核对记录 `experiment-official-pmme-dvgp-only-20261123/horizon/pems-bay/SOFTS/seed1/comparison_to_paper.md`。
