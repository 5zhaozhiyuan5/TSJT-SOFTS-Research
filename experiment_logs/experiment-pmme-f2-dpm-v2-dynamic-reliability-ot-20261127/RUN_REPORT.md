# experiment/pmme-f2-dpm-v2-dynamic-reliability-ot-20261127

运行日期：2026-11-27  
代码提交：`bc2ab8624044f71cdd7531eff0c70a443497ec87`  
目标城市：PEMS-BAY  
源城市：Chengdu、Shenzhen、METR-LA  
模型：SOFTS；seed：1  
运行配置：V2-A strength=0.02；V2-B strength=0.35；两者 Stage I=3000、Stage II=6000、batch=16。

## 正式实验结果

指标来自 support-selected checkpoint；MAE/RMSE 为 36 个预测 horizon 的平均值。

| 变体 | 阶段 | 最佳 support epoch | mean36 MAE | mean36 RMSE | MAE 10 min | MAE 60 min | MAE 120 min | MAE 180 min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V2-A | Stage I | 2999 | 2.447093 | 5.349049 | 1.43368 | 2.35778 | 2.75969 | 3.01980 |
| V2-B | Stage I | 2899 | 2.445798 | 5.316306 | 1.44882 | 2.36694 | 2.77468 | 2.97450 |
| V2-A | Stage II | 5199 | 2.406367 | 5.300607 | 1.37554 | 2.33591 | 2.72295 | 2.96813 |
| V2-B | Stage II | 5999 | 2.403090 | 5.289962 | 1.39190 | 2.33554 | 2.71557 | 2.94479 |

## 运行核验

- V2-A、V2-B 的 Stage I 与 Stage II 均有完成标记，runner 退出码为 0。
- 分支日志在 Stage II 中仍显示 `Epochs x/3000`；训练循环实际使用 `finetune_epochs=6000`，并写出 `2ndStage Source training Ending!!!`。
- 烟测 V2-B（Stage I 3 epoch、Stage II 0）通过，仅用于确认路径与诊断输出，不纳入正式结果表。
- 两次失败的烟测及其日志未同步。同步内容不包含模型 checkpoint。
- 旧汇总器只认 V1 的 `dpm_pair_cost`，无法识别 V2 的 `DPM_V2_DIAG`。本次仅在本地/服务器工作副本中临时放宽汇总器识别条件；训练模型代码未改，临时改动未提交、未推送到远程代码分支。
- 本结果为单 seed。V2-A/V2-B 来自同一代码提交，可作本次配对实验观察；与其他提交的历史结果不构成严格配对，也未在此处与论文数值比较。

## 日志目录

本地完整日志：`D:\模型日志\experiment-pmme-f2-dpm-v2-dynamic-reliability-ot-20261127\logs\runs`  
分支日志：`experiment_logs/experiment-pmme-f2-dpm-v2-dynamic-reliability-ot-20261127/`
## 服务器状态（实验完成时，2026-11-27 16:29 CST）

- RTX 4090：总显存 24,564 MiB；占用 8,997 MiB，空闲 15,220 MiB；GPU 利用率 50%，温度 58°C。
- 唯一 GPU 进程为 Graph-Mamba PID 3930810，占用约 8,984 MiB；本次 PMME 训练已退出，没有仍在运行的 PMME 训练进程。
- 系统负载约 0.10；内存可用约 103 GiB；根分区剩余 86 GB（使用率 91%）。