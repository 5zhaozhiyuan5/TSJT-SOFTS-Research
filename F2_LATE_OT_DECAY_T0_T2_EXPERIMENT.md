# T0/T1/T2 — F2 迁移后期 OT 强度调度实验（2026-11-26）

## 1. 研究问题与假设

H3 检验了原始 F2 Stage-I 的 epoch 2899 checkpoint，在 PEMS-BAY 目标三天支持集上进行 8 组**全参数 PCGrad+真实 AdamW 单步影子更新**。在该特定检查点附近，所有城市关闭 OT 的平均单步探针 SmoothL1 收益为 +0.000623（7/8 正值）；单城 F2→均匀权重整体改善更小。此结果是**同一支持集、同一 checkpoint、单步**的局部现象，不能说明训练从头到尾都不应使用 OT，也不能替代测试集 MAE。

明确提出一个可证伪假设：

> 在模型已学得初步可迁移的交通表示后，固定强度的跨城市 OT 可能继续推动表示对齐，而这种后期对齐对目标预测不再必要，甚至产生优化竞争。前期保留 F2 样本筛选与 OT，中后期降低 OT 的任务影响，可能改善最终 Stage-II 预测精度。

**这一次不是新算法的论文成果，而是先验证因果位置的严格对照。** 如果固定线性退火有效，才考虑后续“目标任务反馈控制强度”的新机制。

## 2. 三组独立配对训练

研究中唯一被改变的变量：Stage-I 的 OT 损失系数随 epoch 的变化。原始 F2 alpha、均匀 beta、SOFTS、官方 GeomLoss OT 代价、目标分组、源城市采样、单侧 PCGrad、AdamW、官方 Stage-II 均保持不变。

令原始实际 OT 损失系数为 \(\lambda_0=\mathrm{DAcoef}\times\mathrm{PMME\_TRANSFER\_OT\_SCALE}=1\times0.01\)，OT 自 epoch 1500 起启用，Stage-I 共 3000 个 epoch（0–2999）。

衰减进度：

\[
r(e)=\operatorname{clip}\!\left(\frac{e-2000}{2999-2000},0,1\right).
\]

以下均指 \(e\geq1500\)；之前三组均无 OT：

- **T0**：\(\lambda(e)=0.01\)，复现完整固定 F2。
- **T1**：\(\lambda(e)=0.01[1-r(e)]\)，epoch 2000 前不衰减，到 epoch 2999 正好为 0。
- **T2**：\(\lambda(e)=0.01[1-0.75r(e)]\)，epoch 2000 前不衰减，到 epoch 2999 正好为 0.0025。

| epoch | T0 固定 | T1 后期退至 0 | T2 后期退至 0.0025 |
|---:|---:|---:|---:|
| 1499 | 0（尚未启用 OT） | 0 | 0 |
| 1500 | 0.0100 | 0.0100 | 0.0100 |
| 2000 | 0.0100 | 0.0100 | 0.0100 |
| 2499 | 0.0100 | 约 0.005005 | 约 0.006254 |
| 2999 | 0.0100 | **0** | **0.0025** |

epoch=2000 与 epoch=2999 的选择是根据现有训练范围事先预设的消融，不是 H3 自动推断的最优时刻。不要按测试 MAE 再修改时间参数。

## 3. 实际代码变更

- `model/Meta_Models/f2_ot_schedule.py`：独立纯数学调度函数；T0 保持原始 F2。
- `model/Meta_Models/PMME.py`：初始化调度配置，并且每个 Stage-I epoch 在已有 OT 损失的系数位置**仅替换实际 OT 缩放值**；后期 T1=0 时直接跳过昂贵的 GeomLoss OT，其他训练分支无变动。
- `run_f2_ot_decay_experiments.sh`：相同 seed、配置与训练入口运行 T0/T1/T2，各自隔离工作目录、checkpoint 和日志；确认 Stage-I/Stage-II 完成才标记 COMPLETED。
- `verify_f2_ot_schedule_log.py`：检查 epoch 0、1500、2000、2999 等实际日志与理论调度一致；失败不标记完成。
- `summarize_f2_ot_decay.py`：从原始官方训练日志中读取**按目标三天支持集验证分数选出的最佳 checkpoint**，提取对应测试集 36-step MAE、10/60/120/180 分钟 MAE/RMSE；给出 seed 对应的 T1/T2−T0，**负数代表改善**。
- `tests/test_f2_ot_schedule.py`、`tests/test_f2_ot_schedule_reporting.py`：无 GPU 数学边界测试、日志检验、最佳 checkpoint 解析测试。

**明确不修改：** `train_DA_OT.py`、`model/Meta_Models/PMME_2ndStage.py`、`model/Meta_Models/transfer_utility.py`、SOFTS、Sinkhorn 和原始 F2 权重评分。

## 4. 运行方法

在已有三座源城市及 PEMS-BAY 数据、PyTorch/GeomLoss/GPU 环境的服务器上：

```bash
git fetch origin
git switch experiment/pmme-f2-late-ot-decay-t0t2-20261126

python -m unittest discover -s tests -p 'test_f2_ot_schedule.py'
python -m unittest discover -s tests -p 'test_f2_ot_schedule_reporting.py'
```

先以**独立目录**进行 4 轮 Stage-I 小规模冒烟验证，含一次正常 OT（e1）、开始衰减（e2）、T1 关 OT（e3）：

```bash
PMME_T_RUN_ROOT=./experiments/f2-t-smoke \
PMME_T_STAGE1_EPOCHS=4 PMME_T_STAGE2_EPOCHS=0 \
PMME_T_DAEPOCH=1 PMME_T_DECAY_START=2 PMME_T_DECAY_END=3 \
PMME_T_VARIANTS='t0 t1 t2' PMME_T_SEEDS=1 \
bash run_f2_ot_decay_experiments.sh
```

正式三组 seed1 完整训练（同样 Stage-I=3000，官方 Stage-II=6000）：

```bash
PMME_T_VARIANTS='t0 t1 t2' PMME_T_SEEDS=1 \
bash run_f2_ot_decay_experiments.sh
```

在核查 seed1 的训练轨迹和对比结果后，如果 T1/T2 相比 T0 有一致或有意义的提升，**不要改退火超参数**，直接补不同 seed：

```bash
PMME_T_VARIANTS='t0 t1 t2' PMME_T_SEEDS='3 5' \
bash run_f2_ot_decay_experiments.sh
```

如只做 Stage-I 快速筛选：

```bash
PMME_T_RUN_ROOT=./experiments/f2-t-stage1-only \
PMME_T_STAGE2_EPOCHS=0 \
bash run_f2_ot_decay_experiments.sh
```

Stage-I-only 输出不能替代完整 Stage-II 最终比较。运行脚本默认按顺序执行，避免并行占用过高显存。

## 5. 输出和日志

各实验独立目录：

```text
experiments/f2-late-ot-decay/t0/seed1/
experiments/f2-late-ot-decay/t1/seed1/
experiments/f2-late-ot-decay/t2/seed1/
```

每组包含：`train.out`、`run_config.txt`、`schedule_check.txt`、Stage-I .pth checkpoint、`logged_files`、`COMPLETED` / `FAILED`。Stage-II 原有 checkpoint/轨迹按作者训练脚本方式保存。

每 25 epoch 日志包含：

```text
f2_ot_schedule epoch ... policy ... phase ... base_scale ...
effective_scale ... DAcoef ... effective_ot_coefficient ...
```

自动结果：

```text
experiments/f2-late-ot-decay/selected_checkpoint_metrics.csv
experiments/f2-late-ot-decay/matched_deltas_vs_t0.csv
```

相同 seed 下，优先比较 Stage-II 全 36 步平均 MAE、10/60/120/180 分钟 MAE/RMSE，同时记录 Stage-I 最佳验证 epoch 和收敛轨迹。查看是否只改善极个别预测步而损害其余预测步。

### 合格证据与停止规则

1. **实验先验**：仅 T0/T1/T2、固定 start=2000、end=2999、尾系数=0/0.0025。禁止根据同一测试集继续试探几十种衰减超参数。
2. 先确认 T0 的指标能复现历史同 seed F2（允许因运行环境差异出现的小幅波动，但必须比较配置/commit）；T0 不对，优先排障而不是宣布 T1/T2 提升。
3. T1/T2 在 seed1 的 36-step 平均 MAE 不劣且多个预测步/时长有改善时，补 seed3/5 并报告三 seed 均值、标准差及每 seed 配对差值；如果 seed1 无改善，则不能把 H3 单步影子结论当成训练结论。
4. 即使 T1/T2 提升，**线性衰减 OT 系数本身很常见，不能直接作为论文创新**；只能作为“后期固定对齐过强”机制有效的实验证据，再研发有针对性的闭环控制。
5. 使用目标支持验证分数选择 checkpoint，目标测试集只作最终离线报告；注意原训练脚本每 100 epoch 仍会打印测试指标，不应据此挑选最佳候选或调整 schedule。
6. H3 checkpoint 单步探针使用的是见过的支持集，不能与这里的测试集 MAE 在数字上直接比较。

此分支提交的是**代码和实验设计**，未在用户服务器上执行实际 GPU 训练，不存在 T0/T1/T2 的新结果。
