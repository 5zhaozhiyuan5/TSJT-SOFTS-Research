# MTPB 一阶段目标引导实验（未验证）

本实验分支从此前 MTPB 集成实验分支继续开发，**不是 main 已验证功能**。main 的默认骨干仍是 Graph WaveNet；这里使用官方 MTPB 的 PatchFSL（包含 Graph WaveNet），不回退到 SOFTS。

## 研究假设

在目标城市 PEMS-BAY 仅前三天有标签数据的情况下，将这些数据用于源梯度方向、源城市权重、旋转角度和残差匹配的训练引导，**不再单独使用目标城市微调阶段**，是否仍能获得有竞争力的 36 步预测？这是待检验假设，不是已获得的性能结论。

## 与上一版的唯一训练协议变化

保留官方 MTPB 的模式库、图重建、Graph WaveNet、长期分支、288→36 预测，以及现有 Support/Query 元梯度、PGrad 源城市交互、目标引导旋转、角度元学习和残差融合。三个源城市 METR-LA、成都、深圳提供预测梯度；PEMS-BAY 前三天数据只参与梯度引导，包括用于更新旋转角度控制器，**不进行额外的目标微调**。因此“只引导梯度”不等于目标标签完全不影响训练参数：目标标签仍通过梯度选择和角度元学习间接影响模型。

训练结束直接保存最后一次源域训练参数并在 PEMS-BAY 测试集评估，不依据目标测试 MAE/RMSE 选择 checkpoint。旧的两阶段模式仍可不带 `--one-stage` 运行作对照。

## 运行（需官方 MTPB 数据和模式库）

```bash
PYTHONPATH=src python scripts/run_mtpb_integrated.py \
  --upstream /path/to/official/MTPB \
  --mode integrated --one-stage \
  --epochs 50 --target-days 3 \
  --output runs/mtpb_one_stage_seed7
```

要求官方 MTPB checkout、原始数据、模式库以及对应依赖。输出 `run_protocol.json`、`diagnostics.jsonl`、`one_stage_model.pt`、`one_stage_test_metrics.json`。36 个 horizon 指标为各测试批次的平均指标，不能在没有真实数据运行的情况下填入数值。

## 重要限制与后续验证

- 这仍沿用 MTPB 事先预训练编码器并生成模式库的准备阶段；所谓“一阶段”仅指**源域梯度训练与目标引导合为一次训练过程、取消目标微调**，不是从原始数据到模式库也完全端到端。
- 目标引导使用 PEMS-BAY 前三天的监督标签，并非零样本。
- 官方 MTPB 的图重建仍使用城市原始邻接矩阵；**本实验不满足严格无外部拓扑的 Graph WaveNet 迁移配置**，后续需要独立处理，不能称其为无拓扑版本。
- 本次只完成源码提交和静态协议测试编写，未运行真实四城市训练，也未得到 MAE/RMSE。
- 下一轮需跑 `pytest -q tests/test_mtpb_one_stage_runner.py tests/test_mtpb_integrated_meta.py`，再运行相同种子、相同模式库的一阶段/原两阶段对照。至少记录 10/60/120/180 分钟 MAE/RMSE 和训练时间。
