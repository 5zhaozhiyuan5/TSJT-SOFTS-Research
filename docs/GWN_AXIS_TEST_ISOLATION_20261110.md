# Graph WaveNet 维度与测试隔离修复（2026-11-10）

基线：`main` @ `a4f72961cb26e8204b9b12587207c0998a0d04c9`，没有把此前未验证的梯度门控实验混入本轮修复。

## 本轮结论

完整读取 main 的 41 个文本文件（包含 16 个 YAML、源码、测试、README、历史文档）。
main 没有提交可核验的真实四城训练输出或 MAE/RMSE。旧文档 60min
MAE≈4.913 / Persistence≈3.104 属于历史诊断，不是当前 Graph WaveNet 结果。
当前训练依赖外部 tsjt_c1 工程及四城市原始数据，不能用合成测试代替实测。

发现两个会妨碍真实模型验证的确定性缺陷：

1. `GraphWaveNetBackbone.forward` 中，`end_conv_2` 的结果为
   `[B,canonical_steps,N,1]`（在 `[..., -1]` 后为
   `[B,canonical_steps,N]`）；多余的 `permute(0,2,1)`
   把时间轴和节点轴交换，随后沿节点轴做 6/12 horizon 切片。
   删除这次转置；新增不同节点数量及 6/12 物理时间的回归测试。
2. `tests/conftest.py` 在导入时把
   `tsjt_pgrad.models.pgrad_model` 全局塞进 `sys.modules`
   替换成 ToyModel，使真正 Graph WaveNet 包装器的测试不再测试真实代码。
   仅移除这一全局替换，保留其他 ToyModel 供旧测试显式使用。

本轮属于 **验证基础设施修复**，不把它宣传成新的梯度创新。
保持原有源城市梯度交互、目标引导旋转、角度元学习、残差融合原样。
后续目标一致性协同门控（单一可证伪假设）应在这个修复通过之后
从最新 main 单独分支实验，比较门控强度 0 和 1，不能把模型修复收益
算作门控收益。

## 数据协议

源：METR-LA（5min 12→12）、成都/深圳（10min 6→6）；
目标：PEMS-BAY（5min 12→12，训练前三天）。
6 步 10min 预测应等价于统一 5min 的 12 步预测取
索引 `[1,3,5,7,9,11]`，覆盖相同的 10/20/.../60min 物理时长。
Graph WaveNet 支持来自历史序列，不增加道路拓扑、POI、LLM
或固定节点嵌入；不使用目标测试标签调参。

## 相关论文及官方仓库

- Graph WaveNet, IJCAI 2019，https://github.com/nnzhan/Graph-WaveNet。
  采用扩张卷积和扩散图卷积；本仓库动态支持改为历史信号相似度，
  不是官方自适应节点嵌入的逐字复现。
- PGrad, ICLR 2023，https://github.com/QData/PGrad。
  顺序虚拟优化与主梯度提取；本轮不改变轨迹算法。
- MTPB, ACM TKDD 2025，https://github.com/zhyliu00/MTPB。
  跨城市多尺度模式库方法；本轮不引入它的原始道路图重建模块。

## 验证

新增 `tests/test_gwn_horizon_axis_regression.py`，
并用 GitHub Actions `gwn-axis-regression.yml` 执行全量
`PYTHONPATH=src python -m pytest -q tests/`。
CI 状态以 Actions 实际记录为准，合成通过不代表真实 MAE/RMSE。
正式实验仍需外部 tsjt_c1 + 四城数据，记录同 seed 的
10/60min MAE/RMSE、6/12 时间轴对应、源目标梯度余弦和训练耗时。

## CI 回归失败的定位（追加）

首次 Actions run 42679860161 返回 **55 passed, 1 failed**。
唯一失败为 `test_gradient_and_virtual_forward`：
`torch.autograd.grad(...)` 默认 `allow_unused=False`，而模型最后
一层扩张卷积在最终 skip 汇总后还有一个 post-skip graph/residual 路径，
其参数对本次预测损失没有计算图依赖。正式 trainer 已使用
`allow_unused=True` 并对 None 梯度补零；因此测试需与正式梯度路径一致。
已在 `tests/test_graph_wavenet_backbone.py` 加入同样处理，下一轮
仍需单独评估最后一层 post-skip 参数是否应该从优化器排除以减少浪费，
但本轮不更改架构，避免混淆维度修复的因果效果。
