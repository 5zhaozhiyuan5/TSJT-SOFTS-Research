# 2026-11-10 · Graph WaveNet 目标一致性协同门控

## 仓库基线审计

本轮以 `main@a4f72961cb26e8204b9b12587207c0998a0d04c9` 为唯一基线，
通过 GitHub 读取 main 树中全部 **41 个文本文件**（源码、配置、测试、README、历史文档、requirements、gitignore）。
main 没有训练结果文件或真实四城市日志；README 和 SOURCE_COMPATIBILITY_CHECK 明确表示没有验证 Graph WaveNet 四城市 MAE/RMSE。
旧历史文档中曾出现 `60min MAE≈4.913` 和 Persistence `≈3.104`，
那是旧版模型诊断，不能冒充本轮 Graph WaveNet 实测。
已知风险：SOFTS 老 checkpoint 不兼容；真实 `tsjt_c1` 依赖和数据不在本仓库；
旧配置仍残留 SOFTS 描述，但实际默认预测器是 Graph WaveNet。
旧版 `gamma_source` 和源预算的对照存在不等超参风险。

## 论文与源码核查

1. **Graph WaveNet**, IJCAI 2019，官方 https://github.com/nnzhan/Graph-WaveNet。
   原始方法使用扩张时间卷积和图扩散卷积以及自适应邻接。
   本仓库采用从历史速度相关性构建动态图的跨城节点数无关变体，
   **不是**原论文固定节点嵌入的精确复现。
2. **PGrad: Learning Principal Gradients for Domain Generalization**, ICLR 2023，
   官方 https://github.com/QData/PGrad。PGrad 从顺序虚拟优化轨迹中提取主梯度。
   本仓库保留城市顺序平衡和虚拟梯度重新计算。
3. **Multi-scale Traffic Pattern Bank for Cross-city Few-shot Traffic Forecasting (MTPB)**,
   ACM TKDD 2025，官方 https://github.com/zhyliu00/MTPB。
   其多尺度模式迁移和 Graph WaveNet 可作为跨城背景对照；
   MTPB 原版涉及模式库和图重建，不能直接宣称符合本项目的
   “不增加道路拓扑”条件。本轮没有将 MTPB 骨干替换进 main。

## 本轮单一可证伪假设

v2 源城市交互奖励 `cos(g_i,g_j)>0` 的源城市正协同，
即使两源都与目标梯度相冲突，也会获得协同加分。
**假设**：只奖励同时对目标具有正梯度对齐的源城市对，
可减少错误的源域协同，在目标 PEMS-BAY 三天训练条件下降低
10/60min 以及全预测步 MAE/RMSE。

设 `c_i=cos(g_i,g_T)`，`r_ij=max(cos(g_i,g_j),0)`，
`a_i=clip(max(c_i,0),0,1)`，则：

```text
synergy_i(lambda) =
  1/(S-1) * sum_{j != i}
    min(r_ij, redundancy_threshold)
    * [(1-lambda) + lambda * a_i * a_j]

lambda=0 : 与 main 的 v2 源协同计算一致
lambda=1 : 完全目标一致性门控（本轮实验配置）
```

只改变 `source_interaction.py` 中的正协同评分，
**不修改**源梯度、PGrad 顺序虚拟优化、负冲突项、冗余项、
目标引导旋转、角度元学习和残差融合的接口/逻辑。
源城市仍为 METR-LA、成都、深圳；目标 PEMS-BAY 仅前三天。
5min 城市使用 12 步，10min 城市使用 6 步，对应物理 60min。
Graph WaveNet 的动态图只由历史数据计算，不引入固定节点嵌入、
额外道路拓扑、POI、LLM 或目标测试标签。

## 额外发现的阻断性问题

运行新分支合成 CI 时暴露 main 已有的两个问题：
- Graph WaveNet 输出原本 `[B,N,H,1]`，但训练要求 `[B,H,N,1]`。
  原因是 `end_conv_2[..., -1]` 已为 `[B,H,N]`，之后误用了 `permute(0,2,1)`。
  本分支删除了错误转置。
- `tests/conftest.py` 将真正的 `TSJTPGrad` 全局覆盖为 `ToyModel`，
  使真实 Graph WaveNet 包装器测试报错。本分支删除该覆盖；
  其他单元测试仍可显式使用 ToyModel。

首次 CI **3 failed / 13 passed**，失败涉及以上两个问题；
后续提交触发重新运行 CI。真实四城市训练尚未执行。
修复是否足以消除其他失败必须以最新 CI 为准。

## 修改文件

- `src/tsjt_pgrad/pgrad/source_interaction.py`：正协同目标一致性门控及诊断字段。
- `src/tsjt_pgrad/training/trainer.py`：向源交互透传开关，不改变下游模块。
- `configs/gwn_target_consistent_synergy_pemsbay_full.yaml`：独立 50 epoch 实验配置，`lambda=1`。
- `tests/test_source_interaction_v2.py`：反目标源城市对、零开关一致性、输入不变测试。
- `src/tsjt_pgrad/models/graph_wavenet_backbone.py`：修正预测张量维度。
- `tests/conftest.py`：删除对真实 Graph WaveNet 包装器的全局伪造。
- `.github/workflows/gwn-target-consistent-synergy.yml`：合成 CI。
- 本研究记录。

## 可复现实验

```bash
PYTHONPATH=src pytest -q tests/test_source_interaction_v2.py tests/test_graph_wavenet_backbone.py
# 需要外部 tsjt_c1 + 四城数据，且源/目标按配置划分
PYTHONPATH=src:/path/to/tsjt_c1 python -m tsjt_pgrad.run \
  --config configs/gwn_target_consistent_synergy_pemsbay_full.yaml
```

与 main 中 `configs/source_interaction_v2_pemsbay_full.yaml` 同 seed、
同源数据和目标三天窗口、同训练步数比较。
先运行 1 epoch smoke 和输入/输出时长检查，再进行 50 epoch 多 seed。
评估时只在训练结束后读取目标测试标签；不许调参或选 checkpoint。
本轮没有真实四城市 MAE/RMSE；任何 synthetic pytest 成功不代表精度提高。
