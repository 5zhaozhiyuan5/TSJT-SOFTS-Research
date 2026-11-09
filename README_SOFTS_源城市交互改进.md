# SOFTS + 源城市交互 v2 + 原有梯度旋转和残差融合

> **版本性质：已完成代码替换与本地合成数据回归测试。未在真实 METR-LA / 成都 / 深圳 / PEMS-BAY 数据上运行训练或报告 MAE。**
>
> 基于之前交付的 `tsjt_source_interaction_v2` 直接修改，仅替换**预测骨干**。源城市交互 v2、PGrad 轨迹、梯度旋转、角度控制器、元梯度反馈、残差融合、训练与评估算法保持原版；新文件 `softs_backbone.py` 替代 `TSJTPaper(Graph WaveNet)`。

## 1. 实际训练架构

```
METR-LA  (5min,12->12) ─┐
成都     (10min,6->6)  ──┼──> 共享 SOFTS backbone ──> 分城市梯度 g_A,g_B,g_C
深圳     (10min,6->6)  ──┘             │
                                     │
PEMS-BAY (5min,12->12) ────> 同一个 SOFTS ───> g_T, Query-R, Query-W
                                     │
                                     ▼
                    原有 source_interaction_v2 / 多顺序 PGrad
                            ┌────────┴────────┐
                            ▼                 ▼
                        源-源协同          六排列虚拟轨迹
                            └────────┬────────┘
                                     ▼
                              公共主梯度 p_new
                                     ▼
                    原 rotation.py + AngleController（不变）
                                     ▼
                       原 training/meta.py（不变）
                                     ▼
                       原 fusion.py（不变）
                                     ▼
                        显式梯度更新 SOFTS（不变）
```

SOFTS 结构直接按照 [SOFTS 官方仓库](https://github.com/Secilia-Cxy/SOFTS) 的 `models/SOFTS.py`、`layers/Embed.py`、`layers/Transformer_EncDec.py` 实现：

- **DataEmbedding_inverted**：沿历史时间轴的线性嵌入，节点作为 token；可选将 time-of-day 和 day-of-week 作为额外 token；
- **STAR**：`gen1/gen2` 产生共享核心表示，节点维度的 aggregate-redistribute，再经 `gen3/gen4`；
- **Encoder**：残差 STAR + LayerNorm + Conv1d FFN；
- **Forecast Head**：`Linear(d_model,12)` 输出未来 12 个 5min 时间点；
- **Instance normalization**：SOFTS 论文代码中的样本级 mean/stdev 归一化和反归一化（`use_norm`）。

这里没有 Graph WaveNet、节点专属 embedding、固定大小邻接矩阵或额外图结构创新。**旧 `model.kwargs` 中的 Graph WaveNet 超参数只是兼容现有 `tsjt_c1` 数据引擎的配置占位，新的 `TSJTPGrad` 包装器会忽略它们；实际预测器只有 SOFTS。** 邻接矩阵作为旧训练函数实参继续传入，但 SOFTS 忽略该值。

## 2. 两种时间频率怎样共享同一个 SOFTS

所有城市在物理时间上都为过去 60 分钟预测未来 60 分钟。

| 城市 | 输入 | SOFTS 输入 | SOFTS 输出 | 还原原有目标张量 |
|---|---|---|---|---|
| METR-LA | 12个5min点 | 直接 12 点 | 12个5min点 | 12点 |
| PEMS-BAY | 12个5min点 | 直接 12 点 | 12个5min点 | 12点 |
| 成都 | 6个10min点 | **历史插值**到 12 个5min点 | 12个5min点 | 取第 2,4,...,12 个预测点，保留 6个10min点 |
| 深圳 | 6个10min点 | 同成都 | 同成都 | 同成都 |

- 历史插值不访问未来标签；对早于首个已知样本的 `-55min` 点做首值保持，不强行使用未来数据。
- 源城市 6 点预测从统一 12 点输出中采样 `+10,+20,...,+60min`。这样 6 和 12 horizon 的预测任务才对齐真实分钟数。
- 对四座城市，没有新增任何节点数相关的可学习参数；共享参数可以直接形成同一个空间的源城市和目标城市梯度。

**提醒：** 原始 SOFTS 没有要求采用这一 6/12 双频率适配。这是为当前四城数据协议增加的时间分辨率适配，需作为实现假设在论文中交代，不应当声称这是官方 SOFTS 的原始设置。

## 3. SOFTS 的训练期随机 STAR 怎样处理

官方 `STAR` 在 `model.train()` 下通过 `torch.multinomial` 按概率采样核心 token；在 `model.eval()` 下使用 softmax 权重求和。为了避免六种 PGrad 轨迹和目标 Query 元反馈不断受到重新采样的噪声影响，本版默认：

```yaml
model:
  kwargs:
    softs_star_pooling: soft
```

此时训练和评估都采用官方评估分支中的**加权聚合**，因此更利于复现固定梯度关系。**这是一项明确的 SOFTS 适配，不等同于严格复现官方随机训练策略。** 如要对照官方随机 STAR：

```yaml
    softs_star_pooling: stochastic
```

注意：SOFTS 的其他 `dropout: 0.1` 仍保留，所以默认版本的训练前向**并非完全确定性**；上面设置仅消除 STAR 的随机多项式采样。如果后续观察到 PGrad 方向噪声，建议另做 `dropout: 0.0` 独立消融，不能将其未经实验当作默认有效优化。

## 4. 文件改动清单

| 文件 | 改动情况 |
|---|---|
| `src/tsjt_pgrad/models/softs_backbone.py` | **新增** SOFTS 结构和双频率适配 |
| `src/tsjt_pgrad/models/pgrad_model.py` | **替换**原 Graph WaveNet 为 SOFTS；AngleController 保持原调用 |
| `configs/*.yaml` | 加入 SOFTS 参数、使用独立输出目录；原梯度超参数保持 |
| `tests/test_softs_backbone.py` | 新增真实 SOFTS 模型级和跨组件集成测试 |
| `pgrad/source_interaction.py` | **完全不变**（仍是源城市协同 v2） |
| `pgrad/pgrad.py` | **完全不变** |
| `pgrad/rotation.py` | **完全不变** |
| `pgrad/fusion.py` | **完全不变** |
| `training/meta.py` | **完全不变** |
| `training/trainer.py` | **完全不变** |

历史旧版 README 和 AUDIT 放在 `docs/历史文档`，内容描述 Graph WaveNet，**不代表本版架构**。

## 5. 必要依赖及使用方式

- 原来的 **`tsjt_c1`** 仍需安装或可被 Python 导入：继续使用 `tsjt_c1.training.engine` 提供的四城数据集加载、训练划分、完整测试集评估、日志及 checkpoint 操作，和 `tsjt_c1.training.tsb` 中的原梯度过滤。**不再依赖 `tsjt_paper` 的 Graph WaveNet 预测模型。**
- `torch`, `numpy`, `pyyaml`；跑本地测试另需 `pytest`。
- 在原 TSJT 项目根目录启动，确保 `data/` 包含原来的四城文件。根据你的具体安装位置，将下列路径替换为实际路径。

在服务器的原项目目录下：

```bash
cd /root/tsjt_c1
# 解压发布包，例如到 /root/tvsgi_softs/ 下
unzip '/root/TSJT_SOFTS_源城市交互保留旋转融合.zip' -d /root
export PYTHONPATH="/root/tsjt_tvsgi_softs/src:/root/tsjt_c1:${PYTHONPATH}"

# 先做 smoke (数据仍使用真实四城数据，少训练步数，并跳过完整测试)
python -m tsjt_pgrad.run --config /root/tsjt_tvsgi_softs/configs/source_interaction_v2_smoke.yaml

# 再完整训练：50 epoch、PEMS-BAY 目标 3 天、完整测试集
python -m tsjt_pgrad.run --config /root/tsjt_tvsgi_softs/configs/source_interaction_v2_pemsbay_full.yaml
```

**重要：** 老的 Graph WaveNet `last.pt` 无法直接作为 SOFTS 的初始化/续训权重；新实验需要 `resume: false` 且独立的 `output_dir`。旧模型对照应继续使用原来的旧代码包运行。

如果 `tsjt_c1` 的实际包路径位于项目的 `src/` 下，请把对应 `src/` 也加入 `PYTHONPATH`。这版代码沿用其已有数据接口，没有尝试复制或改写你原来的四城数据引擎。

## 6. 测试范围与风险

在无 `tsjt_c1` 和真实交通数据的本地容器中：

```bash
PYTHONPATH=src pytest -q tests/
python -m compileall -q src tests
```

新增测试覆盖：四种节点数输入输出、6min/12min 轨迹对齐（实际是 6个10min / 12个5min）、真实 SOFTS 反传、`functional_call` 虚拟更新、SOFTS 软/随机 STAR、实际 SOFTS + 源城市交互 v2 + 原旋转 + 元更新 + 残差融合链路。没有使用真实 GPU 或真实 50 epoch 数据。

研究评价时请记录：Persistence、SOFTS target-only、SOFTS + 原始 TSB、SOFTS + 原 PGrad、SOFTS + 新源城市交互 v2，在**同一 SOFTS 骨干、随机种子和 3天目标协议**下比较。避免直接比较旧 GWN 版本和新版 SOFTS 来证明源城市交互的贡献。

## 7. 参考

- SOFTS 官方仓库: https://github.com/Secilia-Cxy/SOFTS
- PGrad: https://arxiv.org/abs/2305.01134
- PiKE: https://proceedings.neurips.cc/paper_files/paper/2025/hash/f7a94134f1c726796c6f81fb946e489d-Abstract-Conference.html
- Aligned Multi Objective Optimization: https://proceedings.mlr.press/v267/efroni25a.html

## 8. 新提供的 SOFTS 对照配置

- `configs/softs_original_tsb_baseline.yaml`：**同一 SOFTS**，使用 `tsjt_original` 聚合+TSB；用于梯度基线。
- `configs/softs_legacy_pgrad_baseline.yaml`：**同一 SOFTS**，使用旧版单顺序 PGrad、原旋转与融合；用于验证 TV-SGI 的净增益。
- `configs/softs_stochastic_star_ablation.yaml`：源城市交互 v2 不变，只改 SOFTS STAR 为官方 train-time 随机汇聚；用于检查随机采样影响。

如需验证新方法的贡献，应比较**相同 SOFTS、相同 3 天目标数据、同一个 seed 的原始 PGrad 与源城市交互 v2**，而非用旧 GWN 指标直接对比新 SOFTS 指标。
