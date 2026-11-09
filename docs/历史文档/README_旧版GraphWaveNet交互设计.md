# TSJT / PGrad 源城市交互模块改进版（2026-11-09）

## 改动范围（严格）

- **仅改变 PGrad 源城市交互模块**：新增 `src/tsjt_pgrad/pgrad/source_interaction.py`；`training/trainer.py` 仅增加调用分支，将新模块的单一主方向和元数据接回原来同一个入口。
- **完全不改变原有梯度旋转** `src/tsjt_pgrad/pgrad/rotation.py`、角度控制器、`training/meta.py`、残差匹配 `src/tsjt_pgrad/pgrad/fusion.py`、`models/pgrad_model.py`、原 PGrad `pgrad/pgrad.py`。
- **骨干仍为 TSJT / Graph WaveNet**，没有替换为 SOFTS。用户此次明确只修改源城市交互。
- 原 8 种消融路径仍可运行。通过 `train.source_interaction_mode` 控制：`legacy_pgrad` 为原 PGrad；`source_interaction_v2` 为改进源城市交互。

## 为什么改（依据用户上传的 2026-11-08 实验）

实验使用 METR-LA、成都、深圳为源城市，PEMS-BAY 仅 3 天为目标；50 epochs、seed 2025、源系数 1e-4、目标 2e-4、源预算占目标 30%。

- 1,477 条采样诊断中，PGrad 更新记录 1,055 次，均未触发主方向 PCA 回退。
- 平均 `effective_source_ratio≈0.1297`；非零请求更新不是有益更新的证明。
- 残差融合权重的**记录点平均**：METR-LA≈0.6178、成都≈0.0019、深圳≈0.0014；剩余是没有使用的预算，不要求三个权重之和为 1。
- 旋转触发记录：METR-LA 72/1477；成都 1229/1477；深圳 1291/1477。PGrad 参考方向对融合主导城市的直接作用很弱。
- 60 分钟模型 MAE≈4.913，Persistence≈3.104；预测标准差 0.629 vs 标签 9.612。**该问题还可能来自原骨干和预测头，不会因本改动自动解决。**

## 改进后的「源城市交互」算法

### 1. 六顺序均衡的 PGrad 轨迹

保留真正的顺序虚拟优化，不使用固定起点梯度冒充参数更新后的梯度。每次刷新参考方向，默认枚举三个城市的 **全部六种排列**；每个排列调用原 PGrad 模块，获得主梯度 `p^(k)`，对符号和尺度进行统一并合成轨迹参考。避免原来只用一次随机排列造成顺序偶然性；可通过配置缩减为 1 或 3 种顺序做效率消融。

### 2. 源城市之间的正向协同、冲突与冗余分析

令 `u_i = g_i / ||g_i||`，`C_ij = u_i dot u_j`，目标相容性 `a_i = cos(g_i, g_T)`。统计：
- 正向协同：`clip(C_ij, 0, threshold)`；
- 冲突：`max(-C_ij, 0)`；
- 高冗余：`max(C_ij - threshold, 0)`，避免两个非常相似的源城市重复主导。

解析式 `score_i = a_i + β*synergy_i - μ*conflict_i - ν*redundancy_i`，softmax+均匀下限形成逐城市权重，跨刷新周期用仅三元素 EMA 平滑。默认超参数为实验候选，并非调优后的最优配置。

### 3. 只在交互参考向量里处理源—源冲突

将每个标准化源方向按其他源的负余弦进行对称纠偏；由城市权重聚合得到 `p_pair`。这不会修改后面接收的 `g_A,g_B,g_C`，仅用于新的**公共主方向**。

### 4. 专门针对原旋转模块的有效子空间

现有旋转会先移除参考梯度在目标梯度方向上的分量，因此新模块**直接在目标正交子空间**中合成 `p_traj` 与 `p_pair`：

`p_new = normalize((1-alpha)*normalize(project_perp_t(p_traj)) + alpha*normalize(project_perp_t(p_pair))) * mean_PGrad_norm`

默认 `alpha=0.5`。输出与之前 PGrad 一样，仍是逐参数的梯度元组 `principal`；`rotate_source_gradient(...)` 的函数签名和实现不变。

**目标梯度只用来给源城市贡献打分并定义旋转有效子空间，不通过新模块更新预测模型。**

## 关键输出日志

`grad_diagnostics.jsonl` 中的 `pgrad` 增加：

- `pairwise_cosine`：三源城市两两梯度余弦矩阵；
- `source_target_cosine`：逐源目标一致性；
- `source_positive_synergy`、`source_conflict`、`source_redundancy`；
- `source_weights`：由源城市交互决定的内部权重（**不是后续残差融合权重**）；
- `city_orders`、`order_consistency`：六种顺序的主方向一致性；
- `principal_cosine_by_city`、`target_orthogonal_dot`：交互方向在旋转模块中的有效几何关系。

原 `rotation`、`fusion`、`meta` 诊断字段不变。

## 复现说明

本包仍依赖用户原工程中的 `tsjt_c1` 与原始四城数据。本地没有完整的真实数据和独立的 `tsjt_c1` 包，所以只完成合成数据测试，**未声称跑过真实数据，也未声称已有 MAE 提升**。

部署时，将本包的 `src/tsjt_pgrad` 覆盖/安装进包含原 `tsjt_c1` 的实验目录，并保留原工程数据路径。

**第一步：本地 smoke**

```bash
PYTHONPATH=src python -m tsjt_pgrad.run --config configs/source_interaction_v2_smoke.yaml
```

**第二步：完整正式实验（50 epoch、全测试集）**

```bash
PYTHONPATH=src python -m tsjt_pgrad.run --config configs/source_interaction_v2_pemsbay_full.yaml
```

> 启动命令应在原项目根目录下使用。新输出目录与旧实验隔离；不要在旧输出目录上续训，避免污染同设置的比较。

**第三步：消融实验**

- `configs/source_interaction_v2_no_target_feedback.yaml`：取消目标一致性评分，研究源—源关系能否独立起作用；
- `configs/source_interaction_v2_single_order.yaml`：只保留一个固定城市顺序；
- `configs/source_interaction_v2_trajectory_only.yaml`：仅使用顺序轨迹均衡参考；
- `configs/source_interaction_v2_relation_only.yaml`：仅使用显式源—源交互参考。

旧 `configs/pgrad_pemsbay_full.yaml` 使用 `legacy_pgrad`（缺省值），是严格对照。上面消融**都不改变旋转与残差融合**。

## 测试与一致性

```
PYTHONPATH=src pytest -q tests/
# 40 passed（32 原回归 + 8 新测试）
python -m compileall -q src tests
```

文件 SHA256 核对：`rotation.py`、`fusion.py`、`training/meta.py`、`models/pgrad_model.py`、原 `pgrad.py` 与二次核查修正版逐字节相同。

## 论文出处和思路边界

- PGrad: Learning Principal Gradients for Domain Generalization — ICLR 2023，真实多域顺序优化轨迹的研究背景。`https://arxiv.org/abs/2305.01134`
- PiKE: Adaptive Data Mixing for Large-Scale Multi-Task Learning Under Low Gradient Conflicts — NeurIPS 2025，正向梯度交互与贡献权重的启发。`https://papers.nips.cc/paper_files/paper/2025/hash/f7a94134f1c726796c6f81fb946e489d-Abstract-Conference.html`
- Aligned Multi Objective Optimization — ICML 2025，相互一致任务的联合优化启发。`https://proceedings.mlr.press/v267/efroni25a.html`

**本模块是依据上述研究思想提出的跨城市改造，不等同于这些论文的官方算法，也未证明理论或实验上的提升。**
