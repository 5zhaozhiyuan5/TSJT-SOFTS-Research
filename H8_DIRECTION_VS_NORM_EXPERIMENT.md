# H8 — OT 梯度方向与幅度解耦：真实 AdamW 更新配对诊断

**分支：** `experiment/pmme-h8-direction-vs-norm-20261127`  
**基于：** H7 已完成的 full-OT 日志提交 `e35dd64d5cde2f50b9614bdebd5d86731f411bf3`。

> 本阶段只添加独立诊断脚本和单测。**不重新训练 F2、不训练 Stage-II、不修改官方 PMME / SOFTS / OT / 原有 F2 加权代码。** 尚无本 H8 实验数据，代码提交不等于 GPU 验证成功。

## 1. 研究动机和待证问题

H7 的 8 组 **完整 OT、3 天目标支持集内、单步 AdamW 影子更新**观察到：
- `0.5 × OT` 相比 F2 的平均 support-probe MAE 改善 `+0.000223964`。
- `0.5 × Proj(OT,g_t)` 相比 `0.5 × OT` 平均改善 `+0.0000965595`，8/8 批为正。
- 但单独投影不优于减半；全局与分城市投影在三源梯度全冲突时基本数学等价，不能称为城市级独立创新。

**H8 的可证伪假设：** H7 的增量收益是否来自“矢量方向去冲突”，而不是更小的 OT 梯度范数或者由 AdamW 非线性带来的实际更新幅度变化？

不能把 H7/H8 的支持集虚拟收益当作独立测试集指标。F2 checkpoint 已经见过所有三天目标支持窗口。

## 2. 公式、实验变量与控制组

记 `g_s` 为 3 源城市监督预测损失的全参数梯度，`g_t` 为目标 FIT 预测损失梯度，`g_c` 为原始 F2 对城市 c 的真实 GeomLoss OT 梯度，
`g=Σ_c g_c`，目标梯度冲突投影为：

```text
P(g,g_t) = g - min(0, dot(g,g_t)) / (||g_t||² + 1e-8) * g_t
```

原始 PCGrad 依然随后作用于 `g_s + g_policy`，再使用 checkpoint **真实 AdamW 动量/方差/步数/学习率**虚拟执行一步。不会对源/目标数据额外进行训练。

| 策略代码 | 实际传入原始 PCGrad 的 OT 梯度 | 目的 |
|---|---|---|
| `f2` | `g` | 原始 F2 |
| `global_half` | `0.5g` | H6/H7 已成功的固定减半基准 |
| **`projected_half`** | **`0.5P(g,g_t)`** | H7 当前候选 |
| **`norm_matched_raw`** | **`0.5×||P(g)||/||g||×g`** | 与候选原始梯度范数相等、方向不变的**主对照** |
| `direction_norm_matched_half` | `0.5×||g||/||P(g)||×P(g)` | 与固定 OT 减半梯度范数相等、但方向修正的互补对照 |
| **`adamw_step_norm_matched_raw`** | **`α*g`**，α 通过 FIT 梯度和 optimizer state 校准 | 匹配候选相对 no-OT 的 **AdamW 实际参数位移范数** |
| `all_off` | `0` | OT-induced AdamW 位移的零参考 |
| `identity` | 与 `f2` 同一个 OT 向量 | 重放数值噪声对照 |

### 第一个必须回答的问题：方向有无独立作用？

主指标：

```text
Delta_direction =
  MAE_probe(norm_matched_raw) - MAE_probe(projected_half)
```

正值表示**相同原始 OT 梯度范数**时，方向校正更有利。

### 第二个必须回答的问题：是否只是 AdamW 实际参数更新更小？

AdamW 并非单纯的 `theta - lr*g`。用完全相同的模型、当前优化器历史状态和一侧 PCGrad 定义 `theta(g)`。

对每个 FIT batch，不使用 PROBE 标签，解一维 α：

```text
||theta(alpha*g)-theta(0)||₂
  ≈ ||theta(0.5*P(g,g_t))-theta(0)||₂
```

- 首先在 α∈[0,4] 的 9 点粗网格寻找根的包围区间，然后在该区间内执行 16 次二分；
- AdamW 更新范数可能非单调，因此用函数值**跨越目标**的区间二分，不假定全局单调；
- 若无法匹配，则报告 `calibration_limited`、相对误差和 `within_one_percent=false`，**不能声称排除了优化器幅度效应**；
- 若匹配误差不超过 1%，才有效比较 `MAE(adamw_step_norm_matched_raw)-MAE(projected_half)`。

因为原始梯度范数相等 ≠ 真实参数步长相等，所以两种控制缺一不可。

### 配对可复现性

- 仍然使用与 H7 完全相同的 SOFTS 随机池化 seed `681947 + batch_index`；
- 重新计算原始 F2 完整 GeomLoss (`p=1, blur=0.001, debias=False`，`concat(rep, 10*normalized_true_future)`， OT 系数 `0.01`，F2 原始 alpha/beta)；
- 记录源和目标 OT 点预算、样本平铺索引 SHA256、alpha top24/ESS、beta 统计、三段 horizon/10、60、120、180 分钟 MAE；
- 完整 OT 运行时自动对照仓库 H7 的原始 `f2`、`global_half`、`city_projection_half` 三个旧参照。新 `projected_half` 与旧 **city** 投影可能有浮点级差异；日志记录最大绝对 MAE 差，允许 1e-5。

## 3. 固定数据与避免结果泄漏

- 使用已有原始 F2 Stage-I **epoch2899** 完整 checkpoint，包括 optimizer state；
- checkpoint 路径：
```text
/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth
```
- 强制 SHA256：
```text
f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf
```
- 三源：成都、METR-LA、深圳；目标 PEMS-BAY 3 天，SOFTS horizon=36，原始 source batch=16，种子 1；
- 每批 46 个目标支持窗口，按照窗口 index mod 5 拆出 FIT=36、PROBE=10。**FIT** 的未来标签进入 target gradient、F2 OT 目标几何和全部校准；**PROBE** 的未来标签只在虚拟更新全部固定后用于评分；
- 但整体 checkpoint 此前已经见过 PROBE 期间，因此并非真正时间外验证，8 批也不独立；
- 不能读取最终 test 数据进行策略选择。独立时间验证属于后续单独数据切分流程，**本次 H8 并未实现、也不能假称实现**；
- GeomLoss 本次没有显式物化传输矩阵 Π 或完整 cost matrix，**不得声称有这两类完整日志**。

## 4. 运行指令

```bash
git fetch origin
git switch experiment/pmme-h8-direction-vs-norm-20261127

# 先跑CPU单测
python -m unittest discover -s tests -p 'test_h8_direction_norm_audit.py'
# 也检查继承的H7梯度投影公式
python -m unittest discover -s tests -p 'test_h7_ot_component_projection.py'

# 第一步：1批真实 GeomLoss 冒烟 (256源 / 128目标 OT点)
PMME_H8_PROBES=1 PMME_H8_SOURCE_POINTS=256 PMME_H8_TARGET_POINTS=128 \
PMME_H8_RUN_ROOT=./experiments/h8-smoke \
bash run_h8_direction_norm.sh

# 第二步：8批 capped OT，检查校准误差
PMME_H8_PROBES=8 PMME_H8_SOURCE_POINTS=256 PMME_H8_TARGET_POINTS=128 \
PMME_H8_RUN_ROOT=./experiments/h8-pilot-8 \
bash run_h8_direction_norm.sh

# 第三步：正式的8批 FULL OT；用于与H7逐批对照
PMME_H8_PROBES=8 PMME_H8_SOURCE_POINTS=0 PMME_H8_TARGET_POINTS=0 \
PMME_H8_RUN_ROOT=./experiments/h8-full-8 \
bash run_h8_direction_norm.sh
```

**没有 checkpoint 或 SHA 不符合，会立即停止，绝不会补跑/重新训练 F2。**
H8 相比 H7 多了 AdamW shadow 一维匹配，运行耗时会增加，但只执行虚拟更新，不会真正写入模型。

### 每次运行输出

```text
experiments/h8-full-8/
  run_config.txt                     # branch, commit, seed, checkpoint SHA, 点数
  train.out                          # 完整报错/批次结果
  h8_results.jsonl                  # 8候选完整结果、各向量及更新诊断
  h8_results.metadata.json          # 全部输入约束、泄漏说明
  h8_policy_summary.csv             # 相对 F2 / 半强度 / 同范数 / 同 AdamW 的收益
  h8_optimizer_direction_traces.csv # 梯度幅度、余弦、实际位移与匹配误差
  h8_city_gradient_norms.csv        # 三源城市的OT梯度范数
  h8_decision.json                  # 事先约定的停止/通过判据
  h8_vs_h7_reference_check.json    # full-OT与H7一致性检查
  summary.txt
  COMPLETED / FAILED
```

## 5. 筛选门槛以及下一步动作

H8 **主判据**为 `projected_half` vs `norm_matched_raw`：
- 8 批中 ≥6 批正向，前4/后4批均为正均值；
- MAE 改善中位数 >max(`0.00005`, 10 × identity重放噪声)；
- 最大单批次占绝对收益比例 ≤50%；
- 三段 short/mid/long 无集中严重退化。

更严格的第二判据：`adamw_step_norm_matched_raw` 的**全部 8 批更新位移范数校准误差 ≤1%**，且对其仍有稳定、方向一致的 MAE 增益；否则不能排除实际优化器更新幅度的影响。

这只是**假设筛查**，不能作统计显著性或正式测试论据。如果两个方向判据均未通过，停止继续叠加梯度投影/超参，重新检查 OT 匹配目标；若通过，应先设计**模型训练期间未见过的时间验证切分**，再决定是否值得做一次正式 F2 + 新机制训练。不能用最终测试集调参。

## 6. 文件改动范围

**仅新增 6 个文件**，未覆盖 H7、H6 或官方训练代码：

1. `model/Meta_Models/h8_direction_norm_audit.py`
2. `audit_h8_direction_norm.py`
3. `run_h8_direction_norm.sh`
4. `summarize_h8_direction_norm.py`
5. `tests/test_h8_direction_norm_audit.py`
6. `H8_DIRECTION_VS_NORM_EXPERIMENT.md`

**实现前已有实验观察与本次代码不能混淆：H8 实验尚未运行；没有提供新的最终 MAE/RMSE。**
