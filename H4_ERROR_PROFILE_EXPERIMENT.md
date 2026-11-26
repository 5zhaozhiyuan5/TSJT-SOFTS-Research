# H4：F2 预测误差类型 × OT 反事实目标收益诊断（只读实验）

日期：2026-11-26；分支：\`experiment/pmme-f2-error-profile-target-benefit-audit-20261126\`

## 1. 研究假设，不先假定它有效

此前原始 F2/T0 seed1 的 Stage-II 36-horizon 平均 MAE = **2.372713**；PRC (source training prediction accuracy) = **2.394411**，36/36 horizon 都退化。P1–P4、late OT decay、PRC 均没有稳定超过 T0。

**H4 只检验这一命题：在原始 F2 的梯度加权 alpha 已经固定时，源样本本身的预测误差水平和 short/mid/long 误差结构，能否预测“降低该类源样本 OT 权重”对目标 PEMS-BAY 预测误差的局部影响？**

这不是完整 TGRC 新方法、不是新 F2 训练、更不声称达到优于 2.372713 的 test MAE。H4 通过以前已有 ORIGINAL F2 checkpoint 的 **one-step、全参数、stateful AdamW、PCGrad** 影子试验得到探索性证据，先看是否值得实现真实训练算法。

## 2. 确切复用，不新增一组完整训练

- 原模型：原始 F2，SOFTS，源城市 METR-LA、Chengdu、Shenzhen，目标 PEMS-BAY 3-day support。
- 原始 source batch=16；预测 H=36，按 12/12/12 分 short/mid/long。
- **只加载已有原始 F2 Stage-I 完整 checkpoint**，包括 \`model_state_dict\`、\`optimizer_state_dict\`、\`epoch\`。
- 默认读取已有 H3/D1 前置 checkpoint：
  \`/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth\`
- 不运行 F2、PRC 的 Stage-I、Stage-II。若 checkpoint 不存在，脚本报错退出，不自动训练。
- 一组候选完整比较中，模型参数、原始 AdamW 动量、源 batch、目标 fit/probe、源目标 OT 几何、OT 系数、所有随机池化等必须相同。只改变一个城市内部一个预测误差组的 alpha 质量。

## 3. 预测能力分组——**只由源城市训练标签决定**

设源城市 c 内窗口/节点 i 的真实未来为 \`y_i[0:36]\`、当前模型预测为 \`hat_y_i\`、源城市标准差为 sigma_c。

三个 12-step 分量平均绝对误差：

\[
e_i^{(b)}=\frac{1}{12\sigma_c}\sum_{h\in b}|\widehat y_{i,h}-y_{i,h}|,
\quad b \in \{\text{short,mid,long}\},\qquad
e_i=\frac13\sum_b e_i^{(b)}.
\]

在每个源城市的 **同一 OT 采样池** 中，按 \`e_i\` 排名，互斥划分：

1. \`accurate\`：误差最低的前 25%（低误差）
2. \`inaccurate\`：误差最高的后 25%（高误差）
3. 中间 50%：如果三个分量相对误差最大最小值之差小于预先固定阈值 0.15，标记 \`balanced\`；否则把误差最集中的时长分为 \`short_weak\` / \`mid_weak\` / \`long_weak\`。

额外记录“相对于保持最后观测速度不变的 persistence 基线，模型获胜的比例”，它**仅作分析**，不参与分组或候选选择。这样既看当前预测能力，也看三个预测时长的误差结构，而非直接把源城市预测准等同于迁移有益。

**禁止**利用目标 probe/test 标签定义源样本组、OT 权重、评价候选前的组选择。无效组（点数不足 12 或原 F2 alpha mass 不足 2%）固定跳过，不通过测试集调整阈值。

## 4. OT 消融：一组只减半，不直接移除

原始 F2 源边际 \`alpha^F2\` 通过原始梯度余弦计算，目标 \`beta\` 均匀。对城市 c 内一组 G：

\[
\widetilde\alpha_{i}^{(G)}=
\frac{\alpha_{i}^{F2}\big(1-0.5\,\mathbf{1}[i\in G]\big)}
{\sum_k\alpha_{k}^{F2}\big(1-0.5\,\mathbf{1}[k\in G]\big)}.
\]

基线：三城全部原始 F2 alpha。候选：**仅在一个源城市内**对一个组降权 50%，另外两个城市全部原始 F2。这不是 PRC 的直接 \`score = source error\` 权重，也不修改 OT cost、目标 beta、PCGrad、预测骨干或 Stage-II。

**随机分组负对照（已实现）**：对每个城市的 `accurate` 组，生成同样大小的伪随机源样本组，使用 F2 alpha mass 在 24 个候选中选择近似匹配、与真实准确组尽量少重叠的组。候选的产生和选取不读任何目标 probe 标签；仅当对照组 mass 与准确组的相对差小于等于 50% 才纳入影子更新，并在日志中记录剩余质量偏差。这只是近似质量匹配，不是严格配对；若原组与随机组产生相似收益，则拒绝“预测准确性提供了额外信息”的解释。此对照增加若干额外 OT/影子前向开销。

均按同一个原始 \`GeomLoss Sinkhorn(p=1, blur=0.001, debias=False)\` 和 \`concat(rep, 10×RevIN_normalized_future)\` 计算 OT 梯度，OT 系数保持 0.01。

输出样本分组数、源样本采样索引 SHA256、组的 alpha mass 前后值、ESS、3 个时长误差、源样本 persistence 胜率、被跳过的组及原因；**不声称已显式恢复 GeomLoss 的传输矩阵 Π**（原函数只返回 loss）。

## 5. 目标端：相同真实优化器状态的虚拟更新

每批目标城市支持窗口使用 H3 原本的 modulo-5 **整窗口** fit/probe 划分。约 46 窗口中 36 fit、10 probe；三城各对应原始 15/15/16 目标组，组内只用 fit 窗口构建 alpha 和 beta。

对 F2 与每个候选求真实 OT 梯度 \`g_OT(k)\`，统一使用真实源预测损失梯度 \`g_source\` 与目标 **fit-only** 预测损失梯度 \`g_target_fit\`。更新：

\[
g_k=\operatorname{PCGrad}_{\mathrm{one-sided}}
(g_{\mathrm{source}}+g_{\mathrm{OT}}^{(k)},\;g_{\mathrm{target,fit}}),
\quad
\theta_k^+=\operatorname{AdamWShadow}(\theta,\mathrm{state},g_k).
\]

影子更新只在内存里；使用完整模型参数、当前 AdamW 一二阶矩与步数；不用低可信的只更新 output head 的梯度近似。所有候选完全一致的 SOFTS 训练随机数，目标 probe 真值仅用于**影子前向之后**计算损失；影子预测的数据对象中 \`y\` 直接清零，防止隐式窥探。

对每个城市/组，计算

\[
\Delta_{c,G}^{\rm suppress}=
\operatorname{MAE}_{\mathrm{probe}}(\theta^+_{\rm F2})
-
\operatorname{MAE}_{\mathrm{probe}}(\theta^+_{c,G\downarrow}).
\]

- **正号**：此次虚拟更新中，降低这组权重后目标误差下降，该组可能产生有害迁移。
- **负号**：降低后目标误差升高，该组可能有帮助。
- **接近零/方向反复**：不可得出结论，不能启动一个新模块的正式训练。

同时输出 3 个时长的 SmoothL1 gain、10/60/120/180 分钟单步 MAE 与 RMSE、总 MAE/RMSE。候选数随有效组数变化，不进行 test 选择。

## 6. 实验流程与预算

### A. CPU 纯函数单测（无需训练和数据）

\`\`\`bash
git fetch origin
git switch experiment/pmme-f2-error-profile-target-benefit-audit-20261126
python -m unittest discover -s tests -p 'test_h4_error_profiles.py'
\`\`\`

测试：源样本误差类型互斥、城市缩放不改变分组、边际总质量守恒、不合法分组拒绝、影子 PCGrad/AdamW 跑通、模型参数/优化器状态/.grad 不变。

### B. 真实 GPU 冒烟：一批、子采样 OT（非科研结论）

\`\`\`bash
PMME_H4_PROBES=1 PMME_H4_SOURCE_POINTS=128 PMME_H4_TARGET_POINTS=64 \\
  PMME_H4_MIN_GROUP=6 PMME_H4_RUN_ROOT=./experiments/h4-smoke \\
  bash run_h4_error_profiles.sh
\`\`\`

### C. 探索性诊断：原始 F2 checkpoint，8 个重新采样批次

\`\`\`bash
PMME_H4_PROBES=8 PMME_H4_SOURCE_POINTS=256 PMME_H4_TARGET_POINTS=128 \\
  PMME_H4_RUN_ROOT=./experiments/h4-pilot-8 \\
  bash run_h4_error_profiles.sh
\`\`\`

这里对源/目标点做采样，**只说明匹配子集的局部结果**，不能称为原始全量 OT 实验。保留所有样本、目标窗口子集和组分布统计，防止重新采样结果发生人为偏差。

### D. 优先做全量 OT 的有限核验

只有阶段 C 表现出重复可见的分组信号，再针对**事先固定**的分组（例如在阶段 C 的前 4 批选定方向，后 4 批仅检查方向）以 \`SOURCE_POINTS=0 TARGET_POINTS=0\` 做真实全量 OT 的有限核验。该核验的目标 probe 仍然属于训练 checkpoint 见过的支持集，**不能称为独立验证**。

若需要无泄漏、正式发表级别证据：必须另建按时间严格隔离、考虑历史 H=288/预测 H=36 重叠的 fit/probe 分割，训练**未见过探针时段**的 checkpoint，再进行相同测试；本 H4 特意不做这项新的 F2 训练，遵守不重复训练 F2 的要求。

## 7. 预先指定的通过/停止标准

**机制筛选门槛，而非显著性标准：**

- 至少一个非空城市 × 预测组，在不少于 6/8 批次中呈同方向的 MAE gain，且平均 gain 超过重复影子前向的数值噪声；
- 不只是某一城市批次的偶然现象，同一组 short/mid/long gain 的方向有合理一致性；
- 与同计数、近似 alpha 质量匹配的随机分组负对照比较（本次已实现），确实具有**预测误差结构的信息价值**，而不是任意分组都会产生相同效果；
- 影子更新不改变模型/优化器状态，不用任何测试集标签，所有候选同批、同成本、同优化器；
- 如果方向频繁反转、抑制收益比数值噪声还小、或全量 OT 与子采样 OT 方向不一致，**停止 TGRC，不启动 3000+6000 轮训练**。

当前 8 批全来自同一 checkpoint 的三个支持日、且窗口高度相关，**不允许把 6/8 解释成统计显著性、也不允许据此认定能超过 T0 的 2.372713 test MAE**。

## 8. 输出文件和解释

- \`run_config.txt\` 记录代码 commit、checkpoint hash、预算、seed、组和 strength。
- \`h4_results.jsonl\` 每批包含 F2 与干预后的影子预测误差、每城市各组统计、源采样 SHA256、alpha mass、ESS、3 个分量误差、probe/fit 窗口 ID、跳过原因。
- \`h4_results.metadata.json\` 记录原 checkpoint SHA256、是否真正 full OT、无训练、可见数据范围。
- \`h4_summary.csv\` 各城市/误差类型的平均收益、正号比例、3 个时长收益。
- \`summary.txt\` 易读汇总；\`train.out\` 控制台原始明细；\`COMPLETED\` / \`FAILED\` 校验标志。

**H4 只设计和实现诊断，不修改 F2/官方 Stage-II、不训练新的 F2、不宣称新机制已经成立。** 若 H4 真有可信证据，再设计一个独立且足够轻量的 TGRC 控制器，仅加在 F2 的 OT alpha 上，并和 CrossTReS、CMSVC、TAROT、Meta-Weight-Net、先前 G2/G3/H2/H3 明确区分。
