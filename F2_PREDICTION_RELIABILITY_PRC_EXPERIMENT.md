# PRC：F2 源交通模式预测可靠性校准（单模块实验，2026-11-26）

## 一句话目标

原始 F2 通过「源样本梯度与目标任务梯度是否一致」决定源样本 OT 边际权重 alpha。但 **梯度方向相似不代表源样本的当前预测准确**。

本分支在原始 F2 的 alpha 后面**只添加一项** Prediction Reliability Calibration（PRC，预测可靠性校准）：同时考察该源样本在本轮预测中是否预测得准、是否至少优于简单的“保持最后一个观测值不变”的预测；再有界地修正 F2 权重。

**不是**把 P1–P4 模式匹配、OT 代价惩罚、改变 beta 或再次梯度调参叠加进来。官方 Stage-II、SOFTS、PCGrad、Sinkhorn OT 代价、OT 系数及标签来源都不变。

## 1. 输入和输出

源城市每个样本是一个「窗口 × 传感器」：
- 原 F2 的 source alpha：形状 [B*N]，总和 1。
- 预测结果 \hat{y}： [B,N,36]，**物理速度单位**（已反归一化）。
- 真实未来 y：[B,N,36]，仅 **源城市训练标签**。
- 历史速度最后一个值 x_last：[B,N]，原始数据里该值是城市标准化值。
- 源城市 mean/std：用于把 x_last 转换为物理值，并让不同城市误差可比较。

输出：alpha_PRC [B*N]，总和 1。目标 beta 保持 F2 的均匀边际。没有额外的可训练网络。

## 2. 准确性与相对预测技能

对每一个源样本 i：
\[
e_i=\frac{1}{36\sigma_c}\sum_{h=1}^{36}|\hat y_{i,h}-y_{i,h}|,\qquad
p_i=\frac{1}{36\sigma_c}\sum_{h=1}^{36}|x^{last}_{i}-y_{i,h}|.
\]
这里 \sigma_c 是当前源城市标准差，p_i 是 persistence 简单预测误差。

\[
q^{abs}_i=\exp\left(-e_i/\max(\operatorname{median}(e),0.1)\right),
\]
\[
q^{skill}_i=\operatorname{sigmoid}\left(
\frac{p_i-e_i}{\max(\operatorname{median}(p),0.1)}
\right),
\qquad q_i=q^{abs}_i q^{skill}_i.
\]

这两部分是一**个预测可靠性模块**，而不是两个单独训练的组件：
- q_abs 确保源样本绝对预测误差确实较低；
- q_skill 将“预测准确”与“源样本本来就很平稳，简单预测也容易”区分开。

注：评分是 **训练样本上的当前预测准确性**，不是经过独立验证的泛化预测可靠性。不能把 q 解释成真正的迁移收益概率。

## 3. 与原 F2 的结合（只改 alpha）

\[
\tilde q_i=\operatorname{clip}\left(q_i/(\operatorname{mean}(q)+\epsilon),0,2\right)
\]
\[
g_i=(1-\rho)+\rho\tilde q_i,\qquad
\alpha_i^{PRC}=
\frac{\alpha_i^{F2}\,g_i}
{\sum_k\alpha_k^{F2}\,g_k}.
\]

- 对照 F2：\rho=0，alpha 严格保持原始 F2（不会重新归一化改变浮点值）。
- 新方案 PRC：只预先指定一次 \rho=0.5，单样本乘子范围 **[0.5,1.5]**，避免直接砍掉困难样本。
- 其余控制完全不动；不调整 OT 项强度，不增加样本对额外 OT cost，不使用测试标签。

原 F2 的梯度相似性保证「可能适合迁移」，PRC 的预测质量保证「源样本当前是可以预测好的」，**二者缺一不可**。但这属于待验证假设，不能认为 q 越高就一定越能迁移。

## 4. 训练代码和位置

- \`model/Meta_Models/prediction_reliability.py\`：纯 torch 无参数计算 q 与 alpha_PRC。
- \`model/Meta_Models/PMME.py\`：Stage-I source forward 时按城市统计源样本预测误差，F2 alpha 算完后做一次有界校准；其余方法不变。
- \`tests/test_prediction_reliability.py\`：零强度原始 F2 等价性、准确样本获得较大权重、城市缩放一致性、无 NaN/Inf 与异常输入。
- \`run_f2_prediction_reliability.sh\`：**仅运行 F2+PRC**，不重复训练 F2；保持作者原始 Stage-II。
- \`summarize_f2_prediction_reliability.py\`：按支持集验证选出的 checkpoint 汇总 PRC 的 Stage-I、Stage-II 36 步与 10/60/120/180 分钟 MAE/RMSE；与旧 F2/T0 seed1 的 2.372713 只作跨提交的历史参考，不伪称同提交配对实验。

PRC 仅使用训练源城市的历史、未来**训练标签**和当前前向预测；没有测试时可见的未来真实值，也没有新增加推理时输入。

## 5. 运行

\`\`\`bash
git fetch origin
git switch experiment/pmme-f2-prediction-reliability-20261126

python -m unittest discover -s tests -p 'test_prediction_reliability.py'
\`\`\`

先在独立目录以 3 个 Stage-I epoch 验证第一个 active OT 的前/反向能走通（这是功能烟测，不是科学结果）：

\`\`\`bash
PMME_PR_RUN_ROOT=./experiments/f2-prc-smoke \
PMME_PR_STAGE1_EPOCHS=3 PMME_PR_STAGE2_EPOCHS=0 \
PMME_PR_DAEPOCH=1 PMME_PR_MODES='prc' PMME_PR_SEEDS=1 \
bash run_f2_prediction_reliability.sh
\`\`\`

正式的 seed1 **只训练 F2+PRC**（不再训练 F2）：

\`\`\`bash
PMME_PR_MODES='prc' PMME_PR_SEEDS=1 \
bash run_f2_prediction_reliability.sh
\`\`\`

如 seed1 的最终 Stage-II 有明确且非偶然迹象的改善，再保持方案不变，补种子：

\`\`\`bash
PMME_PR_MODES='prc' PMME_PR_SEEDS='3 5' \
bash run_f2_prediction_reliability.sh
\`\`\`

可重复独立目录由环境变量 \`PMME_PR_RUN_ROOT\` 管理；每个 run_config.txt 记录 commit、seed、strength。每组均有独立 checkpoint 和 train.out。

## 6. 诊断与成功标准

每隔 25 epoch（以及 DAepoch）打印每城市：
- \`source_mae\` 和 \`persistence_mae\`（物理速度单位）
- \`model_beats_persistence_frac\`：比最简单预测好的比例
- \`quality_mean/std\`
- F2 与 PRC 的 alpha 有效样本数 ESS
- 样本权重乘子的上下界（以防塌缩）

**首先检查：** PRC 是否真的在重排 alpha，而不是几乎全一样；低误差样本是否更高权重；ESS 是否灾难性下降；是否因只选易样本而退化。原始 F2/T0 seed1 已有完整运行结果，Stage-II 36 步平均 MAE = 2.3727133333。无需重复训练。

**最终目标：** 以固定 0.5 参数，在完整官方 Stage-II 的 36 步平均 MAE 低于历史 F2/T0 seed1 的 2.37271，同时 10/60/120/180 分钟与 RMSE 不普遍退化；先运行 seed1，再保持相同设置补 seed3/5。历史 F2/T0 来自不同代码提交/运行环境记录，不构成严格同提交配对；若结果接近，需要谨慎解释。不要按测试集结果调很多 rho 值。

**局限：** 一次源样本的训练误差不能证明迁移后目标会预测得更准；如果只偏好简单样本导致下降，就否决 PRC，而不是继续叠加复杂机制。

本分支**只有代码和设计，没有在 GPU 服务器实际跑过，也没有宣称指标提升或论文新颖性已被证明**。
