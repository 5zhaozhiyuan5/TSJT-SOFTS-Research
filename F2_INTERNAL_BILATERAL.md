# F2 内部双侧梯度边际优化（实验版）

**目的**：仅改 F2 的权重估计内部机制，避免另接 P1/P2/P3/P4，保持 SOFTS、原 OT ground cost、GeomLoss Sinkhorn、PCGrad、AdamW 和官方 Stage-II 不变。
**原始 F2** `PMME_TRANSFER_VARIANT=gradient` 一字不改，并保留为对照。
**新实验** `PMME_TRANSFER_VARIANT=gradient_coupled` 只允许 `PMME_FEEDBACK_MODE=g0` 和 `PMME_PAIR_VARIANT=f2`。

## 当前与新 F2 的输入输出

输入继续为 `source_features [Bs,Ns,128]`、`source_delta [Bs,Ns,36]`、`target_features [Bt,Nt,128]`、`target_delta [Bt,Nt,36]`，均由当前训练的 SOFTS 预测头及训练损失计算，目标标签只用三天训练支持集。输出仍为 `alpha [Bs*Ns]`、`beta [Bt*Nt]`、诊断字典；没有改变后续 OT 代价和 Stage-II。

**原始 F2**：`alpha0=softmax(temperature*cos(g_i^s, sum_j g_j^t))`; `beta0=uniform`, rank-one `g=delta outer h`。
**新 F2**：将两侧每个 rank-one 梯度方向分别归一化为 `u_i^s, u_j^t`。这里每个梯度方向仍来自同一个 projection2 层，并不声称获得全网络或真实 OT 梯度。

构造隐含的任务兼容性 `S_ij = <u_i^s,u_j^t>_F = cos(delta_i,delta_j) cos(h_i,h_j)`，`S` **不显式物化**。目标为在以下代理优化上进行保守的双侧迭代：

`max_{alpha,beta} alpha^T S beta - KL(alpha||alpha0)/T - KL(beta||beta0)/T`

对于两侧固定另一边时的每个子问题，具有先验相对的 softmax 更新：

`target_score_j = <u_j^t, sum_i alpha_i u_i^s>_F`
`beta_candidate_j = softmax(log beta0_j + T * target_score_j)`
`beta_next = (1-rho) beta0 + rho beta_candidate`

`source_score_i = <u_i^s, sum_j beta_next_j u_j^t>_F`
`alpha_candidate_i = softmax(log alpha0_i + T * source_score_i)`
`alpha_next = (1-rho) alpha0 + rho alpha_candidate`

默认 `T=8.0`（沿用 F2 temperature）、`rho=0.2`、迭代 `2` 次。混合控制器是为了保留原 F2 源先验和广覆盖目标分布；它是**正则化子问题候选解的阻尼更新**，不是上述代理目标的精确联合最优解。

不额外构造密集 `N_source x N_target` 矩阵；每一轮用 `torch.einsum` 聚合和评分，计算规模约随样本数线性增长，依赖固定的 36×128 预测头维度。目标侧权重动态调整，源侧权重通过目标的当前权重反过来细化。

## 为什么称为 F2 内部修改？

- 不动 `g_i=delta_i h_i^T` 的梯度提取原理。
- 首先完整计算**原始 F2** 的 `alpha0`。
- 只把原来的 `(alpha0, uniform beta0)` 输出，换成相互条件化的 `(alpha, beta)`。
- 不添加任何可训练参数或第二次反向传播；`compute_weights` 全程 `@torch.no_grad()`。
- 不修改 OT 匹配代价；与 P1–P4 的 cost 增强有严格区别。
- 不是新的 DVGP / GP 分类器。

## 代码

- `model/Meta_Models/transfer_utility.py`：新增 `gradient_coupled` 权重分支。
- `model/Meta_Models/PMME.py`：增加该实验配置，保留其他模式原有逻辑。
- `tests/test_f2_internal_bilateral.py`：检查概率归一、零梯度、0 混合强度/0 迭代恢复原始 F2、目标权重响应。
- `run_f2_internal_bilateral.sh`：原始 F2 与新 F2 独立日志/目录、同配置对比。

## 运行

```bash
git fetch origin
git switch experiment/pmme-f2-internal-bilateral-20261125
python -m unittest discover -s tests -p 'test_transfer_utility.py'
python -m unittest discover -s tests -p 'test_f2_internal_bilateral.py'

# Stage-I + Stage-II 完整实验；先调通 seed1，再补 seed3/5
PMME_F2_SEEDS="1" bash run_f2_internal_bilateral.sh

# 两轮 Stage-I 的 OT 激活烟雾测试，输出必须用新目录
PMME_F2_RUN_ROOT=./experiments/f2-internal-smoke \
PMME_F2_STAGE1_EPOCHS=2 PMME_F2_DAEPOCH=1 \
PMME_F2_STAGE2_EPOCHS=0 \
PMME_F2_SEEDS="1" bash run_f2_internal_bilateral.sh

# 消融：关闭新内部机制，与原 F2 应严格一致
PMME_F2_RUN_ROOT=./experiments/f2-coupled-zero \
PMME_F2_VARIANTS="gradient_coupled" \
PMME_F2_COUPLED_BLEND=0 \
PMME_F2_SEEDS="1" bash run_f2_internal_bilateral.sh
```

完整实验使用 Stage-I 3000 轮、OT 在 1500 轮激活、官方 Stage-II 6000 轮；原始官方第二阶段文件没有改动。比较同 seed 的 10/60/120/180 分钟 MAE/RMSE。日志记录目标权重 ESS、最大权重与均匀值比、源权重 ESS。强制记录 checkpoint 和运行时长，禁止在目标测试数据上调参。

## 实验地位与局限

这仅是**第一版** F2 内部双侧改进，假设“目标权重应由当前高权重源样本的梯度方向条件化”尚待验证。双侧 importance weighting、KL 约束和交替 Softmax 有先例，不能声明已证明算法原创性。当前没有完成带 GeomLoss、真实交通数据和 GPU 的 Stage-I/II 全流程，也没有任何精度提升结果。此算法只使用预测头梯度代理，不解决预测梯度与真实 OT 更新方向的不一致；这项限制需单独诊断，而不是宣称已经解决。
