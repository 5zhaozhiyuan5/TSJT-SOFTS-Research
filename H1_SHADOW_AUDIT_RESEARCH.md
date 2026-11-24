# H1：F2 梯度加权 OT 的真实优化路径诊断（2026-11-25）

## 研究任务

2026-11-24 的 PEMS-BAY / SOFTS / seed1 对照结果：

| 方案 | 10 / 60 / 120 / 180 分钟 MAE |
|---|---|
| F2 = G1 | 1.37044 / 2.30205 / 2.67932 / 2.92548 |
| G2（旧单头虚拟收益选权） | 1.37936 / 2.31660 / 2.68862 / 2.93983 |
| G3（额外分时长风险惩罚） | 1.39301 / 2.32676 / 2.70598 / 2.95800 |

G2/G3 均未超过 F2。此前 G2 以“只计算 projection1.weight 的局部梯度、再相对于*当前未更新*预测值衡量损失变化”的指标去控制全模型 OT；这个反事实参照点和模型真正执行的 PCGrad+AdamW 更新均不一致。

因此，**下一轮先做 H1，不训练 H2/H3 新策略**：在 F2 训练过程中偶尔比较“真实优化路径下源任务更新”与“真实优化路径下源任务+F2 OT 更新”这两个全模型影子结果，然后判断此前的单头近似是否可靠。

## 现有研究的边界和直接先例

- Ren et al., *Learning to Reweight Examples for Robust Deep Learning*, ICML 2018，https://proceedings.mlr.press/v80/ren18a.html ：用参考损失和虚拟一步更新指导样本加权（H1 思路有明确先例）。
- Koç et al., *Incorporating Importance Weighting in Optimal Transport Based Domain Alignment*, ICML 2026，https://proceedings.mlr.press/v306/koc26a.html ：OT 重要性加权与域对齐已有直接研究。
- *Bi-level unbalanced optimal transport for partial domain adaptation*, Pattern Recognition 2026，https://www.sciencedirect.com/science/article/pii/S0031320325016619 ：双层优化与 OT 的结合已有相近工作。
- PyTorch `torch.func.functional_call` 正式接口：https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html ：使用替代参数执行影子前向，而非篡改真实模型/优化器。

**H1 是诊断消融，不能把“虚拟更新 + 目标损失反馈”称为原创算法。**

## H1 的可复核公式

令：
- `L_s` 为 3 个源城市的监督预测损失之和（实际训练所用）；
- `L_t` 为目标城市前三天支持窗口的监督预测损失；
- `L_OT` 为当前 F2 生成权重、官方完整批次 Sinkhorn、固定缩放 0.01 对应的**真实完整**对齐损失。

计算所有 SOFTS 参数的梯度：

```text
g_s  = grad(L_s)
g_t  = grad(L_t)
g_OT = grad(L_OT)
```

对与目标梯度冲突的部分执行**和原 PMME 完全相同的单侧向量投影** `P(g,g_t)`。

```text
g_A = P(g_s,        g_t)          # 无 OT
g_B = P(g_s + g_OT, g_t)          # F2 OT，当前真实梯度
```

根据**当前 AdamW 优化器每个参数的 exp_avg、exp_avg_sq、step、lr、weight_decay**，分别模拟一步得到 `theta_A` 与 `theta_B`。这不是简单取 `theta - eta*g`，也不更改真实优化器状态。

在**同一组目标支持探测窗口、同一次固定随机池化**下，用 `torch.func.functional_call` 得到两个预测及同一 Huber（beta=5e-4）损失：

```text
full_gain = L_probe(theta_A) - L_probe(theta_B)
```

正号：在这个单步、支持集内、随机数配对的比较中，加上 F2 OT 更好；负号：在该条件下不如不加 OT。

此外用**相同的两种 AdamW 更新结果**，仅重放 `projection1.weight+bias` 对已缓存 SOFTS 编码特征的影响，得到：

```text
head_gain = L_probe(head_A) - L_probe(head_B)
```

该 readout-only 计算在固定编码特征/固定 projection2 的情况下线性精确，不代表全模型。与 `full_gain` 比较可诊断：G2 之前将局部近似代理当成完整模型更新指导时是否存在明显偏差。

### 评判指标

- `sign_agreement_rate`：head_gain 和 full_gain 的符号一致比例。
- `mean_abs_head_full_disagreement`：二者收益数值差异。
- `pearson_full_head_gain`：二者的线性相关（样本数少时仅参考）。
- `full_gain_positive_fraction`：在当前支持集单步对照中 F2 OT 优于 no-OT 的占比。
- `full_band_gains`：分短、中、长期的单步收益；只诊断，不调整权重。
- OT 相对任务/目标梯度范数与夹角。

## 特别重要：什么没有发生

1. **没有改变 F2 的 alpha/beta、OT 强度、Sinkhorn 参数、源任务损失、PCGrad、AdamW 的真实训练分支或 Stage-II**。
2. H1 虚拟模型使用所有预测模型参数及其 AdamW 当前状态，但结果不调用真实 `optimizer.step`，不回写任何参数、动量、`.grad` 或随机状态；模型临时缓存、training flags 也恢复。
3. H1 没有对 G2 原来 rho=0 / 0.5 / 1 的**小样本候选 OT**逐一做全模型 shadow 回放；H1 只回答**F2 当前真实完整 OT 对比 no-OT**的单步问题。所以即使 H1 很好，也不能直接宣称 G2 离散策略评价已经可靠。
4. 用于审计的探测窗口仍来自前三天训练支持数据，**不是独立验证集**。目标时间窗口有重叠，模型在先前训练中已见过这些时间段，H1 的收益不等于未来三天外的预测收益。
5. H1 的两个 shadow forward 固定相同随机池化，以消除这两次前向本身的随机差异；它并未保证未来 1000 次更新也有效。
6. H1 每 200 轮审计一次（Stage-I OT 从 epoch1500 开始），共约 8 次，这有助于减少额外计算，但不足以形成可靠统计推断；相关性应结合更多 seed、更多时点讨论。
7. **源代码已提交，H1 的真实 GPU 端到端与 CPU 单测尚待服务器执行。** 不能将其视为已经跑出的实验结果。

## 运行命令

```bash
git fetch origin
git switch experiment/pmme-h1-shadow-ot-audit-20261125

# 先执行数学单元测试，再跑完整 seed1
python -m unittest discover -s tests -p 'test_h1_shadow_audit.py'
python -m unittest discover -s tests -p 'test_transfer_utility.py'

PMME_SEEDS="1" bash run_h1_shadow_audit.sh
```

该脚本保持官方 PEMS-BAY / SOFTS / 三天目标训练、Stage-I 3000 epoch + Stage-II 6000 epoch、seed1、固定 DAcoef=1、DAepoch=1500、xySquaredRatio=100、projection=1、MemSize=10000、OT 缩放 0.01、预测梯度 F2 温度8。**H1 训练精度应接近 G1/F2 的对应 seed1**；若差异显著，优先排查影子前向的状态恢复/随机数保护，而不是解释为模型创新。

日志目录：
`experiments/h1-shadow-audit/scale0p01/interval200/seed1/`

脚本自动生成：
- `train.out`: 原训练信息及每 200 轮 `h1_shadow_audit {...}`
- `h1_shadow_records.csv`: 8 条左右的逐轮收益与分时长指标
- `h1_shadow_summary.txt`: 相关性、方向一致率、OT 相对范数
- `run_config.txt`: seed / OT 系数 / audit interval / git commit
- `COMPLETED` / `FAILED`: 标志

独立重算：
```bash
python summarize_h1_shadow.py experiments/h1-shadow-audit/scale0p01/interval200/seed1/train.out
```

若有证据表明局部代理不可靠，下一阶段改进全模型/低秩梯度影响近似；若全模型影子收益和后续效果也没有关联，应考虑抛弃 G2 的在线反馈，不再继续在此基础上堆模块。**无论如何，先不动已经有正向证据的 F2。**
