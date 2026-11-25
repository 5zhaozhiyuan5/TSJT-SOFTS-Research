# H2-A：在 H1 之上做三个连续、可检验的候选 OT 实验

目标：解释为什么原 G2/G3 的预测效果比 F2 差；当前只记录诊断，不改变真正的 F2 训练。

## 0. 研究约束与既有结果

- 目标 PEMS-BAY，只使用前三天目标支持数据；源城 Chengdu、Shenzhen、METR-LA；SOFTS、36预测步。
- 论文原始 Stage-II 不变；原始 `train_DA_OT.py`、原来的 SOFTS、F2 (`transfer_utility.py`) 权重逻辑一字不改。
- 上轮 F2/G1 seed1: MAE(10/60/120/180min) = 1.37044, 2.30205, 2.67932, 2.92548；G2/G3 不及 F2。
- H1 已实现全参数 PCGrad + 当前 AdamW 状态的一步影子收益，但只比较 F2 OT vs no-OT。这一轮从 H1 延伸到**多候选 / 源城市贡献 / 预测窗口稳定性**。
- 元样本权重与 OT 重要性加权已有研究：
  [Ren et al., ICML 2018](https://proceedings.mlr.press/v80/ren18a.html)；
  [Koç et al., ICML 2026](https://proceedings.mlr.press/v306/koc26a.html)。
  当前 H2-A 是诊断实验，不声明新算法原创性。

## 1. 三个实验：同一次训练，三份独立报告

| 实验 | 候选 / 对照 | 目的 | 日志 |
|---|---|---|---|
| **H2-A1：全批次 OT 权重对照** | 不用 OT、均匀 OT、均匀/F2 50% 混合 OT、完整 F2 OT | 确定单步全模型收益是否偏好梯度加权 | `h2_a1_global.csv` |
| **H2-A2：源城市贡献消融** | 固定另两城的 F2；仅把一座源城改成均匀 OT，分别对三城实验 | 检查是否所有源城市都应使用同样强度的 F2 梯度权重 | `h2_a2_citywise.csv` |
| **H2-A3：目标时间窗口稳定性** | 使用 A1/A2 同一组影子模型，在 3 个连续目标支持窗口块上分别评分 | 检查不同支持时间段的候选排序是否稳定 | `h2_a3_folds.csv` |

三个实验都在同一次训练、同一模型权重和同一批随机状态下得到；无需把相同的 Stage-I + Stage-II 分别重新训练三次。

### 权重与真实 OT

F2 源 marginal `alpha_f2` 是原算法计算的分布；均匀 marginal `alpha_uniform=(1/N,...,1/N)`。

- no-OT：`L_s`
- uniform：`L_s + L_OT(alpha_uniform)`
- mixed：`L_s + L_OT(0.5 alpha_uniform + 0.5 alpha_f2)`
- F2：`L_s + L_OT(alpha_f2)`
- A2 的三种方案：两座源城市用 F2，一座指定源城市用 uniform。

**每个 OT 都使用实际完整批次的源/目标表示及真实未来值，官方 GeomLoss Sinkhorn（p=1，blur=0.001，debias=False），同一乘积 `DAcoef * PMME_TRANSFER_OT_SCALE=0.01`。** 不采用旧 G2 的 256/128 代理子集，不改变 OT 传输算子。

使用 H1 的全参数单侧 PCGrad 与 AdamW 动量状态，只模拟一步，并用 `torch.func.functional_call` 在**同一组目标支持窗口 / 同一 STAR 随机池化抽样**上比较结果；不做真实 optimizer.step、不回写 .grad 或模型参数。

定义相对无 OT 的收益：

```text
gain(candidate)=Huber_target[ shadow(no-OT) ]
               -Huber_target[ shadow(candidate) ]
```

正：这个支持集上的一步影子结果更好。负：相反。A2 也计算把单城改 uniform 的 `gain_vs_f2`，正说明该城在当前单步对照中用 uniform 更好。

A3 的三段只作为**同一已使用目标支持数据中的描述性稳定性分析**。三个时间段可能共享相邻预测区间，且建模与 OT 梯度使用了整个目标支持数据，**并不是独立验证集**；不能声称获得了未见数据上的样本外收益。

### 容易误解的结论边界

- H2-A1/A2/A3 不会改变实际的 F2 OT 权重或训练更新，也没有训练新的收益控制器。Stage-II 仍是官方实现。
- A2 是**源城市级权重替换**，不是给每条样本对学新权重；不能冒称解决了完整样本对级对齐。源城市抽样顺序随机，每座源城市所配对的目标 15-window 块可能跨 epoch 改变；日志记录 `target_group`，分析城市效应时必须分层查看，不能混淆城市差异与目标窗口组差异。
- 一步当前支持集目标损失不是未来交通预测质量；如果 H2 单步赢家不稳定，不能立刻让它在线控制真实 OT。
- 不应因为 A3 三段结果不同，就直接重做 G3 的分时长风险惩罚。
- 当前分支是 H1 的扩展；H2 新代码没有覆盖 H1 的已有文件，H1 模式仍可使用。
- H2 计算真实完整 OT 代价较高，仅每隔默认 200 个 Stage-I epoch 执行一次；额外的 Sinkhorn 图、影子模型按候选逐项释放，避免一次保留多个大模型。

## 2. 如何运行

```bash
git fetch origin
git switch experiment/pmme-h2-ot-candidates-20261125

python -m unittest discover -s tests -p 'test_h2_candidate_audit.py'
python -m unittest discover -s tests -p 'test_h1_shadow_audit.py'
python -m unittest discover -s tests -p 'test_transfer_utility.py'

PMME_SEEDS="1" bash run_h2_candidate_audit.sh
```

**默认只跑 Stage-I 3000 轮**，因为此次三个实验只做 Stage-I 单步收益诊断，并不改变预测模型；之前 F2/H1 已完整运行官方 Stage-II。Stage-I 的官方 checkpoint/target support 选择路径不变。设定 `--finetune_epochs 0` 只跳过末尾本轮不必要的 Stage-II，**并不改变 Stage-I 的训练算法**；如果希望独立复核完整预测：

```bash
PMME_H2_STAGE2_EPOCHS=6000 PMME_SEEDS="1" bash run_h2_candidate_audit.sh
```

另外可用 `PMME_H2_INTERVAL=100` 加密 Stage-I 检查，或 `PMME_H2_CITYWISE=0` 仅运行 A1 + A3 的影子前向（仍计算三个梯度候选）。如需 seed3/5，执行 `PMME_SEEDS="3 5" bash run_h2_candidate_audit.sh`。分支、代码提交和配置记录于每个单独输出目录。

默认日志位置：

```text
experiments/h2-candidate-audit/scale0p01/interval200/citywise1/stage20/seed1/
  run_config.txt
  train.out
  h2_a1_global.csv
  h2_a2_citywise.csv
  h2_a3_folds.csv
  h2_summary.txt
  COMPLETED
```

### 我们下一步如何根据结论推进 H2-B？

1. 如果 global F2 经常显著优于 uniform，而单城替换不稳定：保留 F2，不要激进控制权重。
2. 如果某座源城改 uniform 在多个轮次/时间块上 consistently 更好：才有动机做**源城条件化的保守 OT 权重**，随后跑真实训练验证。
3. 如果三个目标支持窗口上的最优候选变化剧烈、margin 接近 0：说明收益决策信号偏弱，先研究稳健估计而非直接上线 H2-B。
4. 如果全模型收益和 H1 原来的局部收益经常相反：优先修正局部代理；但 H2-A 本身已经使用全参数影子更新。
5. 如果验证 H2-B，才用 Stage-I 3000 + Stage-II 6000 的真正训练，严格与相同 seed 的原 F2 做 MAE/RMSE 配对对照，不用 Stage-II 误差代替阶段一诊断。

当前代码和测试已提交，但没有访问用户 GPU/真实交通数据，**尚未实际跑出 H2 实验数字**。完整 CPU 测试也需在服务器环境执行，以便验证 PyTorch/GeomLoss 版本。
