# H3: 城市级 F2 迁移收益反事实诊断

## 研究目标

D1/D2（原始 F2 Stage-I checkpoint、seed1、16 probe batches）发现：METR-LA 上高梯度余弦对应更有利的局部 OT 梯度，成都、深圳往往相反；GradDrop 符号冲突与梯度余弦相关约 −0.9989，无法证明独立贡献。

**H3 解决更精确的问题：**
原始 F2 的源城市问题是“该城市完全不应做 OT”，还是“该城市的源样本选择/质量分配不合理”？

H3 是**只读反事实诊断**，不是重新训练 F2，也不是新预测模型。它直接加载上轮生成的原始 F2 **完整 Stage-I checkpoint（包括 AdamW optimizer state）**，用真实 GeomLoss OT 梯度、原始单侧 PCGrad、原始 AdamW 动量模拟一次完整参数更新。任何候选方案都不更改实际 checkpoint。

## 七种候选策略 + 一种诊断参照

| 策略 | 成都 | 深圳 | METR-LA | 解释 |
|---|---|---|---|---|
| f2 | F2 | F2 | F2 | 原始基线 |
| city_uniform:chengdu_m | uniform | F2 | F2 | 成都 F2 样本权重是否有问题 |
| city_uniform:shenzhen | F2 | uniform | F2 | 深圳样本权重问题 |
| city_uniform:metr-la | F2 | F2 | uniform | METR-LA 样本权重问题 |
| city_off:chengdu_m | OFF | F2 | F2 | 成都 OT 是否需要启用 |
| city_off:shenzhen | F2 | OFF | F2 | 深圳 OT 是否需要启用 |
| city_off:metr-la | F2 | F2 | OFF | METR-LA OT 是否需要启用 |
| all_off（额外参照） | OFF | OFF | OFF | 整体 OT 是否有局部贡献 |

uniform **保留该城市 OT**，只是将源 alpha 设成均匀；OFF **不计算该城市 OT 对梯度更新的贡献**。这是两种不同的干预。

## 公平性与方法

训练好的 checkpoint 已有原始源城市/目标支持数据训练经历。H3 每次重新采样三城各一个源 batch 和目标 46-window 支持 batch，只在本次反事实分析中将目标窗口 index % 5 == 0 的 10 个窗口作为探针；其余 36 个窗口为 fit。按原模型 [15,15,16] 的目标分组，组内 fit 各 12 窗口。

所有候选方案都使用：
- 相同的源 batch、目标 FIT batch、目标未来标签、SOFTS 预测表示及 GeomLoss cost；
- 相同目标 fit-only 梯度来计算原始 F2 alpha：softmax(8*cosine(rank-one source head gradient, target FIT pooled head gradient))；
- 相同的源任务预测损失梯度、相同的目标 FIT 预测梯度；
- 相同的源城市 OT 系数 0.01，平方未来权重 100 的 sqrt=10，GeomLoss(p=1, blur=0.001, debias=False)；
- 完整参数空间的单侧 PCGrad，加载的真实 AdamW exp_avg / exp_avg_sq / step；
- 同一模型参数、优化器状态、随机池化状态。

仅目标 PROBE 的真实标签用于模拟更新后的预测损失打分，不参与候选 OT 权重、OT 梯度、目标 fit 梯度或更新方向。影子模型接收到的是目标 y=0（预测本来只依赖 x），进一步显式阻断此路径。

真实候选实现：
  g_source = grad(source task losses, ALL parameters)
  g_fit = grad(target FIT prediction loss, ALL parameters)
  g_OT,c^F2 = grad(full original OT with F2 alpha, ALL parameters)
  g_OT,c^U = grad(full original OT with uniform alpha, ALL parameters)

组合七种 OT 梯度，每种分别
  g_k^PCGrad = PCGrad_one_side(g_source + sum_c g_OT,c^k, g_fit)
  theta_k = AdamW_shadow(theta_original, optimizer_state_original, g_k^PCGrad)

相同随机种子下对 PROBE 窗口输出预测，得 loss_k。
  gain_k = loss_F2 - loss_k.
正 gain 表示该干预在这个 probe 批次的一次影子更新后优于 F2。

记录每组的平均 SmoothL1/MAE/RMSE、短中长三个 horizon band 的 SmoothL1，10/60/120/180min MAE/RMSE，以及城市 OT 梯度与 target fit 梯度夹角。

**论文研究限制**：
- 目标探针仍来自三天已训练过的支持集（可能与 fit 窗口时间重叠），因此绝不是独立的测试集/无泄漏的完整训练验证。
- 一次虚拟更新不等于经过 3000+6000 轮训练的最终 MAE，也不保证长期迁移获益。
- 同一 checkpoint 的多次窗口采样相互相关，不能作为独立随机种子试验。
- 默认正式诊断 source_points=0, target_points=0，使用每个城市的原始完整批次 (16×city_nodes) 与 fit 12×325 个目标点。缩小 subset 只用于 smoke，不能等同 full OT。

## 文件与提交

- model/Meta_Models/h3_city_counterfactual.py：策略组合、全参数影子 PCGrad/AdamW 和评价
- audit_h3_city_checkpoint.py：加载 F2 训练完成 checkpoint（含 optimizer state），构造 fit/probe，仅采样原有支持集
- run_h3_city_counterfactual.sh：冒烟、完整诊断脚本，结果隔离
- summarize_h3_city_audit.py：策略级胜率/平均差值和按 horizon 汇总 CSV
- tests/test_h3_city_counterfactual.py：候选组合、fit/probe 隔离、模型及优化器不变等单测

## 服务器运行步骤

前提：此前 D1/D2 分支训练生成的原始 F2 Stage-I 完整 checkpoint 仍在服务器上。默认路径：
/home/zhaozhiyuan/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/l1_pems-bay_SOFTS_seed1.pth

1. 切换：
   git fetch origin
   git switch experiment/pmme-f2-h3-city-counterfactual-20261126

2. CPU 小样本单元测试：
   python -m unittest discover -s tests -p 'test_h3_city_counterfactual.py'

3. 先真实 GeomLoss GPU smoke（使用同一 OT 实现，但缩小样本点数以排障，不是最终实验）：
   PMME_H3_PROBES=1 PMME_H3_SOURCE_POINTS=256 PMME_H3_TARGET_POINTS=128 \
     bash run_h3_city_counterfactual.sh

4. 通过后做正式 H3（全 OT 几何，七策略 + all_off）：
   PMME_H3_PROBES=8 PMME_H3_SOURCE_POINTS=0 PMME_H3_TARGET_POINTS=0 \
     bash run_h3_city_counterfactual.sh

5. 若性能允许，再设置不同输出路径扩充诊断批次：
   PMME_H3_PROBES=16 PMME_H3_RUN_ROOT=./experiments/h3-city-full-16 \
     bash run_h3_city_counterfactual.sh

指定别的 checkpoint：
   bash run_h3_city_counterfactual.sh /absolute/path/to/l1_pems-bay_SOFTS_seed1.pth

输出：
  experiments/h3-city-full/{run_config.txt,train.out,h3_city_results.jsonl,
                            h3_city_summary.csv,summary.txt,COMPLETED}
  experiments/h3-city-smoke/...（独立目录）

服务器 GPU 运行速度和显存取决于实际数据与 GeomLoss；上述脚本**没有在服务器上执行**，请先完成单测与一批 smoke。任何完整训练结果、MAE 提升都不能通过这次代码提交宣称实现。

## 解释方式

- city_uniform:chengdu_m 显著改善，而 city_off:chengdu_m 退化：
  倾向说明成都的 OT 知识有用，但 F2 的内部样本加权可能不合适。
- city_off:chengdu_m 改善，而 uniform 效果不稳定：
  倾向说明该训练状态下成都 OT 强度应被抑制，需进一步验证。
- city_uniform:metr-la 退化：
  可能说明原始 F2 在该城市的样本选择是有效的。
- all_off 显著改善：
  需要重新审查 OT 整体目标与预测任务兼容性。
以上仅是可检验的假设，严禁根据单步/单种子结果直接推断最终负迁移或论文创新有效。

H3 之后如果有持续、方向一致的城市场景差异，优先研究城市级反馈权重系数 λ_city（不直接乘到城市内部单独归一化的 alpha 上）；如果没有，不继续叠加梯度评分，考虑匹配目标本身。
