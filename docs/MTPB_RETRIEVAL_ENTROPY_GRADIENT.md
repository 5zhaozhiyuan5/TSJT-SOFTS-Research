# 五尺度模式检索熵 × 目标引导源梯度（2026-11-10）

基线：main a4f72961cb26e8204b9b12587207c0998a0d04c9。
本研究分支 `iteration/20261110-mtpb-retrieval-entropy-gradient` 已于 2026-11-10 快进更新至 `main`；该提交在 MTPB 一阶段适配路径上增加一个可证伪交叉改动。

## 论文及源码依据

- Liu et al., *Multi-scale Traffic Pattern Bank for Cross-city Few-shot Traffic Forecasting*,
  ACM TKDD 2025，https://arxiv.org/abs/2402.00397 ，
  官方 https://github.com/zhyliu00/MTPB ，核查固定提交
  5676a38a34bf3aebbd83cef3cb0d617a9a153b15 的
  model/Meta_Models/meta_patch.py 第176–234行及 rep_model_final.py：
  PatternEncoder_patternkeyv2 逐尺度计算 [B,N,L/h,K] softmax，
  再用该权重聚合各尺度模式；1/3/6/12/24 小时。
- Zhang et al., *Drawing Informative Gradients from Sources: A One-stage Transfer
  Learning Framework for Cross-city Spatiotemporal Forecasting*, AAAI 2025，
  https://ojs.aaai.org/index.php/AAAI/article/view/32102 。
  本轮沿用目标梯度仅作源梯度引导的一阶段原则，未宣称原论文使用模式熵。
- *PGrad: Learning Principal Gradients for Domain Generalization*, ICLR 2023，
  https://github.com/QData/PGrad 。本轮不修改 PGrad 虚拟优化轨迹。

这些论文不证明本交叉机制能够改善 PEMS-BAY 精度；下述门控是待证伪假设。

## 旧路径 / 新路径

旧路径：官方 MTPB 模式库/检索/聚合/重建图/GWN/长期分支预测 →
源城市一阶 Support/Query 梯度 → PGrad 与源城市交互 →
目标梯度引导旋转与残差融合 → 只执行源梯度组合的共享参数更新。

新路径：在**同一次目标训练窗口参考梯度前向**通过官方
PatternEncoder_patternkeyv2 的 pre-hook，读取其输入 X；
以原模型 Qnet_dict[h] 和 qry_mat_dict[h] 复算
p_h=softmax(q_h(X)W_h)，不新增参数，不修改模式检索结果。
令 H_h=E[-sum_k p_hk log(p_hk)]，C_h=clip(1-H_h/log K,0,1)，
C=mean_h C_h。新的源梯度目标对齐评分系数为
a_T=(1-alpha)+alpha*C，其中 alpha∈[0,1]。
将 a_T 作为 source_interaction_principal_gradient(target_strength=...)
传入真实训练路径。**其余正协同、冗余、PGrad、旋转、角度学习、
残差融合及预测主干保持不变**。

alpha=0：不注册 hook、target_strength=1，回退旧路径；
alpha=1：使用实际检索置信度 C；K<2、无官方编码器或非有限值时
直接失败，不允许静默假造模式信息。

该置信度只是检索集中度，**不是经校准的概率正确性**。
均匀检索可能是多模式真实混合而非不可靠；这是机制的主要风险。
单步额外代价约 O(sum_h B*N*(L/h)*K*D*h) 的检索 logits
重算，且 no_grad，不反传新的梯度；空间按尺度持有一个权重张量。
不新增固定城市 embedding、道路拓扑、POI 或目标测试标签。

## 训练协议

保持官方 MTPB 288→36、五尺度离线源模式库、Graph WaveNet、
图重建、长期分支；METR-LA/成都/深圳为源，PEMS-BAY 前三天
target_maml 仅计算目标参考/查询梯度，无 target Support 内层适配，
无第二阶段微调。目标梯度不会作为单独 SGD 更新；目标测试仅训练后
直接评估。官方离线预训练仍然存在。

## 合成 smoke（不能当真实交通结果）

```bash
PYTHONPATH=src pytest -q tests/test_mtpb_retrieval_confidence.py \
  tests/test_mtpb_one_stage_gradient.py \
  tests/test_mtpb_one_stage_eval.py tests/test_mtpb_train_only_norm.py
```

CPU 本地独立辅助模块测试（同内容）已运行：3 passed。
完整仓库 CI 结果以 Actions 为准；由于本地 GitHub DNS 无法解析，
未在本地执行完整训练路径。无真实四城市 MAE/RMSE。

## 可复现实验（真实数据准备好之后，勿擅自启动长训练）

```bash
PYTHONPATH=src python scripts/run_mtpb_one_stage.py \
 --upstream /path/to/MTPB --mode target_guided --epochs 1 \
 --order-count 1 --retrieval-guidance-strength 0 \
 --skip-eval --output outputs/mtpb_entropy_off_smoke
PYTHONPATH=src python scripts/run_mtpb_one_stage.py \
 --upstream /path/to/MTPB --mode target_guided --epochs 1 \
 --order-count 1 --retrieval-guidance-strength 1 \
 --skip-eval --output outputs/mtpb_entropy_on_smoke
```

正式 50 epoch 仅将 --epochs 改为 50 并移除 --skip-eval，
保持 seed、数据、模式库、训练预算一致。需记录每尺度置信度、
源-目标梯度余弦、源权重、旋转次数、残差融合、耗时，
以及 10/60/120/180min MAE/RMSE、Persistence。没有结果前
**不得声称提升**。

下一轮唯一优先问题：检验“检索熵低 ≠ 模式迁移可靠”的失效情形，
比较熵门控与实际源-目标模式分布相似度能否更好预测梯度冲突。
