# MTPB + Target-Guided Gradient: 一阶段研究版（无目标城市微调）

基线：5zhaozhiyuan5/TSJT-SOFTS-Research 的 `main`
`a4f72961cb26e8204b9b12587207c0998a0d04c9`。
本研究改动在独立 `iteration/mtpb-one-stage-target-gradient-no-finetune` 分支，
没有合并 `main`，也没有触碰任何其他仓库。

## 关键定义：这里的“一阶段”

- **一阶段跨城市适应**：利用源城市数据计算预测学习梯度，目标城市前三天有标签
  数据只用来计算参考梯度、源城市权重、目标正交旋转、残差融合权重和角度控制器
  的元反馈；**不执行任何目标模型参数的微调阶段**。
- 这里的“无微调”指 **不对 MTPB 的预测模型单独执行 target-only SGD/Adam**。
  实际训练仍可用目标标签来影响源梯度选择；这不是零样本预测，也不是完全不用目标监督。
- **离线模式库预训练仍保留**：官方 MTPB 的预训练编码器/模式库应事先从源城市生成。
  因此这里的一阶段是“源域+目标引导联合训练 → 直接测试”的一阶段**迁移协议**，
  不是取消交通模式库离线预训练。若要端到端只训练一次模式库，是后续另一个研究任务。

## 与之前 MTPB 版本的差别

上一版 `iteration/mtpb-replace-meta-update-288x36` 虽已替换源域元更新，
运行入口默认仍在结束后调用 `rep.finetuning(target_train,target_test,...)`。
这一版做两个小而明确的修改：

1. 在 **MTPB 预测模型参数 `theta` 不变**的情况下，
   从 PEMS-BAY 前三天训练窗口计算两个直接梯度 `g_T` 和 `g_Q`，
   不再对 PEMS-BAY 执行 Support 内层参数更新。
2. 源域元训练完成后直接保存 `source_trained_final.pt` 并在目标测试集上
   运行 **无优化器、无微调、无目标测试集选模**的前向评估。严禁调用
   `STRep.finetuning`。

未改动 MTPB 源模式库、模式检索、图重建、Graph WaveNet、长期表示、
融合预测头与 288→36 任务设置。Graph WaveNet 使用 MTPB 官方从模式和原始
图得到的重建图（此处不增加新的道路拓扑、POI、LLM 或固定节点 embedding）。
官方图路径与本仓库 main 既有“历史信号动态图”实现并不相同，
因此两者应视为不同预测模型，切勿直接混淆实验结果。

## 真实更新路径

源城市分别是 METR-LA、成都、深圳。对每个源城市执行
`K=2` 步 Support/Query 一阶元梯度计算，得到
`g_LA, g_CD, g_SZ`。

**目标梯度只在共享参数上计算**：
`g_T = d L_T(theta)/d theta` 和
`g_Q = d L_Q(theta)/d theta`，采自目标前三天不重复的窗口索引。
目标城市不执行虚拟 Support 适应，也不调用真实微调器。
由于窗口是滑动截取，索引不同并不保证历史时间不重叠；
该限制需要通过时间块隔离消融进一步验证。

保留之前的梯度方法接口：

- `source_interaction_principal_gradient()`：城市轨迹 PGrad
  + 正向协作、冲突和冗余建模；
- `rotate_source_gradient()`：目标正交空间旋转；
- `residual_matching_fusion()`：目标查询梯度残差的非负融合；
- `AngleController`：用目标训练窗口查询反馈更新旋转角度。

设 `w_i` 为源权重、`v_i` 为残差融合权重，
`R_i` 为旋转后的目标正交源梯度，最终 **模型参数仅沿源梯度形成的方向** 更新：

```text
g_parallel = sum_i w_i * max(<g_i, g_T>/||g_T||^2, 0) * g_T
g_residual = lambda * sum_i v_i * R_i
g_source_only = g_parallel + g_residual
theta <- theta - meta_lr * clip(g_source_only)
```

虽然 `g_parallel` 写成 `g_T` 方向，但其系数来自源梯度在目标方向上的
**正投影**。源梯度全零时 `g_parallel` 也为零，并不会单独拿目标梯度训练模型。
目标的 Query 反馈**只更新角度控制器**，不触发预测模型的目标监督微调。
实际生成的 `theta` 显然会依赖目标梯度，这是用户希望的引导作用。

## 数据和严格对照

- 输入：288 步历史，输出：36 步未来，统一 5min 采样网格。
- 原 10min 的成都/深圳通过 MTPB 官方 `datasets.py` 的插值逻辑
  转到 5min；需在真实数据上检查对齐。
- `source_train(... add_target=False)`：源域只包括上述三个城市。
- PEMS-BAY 仅前三天 `target_maml` 有监督窗口用于梯度指导；
  `test` 仅用于训练结束后的最终前向评估。
- 目标归一化均值/方差按**前三天的原始速度**统计，避免官方实现
  在划分训练/测试前归一化造成测试期统计信息进入模型。
- 同一 runner 提供 `--mode native`（官方源域元更新 + 直接测试）
  和 `--mode target_guided`（新源梯度更新 + 直接测试）。
  两者都**没有目标微调**；但每轮梯度评估次数不同，
  暂不属于等算力公平对比，应额外记录 wall-clock / FLOPs。
- `direct_target_test.json` 输出真实运行之后的各 horizon 和 overall MAE/RMSE；
  RMSE 使用全量误差平方累计后开根号，而不是平均各 batch RMSE。
- 测试期推理不给优化器，检查参数与 buffer 均无变化。
  官方 `PatchFSL.forward(stage='test')` 会把长期分支重新设为 train，
  因此测试入口加入长期分支 pre-hook 强制 eval，避免测试期 BN/Dropout 更新。
- 不使用目标测试集调参或挑 checkpoint。

## 一次训练和测试

首先按官方 https://github.com/zhyliu00/MTPB 获取数据及五种尺度模式库。
官方原始环境为 PyTorch 1.10；本项目使用 `torch.func.functional_call`
要求 PyTorch 2.x，因此必须先核实 PyG 依赖兼容性。

```bash
# 运行研究代码的核心合成测试
PYTHONPATH=src pytest -q \
  tests/test_mtpb_one_stage_gradient.py \
  tests/test_mtpb_one_stage_eval.py \
  tests/test_mtpb_train_only_norm.py

# 只看脚本参数，不访问真实数据
PYTHONPATH=src python scripts/run_mtpb_one_stage.py --help

# 官方代码和数据准备好之后，先分别跑一轮 smoke:
PYTHONPATH=src python scripts/run_mtpb_one_stage.py \
  --upstream /path/to/MTPB --mode native \
  --epochs 1 --skip-eval --output outputs/mtpb_source_only_smoke

PYTHONPATH=src python scripts/run_mtpb_one_stage.py \
  --upstream /path/to/MTPB --mode target_guided \
  --epochs 1 --order-count 1 --skip-eval \
  --output outputs/mtpb_guide_only_smoke

# 再运行 50 个源域元训练 epoch 并直接测试（均无目标微调）：
PYTHONPATH=src python scripts/run_mtpb_one_stage.py \
  --upstream /path/to/MTPB --mode target_guided \
  --epochs 50 --output outputs/mtpb_one_stage_guided_seed7
```

脚本没有 `--finetune-epochs` 参数，也没有 `rep.finetuning()` 调用。
`--skip-eval` 仅用于计算资源受限时验证单步训练，不用于正式结果报告。

## 可证伪假设与实验

**单一假设**：保留 MTPB 的模式库+GWN 预测结构，仅将目标三天标签
用于引导多源元梯度，在**不做目标微调**的条件下，能比原生 MTPB 的
源域元更新+直接测试取得更低的 PEMS-BAY MAE/RMSE。

应报告：10/60/120/180 分钟 MAE/RMSE、全 36 步指标、
source-target cosine、源权重、融合权重、旋转次数、
训练耗时、目标模型参数是否只经源梯度更新，
以及无目标引导消融、固定目标引导强度消融。
考虑随机性至少使用多个 seed。
不得用原 MTPB 的**先源训练再目标微调**指标作为一阶段效果的同协议基线。
还应提供 Persistence 和单纯 Graph WaveNet 对照。

## 论文依据（不是新方法效果的证明）

1. Liu et al., *Multi-scale Traffic Pattern Bank for Cross-city Few-shot Traffic Forecasting*,
   ACM TKDD 2025，https://arxiv.org/abs/2402.00397，
   官方代码 https://github.com/zhyliu00/MTPB。
2. Zhang et al., *Drawing Informative Gradients from Sources:
   A One-stage Transfer Learning Framework for Cross-city Spatiotemporal Forecasting*，
   AAAI 2025，https://ojs.aaai.org/index.php/AAAI/article/view/32102。
3. Wang et al., *PGrad: Learning Principal Gradients for Domain Generalization*，
   ICLR 2023，官方代码 https://github.com/QData/PGrad。
4. Li et al., *PiKE: Adaptive Data Mixing for Large-Scale Multi-Task Learning Under Low Gradient Conflicts*，
   NeurIPS 2025，
   https://papers.nips.cc/paper_files/paper/2025/hash/f7a94134f1c726796c6f81fb946e489d-Abstract-Conference.html。
   本轮仅复用已有源交互模块，没有追加 PiKE 新算法。

## 本轮验证状态

这是**需要正式验证**的最小一阶段改动。没有获取官方四城市数据、
没有运行官方 PatchFSL 端到端训练，不存在本轮可核实的目标 MAE/RMSE。
即便合成逻辑测试通过也不能替代真实交通结果。
