# H10 — 官方 F2 更新一致性 + 预测退化来源定位（含强制失败门槛）

**分支：** `experiment/pmme-h10-update-provenance-20261127`  
**基于：** H9 完整日志提交 `dbc63c5cc14e76e1db1c89b53bff260c69f05153`。

> **实现已提交，服务器测试和 GPU 结果尚未运行。** 本实验不重新训练 F2，不改变原始 checkpoint、SOFTS、数据加载器或官方 Stage-II；仅对单独复制的模型及 AdamW 状态执行更新。

## 0. 重要勘误：原始 F2 确实使用 PCGrad

**必须依据 checkpoint 的真实训练日志，而不能根据 train_DA_OT.py 中参数的默认值猜测。**

原始 3000 epoch F2 日志在：
`experiment_logs/experiment-pmme-f2-gradient-utility-diagnostics-20261126/prerequisite_original_f2_stage1/seed1/train.out`

其中训练命令的配置显示 **`projection: 1.0`**，每个 epoch 都打印 `self.projection_args 1.0`。因此其实际训练方向为：

```text
g_s = grad(source forecasting losses)
g_t = grad(target forecasting loss across ALL 46 support windows)
g_ot = grad(original F2 full GeomLoss OT, with ALL 46 support windows)

PCGrad_one_side(g_s + g_ot, g_t)
  = g_s+g_ot - min(0, <g_s+g_ot,g_t>)/(||g_t||²+1e-8)*g_t

AdamW_original_loaded_state.step(projected_gradient)
```

原始 `PMME.py` 的 `projection_args==0` 分支确实不执行 PCGrad，但**不是这份原始 F2 checkpoint 所对应的训练分支**。H10-T5 使用 **projection=1** 的真实更新，H10-T6 则刻意关闭 PCGrad 做消融。

H9 虽然也使用了 PCGrad，计算 OT 和目标梯度却将 46 个支持窗口按 modulo-5 拆分，排除了 10 个窗口。这是 H6–H8 诊断允许的 FIT/PROBE 设计，但不忠实于原始训练。H10 所有梯度组恢复使用**完整 46 支持窗口**。不再称旧 H9 的 F2 策略为严格官方复现。

## 1. H10-A：数值一致性 HARD GATE

同一个原始 Stage-I F2 epoch2899 checkpoint，SHA：

```text
f04c73058382a49204cf454097d41dcf22df3a262b9636484e13e4e8d3346baf
```

相同 seed1、源城市和 target 46-window support batch、训练前向随机种子、SOFTS、梯度/OT 代价、`alpha,beta`、AdamW 学习率、权重衰减和**全部历史一阶/二阶状态**。

A1 对独立复制的 STRep wrapper 调用 `meta_train_revise(...,epoch=2899)`，真正沿用原始 F2 的 `projection=1` 实现；A2 独立复制 PatchFSL 和 AdamW，显式求 `g_s, g_t, g_OT`，按原始单侧 PCGrad 与**补零梯度**语义运行真实 AdamW。两者都执行一次真实克隆更新。源/目标样本索引相同。

**强制条件：**
- 全部 source/target OT 点 `source_points=0,target_points=0`；抽样 OT 不允许用于与官方全量梯度做一致性判断。
- 目标支持历史与未来全部 46 窗口参与 target gradient 与 F2 alpha/OT。没有 36/10 拆分。
- 对比每个参数的最大绝对差、每个模型 buffer、AdamW 每一条状态。
- 每项误差不得超过预注册 `2e-5`，optimizer 参数组超参必须相等，原始模型+AdamW 对比前后完全不变。
- H10-A **不通过**时报告 JSON 仍保留以排障，但是 shell 标记 `FAILED`，**不运行 H10-B，不报告假设机制有效**。
- 若出现数值差异，优先排查：随机前向、F2 全46权重、target-gradient scale、原始损失乘数和 PCGrad None/zero 语义。不能在未查明原因前增加实验变量。

## 2. H10-B：9 组因果归因对照（T0/T0-F/T1–T7）

在 A 通过后，从**同一原始 checkpoint**及其 AdamW 状态，每个方案独立复制，所有方案每步共用 source/target batch 随机种子与样本指纹；每一步分别重算当前模型的梯度。只在开发区间评分，不使用开发标签训练。

| Policy | 实际动作 | 用来检验 |
|---|---|---|
| `t0_no_update` | 不做任何训练前向或参数更新 | 原始已有 checkpoint 的开发预测能力 |
| `t0f_train_forward_only` | 执行相同训练前向但不反传/step | 训练模式、buffers、缓存等状态影响 |
| `t1_zero_grad_adamw` | 给每个参数**显式零梯度**后 AdamW.step | checkpoint 动量 / 权重衰减 / 历史状态漂移 |
| `t2_source_only` | (g_s) | 源预测梯度的单独贡献 |
| `t3_target_only` | (g_t)（46 全支持） | 目标适配梯度的单独贡献 |
| `t4_source_pcgrad` | PCGrad((g_s,g_t)) | 无 OT 时的目标引导投影 |
| **`t5_official_f2`** | **PCGrad((g_s+g_{OT},g_t))** | **真实训练分支对应的 F2 更新** |
| `t6_full_pcgrad` | (g_s+g_{OT}) **不经过 PCGrad** | 移除原始 PCGrad 的消融；文件中沿用 ID，含义明确 |
| `t7_project_ot_half` | PCGrad((g_s+0.5P(g_{OT},g_t),g_t)) | 仅在 OT 分量去冲突，同时保留官方整梯度 PCGrad |

注意 T1 的 zero-gradient 与 None-gradient 不同：原始 AdamW 在零梯度步骤仍然可能通过动量和权重衰减更新参数。官方 `projection=1` 的 `flatten_and_pad_grads`+`write_back_flat_grads` 使**每个参数**拥有显式梯度张量（包括缺失→零）；H10-T5/T4/T7 按此语义复现。T6 则保持直接 backward 时非参与参数为 None 的语义。

### 多步对比

阶段一优先 1 步全量 OT，只有完成 H10-A 保真并分析 1 步结果后，才选择 5 步完整 OT 或 5 步 capped OT 作为初筛。

- 最多 5 步，每次在当前策略模型上重算 source/target/OT。
- 每个策略独立 checkpoint 和 AdamW 的深拷贝，不覆盖原始文件。
- 结果包含：每步模型参数位移、AdamW 学习率、g_s/g_t/g_OT 范数与余弦、各源城市 OT 范数、alpha/beta mass/ESS/top24、OT sample IDs hash、全开发集 MAE/RMSE、短中长期及分天指标。
- 未显式生成完整 OT 传输矩阵 Π/C。GeomLoss 内部隐式求解，**不能据此声称有完整 Π/C 矩阵日志**。

## 3. 时间切分和限制

沿用 H9 的开发集，只用 PEMS-BAY 目标时间：

```text
target raw time  [0, 864)      = 3天训练支持窗口
target raw time  [864, 2016)   = 固定4天开发（70个重叠滚动窗口）
target raw time  [2016, end)   = 本次不作为开发打分，也不用于策略选择
```

开发历史/标签不进入梯度；model 前向看到的开发标签始终为零，真实标签只在外部用于评分。

**不能叫作全新独立测试集：** 官方原始训练在 checkpoint 间隔曾计算支持集之后的 test 指标，且归一化统计量来源于目标全序列。当前阶段用于问题定位/选方案，不直接与论文标准 test 数字比较；策略选择后另立严格的验证/测试协议。

## 4. 运行流程

```bash
git fetch origin
git switch experiment/pmme-h10-update-provenance-20261127

# CPU测试（不需要 F2 数据、GeomLoss 或 GPU）
python -m unittest discover -s tests -p 'test_h10_update_provenance.py'

# STEP 1: H10-A 真·官方一整步 vs 完整F2手动一整步，硬性一致性检查
PMME_H10_PHASE=a \
PMME_H10_SOURCE_POINTS=0 PMME_H10_TARGET_POINTS=0 \
PMME_H10_RUN_ROOT=./experiments/h10-a-full \
bash run_h10_update_provenance.sh

# STEP 2: 只有 H10-A passed 且有 h10_results.json 才跑 B
# 先执行一批轻量级的 T0–T7 同条件消融（capped 256/128）
PMME_H10_PHASE=b \
PMME_H10_PARITY_REPORT=./experiments/h10-a-full/h10_results.json \
PMME_H10_SOURCE_POINTS=256 PMME_H10_TARGET_POINTS=128 \
PMME_H10_MAX_UPDATES=1 PMME_H10_DEV_MAX_WINDOWS=16 \
PMME_H10_RUN_ROOT=./experiments/h10-b-smoke \
bash run_h10_update_provenance.sh

# STEP 3: 正式 H10-B，9组方案各执行1次 FULL OT 更新
PMME_H10_PHASE=b \
PMME_H10_PARITY_REPORT=./experiments/h10-a-full/h10_results.json \
PMME_H10_SOURCE_POINTS=0 PMME_H10_TARGET_POINTS=0 \
PMME_H10_MAX_UPDATES=1 PMME_H10_DEV_MAX_WINDOWS=0 \
PMME_H10_RUN_ROOT=./experiments/h10-b-full-1 \
bash run_h10_update_provenance.sh

# STEP 4 (可选): 若前三步排障一致、单步诊断有解释价值，再跑 5步
PMME_H10_PHASE=b \
PMME_H10_PARITY_REPORT=./experiments/h10-a-full/h10_results.json \
PMME_H10_SOURCE_POINTS=256 PMME_H10_TARGET_POINTS=128 \
PMME_H10_MAX_UPDATES=5 PMME_H10_DEV_MAX_WINDOWS=0 \
PMME_H10_RUN_ROOT=./experiments/h10-b-pilot-5 \
bash run_h10_update_provenance.sh
```

**请顺序运行，不要同时跑多个任务；一致性不过关不要启动下一步。** 每个 run 目录带 git SHA/参数/原 checkpoint SHA，禁止跨参数覆盖旧结果。

## 5. 输出目录和诊断

```text
experiments/h10-a-full/
  h10_results.json                   # 官方 A1/A2 原始参数/优化器误差证据
  h10_decision.json                  # A gate pass / fail
  train.out
  summary.txt
  run_config.txt
  COMPLETED / FAILED

experiments/h10-b-full-1/
  h10_results.json                   # B各策略步骤、真实 OT、梯度、开发指标
  h10_policy_development.csv         # MAE/RMSE分步，正值为改善
  h10_daywise_development.csv        # 按开发预测天
  h10_decision.json
  train.out, summary.txt, run_config.txt
  COMPLETED / FAILED
```

H10 的首要判据不是先追求极小的 MAE 改善，而是能够**以保真的原始官方梯度步骤**分辨：预训练权重已经好到不适合继续更新？还是源梯度、OT 或已有 PCGrad/AdamW 历史造成了失效？如果所有更新策略都使开发集退化，那么阶段末 checkpoint 的少步适配不应继续作为论文主要创新场景；应改为在 Stage-I 训练过程中研究调整，或重审 OT 样本匹配目标。

## 6. 已新增文件（仅新文件，官方代码未修改）

1. `model/Meta_Models/h10_original_f2_bank.py`：修复 H9 的 target 36→46，记录真实 F2 OT 边际与梯度及 None mask。
2. `model/Meta_Models/h10_update_provenance.py`：T1–T7 更新方向、真实 AdamW、官方 wrapper 克隆一致性。
3. `audit_h10_update_provenance.py`：H10-A 官方一致性硬门槛和 H10-B 各实验工作流。
4. `run_h10_update_provenance.sh`：固定 SHA、分阶段运行、防覆盖/失败终止。
5. `summarize_h10_update_provenance.py`：结果摘要与分天诊断。
6. `tests/test_h10_update_provenance.py`：CPU梯度、optimizer、克隆与一致性相关测试。
7. `H10_UPDATE_PROVENANCE_EXPERIMENT.md`：实验预注册及执行命令。

**提交代码不等于 GPU 已验证。务必等 H10-A 一致性结果合格后再讨论后续实验收益。**
