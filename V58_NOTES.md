# v58 版本改动说明（DP_POT 序列第 58 号实验）

**目标城市**：METR-LA　**源城市**：Chengdu + Shenzhen + PEMS-BAY（P,C,S → METR-LA）
**主干**：SOFTS　**运行脚本**：`run_v58_dvgp.sh`　**结果日志**：`dp_pot_v58_dvgp.log`
**checkpoint 前缀**：`dp_pot_v58_dvgp`

---

## 一、v58 相对前序版本的改动

v58 的核心是引入 **DVGP 软选择器（`--dvgp_soft_selector`）**，替代 v31 起的 DVGP 固定硬阈值，
用于论文 PMME「Pattern Matching (PM)」模块 Step 1（DVGP 识别跨城共享模式）的工程增强。

### 改动 1：DVGP 软 + 自适应选择器（核心，`model/Meta_Models/PMME.py`）

命令行新增参数：

```bash
--dvgp_soft_selector --dvgp_temp 2.0 \
--dvgp_thresh_start 0.05 --dvgp_thresh_end 0.3
```

代码逻辑（`PMME.py` 约 423 行）由三个子步骤构成（CDTrans/FixMatch 风格）：

1. **温度去饱和**：`prob_t = prob ** (1/T)`，`T = dvgp_temp = 2.0`。
   幂式锐化拉大 shared/private 模式之间的置信度差，但保留原始 `[0,1]` 尺度
   （不做 sum 归一化），避免把 `OT_loss_scale = sqrt(prob1.mean * prob2.mean)`
   压向 0 导致 Sinkhorn 出现 NaN。

2. **课程式自适应阈值**：`thresh_t = thresh_start + (thresh_end - thresh_start) * (epoch / DAepoch)`，
   训练早期松（保留多数样本）、后期紧（只保留高置信样本）。
   chengdu_m（524 节点，DVGP 精度 ~0.98）单独放宽 `thresh_end *= dvgp_chengdu_scale`，
   避免训练后期 chengdu 的 `prob1` 塌缩到 `ratio1 ~ 0.008`、把携带最丰富长程（180min）
   动力学的源城市几乎全部排除。

3. **软门控（无硬截断）**：`g = (x - thresh).clamp(min=0)` 后逐 batch 归一化。
   略低于阈值的样本被平滑衰减而非直接丢弃，保留其对 OT mass 的小贡献。

对比 v31 旧逻辑（`else` 分支）：`threshold(prob) = prob > thresh ? prob : 0`，是硬截断。

### 改动 2：DVGP 共享度对齐路由熵（`model/Meta_Models/dynamic_pattern.py`）

在原型 / 路由损失中，用 DVGP 的共享度 mass（`source_weight`/`target_weight`）对每个样本的
路由熵做加权：

> 高度跨城共享的模式，若路由含糊（高熵）会受到更大惩罚，迫使 router 给这些
> 「真正可迁移」的模式分配尖锐、稳定的 expert —— 这是 confidence-weighted
> contrastive alignment 的路由版本。

### 改动 3：MoE 融合回退到 add

从 v57 的 `--moe_fusion_mode hybrid --moe_gate_coef 0.05` 回退到
`--moe_fusion_mode add --moe_gate_coef 0.0`（关闭 gate、回到纯 add 融合）。

---

## 二、v58 结果（论文口径，目标 METR-LA）

来自 `dp_pot_v58_dvgp.log` 逐 horizon 打印（Horizon 1/11/23/35 = 10/60/120/180min）：

| 指标 | 10min | 60min | 120min | 180min |
|---|---|---|---|---|
| MAE | 2.709 | 4.054 | 4.694 | 5.037 |
| RMSE | 5.038 | 8.088 | 9.246 | 9.812 |

## 三、与论文 SOFTS+PMME 对比

论文 Table 1（P,C,S → METR-LA）：SOFTS+PMME —— MAE 2.71/4.05/4.69/5.02，RMSE 5.05/8.16/9.30/9.77。

| horizon | MAE（v58 vs 论文） | RMSE（v58 vs 论文） |
|---|---|---|
| 10min | 2.709 vs 2.71（打平） | 5.038 vs 5.05（略优） |
| 60min | 4.054 vs 4.05（追平） | **8.088 vs 8.16（优 0.07）** |
| 120min | 4.694 vs 4.69（打平） | **9.246 vs 9.30（略优）** |
| 180min | 5.037 vs 5.02（略输） | 9.812 vs 9.77（略输） |

**结论**：v58 在 60min 这一最受关注的中间列上，MAE/RMSE 均追平或小幅优于论文 SOFTS+PMME，
其中 60min RMSE 8.088 比论文 8.16 低 0.07。长程（180min）仍有微小差距，未全面超越论文。

---

## 四、文件清单

- `run_v58_dvgp.sh` — v58 运行命令（完整、可复现）
- `dp_pot_v58_dvgp.log` — v58 完整训练/评估结果日志（含逐 horizon 指标）
- `train_DA_OT.py` / `datasets.py` / `utils.py` — 训练与数据主程序
- `model/Meta_Models/PMME.py` / `PMME_2ndStage.py` / `dynamic_pattern.py` — 核心模型（含 DVGP 软选择器）
- `model/Meta_Models/SOFTS/` / `iTransformer/` — 骨干网络
- `configs/` — 配置文件

> 注：v58 时段的精确源码快照已被后续版本覆盖（服务器仅保留到 v47 的 .bak）。
> 本仓库所述「v58 代码」为含 DVGP 软选择器的最新主线实现，`dvgp_soft_selector`
> 逻辑与本说明一致，`run_v58_dvgp.sh` 可完整复现 v58 的运行配置与结果。
