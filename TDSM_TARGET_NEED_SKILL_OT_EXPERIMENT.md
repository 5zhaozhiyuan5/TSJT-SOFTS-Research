# TDSM-OT：基于目标预测需求与源模式预测技能匹配的双侧 OT 校准

**项目**：跨城市少样本交通预测 / PMME Stage-I / SOFTS / F2  
**远程分支**：`experiment/pmme-f2-target-need-skill-matching-20261127`

> **这是一个实际参与 Stage-I 优化的新增模块，不是 H6-H11 checkpoint 梯度排障。** 默认 strength=0，F2 原有代码路径不变。只有显式启用 TDSM 时，才修改每对源城市—目标支持组的 OT 边际权重 `alpha, beta`，随后仍使用原来的 GeomLoss Sinkhorn、source task loss、PCGrad 和 AdamW。**官方 Stage-II 未修改**。

## 1. 先核对已有实验，避免重新发明旧方法

历史已有机制和观测：

- **F2**：源模式 projection2 梯度与目标任务梯度对齐，产生源边际 `alpha`；目标边际 `beta` 均匀。
- **PRC**：仅在 F2 `alpha` 上追加源预测误差和 persistence 技能校准。seed1 正式 Stage-I+Stage-II 完成；Stage-II 36-step MAE **2.394411**，高于历史 F2/T0 **2.372713**（不同提交，非严格配对），故简单偏爱准确/容易源样本未表现出收益。
- **P2**：历史—真实未来变化形态与输出梯度对用于加权的 **OT 样本对 cost**；不显式估计目标在哪些预测时长更需要迁移帮助。
- **BIOT**：利用特征/梯度组耦合重新估计 `alpha,beta`，没有将 *源样本当次预测技能 × 目标当前预测误差需求 × 历史—未来动态匹配* 结合到双方权重。
- **H6–H10**：已训练 checkpoint 的梯度诊断，不是正式训练中的新模块，不能替代最终模型效果实验。

本次目标不是把 PRC 再运行一次，也不是单独增加双侧 beta 或轨迹距离；**新的交互项必须同时依赖源样本预测技能、目标需求、短中长期历史—未来响应匹配**，不满足三项条件就不能单靠“准确”或“困难”获得很高的权重。

## 2. 新模块位置

```text
目标城市三天 SUPPORT 历史+真实未来+当次预测
源城市历史+真实未来+当次预测
                  |
          SOFTS Stage-I 原始前向
                  |
     F2 projection2 梯度匹配  --> alpha_F2
                           beta_F2 = 均匀
                  |
  TDSM-OT 双侧预测需求-技能匹配模块（新增）
  [源预测技能 q_i^h]  [目标误差需求 d_j^h]
  [3段历史→未来响应 r_i^h / r_j^h]
  [32 prototypes / side, conditional match]
                  |
              alpha* , beta*
                  |
      未修改 GeomLoss Sinkhorn OT 损失
                  |
   原 source task loss + OT + one-sided PCGrad
                  |
    原 Stage-I 权重 -> 官方 Stage-II
```

未来真值只在**训练期间可用的源/目标 SUPPORT**中参与 Stage-I 的原始 OT 和新权重估计。开发/正式测试未来标签**不会**进入新模块，预测时不需要这些标签。

## 3. 模块内部公式与解释

源/目标每个样本为**一个历史窗口下的一个传感器节点**；每个源城市分别与 15/15/16 目标窗口组成一个 OT 子问题。预测长度36，三段 (h\in\{1,2,3\}) 每段12步。

### A. 源模式预测技能

源城市物理速度单位的 MAE 按城市 `std` 标准化：

```
e_i^h = mean_t |ŷ_i,t - y_i,t| / std_source
p_i^h = mean_t |x_last,i - y_i,t| / std_source

q_i^h = exp(-e_i^h / median_source(e^h))
         * sigmoid( (p_i^h - e_i^h) / median_source(p^h) )
```

这里同时检验当前源预测是否准确，以及是否好于“未来保持最后观测速度”的简单预测。注意：`q` 来自源训练误差，**不是**外部真实泛化技能概率。

### B. 目标预测需求

```
e_j^h = mean_t |ŷ_target,j,t - y_target,j,t| / std_target
d_j^h = sigmoid( (e_j^h - median_target(e^h))
                  / median_target(e^h) )
```

目标当前较难预测的短/中/长期段 `d` 较高。但目标需要并不自动意味着存在可靠源知识，因此还要乘动态匹配度及源技能。

### C. 历史—未来响应形态

对每个样本、每段12步预测，构建长度3的响应向量：

```
r_i^h = [
  (future_band - last12_history_mean)/history_std 的均值,
  (future_band_last4 - future_band_first4)/history_std,
  (future_band - last12_history_mean)/history_std 的波动
]

sim_ij^h = exp(-||r_i^h - r_j^h||_2 / temperature)
```

这样的比较区分不同城市绝对速度尺度，关注历史状态如何演变成未来短/中/长期行为。真实未来仅用于 Stage-I SUPPORT，**不用于测试阶段匹配**。

### D. 双侧条件收益

为了避免现有 GeomLoss 大规模 OT 再增加一个 (N_s\times N_t) 矩阵，分别选择最多32个源和32个目标代表样本：一部分从先验×质量的重要点中选，一部分覆盖原始样本索引。匹配成本大致 (O(3K(N_s+N_t)))。

```
U_s(i) = (1/3) Σ_h q_i^h *
          mean_{j in target_prototypes}[sim_ij^h * d_j^h]

U_t(j) = (1/3) Σ_h d_j^h *
          mean_{i in source_prototypes}[sim_ij^h * q_i^h]
```

**核心区别**：既不能仅因源样本预测准确就提高 `alpha`，也不能仅因目标预测难就提高 `beta`。必须具有“可匹配的、当前预测技能可靠的源样本”。

### E. 输出进入原 Sinkhorn 的两个边际

对于先验 (a)（`alpha_F2`）或 (b)（`beta_F2`）：

```
relative(v; p) = clip(v / Σ_i p_i*v_i, 0, 2)
candidate_i = p_i * relative_i / Σ_k p_k*relative_k
new_p = (1-strength)*p + strength*candidate
```

正式默认 `strength=0.35`：保留65%原始质量先验，因此不会将 F2 样本全部丢弃。两个边际分别计算、分别记录 ESS 和 L1 位移；原始 OT cost 不变。

如果相似性×技能都是0，保持原始先验；`strength=0` 直接返回原始 `alpha,beta` 对象，不执行额外匹配。

## 4. 实际修改内容

**新增：**
- `model/Meta_Models/task_need_skill_ot.py`：动态响应、源预测技能、目标误差需求、双侧原型匹配、边际输出。
- `tests/test_task_need_skill_ot.py`：质量排序、短中长期、双侧变动、零强度 F2 恢复、全零/非法值稳定性。
- `run_f2_target_need_skill.sh`：真正运行 Stage-I 3000 epoch，Stage-II 原始6000 epoch，并校验训练中出现有效 OT 权重日志。
- `summarize_f2_tdsm.py`：按原支持集验证选择 checkpoint，汇总 Stage-I/Stage-II 36-step MAE/RMSE 及10/60/120/180分钟结果。

**修改：**
- `model/Meta_Models/PMME.py`：只在 Stage-I 原 F2 产生 `alpha,beta` 后，调用新 TDSM 模块；原 PRC/P1-P4/BIOT 等机制要求与新模块互斥，防止多个旧机制叠加造成无法识别归因。
- **未修改** `SOFTS`、`train_DA_OT.py`、`datasets.py`、`GeomLoss` 或官方 Stage-II。

## 5. 运行

```bash
git fetch origin
git switch experiment/pmme-f2-target-need-skill-matching-20261127

python -m unittest discover -s tests -p 'test_task_need_skill_ot.py'

# 先3个Stage-I epoch烟测，确保第1个OT active epoch有真实训练更新
PMME_TDSM_STAGE1_EPOCHS=3 PMME_TDSM_STAGE2_EPOCHS=0 \
PMME_TDSM_DAEPOCH=1 PMME_TDSM_RUN_ROOT=./experiments/tdsm-smoke \
bash run_f2_target_need_skill.sh

# 完整新模块 Stage-I + 官方 Stage-II / seed1
PMME_TDSM_RUN_ROOT=./experiments/tdsm-formal-seed1 \
bash run_f2_target_need_skill.sh
```

只有第一个冒烟测试通过后才执行3000/6000 epoch正式训练。若结果改善再做相同预注册参数下 seed3/5 复现。每个独立运行目录包含 `run_config.txt`、Stage-I/II训练日志、完成标记、汇总CSV，原有 F2 checkpoint 不覆盖。

## 6. 评价和必要消融（**不是**新一轮 H11 optimizer experiments）

目标首先是 F2+TDSM 36-step Stage-II MAE/RMSE 能否改善，尤其要比已失败的 source-only PRC 更好。不能只看 `alpha,beta` 是否非均匀。

需要检查：
1. **实际发生双侧校准**：source/target L1 shift >0，并记录 ESS、是否塌缩。
2. **与原始 F2 的相同配置比较**（对历史不同提交的 F2 数字，只作参考）；Stage-II 固定官方实现、支持集验证选模。
3. **必要消融**：同代码固定强度下，只有源技能、只有目标需求、移除两侧动态匹配。这些消融只为证明“条件匹配交互”必要性，不能用测试集调参。
4. **多种子**：seed1 有可信改善才考虑 seed3/5；不能把偶然一个 seed 视为创新成立。
5. **文献查新**：现阶段不能声称 TDSM 三项交互已全球唯一。必须再比对近期公开论文与源码（预测收益OT、双侧学习边际、交通模式对齐、任务引导迁移），确定机制级区别。

**模块已实现不等于实验成功。是否能形成小论文创新取决于真实 Stage-I+Stage-II 改善、多种子稳定性及后续查新。**
