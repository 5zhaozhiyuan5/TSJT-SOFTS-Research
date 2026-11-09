# TSJT-PGrad 修复包（基于用户上传的 2026-11-08 源码）

## 当前交付状态

- **已完成**：重构 `pgrad.py`、`grad_utils.py`、`rotation.py`、`fusion.py`、`meta.py`、`trainer.py`；保留 `pgrad_model.py` 与 `run.py` 的接口和原始 TSJT 预测骨干。
- **已本地通过**：32 项 PyTorch 合成数据测试，包括 8 个训练模式的一步 smoke、完整方法三步 smoke、真正的顺序梯度重算、角度目标损失（`mu_phi=0`）梯度链、非负权重、目标正交性、查询标签窗口不重叠。
- **未完成/无法声称**：没有原始 `tsjt_c1`、`tsjt_paper` 工程和 PEMS-BAY 数据，本环境**无法验证真实 TSJT 50 epoch、真实指标或服务器 GPU 联调**。32 项测试**不是** TSJT 真数据端到端测试。

## 已修复的实质性问题

1. `full_method` 与 `pgrad_rotation_only` 分支不再重合；在 `full_method` 中真实调用 PGrad、逐源旋转、目标残差匹配。
2. PGrad 的后续城市梯度在 **前一城市虚拟更新后的参数** 上重新计算（`torch.func.functional_call`）；**不是**在固定梯度上累加。
3. 轨迹 PCA 先利用 `theta_k - theta_0` 建立位移，再中心化；从小型 Gram 矩阵得到方向。
4. 主梯度符号校准到 `theta_0 - theta_last`，并除以 PGrad 内层学习率消除非预期的虚拟步长尺度压缩。
5. 梯度旋转的次要轴来自 `Orth(g_P, g_T, p_i)`，不再来自源梯度的目标平行分量，因此冲突梯度不会被旋转回负目标方向。
6. 每个源城市单独输出一个候选方向和角度；残差匹配输入是 **M 个城市方向**，而不是唯一的 `rotated`。
7. Query-R / Query-W 从目标**训练集内部**按步采样，保证当前步 Support、Query-R、Query-W **标签区间无重叠**。不同训练步之间允许轮换用途；这是一种 episode 内划分，不是独立的永久保留验证集。
8. Query-R 目标预测损失通过可微虚拟更新学习角度，`mu_phi=0` 时仍有梯度才算通过；禁止正则项独自导致的假成功。
9. Query-W 提供独立梯度，不再把 `g_T` 冒充 `g_Q`；融合解使用非负 l1 球上的投影梯度法。
10. 融合函数向训练器返回**纯源方向**，避免再与 `gamma_target * g_T` 一起使目标梯度重复计算。
11. 增加 `effective_source_ratio`、融合权重、旋转有效率、PGrad 特征值、Query 元梯度及参数更新量等日志（`grad_diagnostics.jsonl`）。
12. 检查旧 `complete.json` / `last.pt` 配置哈希，不会无声地复用不同配置的旧结果。独立输出目录避免与旧实验冲突。

## 放回原项目

请在原 TSJT 工程根目录执行（按服务器实际路径调整）：

```bash
# 解压后，将此包的 src/tsjt_pgrad 覆盖到原工程对应目录
cp -r tsjt_pgrad_fixed/src/tsjt_pgrad ./src/
cp tsjt_pgrad_fixed/configs/*.yaml ./configs/

# 先测试小模块（无需真实数据，但使用测试专属模拟骨干）
pytest -q tsjt_pgrad_fixed/tests/test_fixed_core.py

# 在原工程真实数据下短跑 5 step，关闭正式测试集评价
PYTHONPATH=src python -m tsjt_pgrad.run --config configs/pgrad_pemsbay_smoke.yaml
```

**注意**：需要原始工程的 `tsjt_c1`、`tsjt_paper`，否则上面的真实训练命令无法执行。工程如未安装为 package，请确保 `PYTHONPATH=src`。

## 配置解释

- `pgrad_pemsbay_smoke.yaml`：只跑 1 epoch 的前 **5 step**，`skip_final_evaluation=true`，新目录 `outputs/tsjt_pgrad_reviewed_smoke_seed2025`。
- `pgrad_pemsbay_full.yaml`：50 epoch 正式训练配置，**只有短跑联调全部通过以后再使用**；全测试集评估在训练后执行。
- `pgrad_pemsbay_paper_scale.yaml`：保留原历史 `gamma_source=1e-6` 的对照配置，单独输出目录。

`smoke` 与 `full` 的 `gamma_source` **暂设为 1e-4**，同时 `source_budget_ratio=0.3` 对源更新范数设置**上界**，以便诊断源贡献。**这属于待验证超参，绝非论文原始设置，也不保证提高性能**。用于论文比较时，必须对各消融使用相同系数、相同数据协议，另外保留论文系数的 baseline。

`source_budget_ratio` 只**裁剪**源贡献，不自动放大源梯度；如果融合权重全零，增大 `gamma_source` 依然没有作用。

## 验收日志

真实短跑中应检查 `outputs/tsjt_pgrad_reviewed_smoke_seed2025/grad_diagnostics.jsonl`：

- full_method 的 `path` 包含 `[per_city_gradient,pgrad,rotation,residual_matching]`；
- `num_source_directions=3`；
- `pgrad.city_order` 记录真正顺序采样、`trajectory_len=4`；
- 融合 `weights` 有三个值；`fusion.residual_norm` 不应因误传同一梯度而必然为零；
- `rotation[*].target_dot` 接近零；
- `meta.query_grad_norm`、`meta.meta_updated` 和 `meta.param_delta` 能区分真实 Query 反馈与正则项更新；
- `effective_source_ratio` 统计实际使用的 `gamma_s * source` 与 `gamma_t * target` 的范数比例。

若出现 `meta_updated=false`、大量零权重或全局预测仍几乎为常数，**停止 50 epoch**，先核查真实数据和 TSJT 骨干，再调整实验。

## 已知限制／需服务器核对

1. 本次没有原始 `tsjt_c1` / `tsjt_paper` 源文件，无法保证原始 `tsjt_original` 与用户服务器基线逐参数、逐样本完全相等；短跑中应单独比较数值。
2. `TrainOnlyQuerySampler` 依赖 TSJT 目标训练集存在 `.indices`，其元素是时序预测窗口起点，且 `history/horizon` 与 YAML 一致。若原项目字段不同，应根据真实数据加载器做适配，**不要移除时间隔离检查**。
3. 角度 Query 优化通过单步显式参数更新，保留有限的噪声和实际梯度尺度问题。尤其 `gamma_source` 很小时，即使数学正确，Query 对角度的梯度也可能太弱。
4. PGrad 是**受原论文启发的轻量化实现**，不是作者官方实现的逐行精确复现；未采用官方完整的分块多轨迹配置。
5. 当前梯度操作对所有**可训练的共享 backbone 参数**进行。若有城市专属可训练参数，服务器需按实际模型明确标注并单独更新，不能默认“全部 backbone 参数均共享”。
6. 完整骨干依然依赖 TSJT 的图邻接信息（`use_knowledge_graph: true`）；此次改造只保证额外梯度机制本身不新增跨城拓扑依赖。
7. 断点继续训练能恢复模型、PGrad 缓存、权重和步数，但**尚未保存 Python/CUDA 随机数生成器状态**；断点后的随机轨迹采样不保证与完全不中断的运行逐步一致。

## 本地测试

```text
$ pytest -q tests/test_fixed_core.py
32 passed
```


## 二次审查新增修补（2026-11-08）

- 修复 `run.skip_complete=false` 被忽略的问题：目录存在 `complete.json` 时必须新建输出目录（如确需复用，应显式设置 `skip_complete=true` 且核对配置哈希和结果文件）。
- 当目录有未完成的 `last.pt` 且 `resume=false` 时，不再覆盖 checkpoint、在旧 `train.jsonl` 后追加重复日志；直接提示选择新目录。
- 增加 `epochs/max_epochs/evaluate_once_after_epoch` 一致性校验，防止类似 smoke 的 1/50 冲突和实际训练长度误判。
- 新增 7 项回归测试，总计 32 项本地合成/逻辑测试。
- **尚未覆盖**：真实 `tsjt_c1`/`tsjt_paper` 的数据加载、校验细节、GPU 训练及实验指标。
- **实验公平性风险**：诊断配置把 `gamma_source` 从论文值 `1e-6` 提高至 `1e-4`，同时新方法有 `source_budget_ratio=0.3`，而 `tsjt_original` 绕过了该上限。不要将新配置下的 `tsjt_original` 当作历史论文超参的完全复现，也不要将不对等配置的结果用于消融结论。
- **元学习稳定性风险**：当前 `meta_step_scale=1`、`gamma_source` 很小时，Query-R 的角度梯度可能极小；源贡献上限系数通过 Python float 计算，不对上限系数求导，属于对元更新的近似。需在真实梯度日志核对。
- **Smoke 限制**：5 step 只覆盖运行分支，不保证至少有一个有效的冲突旋转，更不保证 `meta.meta_updated=true`；若所有旋转回退，应扩大短测样本再判断。

- **梯度日志精度修复**：角度控制器的 `meta.param_delta` 改为实际参数变化的 L2 范数，不再把 `lr * grad_norm` 冒充实际更新；若 fp32 下全部舍入为零，记录 `meta_updated=false` 及原因。新增 `realized_source_effect_norm`、`realized_source_effect_ratio`，测量本次实际参数更新与仅目标更新的浮点数差异。由于实际 `gamma_source` 可能极小，这些指标比请求梯度范数更重要。
