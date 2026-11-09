# TSJT-PGrad 二次代码核查结果

日期：2026-11-08。基于用户已提供的 `TSJT_PGrad_修复代码_smoke校正版.zip`，对 Python 代码、三份 YAML、八个消融路径、模拟端到端逻辑进行了核查。未连接真实 TSJT 工程或服务器。

## 已证实并修改

1. **已修复 smoke YAML**：`epochs=1` 与 `evaluate_once_after_epoch=1` 一致；仍保留 `skip_final_evaluation=true`。
2. **`run.skip_complete=false` 无效**：原代码只要 `complete.json` 存在就直接返回旧结果。现在报 `FileExistsError`，只有显式 `skip_complete=true`、同配置哈希且文件齐备时才复用。
3. **旧日志混入**：原代码在 `resume=false` 且只有旧 `last.pt` 时可能重新训练并继续向旧 JSONL 追加。现在阻止静默覆盖/混合并要求新目录。
4. **配置中存在多个训练时长字段**：现在显式检查 `epochs=max_epochs=evaluate_once_after_epoch`，避免类似本次 1/50 不一致。
5. **角度元学习假阳性诊断**：原代码把 `lr * 梯度范数` 称为 `param_delta`，实际上 FP32 更新可能完全被舍入。现在记录真实权重变化，并保留理论提议量为 `proposed_param_delta`。
6. **源梯度被 FP32 舍入时无法从日志判断**：新增 `realized_source_effect_norm`、`realized_source_effect_ratio`、`realized_total_update_norm`，计算实际更新相对于“只用目标梯度更新”产生的数值差别。

## 复核情况

- `python -m pytest -q tests/test_fixed_core.py` → `32 passed`（本地、合成数据/模拟 TSJT 骨干）。
- `python -m compileall -q src tests` → 通过。
- YAML 检查：smoke 1/1/1，full 50/50/50，paper_scale 50/50/50。
- 使用模拟数据加载器与 ToyModel 完成了 `_run_all_pgrad` 中一次 5-step 训练编排，并观察到 `full_method` 路径以及 3 个融合权重；不代表真正执行了原工程的 `run()`/数据管线。
- 数值示例出现 Query 梯度范数约 1e-7、角度参数实际变化约 1e-9，表明元学习仍可能非常弱；真实数值以服务器日志为准。

## 尚未证实 / 实验风险

- **未提供原始 `tsjt_c1`、`tsjt_paper` 项目与真实数据**：真实 `_validate_config`、`_make_data`、`_CyclingCursor`、真实 TSJTPaper 接口与 GPU OOM 风险尚未验证。
- **Query 采样器假设**：目标训练数据集必须公开 `.indices`，且它是窗口开始时间戳/索引；否则会在训练开始时抛错。需要服务器上用真实数据检查。
- **5-step 并不能确保角度元学习成功**：若没有有效冲突源方向，代码会跳过元更新，日志 `meta_updated=false`；即使非零梯度，实际参数变化可能太小。
- **基线公平性**：诊断配置把 `gamma_source` 从 `1e-6` 改为 `1e-4`；`tsjt_original` 模式不使用新方法的 `source_budget_ratio=0.3` 上限。比较时必须统一超参数口径或明确这是不同设置，不可据此直接宣称优于论文基线。
- **元学习近似**：源预算 `cap` 由 Python float 计算，是非可微截断系数；元学习角度时没有对 cap 的变化求导。
- **恢复训练复现性**：checkpoint 未保存 Python/CUDA 随机数状态；仅能恢复模型与缓存，不能保证续训轨迹逐步重现。
- **模型表现**：还没有真实训练，不保证修复后 MAE/RMSE 优于原始 TSB，也尚不能解释之前近常数预测的全部原因。

## 建议执行步骤

1. 本包的三份 YAML 已改用 `outputs/tsjt_pgrad_reviewed_*` 新目录；部署二次审查包时不要覆盖原实验目录。为避免 `complete.json` 冲突，务必采用全新目录；不建议删除旧证据。
2. 在原工程先运行 `PYTHONPATH=src python -m tsjt_pgrad.run --config configs/pgrad_pemsbay_smoke.yaml`。至少确认 5 step 完成、`grad_diagnostics.jsonl` 写入。
3. 检查 `pgrad_updated`、`pgrad.pgrad_fallback`、`rotation[*].rotated`、`meta.query_grad_norm`、`meta.param_delta`、`fusion.weights`、`realized_source_effect_ratio`。
4. 如果角度全被回退或元更新变化近零，先检查 50–100 step 日志并校验数据与梯度尺度，不要直接跑完整 50 epoch。
