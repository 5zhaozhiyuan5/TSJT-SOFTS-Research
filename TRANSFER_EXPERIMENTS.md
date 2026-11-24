# F1–F4 预测效用导向 OT：第一轮机制筛选

新分支：`experiment/pmme-task-utility-20261124`

**请先读**：`LITERATURE_REVIEW_F1_F4.md`。F1–F4 均属于最小诊断/消融，不代表最终创新；相关思想与 Ren (ICML 2018)、Grad-Match (ICML 2021)、TSJT (AAAI 2025)、GradVac (ICLR 2021) 和 OT 重要性加权 (ICML 2026) 已存在明显交集。

## 改动范围

只在 Stage-I `model/Meta_Models/PMME.py` 替换原 DVGP/神经域分类器生成 OT 权重的路线；没有 DVGP、分类 BCE、源/目标域标签；没有改官方 `train_DA_OT.py`、SOFTS/iTransformer、数据划分、Sinkhorn OT 算子、梯度投影、支持集选模规则或 Stage-II。

| 名称 | 环境变量 `PMME_TRANSFER_VARIANT` | OT 源侧边际 | OT 目标侧边际 |
|---|---|---|---|
| F1 | `uniform` | 均匀 | 均匀 |
| F2 | `gradient` | 36 步预测头节点级梯度与目标梯度的余弦，softmax | 均匀 |
| F3 | `horizon` | 3 个 12 步区间的梯度余弦等权平均，softmax | 均匀 |
| F4 | `reliable` | F3 + 目标支持子批次一致性权重 + ESS 下界 | 均匀 |

F2–F4 使用对共享预测输出层 `projection2.weight` 的**解析一阶梯度代理**，并非对完整 SOFTS 权重的逐节点梯度。不计算单独的传输矩阵；仍交给官方 Sinkhorn。F3 也不做三次独立 OT。

## 修改后的 OT 强度控制（非常重要）

之前四个方案全部采用 `OT_loss_scale=1`，这和旧 DVGP 筛选后的实际缩放不同，可能让对比受到 OT 强度影响。

现在 `PMME_TRANSFER_OT_SCALE` 用于 F1–F4 的共同固定缩放，**默认 0.01**（保守的探索尺度，**并非**“匹配 E0”的已知值）。可以另行使用 1.0 做敏感性实验：

- 固定缩放与样本边际权重彼此独立，比较 F2/F3/F4 **相对于 F1** 时保持二者相同。
- E0 的缩放是概率驱动、每批动态变化，因此 F1–F4 与 E0 不是只有一处变化的严格配对消融。
- 诊断 `task_utility_gradient_diagnostic ot_to_target_norm_ratio ... ot_vs_target_cosine ...` 记录实际 OT 梯度的量级。选择 **projection1.weight** 而不是 projection2.weight：后者不在 OT 路径上。
- 可选 `PMME_TRANSFER_TEMPERATURE=8.0` 和 `PMME_TRANSFER_MIN_ESS=0.5`，第一轮必须保持固定。

## 运行步骤

进入仓库根目录，确保官方数据位于 `data/`，并已安装完整官方 PMME 依赖：

```bash
git fetch origin
git switch experiment/pmme-task-utility-20261124

# 1. CPU 数学/接口单元测试
python -m unittest discover -s tests -p 'test_transfer_utility.py'

# 2. 第一轮四种机制先各跑 seed 1，按序执行
PMME_SEEDS="1" bash run_transfer_experiments.sh

# 3. OT 强度敏感性：F1 和 F2 用相同的两个 scale
#    已跑过的 scale0p01/seed1 自动跳过，无需重跑
PMME_VARIANTS="uniform gradient" PMME_SEEDS="1" \
PMME_OT_SCALES="0.01 1.0" bash run_transfer_experiments.sh

# 4. 若稳定合理，再补 seed 3、5（仍按序执行）
PMME_SEEDS="3 5" bash run_transfer_experiments.sh
```

或者直接运行全部 3 seed × 4 variant（默认 scale 0.01）：

```bash
bash run_transfer_experiments.sh
```

其他设置：`PMME_GPU=0`、`PMME_THREADS=4`，`PMME_RUN_ROOT=/path/to/destination`。

默认每组训练为 `PEMS-BAY / SOFTS / 3000 Stage-I + 6000 Stage-II / 三天目标支持集`，使用官方 PEMS-BAY/SOFTS `DAcoef=1`, `DAepoch=1500`, `xySquaredRatio=100`, `projection=1`, `MemSize=10000`。保留命令行 `--classifier 0.5` 仅为配置可追溯，F1–F4 不使用原 DVGP 阈值。

## 结果保存与检查

每组独立目录：

`experiments/transfer-utility/<variant>/scale<数字中小数点改p>/seed<seed>/`

例如 `.../gradient/scale0p01/seed1/train.out`。内有 Stage-I 检查点、Stage-II 输出、`run_config.txt` 和 `COMPLETED` / `FAILED` 标记。失败即停；已完成则跳过，不覆盖此前 scale 和 seed。这里的 `COMPLETED` 指有 Python 成功退出和 Stage-II 结束标记，不能替代结果质量核查。

重点核查：

- `task_utility`：源样本梯度效用均值、正值比例、ESS、三段稳定性
- `task_utility_gradient_diagnostic`：OT 与目标预测梯度范数比、方向余弦
- Stage-I / Stage-II 由官方支持集 18 个奇数预测步的评价规则选择最佳检查点
- 测试指标仅在官方选模结果确定后汇报；每一模型比较相同 seeds 的均值 ± 标准差

**验证状态**：已写入 GitHub 源码和单元测试，尚未在用户服务器真实数据上完成 GPU 端到端训练；不能承诺无兼容性问题或精度改善。
