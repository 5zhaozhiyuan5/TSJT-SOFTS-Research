# Hybrid nHMM–Copula — 智利多地点零膨胀连续降水（2026）

**正式论文**：*Characterizing Climate-Driven Shifts in Chilean Rainfall Regimes with a Hybrid Hidden Markov–Copula Framework*

**期刊**：Hydrology and Earth System Sciences (2026), 30, 6531–6547；DOI https://doi.org/10.5194/hess-30-6531-2026

**代码仓库**：https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework

**研究内容**：智利降雨状态的动态转移与不同地点之间的相关性。使用非齐次隐马尔可夫模型（nHMM）、DTW 分区、零膨胀降雨边缘分布、极值 GPD 与 vine copulas。不是道路时空图 GNN。

## 论文实际数据：作者提供 33,960,384 字节 ZIP

通过 **Git 子模块** 引用的作者原始文件：

```text
upstream/Data/mapa_estaciones_precipitaciones_clean.csv.zip
```

原文件 Git blob SHA-1：`0d007e8f179c4602ca5d44a294cf5c2defda623f`。

解压所得 `mapa_estaciones_precipitaciones_clean.csv` 是处理好的研究输入，**216,657,356 字节（实际读取 ZIP 目录测得）**，约 217 MB。字段：

| 字段 | 含义 |
|---|---|
| `precipitation` | 日降水量，单位 mm，连续正值和真实零值 |
| `dates`, `year`, `month`, `day`, `Year` | 时间 |
| `codigo_estacion`, `location` | 站点标识和地点 |
| `x`, `y`, `z` | 空间坐标与海拔 |
| `labels` | 预先计算的气候分区 |
| `medoids` | DTW 聚类代表点标识 |
| `States` | 隐藏降雨状态（预先计算） |
| `year1980` | 历史时间索引 |

解压（需先 `git submodule update --init --recursive`）：

```bash
python -m zipfile -l data/zero_inflated_papers/hybrid_nhmm_copula/upstream/Data/mapa_estaciones_precipitaciones_clean.csv.zip
```

如需进一步解压请使用安全受限的 ZIP 处理方式并预留至少 500 MB 磁盘空间。作者 R 代码：`upstream/Scripts/Hybrid_nHMM_Copula_Simulation_Validation_2022.R`。作者说明的训练时期截至 2021-08-31，2022 年 5–8 月用于外推验证。

**版权/完整性提醒**：作者 `LICENSES.md` 明确表示，处理后的 CSV 可能包含多个上游数据源，需要独立确认完整再分发许可。因此**本分支不二次上传 ZIP**，而通过 Git 子模块固定到作者原始发行文件。软件 MIT 不等于数据可无限制再分发。完整的原始 CR2MET 数据构建与 nHMM 初始估计不全部包含在该存档中。

## 实际数据验证（2026-11-26）

通过固定源码子模块克隆，作者 ZIP 文件 SHA 校验及 CRC 校验均通过。对 ZIP 内主 CSV 的 `precipitation` 字段逐行统计：

- 有效记录：**2,432,940** 条；缺失/无效：**0** 条
- 精确零值：**793,844** 条，比例 **32.629%**
- 正值：**1,639,096** 条，其中 **1,638,991** 条有小数部分，证明并非单纯的计数型整数数据
- 校验工作流：https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research/actions/runs/45774197445

以上为作者处理后 CSV 的实际统计，不代表原始未处理 CR2MET 中的总体零值比例。
