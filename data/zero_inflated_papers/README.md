# 连续域零膨胀论文数据集（已核实来源）

本目录专门收集与**大量真实零值 + 连续型正值**相关的、**正式发表**的论文数据。2026-11-26 建立。与本仓库 PMME / TSJT 实验隔离，不更改原有训练流程。

## 论文、源码与数据状态

| 目录 | 论文及发表场所 | 数据 / 研究任务 | 数据获取及当前存放方式 |
|---|---|---|---|
| `saeczi/` | *Small area estimation of forest biomass via a two-stage model for continuous zero-inflated data*（Canadian Journal of Forest Research, 2025） | 美国俄勒冈州森林生物量，零值 + 连续正生物量；空间统计 | **两份真实 R 数据已直接提交**：`saeczi/raw/samp.rda`、`saeczi/raw/pop.rda`（俄勒冈示例，不是论文所有内华达研究数据） |
| `rainfallbench/` | *How Effective Are Time-Series Models for Precipitation Nowcasting? A Comprehensive Benchmark for GNSS-Based Precipitation Nowcasting*（IEEE TGRS, 2026） | GNSS/气象站降雨预测；连续降水量 `tp`、大量零值 | **完整作者仓库以固定 Git 子模块引用**于 `rainfallbench/upstream/`（含各大洲公开 CSV）；另已直接入库 3 个严格零膨胀代表站点到 `rainfallbench/stations/`，实际结果以文件清单/校验报告为准 |
| `hybrid_nhmm_copula/` | *Characterizing Climate-Driven Shifts in Chilean Rainfall Regimes with a Hybrid Hidden Markov–Copula Framework*（Hydrology and Earth System Sciences, 2026） | 智利多站点日降水；零值质量、正值边缘分布与极端值 | **作者仓库以固定 Git 子模块引用**于 `hybrid_nhmm_copula/upstream/`，其中 `Data/mapa_estaciones_precipitaciones_clean.csv.zip` 为约 34 MB 真正的加工数据；因衍生数据的再分发许可不确定，不另行复制该压缩包 |

**重要区别**：Git 子模块是对作者原始仓库指定提交的正式引用。GitHub 页面上会显示子模块入口，`git clone --recurse-submodules` 或 `git submodule update --init --recursive` 后，**全部原始文件将真实出现在本地目录**。这与仅在 README 中放一个下载链接不同；但数据字节并非重新上传到本仓库 Git 对象库。这里绝不把子模块说成“已重新上传的文件”。

## 获取三组数据

```bash
git clone --branch datasets/continuous-zero-inflated-verified-20261126 \
  --recurse-submodules \
  https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research.git
cd TSJT-SOFTS-Research
python data/zero_inflated_papers/verify_datasets.py
```

已有克隆的用户：

```bash
git fetch origin
git switch datasets/continuous-zero-inflated-verified-20261126
git submodule update --init --recursive
python data/zero_inflated_papers/verify_datasets.py
```

如果部分作者原始 GitHub 数据仓库较大，请预留足够磁盘空间。最少查看三组主要数据：

```text
data/zero_inflated_papers/saeczi/raw/
  samp.rda
  pop.rda
data/zero_inflated_papers/rainfallbench/upstream/dataset/
  gnss/Asia/J340_data.csv
  gnss/Europe/ZIMM_data.csv
  gnss/North_America/P095_data.csv
  Rain/JFNG_data_15min.csv
data/zero_inflated_papers/hybrid_nhmm_copula/upstream/Data/
  mapa_estaciones_precipitaciones_clean.csv.zip
```

RainfallBench `stations/` 已提交 J340 / ZIMM / P095 三个真实 CSV，并经原作者 Git blob SHA 校验。**JFNG 虽来自该论文，但真实 `tp` 列 171,069 条记录中精确零值数为 0，因此不纳入严格零膨胀子集，仍可在完整 `upstream/dataset/Rain/` 查阅。**

## 为什么不把三篇的“完整论文训练数据”一概称为已复现？

- SAECZI 公开 `samp/pop` 是俄勒冈示例，而正式论文还包含美国内华达州应用，部分保密清查字段未公开。
- RainfallBench 包含作者发布的更多站点和模型脚本。三个已经核实含大量精确零值的代表站点仅是便于快速分析的子集，完整公开数据请从 `upstream/dataset` 读取。
- 智利项目作者明确指出压缩 CSV 中有上游派生内容，尚未全面确认再分发授权；本仓库通过 Git 子模块引入作者原始发布文件，不擅自再次发布数据。
- 所有“零值占比”须以真实 CSV / RData 的实测值为准，不得用论文总体零值比例代替某个具体站点的实测结果。
- 这些数据属于**连续域零膨胀**，不是离散计数型 ZIP 数据；但均非传统图神经网络模型数据，不能直接假设已有道路拓扑图。

具体字段、论文、源码、原文件及引用方法见各目录 README、`sources.json`。

## 数据文件真实校验结果（2026-11-26）

- SAECZI：`samp.rda` 27,899 B、`pop.rda` 124,974 B；均通过作者 Git blob SHA 校验。
- RainfallBench：三份 GPM 连续降雨 CSV 的精确零值比例分别为 **J340 80.612%**、**ZIMM 84.936%**、**P095 94.711%**；所有原始文件大小和 Git SHA 与作者一致。JFNG 原始 `tp` 列 **0/171069 精确零值**，因此不属于本次选出的严格零膨胀站点。
- Hybrid nHMM–Copula：作者 ZIP **33,960,384 B**，完整解压可得到研究 CSV **216,657,356 B**；在 **2,432,940** 条有效 `precipitation` 记录中，**793,844** 条精确等于 0（**32.629%**），**1,639,096** 条是正值，其中 **1,638,991** 条带小数，确认存在零值点质量与连续正值。原始 ZIP 与作者 SHA 相同且通过 ZIP CRC 检查。
- 全部以上文件及子模块读取均经 GitHub Actions 真实检出与校验，参考 [验证记录](https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research/actions/runs/45774197445)。智利作者数据仍由子模块提供，**不是本仓库再次上传的 ZIP 副本**。
