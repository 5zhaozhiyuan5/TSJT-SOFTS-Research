# 零膨胀连续域论文数据总览

本目录只有三组数据集及其论文、源码、文件说明、数据校验文件。**智利降水压缩包现在是真实上传到本分支的普通 Git 文件**，不再通过作者子模块间接获取。

| 目录 | 研究内容及论文 | 已直接上传的真实数据 | 论文原始源码 |
|---|---|---|---|
| `saeczi/` | 2025 年 Canadian Journal of Forest Research，连续零膨胀森林生物量空间估计 | `raw/samp.rda`、`raw/pop.rda` | https://github.com/harvard-ufds/saeczi |
| `rainfallbench/` | 2026 年 IEEE TGRS，零膨胀降水时间序列预测 | `stations/J340_data.csv`、`ZIMM_data.csv`、`P095_data.csv`；另有完整数据 Git 子模块 | https://github.com/yfangZhang/RainfallBench |
| `hybrid_nhmm_copula/` | 2026 年 Hydrology and Earth System Sciences，智利时空降水零值/连续正值和空间相关性 | **`raw/mapa_estaciones_precipitaciones_clean.csv.zip`**（33,960,384 字节，解压 CSV 为 216,657,356 字节），`raw/download_verified.json` | https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework |

## 文件是否真实且零膨胀？

- SAECZI 两个 RData 文件与作者仓库同一 Git blob SHA；这些是**俄勒冈州公开示例数据**，不等于全部论文保密原始输入。
- RainfallBench 已提交三个真正存在精确零值的 GPM 降水 CSV：每份 59,160 条有效降水记录，J340 零值 **47,690（80.612%）**、ZIMM 零值 **50,248（84.936%）**、P095 零值 **56,031（94.711%）**。JFNG 的公开 `tp` 列精确零值为零，因此不计入严格零膨胀样本。
- Chile `precipitation` 字段实际有 **2,432,940** 条有效记录，**793,844** 条精确为 0（**32.629%**），**1,639,096** 条正值，其中绝大多数含小数。ZIP 与原作者对应源提交的 Git blob SHA 一致、ZIP CRC 校验通过。

## 一键下载、校验

```bash
git clone --branch datasets/continuous-zero-inflated-verified-20261126 https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research.git
cd TSJT-SOFTS-Research
python data/zero_inflated_papers/verify_datasets.py
```

上述克隆不需要子模块即可取得 **全部已直接上传的数据，包括智利 ZIP**。如需另外获取 RainfallBench 原作者发布的全部公开 GNSS 数据，再运行 `git submodule update --init --recursive`。

### 权限和复现限制

- 智利数据 ZIP 由原作者仓库公开发布，现按要求复制到本分支，**保留原始字节和明确来源**；由于作者 `LICENSES.md` 未完整确认上游加工数据二次分发权限，使用者还需核实相应许可（参见 `hybrid_nhmm_copula/DATA_RIGHTS_NOTICE.md`）。
- 智利作者处理后数据已经附带 `labels`、`States` 等预处理结果；论文没有完整公开从原始 CR2MET 重新计算全部中间结果的工作流。
- 数据是时空观测，不提供标准道路时空图邻接矩阵；若作为 STGNN 任务需构建图和严格时间划分。
