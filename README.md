# 连续域零膨胀论文数据集 — 专用分支

本分支**只包含三组论文数据及相关说明、校验文件**，没有 PMME、TSJT 或其他工程代码。智利数据 ZIP 已直接放入本分支（不是子模块），而 RainfallBench 的完整大规模作者仓库继续采用子模块引用。

## 三组数据

| 数据 | 论文及发表信息 | 官方源码仓库 | 本分支的真实数据 |
|---|---|---|---|
| **SAECZI** | *Small area estimation of forest biomass via a two-stage model for continuous zero-inflated data* — Canadian Journal of Forest Research (2025), [DOI](https://doi.org/10.1139/cjfr-2024-0149) | https://github.com/harvard-ufds/saeczi | [samp.rda](data/zero_inflated_papers/saeczi/raw/samp.rda)、[pop.rda](data/zero_inflated_papers/saeczi/raw/pop.rda)，俄勒冈州示例 |
| **RainfallBench** | *How Effective Are Time-Series Models for Precipitation Nowcasting?* — IEEE TGRS (2026), [DOI](https://doi.org/10.1109/TGRS.2026.3671768) | https://github.com/yfangZhang/RainfallBench | [三个真实站点 CSV](data/zero_inflated_papers/rainfallbench/stations)，完整作者公开数据另可在 `rainfallbench/upstream` 子模块获取 |
| **Hybrid nHMM–Copula** | *Characterizing Climate-Driven Shifts in Chilean Rainfall Regimes with a Hybrid Hidden Markov–Copula Framework* — Hydrology and Earth System Sciences (2026), [DOI](https://doi.org/10.5194/hess-30-6531-2026) | https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework | **[原作者 33.96 MB ZIP 已直接上传本分支](data/zero_inflated_papers/hybrid_nhmm_copula/raw/mapa_estaciones_precipitaciones_clean.csv.zip)**，解压后为 216.66 MB CSV |

## 目录结构

```text
README.md
.gitmodules                                      # 仅 RainfallBench 作者原始仓库
data/zero_inflated_papers/
  README.md                                     # 三个数据集概述
  sources.json                                  # 论文、源码、数据文件与验证元信息
  verify_datasets.py                            # Git blob SHA、大小、ZIP CRC 检验
  saeczi/
    README.md
    raw/
      samp.rda                                   # 27,899 B
      pop.rda                                    # 124,974 B
  rainfallbench/
    README.md
    verification.json
    stations/
      J340_data.csv                             # 零值 80.612%
      ZIMM_data.csv                             # 零值 84.936%
      P095_data.csv                             # 零值 94.711%
    upstream/                                    # 唯一仍保留的 Git 子模块
  hybrid_nhmm_copula/
    README.md
    DATA_RIGHTS_NOTICE.md                       # 原作者数据来源与使用条件提醒
    raw/
      mapa_estaciones_precipitaciones_clean.csv.zip # 33,960,384 B
      download_verified.json                    # 来源哈希和压缩包 CRC
```

## 下载 / 校验

普通克隆即可得到**智利 ZIP、SAECZI 数据和三个 RainfallBench CSV**，不需要初始化子模块：

```bash
git clone --branch datasets/continuous-zero-inflated-verified-20261126 https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research.git
cd TSJT-SOFTS-Research
python data/zero_inflated_papers/verify_datasets.py
```

仅当你需要 RainfallBench **全部公开站点与完整基准源码**时，再运行：

```bash
git submodule update --init --recursive
```

## 实测零膨胀程度

- 智利 `precipitation`：2,432,940 条有效记录，其中 793,844 条精确零值（**32.629%**），1,639,096 条连续正值。
- RainfallBench `gpm_precip_sum_1h`：日本 J340 **80.612%**、瑞士 ZIMM **84.936%**、美国 P095 **94.711%** 精确零值。
- SAECZI：森林生物量的连续零膨胀样例，实际 RData 在仓库内。
- 不包含 JFNG 文件作为严格零膨胀样本，因为其公开的 `tp` 列没有精确零值。

**版权/来源提醒**：智利数据 ZIP 是从[作者的 GitHub 原始发布文件](https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework/blob/c09e5c46c6c5c237c86a3d214768febfc9e0dbac/Data/mapa_estaciones_precipitaciones_clean.csv.zip)直接复制、字节哈希一致，并非本仓库原创。作者 `LICENSES.md` 提醒，CR2MET 及其他上游衍生数据的**再分发权限还需要核实**；本次拷贝不代表已得到额外许可，公开分享前请检查相关条款并保留原作者引用。
