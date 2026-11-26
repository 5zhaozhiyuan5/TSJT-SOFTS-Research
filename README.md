# 零膨胀连续域论文数据集 — 专用分支

**这个分支只存放数据集及与数据有关的说明/校验文件**，不包含 PMME、TSJT 或其他模型代码。

## 三篇正式发表论文及数据

| 数据集 | 论文 | 论文源码地址 | 本分支里的数据 |
|---|---|---|---|
| SAECZI | *Small area estimation of forest biomass via a two-stage model for continuous zero-inflated data*，Canadian Journal of Forest Research (2025)，[DOI](https://doi.org/10.1139/cjfr-2024-0149) | https://github.com/harvard-ufds/saeczi | `data/zero_inflated_papers/saeczi/raw/samp.rda` 和 `pop.rda`，真实俄勒冈州样例 |
| RainfallBench | *How Effective Are Time-Series Models for Precipitation Nowcasting?*，IEEE TGRS (2026)，[DOI](https://doi.org/10.1109/TGRS.2026.3671768) | https://github.com/yfangZhang/RainfallBench | `data/zero_inflated_papers/rainfallbench/stations/`：三个经过精确零值验证的真实 CSV；作者完整仓库位于 `upstream` 子模块 |
| Hybrid nHMM–Copula | *Characterizing Climate-Driven Shifts in Chilean Rainfall Regimes*，Hydrology and Earth System Sciences (2026)，[DOI](https://doi.org/10.5194/hess-30-6531-2026) | https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework | 作者完整仓库及其 33.96 MB 加工数据 ZIP 位于 `data/zero_inflated_papers/hybrid_nhmm_copula/upstream` 子模块 |

更详细的研究问题、文件用途、字段、下载及零值比例见 [data/zero_inflated_papers/README.md](data/zero_inflated_papers/README.md)。

## 文件结构

```text
README.md
.gitmodules                  # 两个作者原始仓库的 Git 子模块配置
data/
  zero_inflated_papers/
    README.md                # 三组数据总说明及实测校验结果
    sources.json             # 论文、源码、数据和来源的结构化记录
    verify_datasets.py       # 实际数据完整性检查
    saeczi/
      README.md
      raw/samp.rda
      raw/pop.rda
    rainfallbench/
      README.md
      stations/J340_data.csv
      stations/ZIMM_data.csv
      stations/P095_data.csv
      verification.json
      upstream/              # 子模块，克隆后有更多原始站点数据
    hybrid_nhmm_copula/
      README.md
      upstream/              # 子模块，克隆后有原作者 33.96 MB ZIP
```

## 下载时包含作者真实数据文件

```bash
git clone --branch datasets/continuous-zero-inflated-verified-20261126 --recurse-submodules https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research.git
cd TSJT-SOFTS-Research
python data/zero_inflated_papers/verify_datasets.py
```

已经克隆过的仓库可切换本分支并执行 `git submodule update --init --recursive`。

**数据真实性**：SAECZI 的两份 R 数据与 RainfallBench 三份 CSV 已直接提交到本分支并校验来源 Git SHA；智利 ZIP 并未在本仓库二次再分发，而是通过固定作者原始仓库的 Git 子模块访问（作者数据再分发许可尚需进一步确认）。该数据集解压后有 2,432,940 条有效降水记录，精确零值占 32.629%；三个 RainfallBench CSV 的真实零值比例分别为 80.612%、84.936%、94.711%。

**注意**：该分支最初从原仓库 `main` 派生，因此历史提交仍有共同祖先；**当前分支最新文件树**已清理，仅保留本页列出的数据相关内容，`main` 分支没有任何改动。
