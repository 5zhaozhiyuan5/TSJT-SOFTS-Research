# 连续域零膨胀研究数据集 — 数据专用分支

这个分支只保留三组正式发表论文的真实数据、相关来源说明与校验脚本。**智利降水数据已经解压成完整 CSV，不再需要 ZIP 或智利作者子模块。**

| 数据集 | 正式论文 | 源码 | 已存放的数据 |
|---|---|---|---|
| **SAECZI**（森林生物量） | *Small area estimation of forest biomass via a two-stage model for continuous zero-inflated data*，Canadian Journal of Forest Research (2025)，[DOI](https://doi.org/10.1139/cjfr-2024-0149) | https://github.com/harvard-ufds/saeczi | `data/zero_inflated_papers/saeczi/raw/samp.rda`、`pop.rda`，俄勒冈州示例 |
| **RainfallBench**（多气象站降雨） | *How Effective Are Time-Series Models for Precipitation Nowcasting?*，IEEE TGRS (2026)，[DOI](https://doi.org/10.1109/TGRS.2026.3671768) | https://github.com/yfangZhang/RainfallBench | `data/zero_inflated_papers/rainfallbench/stations/` 里三个真实、严格零膨胀 CSV；其他原始公开站点在 `upstream` 子模块 |
| **Hybrid nHMM–Copula**（智利多站点降雨） | *Characterizing Climate-Driven Shifts in Chilean Rainfall Regimes*，Hydrology and Earth System Sciences (2026)，[DOI](https://doi.org/10.5194/hess-30-6531-2026) | https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework | **`data/zero_inflated_papers/hybrid_nhmm_copula/raw/mapa_estaciones_precipitaciones_clean.csv`（216,657,356 字节，已解压）** |

## 数据集目录

```text
README.md
.gitattributes                 # 仅对智利大 CSV 开启 Git LFS
.gitmodules                    # 仅保留 RainfallBench 作者源码及完整数据
data/zero_inflated_papers/
  README.md
  sources.json
  verify_datasets.py
  saeczi/
    README.md
    raw/samp.rda
    raw/pop.rda
  rainfallbench/
    README.md
    stations/J340_data.csv
    stations/ZIMM_data.csv
    stations/P095_data.csv
    upstream/                  # 需要时可初始化的作者仓库子模块
    verification.json
  hybrid_nhmm_copula/
    README.md
    DATA_RIGHTS_NOTICE.md
    raw/
      mapa_estaciones_precipitaciones_clean.csv  # 216.66 MB（Git LFS）
      extraction_verified.json                   # SHA-256 和作者原文件信息
```

## 怎样直接下载完整 CSV？

GitHub 普通 Git 文件超过 **100 MB** 会被拒绝，因此智利 CSV 通过 **Git LFS** 存储。GitHub 网页可能显示一个很小的指针文件，**不能将指针当作真实 CSV 下载**。需要在已安装 [Git LFS](https://git-lfs.com/) 的电脑上运行：

```bash
git lfs install
git clone --branch datasets/continuous-zero-inflated-verified-20261126 https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research.git
cd TSJT-SOFTS-Research
git lfs pull
python data/zero_inflated_papers/verify_datasets.py
```

克隆后 `data/zero_inflated_papers/hybrid_nhmm_copula/raw/mapa_estaciones_precipitaciones_clean.csv` 就是真正的 **216,657,356 字节** CSV，Python Pandas/R 可以直接读取。**已在 GitHub Actions 全新 LFS 检出中验证文件大小、SHA-256 和 2,432,940 条记录**：[验证记录](https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research/actions/runs/45777509526)。

如果还需 RainfallBench 全部原始公开数据，可额外运行 `git submodule update --init --recursive`。

## 严格零膨胀数据的实测结果

- **智利**：`precipitation` 2,432,940 条有效记录，793,844 条为零（**32.629%**），其余为连续正值。
- **RainfallBench**：`gpm_precip_sum_1h` 列，J340 为 **80.612%**，ZIMM 为 **84.936%**，P095 为 **94.711%** 精确零值。
- **SAECZI**：俄勒冈州森林生物量研究样例。具体响应变量见其 README。

**使用权限提醒**：智利 CSV 属于作者公开发布的加工数据。原作者 `LICENSES.md` 提醒，上游 CR2MET 及其他来源的再分发权限尚需核实，不能把本次镜像上传视为获得进一步授权；详见 `DATA_RIGHTS_NOTICE.md`。
