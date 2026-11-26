# Hybrid nHMM–Copula — 智利多地点零膨胀连续降水数据（2026）

**论文**：*Characterizing Climate-Driven Shifts in Chilean Rainfall Regimes with a Hybrid Hidden Markov–Copula Framework*，Hydrology and Earth System Sciences (2026), 30, 6531–6547，DOI https://doi.org/10.5194/hess-30-6531-2026

**论文对应源码**：https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework

**研究内容**：用非齐次隐马尔可夫模型 (nHMM)、DTW 分区、零膨胀连续边缘分布、GPD 极端值和 vine copulas，建模智利多个空间位置的降雨状态、降水强度及空间相关性。并非已经提供邻接矩阵的道路 STGNN 数据。

## 现在存储的是解压后的真实 CSV，而不是 ZIP

```text
raw/
  mapa_estaciones_precipitaciones_clean.csv   # 解压后的完整文件，216,657,356 字节，Git LFS
  extraction_verified.json                    # 原始 ZIP 哈希、解压后 CSV 哈希、数据来源
```

[查看 GitHub 中的 CSV 文件](raw/mapa_estaciones_precipitaciones_clean.csv)。文件超出普通 Git 100MB 限额，采用 Git LFS。**GitHub 网页上看到的 134 字节文本只是 LFS 指针；通过 `git lfs pull` 得到真正的 216.66 MB CSV**。

原作者发布 ZIP：https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework/blob/c09e5c46c6c5c237c86a3d214768febfc9e0dbac/Data/mapa_estaciones_precipitaciones_clean.csv.zip

## 实际 CSV 字段

| 列 | 内容 |
|---|---|
| `precipitation` | 每日降水量 mm，含严格零值与连续非零值 |
| `dates`, `year`, `month`, `day`, `Year` | 时间维度 |
| `codigo_estacion`, `location` | 位置/站点标识 |
| `x`, `y`, `z` | 经度、纬度和海拔 |
| `labels` | 预计算的气候分区 |
| `medoids` | DTW 聚类代表位置标识 |
| `States` | 预计算的隐降雨状态 |
| `year1980` | 历史时间索引 |

## 已经执行的真实检验

- CSV 完整大小为 **216,657,356 字节**；
- **2,432,940** 条有效降水记录，其中 **793,844** 条精确 0（**32.629%**）；
- 其余 **1,639,096** 条正值中 **1,638,991** 条带小数；
- 完整 CSV 的 SHA-256：`93609b0eacd2809f0d7bd86fa44ab0d54034faf7d306180fc02108307502d92a`；
- 经 GitHub Actions **全新 Git LFS checkout** 后再次确认 CSV 大小、SHA-256 和记录数：[验证日志](https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research/actions/runs/45777509526)。

## 怎样取得解压后的 CSV？

```bash
git lfs install
git clone --branch datasets/continuous-zero-inflated-verified-20261126 https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research.git
cd TSJT-SOFTS-Research
git lfs pull
python data/zero_inflated_papers/verify_datasets.py
```

然后直接用 Pandas 读取 `data/zero_inflated_papers/hybrid_nhmm_copula/raw/mapa_estaciones_precipitaciones_clean.csv`，不再需要解压。若读取到只有 134 字节，说明 Git LFS 客户端尚未获取完整文件。

**许可/溯源**：本分支存放的是原作者经过加工的数据，原始 ZIP 的 Git blob SHA 是 `0d007e8f179c4602ca5d44a294cf5c2defda623f`。作者 `LICENSES.md` 提醒加工数据可能有第三方来源，对外再分发权限需要进一步核实；见 `DATA_RIGHTS_NOTICE.md`。
