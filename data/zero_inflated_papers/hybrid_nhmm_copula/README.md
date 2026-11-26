# Hybrid nHMM–Copula — 智利多地点连续域零膨胀降水（2026）

**论文**：*Characterizing Climate-Driven Shifts in Chilean Rainfall Regimes with a Hybrid Hidden Markov–Copula Framework*。Hydrology and Earth System Sciences（2026），30，6531–6547，DOI https://doi.org/10.5194/hess-30-6531-2026

**原作者代码仓库**：https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework

**论文代码脚本**：https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework/blob/main/Scripts/Hybrid_nHMM_Copula_Simulation_Validation_2022.R

## 数据在本仓库的真实位置

```text
raw/
  mapa_estaciones_precipitaciones_clean.csv.zip  # 33,960,384 字节，已上传，不是链接和子模块
  download_verified.json                         # 下载源、原始 Git 哈希、SHA256、ZIP 校验
```

[查看本分支实际 ZIP](raw/mapa_estaciones_precipitaciones_clean.csv.zip)

**原作者文件**：https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework/blob/c09e5c46c6c5c237c86a3d214768febfc9e0dbac/Data/mapa_estaciones_precipitaciones_clean.csv.zip

**原始 Git blob SHA-1**：`0d007e8f179c4602ca5d44a294cf5c2defda623f`；本仓库 ZIP 的 Git blob SHA 相同。解压可得主 CSV `mapa_estaciones_precipitaciones_clean.csv`，大小 **216,657,356 字节**。

## 字段说明

| 字段 | 解释 |
|---|---|
| `precipitation` | 连续非负日降水量，单位 mm，含精确零值 |
| `dates`, `year`, `month`, `day`, `Year` | 日期和日历信息 |
| `codigo_estacion`, `location` | 地点/网格位置标识 |
| `x`, `y`, `z` | 空间坐标与海拔 |
| `labels` | 预计算的气候分区 |
| `medoids` | 预计算的 DTW 聚类代表点 |
| `States` | 预计算的降雨隐藏状态 |
| `year1980` | 历史时间索引 |

## 实测数据质量（原作者处理后的 CSV）

- 总有效降水记录 **2,432,940** 条，缺失/非法值为 **0**；
- 精确零值 **793,844** 条，比例 **32.629%**；
- 正值 **1,639,096** 条，其中 **1,638,991** 条有小数部分；
- Git SHA 与作者固定源码一致， ZIP CRC 校验通过。

## 下载与解压

常规 `git clone` 本分支即可获得 ZIP，不需要初始化作者仓库子模块。运行：

```bash
python -m zipfile -l data/zero_inflated_papers/hybrid_nhmm_copula/raw/mapa_estaciones_precipitaciones_clean.csv.zip
python -m zipfile -e data/zero_inflated_papers/hybrid_nhmm_copula/raw/mapa_estaciones_precipitaciones_clean.csv.zip ./
python data/zero_inflated_papers/verify_datasets.py
```

**作者代码与来源**：本分支只存放真实 ZIP、校验和说明，不复制作者 R 源码。原论文用于多地点连续降水的零膨胀分布、非齐次隐马尔可夫降雨状态、DTW 分区、GPD 极值和 vine copula。作者软件存档包含 2022 年留出验证示例，但原始 CR2MET 重建流程不完整。

**版权提醒**：本次按仓库所有者要求复制了原作者发布的 ZIP。原作者 `LICENSES.md` 提醒，压缩包是多个来源加工的研究数据，上游再分发许可尚需要确认；本次拷贝不等于获得额外授权。详情见 `DATA_RIGHTS_NOTICE.md`。
