# 三组已核实的连续域零膨胀论文数据

本目录只存放三组论文的真实数据及必要说明/校验文件，**智利数据已经解压成完整的 CSV，并通过 Git LFS 上传至本分支**。

| 目录 | 论文及研究任务 | 真实文件（本分支） | 作者源码 |
|---|---|---|---|
| `saeczi/` | 2025 Canadian Journal of Forest Research，零膨胀森林生物量空间估计 | `raw/samp.rda`、`raw/pop.rda` | https://github.com/harvard-ufds/saeczi |
| `rainfallbench/` | 2026 IEEE TGRS，气象站连续降水预测 | `stations/J340_data.csv`、`ZIMM_data.csv`、`P095_data.csv` | https://github.com/yfangZhang/RainfallBench |
| `hybrid_nhmm_copula/` | 2026 Hydrology and Earth System Sciences，智利多地点连续零膨胀降水、空间相关性 | **`raw/mapa_estaciones_precipitaciones_clean.csv`（216,657,356 B，Git LFS）** | https://github.com/mauricio-herrera/hybrid-nhmm-copula-precipitation-framework |

### 完整 CSV 文件和检验方式

智利文件是解压后的完整 CSV。**普通 GitHub 网页显示 Git LFS 指针不等于数据未上传**，需 Git LFS 客户端获取真实字节：

```bash
git lfs install
git clone --branch datasets/continuous-zero-inflated-verified-20261126 https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research.git
cd TSJT-SOFTS-Research
git lfs pull
python data/zero_inflated_papers/verify_datasets.py
```

`verify_datasets.py` 会校验作者 RData 和三份气象 CSV 的 Git blob SHA、智利完整 CSV 的 SHA-256 与实际文件大小；若 LFS 只下载到指针而没有完整 CSV，则会明确报错。

**实测的零值比例**：智利 `precipitation` 32.629%（793,844 / 2,432,940）；RainfallBench GPM 站点 J340 80.612%、ZIMM 84.936%、P095 94.711%。JFNG 公开的 `tp` 列不含精确零值，因此不选为严格连续域零膨胀样本。

智利数据处理后 CSV 的再分发许可尚需独立确认；请保留作者来源与引用，详情见 `hybrid_nhmm_copula/DATA_RIGHTS_NOTICE.md`。
