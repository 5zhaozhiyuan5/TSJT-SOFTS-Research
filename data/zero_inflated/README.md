# 连续域零膨胀数据集整理（DEMM / ZIDF / SAECZI）

> 研究数据隔离目录；不属于 TSJT/PMME 现有训练数据，不修改原有代码。整理日期：2026-11-26。

## 数据状态（请先看这个表）

| 数据集 | 研究对象 | 原始数据形式 | 当前 Git 分支的实际内容 | 获取方式 |
| --- | --- | --- | --- | --- |
| DEMM / KDD 2022 | 美国本土降水时空预测 | 3 天降水、SubX 预报集合 | 文件说明 + 获取脚本，**大数据未上传** | 作者 Google Drive 约 400 MB 压缩包 |
| ZIDF / ECAI 2025 | 澳大利亚南部降水时间序列 | NOAA NCEP/NCAR NetCDF + 合成 Gamma 正值 | 文件说明 + 生成/下载脚本 + **1 万条合成 CSV 样本已上传**；真实 NetCDF 未上传 | NOAA 官方数据下载或本地生成合成数据 |
| SAECZI / CJFR 2025 | 俄勒冈州县域森林生物量 | R 的 \`.rda\` 文件 | **已提交真实 \`samp.rda\`、\`pop.rda\`** | 作者 Harvard-UFDS GitHub / CRAN R 包 |

**文件已入库**不等于**原论文全实验环境已复现**。DEMM 和 ZIDF 的真实训练数据尚未在 Git 仓库中；不应将合成数据误写为真实数据，也不应把 SAECZI 的俄勒冈示例数据误当论文的全部内华达实验。

## 目录结构

\`\`\`text
data/zero_inflated/
├── README.md                           # 本总览；获取现状与用途
├── sources.json                        # 统一数据来源、论文与文件清单（机器可读）
├── .gitignore                          # 防止大文件被直接推送 Git
├── demm/
│   └── README.md                       # DEMM 文件含义、维度、作者网盘
├── zidf/
│   └── README.md                       # ZIDF 真实/合成数据、预处理与陷阱
├── saeczi/
│   ├── README.md                       # SAECZI 字段含义、R/CSV 使用方法
│   └── raw/
│       ├── samp.rda                    # 已包含；真实森林样地样本
│       └── pop.rda                     # 已包含；真实像元级辅助变量
└── scripts/
    ├── manage_datasets.py             # DEMM 下载、NOAA 原始文件获取、ZIDF 合成生成、校验 SAECZI
    └── export_saeczi.R                # 将 .rda 导出 CSV；需安装 R
\`\`\`

运行命令（在 Git 仓库根目录）：

\`\`\`bash
# 只需 numpy；生成严格非负、零点质量 + 连续非零值的模拟数据
python data/zero_inflated/scripts/manage_datasets.py zidf-synthetic --samples 10000 --zero-ratio 0.70 --seed 42

# 检查已入库真实 R 数据的 Git blob 完整性
python data/zero_inflated/scripts/manage_datasets.py verify-saeczi

# 下载 DEMM 作者数据包；需先 pip install gdown
python data/zero_inflated/scripts/manage_datasets.py demm

# 下载 NOAA 1948 年 6 个原始 NetCDF 文件作为小规模验证
python data/zero_inflated/scripts/manage_datasets.py zidf-noaa --start-year 1948 --end-year 1948

# 安装 R 后导出 SAECZI CSV（包含全量样本与辅助变量）
Rscript data/zero_inflated/scripts/export_saeczi.R
\`\`\`

扩展下载全部 1948–2014 年 NOAA 原始文件可更改年份，但**规模大、下载耗时长**，而且跨网格、单位、时间对齐和气象变量预处理必须与论文逐项核实。直接运行 NOAA 下载命令得到的是原始数据，不是复现论文最终表格的保证。

## 数据放置与治理

- Git 仅追踪两份小型真实 R 数据及文本脚本。
- DEMM 下载归档位于 \`demm/raw/\`，ZIDF NOAA 原始文件位于 \`zidf/raw/\`，ZIDF 合成数据位于 \`zidf/synthetic/\`，均被本目录的 .gitignore 忽略。
- 不要删除或覆盖已有项目根目录 \`data/\` 中 TSJT/PMME 使用的文件。
- 原论文、数据供应方的版权、引用、重分发义务仍然适用；引用见各目录 README 和 sources.json。
- **实测零值比例、非零分布与预测精度**应在下载完整数据后重新统计，不能直接把论文数值当成本地结果。

## 本次补充（2026-11-26）

- `zidf/example_data/zidf_synthetic_10000_zero70_seed42.csv`：可下载的 10,000 条连续零膨胀合成数据（约 653 KB），5 个输入特征加 `target`，实际零值比例 69.52%。
- `zidf/example_data/metadata.json`：随机种子、生成规则及数据类型。
- **DEMM 实验数据和 ZIDF NOAA 真实降雨数据仍未成功下载到仓库**；不可将合成样例描述为原论文的真实实验数据。
