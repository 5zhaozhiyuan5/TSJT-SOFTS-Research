# ZIDF — Zero Inflation Diffusion Framework（ECAI 2025）

论文：*From Noise to Precision: A Diffusion-Driven Approach to Zero-Inflated Precipitation Prediction*

- 官方源码：https://github.com/Wentao-Gao/ZIDF-from-noise-to-precision
- 期刊/会议 DOI：https://doi.org/10.3233/FAIA250921
- NOAA NCEP/NCAR 数据：https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
- 作者配置：https://github.com/Wentao-Gao/ZIDF-from-noise-to-precision/blob/main/configs/sa_precip.yaml
- 原始数据为南澳大利亚地区的 NCEP/NCAR 气象再分析，真实样本具有大量零降雨量；仓库没有随附原始 NetCDF 文件。

## 文件及字段

| 文件名 / 位置 | 是否已入 Git | 作用 |
| --- | --- | --- |
| \`raw/<年份>/prate.sfc.gauss.<年份>.nc\` | 否；可通过脚本获取 | NOAA 逐日地表降水率 \`prate\`（通常单位 kg/m²/s，**不是 mm**） |
| \`raw/<年份>/air.sig995.<年份>.nc\` | 否 | 近地面气温 \`air\` |
| \`raw/<年份>/uwnd.sig995.<年份>.nc\` | 否 | 近地面东西向风速 \`uwnd\` |
| \`raw/<年份>/vwnd.sig995.<年份>.nc\` | 否 | 近地面南北向风速 \`vwnd\` |
| \`raw/<年份>/rhum.sig995.<年份>.nc\` | 否 | 近地面相对湿度 \`rhum\` |
| \`raw/<年份>/pres.sfc.<年份>.nc\` | 否 | 地表气压 \`pres\` |
| \`synthetic/synthetic_zero70_n10000_seed42.npz\` | 否；可本地生成 | \`features\`、\`targets\` 两个数组（零值 + Gamma 连续正值） |
| \`synthetic/...metadata.json\` | 否；可本地生成 | 合成数据设置、实际零值比例和来源标记 |
| \`README.md\` | 是 | 说明上述全部文件的角色与实际状态 |

**注意**：上面的 NOAA 文件名是公开 NCEP dailyavgs 的原始文件模式，并非作者仓库中已预处理好的最终输入。作者 \`configs/sa_precip.yaml\` 将路径抽象为 \`prate.nc\`、\`air.nc\` 等，需自行合并年份、区域裁剪、时间对齐和单位处理。原始数据下载不等于论文严格复现。

## 生成连续域零膨胀合成数据

\`\`\`bash
pip install numpy
python data/zero_inflated/scripts/manage_datasets.py zidf-synthetic --samples 10000 --zero-ratio 0.7 --seed 42
\`\`\`

此脚本按作者 \`data/preprocessing.py\` 中的 \`generate_synthetic_data\` 数学流程独立实现：周期项、趋势项、随机协变量；正值从 Gamma(shape=2, scale=0.05) 抽样；按 Bernoulli 掩码置零；非零部分再施加周期性影响。**这是人工合成数据，不是 NOAA 实测值**。

- \`features\` 形状：\`(n_samples, n_features)\`，默认 5 个特征。
- \`targets\` 形状：\`(n_samples,)\`，非负，部分点精确等于零。
- \`zero_ratio\` 是采样概率，实际比率不会严格等于请求值。
- 作者预测脚本默认历史长度 48、预测长度 24；构造预测窗口需注意按时间切分避免泄露。

## 获取 NOAA 原始数据

\`\`\`bash
# 先仅下载 1948 年进行小规模验证
python data/zero_inflated/scripts/manage_datasets.py zidf-noaa --start-year 1948 --end-year 1948

# 再根据磁盘、网络、变量定义决定是否下载 1948–2014 全部年度
python data/zero_inflated/scripts/manage_datasets.py zidf-noaa --start-year 1948 --end-year 2014
\`\`\`

官方配置 South Australia 空间范围：纬度 -38 到 -26，经度 129 到 141。原始 \`lat\` 可能是降序坐标，因此直接使用 \`slice(-38,-26)\` 可能选不到区域；请确认排序方向。区域平均后的 \`prate\` 可能不再具有原始网格点同样的零值比例，必须本地重算并与论文描述对比。对降水率到日累计降水的换算需要核实源文件的 \`units\` 和时间定义，例如 kg/m²/s 对应的全天量需乘以 86400 秒（1 kg/m² 水约等于 1 mm）。

**本次未上传原始 NOAA NetCDF，也未声称可直接得到作者表格中的 68.3% 零值比例。**

## 已提交的合成样例文件（不是 NOAA 原始数据）

- `example_data/zidf_synthetic_10000_zero70_seed42.csv`：10,000 条真实生成并已上传的零膨胀合成记录；字段为 `periodic,linear,random1,random2,random3,target`，实际零值占比 69.52%。
- `example_data/metadata.json`：生成参数和来源标记。
- `example_data/README.md`：逐文件说明。可用于测试数据加载，但不能代替 1948–2014 年南澳大利亚真实降水数据。


## 2026-11-26: 一键下载 NOAA 真实南澳大利亚区域数据（新脚本）

推荐使用新脚本 \`../scripts/download_zidf_noaa.py\`，默认优先通过 NOAA THREDDS **NCSS 区域裁剪服务**，只请求南澳大利亚附近网格，避免拉取每年的全球原始大文件。变量映射依据 NOAA THREDDS 官方 annual 数据目录：

- \`prate\` → \`surface_gauss/prate.sfc.gauss.YYYY.nc\`
- \`air\` → \`surface/air.sig995.YYYY.nc\`
- \`uwnd\` → \`surface/uwnd.sig995.YYYY.nc\`
- \`vwnd\` → \`surface/vwnd.sig995.YYYY.nc\`
- \`rhum\` → \`surface/rhum.sig995.YYYY.nc\`
- \`pres\` → \`surface_gauss/pres.sfc.gauss.YYYY.nc\`

\`\`\`bash
# 先测试 1948 年六个变量是否能获取
python data/zero_inflated/scripts/download_zidf_noaa.py download --start-year 1948 --end-year 1948

# 完整 1948–2014，共 67 年 × 6 变量 = 402 个 NetCDF
python data/zero_inflated/scripts/download_zidf_noaa.py download --start-year 1948 --end-year 2014

# 检查是否每个年份每个变量都拿到了 NetCDF
python data/zero_inflated/scripts/download_zidf_noaa.py verify --start-year 1948 --end-year 2014

# 若需要将区域均值重新整理为可训练 CSV / NPZ：
pip install numpy pandas xarray netCDF4
python data/zero_inflated/scripts/download_zidf_noaa.py prepare --start-year 1948 --end-year 2014
\`\`\`

- 下载的真实 NOAA 文件放于 \`zidf/raw/noaa_sa/subset/YEAR/\`，不加入 Git 历史；每个文件都有 NetCDF 文件头检查（不等于全内容校验）。
- 输出 \`zidf/raw/noaa_sa/download_manifest_subset.jsonl\` 用于记录具体文件、请求 URL、状态、字节数。
- 按文件粒度跳过已经检查通过的 NetCDF，失败后重运行相同命令即可补齐缺失文件。
- 如果 NCSS 区域裁剪服务发生错误，可添加 \`--mode global\` 直接下载年度完整全球网格 NetCDF。
- \`prepare\` 输出至 \`zidf/processed/\`，按各变量独立区域均值、每日对齐， \`prate\` 检测单位后转换为 mm/day，最后保存 \`csv/npz/json\`。**这是我们独立实现的区域聚合方法，不保证与 ZIDF 论文数据加工完全相同；区域均值可能降低或改变精确零值比例。**
- 当前助手所在环境的 DNS 解析失败，没有在这里成功下载 NOAA 原始文件；请在能够访问 NOAA 的 Windows/Linux 主机执行。不能将只有脚本的目录当作完整数据集。


## 实际下载测试记录 — 2026-11-26

- 在当前助手计算环境通过请求 NOAA \`downloads.psl.noaa.gov\` / \`psl.noaa.gov\` 时，出现 DNS 解析失败，未下载到文件。
- 在 GitHub Actions 工作流 [run 45772146229](https://github.com/5zhaozhiyuan5/TSJT-SOFTS-Research/actions/runs/45772146229) 实际执行 1948 年 6 个官方 NetCDF 区域裁剪请求，**六个均收到 HTTP 429 Too Many Requests**；NetCDF 校验步骤未执行，**成功下载数为零**。
- 该失败运行曾自动上传一个只有 724 字节的 artifact，其实**只包含失败日志，不包含 \`.nc\` 数据**，不要将其作为论文数据下载。
- 现在工作流改为手动执行，且只有 NetCDF 完成校验后才允许上传真实 .nc 文件，以防再出现误导性 artifact。
- 正常下载建议从本地网络或你的 Linux 服务器运行 \`scripts/download_zidf_noaa.py\`；如收到 429，请尊重 NOAA 的请求限制，稍后重试、降低请求频率，或直接从官方目录用浏览器下载，不应高频请求。
