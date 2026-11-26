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
