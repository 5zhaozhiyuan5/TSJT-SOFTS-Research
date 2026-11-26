# SAECZI — Continuous Zero Inflated Small Area Estimation（CJFR 2025）

论文：*Small area estimation of forest biomass via a two-stage model for continuous zero-inflated data*

- 期刊：https://doi.org/10.1139/cjfr-2024-0149
- R 包源码：https://github.com/harvard-ufds/saeczi
- CRAN：https://cran.r-project.org/package=saeczi
- 数据来源：美国 USDA Forest Inventory and Analysis（FIA）森林样本示例；**这里提交的是俄勒冈州示例，不是论文内华达州完整实验原始数据**。

## 已真正提交的文件

| 文件名 | 大小（作者仓库） | 作用 | 关键字段 |
| --- | ---: | --- | --- |
| \`raw/samp.rda\` | 27,899 B | 样地层面的真实采样数据；用于学习零值发生与正生物量大小 | \`DRYBIO_AG_TPA_live_ADJ\`（地上活生物量）, \`COUNTYFIPS\`（县域分组）, \`tcc16\`（树冠覆盖率）, \`elev\`（海拔） |
| \`raw/pop.rda\` | 124,974 B | 俄勒冈各县像元级总体辅助变量；在全部目标区域上应用拟合模型 | \`COUNTYFIPS\`, \`tcc16\`, \`elev\` 等 |
| \`README.md\` | 文本 | 本文件、来源及字段说明 | — |

源文件 Git blob SHA-1（可用于校验所提交二进制是否与作者仓库一致）：

- \`samp.rda\`: \`6df7eb79d82f62777e2fe2f4fa02a18d7bf76bd3\`
- \`pop.rda\`: \`e42b6a1ac2484069f0d2fadeca5ddd1b58cf74df\`

该数据是**零膨胀连续型响应**，但用途是**森林生物量空间统计估计**，不能误标为时序降水数据。

## 用 R 加载当前已提交的原始文件

从仓库根目录：

\`\`\`r
env <- new.env(parent = emptyenv())
load("data/zero_inflated/saeczi/raw/samp.rda", envir = env)
load("data/zero_inflated/saeczi/raw/pop.rda", envir = env)
names(env)
head(env$samp)
head(env$pop)
\`\`\`

若你已安装 R，则从仓库根目录执行：

\`\`\`bash
Rscript data/zero_inflated/scripts/export_saeczi.R
\`\`\`

结果将存储于 \`saeczi/exported/samp.csv\` 与 \`saeczi/exported/pop.csv\`（默认忽略，不写入 Git）。转换只改变文件格式，不会模拟或造出样本。R 包版本可用 \`install.packages("saeczi")\` 安装，也可以直接加载提交的 \`.rda\` 文件，无需安装整个 R 包。

注意：\`samp\` 的真实零值须结合 \`DRYBIO_AG_TPA_live_ADJ\` 实测；\`pop\` 是辅助变量表，不是目标标签。
