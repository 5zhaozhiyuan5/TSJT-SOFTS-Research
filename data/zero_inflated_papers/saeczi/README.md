# SAECZI — 连续零膨胀森林生物量（2025）

**正式论文**：*Small area estimation of forest biomass via a two-stage model for continuous zero-inflated data*

**期刊**：Canadian Journal of Forest Research（2025），DOI https://doi.org/10.1139/cjfr-2024-0149

**代码/数据仓库**：https://github.com/harvard-ufds/saeczi

**方法**：对森林地上生物量以两个模型分别估计非零发生概率与非零连续生物量大小；适用于空间小区域估计。

## 已直接入库的真实文件

| 文件 | 原仓库大小 | 用途 |
|---|---:|---|
| `raw/samp.rda` | 27,899 字节 | 美国俄勒冈州 FIA 森林样地的实测样本，响应变量 `DRYBIO_AG_TPA_live_ADJ`（地上活生物量）具有零值/正值 |
| `raw/pop.rda` | 124,974 字节 | 俄勒冈州各县像元级总体辅助变量，用于对全域进行估计；不是生物量真实标签 |

**代表字段**：`COUNTYFIPS` 县名/县代码分组，`tcc16` 树冠覆盖率，`elev` 海拔。`samp` 的 `DRYBIO_AG_TPA_live_ADJ` 才是零膨胀连续响应。

原作者 Git blob SHA-1：
- samp.rda: `6df7eb79d82f62777e2fe2f4fa02a18d7bf76bd3`
- pop.rda: `e42b6a1ac2484069f0d2fadeca5ddd1b58cf74df`

## 读取示例

```r
e <- new.env()
load("data/zero_inflated_papers/saeczi/raw/samp.rda", envir=e)
load("data/zero_inflated_papers/saeczi/raw/pop.rda", envir=e)
head(e$samp)
head(e$pop)
mean(e$samp$DRYBIO_AG_TPA_live_ADJ == 0, na.rm=TRUE)
```

重要：这是作者 R 包公开提供的**俄勒冈州示例数据**，不能当作论文全部内华达州真实研究输入。2025 论文提到部分保密 FIA 数据不可公开，本仓库没有这些受限数据。
