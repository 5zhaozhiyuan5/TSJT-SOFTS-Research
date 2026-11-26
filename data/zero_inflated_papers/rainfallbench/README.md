# RainfallBench — 连续零膨胀降雨时间序列（2026）

**正式论文**：*How Effective Are Time-Series Models for Precipitation Nowcasting? A Comprehensive Benchmark for GNSS-Based Precipitation Nowcasting*

**期刊**：IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026。DOI https://doi.org/10.1109/TGRS.2026.3671768

**论文对应源码**：https://github.com/yfangZhang/RainfallBench

**预测任务**：利用气象站历史序列预测未来 0–6 小时降水。核心目标列 `tp`（Total precipitation）是非负连续量，包含大量真实零降水时刻；其他协变量为 `t2m`（2m 气温）、`sp`（地表气压）、`rh`（相对湿度）、`wind_speed`、`pwv`（GNSS 可降水量）。

## 数据文件

完整作者仓库固定版本：`upstream/`（Git 子模块）。**这是完整公开数据集的路径**，而不是只取一段示例。

四个用于快速实测/对比的代表 CSV（以源文件 Git blob SHA-1 逐字节校验）：

| 复制到的文件 | 原始来源（位于 `upstream`） | 原文件大小 | 区域 |
|---|---|---:|---|
| `stations/J340_data.csv` | `dataset/gnss/Asia/J340_data.csv` | 4,652,001 B | 日本 |
| `stations/ZIMM_data.csv` | `dataset/gnss/Europe/ZIMM_data.csv` | 4,680,618 B | 瑞士 |
| `stations/P095_data.csv` | `dataset/gnss/North_America/P095_data.csv` | 4,536,152 B | 美国 |
| `stations/JFNG_data_15min.csv` | `dataset/Rain/JFNG_data_15min.csv` | 9,325,353 B | 中国 |

这些副本如 `stations/` 中不存在，可通过 `git submodule update --init --recursive` 从 `upstream` 直接访问对应原始路径；已提交的副本仍须通过 `verify_datasets.py` 校验。不同站点采样频率可能不同（JFNG 标为 15min），不要强行按同一时间间隔处理。

**免责声明**：论文总体描述了明显零膨胀现象，但每个站点的实际零值比例需要读取并计算；不会将“总体 >70%”直接视为某站点结论。RainfallBench 的代码 MIT 许可不应自动等同于第三方气象来源的所有数据许可，请保留来源与作者引用。

主要预处理：时间排序、按时间划分 train/val/test，避免滑窗跨划分泄露；如果预测 `tp`，注意连续值与零点保持原样，不要随意填补零值。
