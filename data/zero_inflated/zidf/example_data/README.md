# ZIDF 示例合成数据（不是 NOAA 真实数据）

- `zidf_synthetic_10000_zero70_seed42.csv`: 10000 行，5 列特征 + 1 列连续非负 `target`。利用作者 `data/preprocessing.py` 公开的合成规则，在 NumPy RandomState(seed=42) 下生成；实际零值占比 69.52%。
- `metadata.json`: 数据生成规则、采样数、随机种子和是否真实数据的标记。
- **仅用于测试 ZIDF 数据管道，不代表论文真实南澳大利亚降水数据或论文全部合成实验。**
- 南澳大利亚真实 NOAA 原始 NetCDF **仍未下载及提交**，见 `../README.md`。
