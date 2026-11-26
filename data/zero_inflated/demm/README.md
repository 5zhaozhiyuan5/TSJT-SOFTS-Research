# DEMM — Deep Extreme Mixture Model（KDD 2022）

论文：*Beyond Point Prediction: Capturing Zero-Inflated & Heavy-Tailed Spatiotemporal Data with Deep Extreme Mixture Models*

- 作者源码：https://github.com/ampersandmcd/DeepExtremeMixtureModel
- 作者数据说明：https://github.com/ampersandmcd/DeepExtremeMixtureModel/blob/main/data/README.md
- **作者预处理数据包**：https://drive.google.com/file/d/1vy9h3uiarpwrFCGFgr9Ex81q2_1v0f34/view?usp=sharing
- 数据源：SubX（预报输入）和对应观测降水标签；主要面向美国本土时空预测。

## 原论文数据结构

| 文件名 | 状态 | 作用 |
| --- | --- | --- |
| \`raw/demm_author_bundle.zip\` | 尚未加入 Git；脚本从作者 Google Drive 获取 | 作者约 400MB 数据压缩包，需下载后检查内部文件名 |
| \`raw/processed_data.pickle\` | 下载/解包后预期存在，尚未本地验证 | 预处理数据字典；X 已做 \`log(1+x)\` 和标准化，Y 是目标观测降水 |
| \`raw/rand_thresholds.pickle\` | 下载/解包后预期存在，尚未本地验证 | 训练可变阈值 DEMM 时预先生成的随机阈值 |
| \`raw/all_data.pickle\` | 作者 README 提到的未标准化版本，但不保证压缩包包含 | 原始预报特征，尚未进行对数变换/标准化 |
| \`README.md\` | 当前已提交 | 本数据说明 |

作者数据 README 中列出的维度：

- 输入 \`X\`：\`[939, 11, 7, 29, 59]\`；939 个每周模拟、11 个集合成员、7 个预报平均周期、29×59 空间网格。
- 标签 \`Y\`：\`[939, 1, 29, 59]\`；向前约 10 天的 3 天平均降雨量，额外通道维度为方便实现。
- 实际解压的内部目录层级可能是 \`subx/\`；按作者 \`src/train.py\`、\`src/test.py\` 的路径 \`../data/subx/...pickle\` 校对。
- \`.pickle\` 是 Python 序列化文件，**不要从不可信来源执行/加载**；只使用作者官方文件，并通过检查结构与哈希确认来源。

下载：

\`\`\`bash
pip install gdown
python data/zero_inflated/scripts/manage_datasets.py demm
\`\`\`

脚本默认将 zip 保存至 \`data/zero_inflated/demm/raw/\`，**不会自动解压或推送到 Git**。需在本地确认归档结构并手动解压，再根据 DEMM 官方训练脚本放到要求的位置。

本目录暂未提交 DEMM 实际 .pickle 数据，不能据此声称 DEMM 数据已经复制进仓库。
