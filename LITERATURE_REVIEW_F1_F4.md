# F1–F4 第一轮机制探索：文献核查与实验决策（2026-11-24）

## 范围与判断

本轮不是最终论文创新查重，而是评估四个最小机制消融是否**有实验意义、与 DVGP 技术路径不同、且可解释地组合**。检索覆盖 2017–2026 年 OT 域适应、实例重加权、源域梯度选择、多任务优化、时间序列梯度可靠性。只使用检索所得的一手会议页面或机构记录。**不保证没有未检索到的同类方法**。

已发现 F2–F4 的关键数学操作均有接近先例，**不得把“梯度余弦/分时长平均/ESS”单独宣称新颖**。下一步只有在同 seed 验证机制有用之后，才能围绕“实际 OT 更新对于目标预测的边际效用”设计新的目标函数并再次查重。

## F1：无分类器、均匀边际 OT

- 依据：Courty et al., *Optimal Transport for Domain Adaptation*, TPAMI 2017；Courty et al., *Joint Distribution Optimal Transportation for Domain Adaptation*, NeurIPS 2017。
- 相关更新：Naram et al., *Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport*, ICML 2025, https://proceedings.mlr.press/v267/naram25a.html ; Koç et al., *Incorporating Importance Weighting in Optimal Transport Based Domain Alignment*, ICML 2026, https://proceedings.mlr.press/v306/koc26a.html
- 机制：只保留原始 Sinkhorn OT，alpha/beta 均匀；不输出城市归属概率。
- 必要性：隔离“在同样 OT 系数下，按预测效用加权是否比不加权好”。
- 可证伪点：若 F1 优于 F2–F4，则任务梯度加权可能不提供价值；不能归因 DVGP 一定无用。
- 论文本身创新性：无。

## F2：目标指导的预测头梯度效用

- 直接先例：Ren et al., *Learning to Reweight Examples for Robust Deep Learning*, ICML 2018, https://proceedings.mlr.press/v80/ren18a.html ; Killamsetty et al., *GRAD-MATCH*, ICML 2021, https://proceedings.mlr.press/v139/killamsetty21a.html
- 场景直接相关：Li et al., *GradMix: Multi-source Transfer across Domains and Tasks*, WACV 2020, https://arxiv.org/abs/2002.03264 ; Zhang et al., *Drawing Informative Gradients from Sources*, AAAI 2025, https://mlanthology.org/aaai/2025/zhang2025aaai-drawing/
- 相似性：以上方法均以参考/目标梯度指导源样本或源梯度选择；**目标引导梯度余弦不是创新**。
- 本次不同的验证对象：将节点级共享预测头一阶梯度方向转成 **OT 源边际**，而不是训练损失权重或直接替换梯度。目标边际均匀。
- 计算口径：真实 `projection2` 的输出权重梯度代理，非完整骨干的梯度；真实 OT 作用于 `rep`，不经过 `projection2.weight`；此梯度代理和实际 OT 更新可能不一致。
- 可证伪点：与 F1 同规模 OT 下若不改善，说明“源监督梯度≈有益 OT”不成立或代理过弱。

## F3：36 步分成三段的任务效用

- 参考：TSJT (AAAI 2025) 目标梯度引导；多任务梯度冲突 CAGrad, NeurIPS 2021；2026 预印本 *Not All Variables Agree: Reliability-Aware Variable-Wise Gradient Surgery for Multivariate Time-Series Forecasting*, https://arxiv.org/abs/2609.08554
- 已有工作警告：2026 预印本已涉及按变量分组、可靠性与输出层梯度代理，相关概念高度相似。其发表级别尚不能由预印本断言。
- 本次 F3 的实质仅是三段分别计算源—目标方向相似度并等权合并成**一个共享 OT 边际**；不训练三套预测骨干，也不执行三次独立 OT。
- 严格限制：不同预测区间在最后一层对应不同参数切片，不能用输出头不同区间的余弦来论证共享层跨区间梯度冲突。
- 可证伪点：F3 vs F2 是否改善 120/180 分钟且不牺牲 10 分钟；记录每段梯度余弦，不能只看最终总平均。

## F4：子批次可靠性与 ESS 约束

- 参考：Wang et al., *Gradient Vaccine*, ICLR 2021, https://research.google/pubs/gradient-vaccine-investigating-and-improving-multi-task-optimization-in-massively-multilingual-models/ ; Michalkiewicz et al., *Domain Generalization Guided by Gradient Signal to Noise Ratio of Parameters*, ICCV 2023, https://mlanthology.org/iccv/2023/michalkiewicz2023iccv-domain/ 。
- F4 两半目标支持集方向余弦是**批次一致性**，不是跨 epoch 动态稳定性。ESS 是已有重要性采样术语而非新方法。
- 强度与集中度不同：ESS 约束只控制 alpha 集中程度，不能证明哪些样本是对的；0.5 是预先指定的消融超参数。
- 可证伪点：F4 vs F3 对不同种子、不同预测区间是否稳定有收益，及其 ESS 分布是否与 F3 区分。

## 其他相似工作的警告

- Lyu et al., *Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint*, ICLR 2025, https://mlanthology.org/iclr/2025/lyu2025iclr-crossdomain/ 已研究在少量目标数据下用 OT 筛选源域样本；不能宣称“少目标样本+OT 数据过滤”是全新。
- Koç et al. ICML 2026 上述工作理论化重要性加权 OT；不得把“weighted OT”单独包装为创新。
- 模式选择、目标梯度稳定性、条件化传输的**新颖组合**仍需在第二轮架构确定后单独做更窄检索，不能提前保证唯一性。

## 第一轮严格实验协议（先检验，后创新）

固定：PEMS-BAY、目标前 3 天；SOFTS；训练 3000 + 6000；seeds 1/3/5；作者对应参数 DAcoef=1、DAepoch=1500、xyR=100、projection=1、MemSize=10000。所有 F1–F4 使用完全相同的 **固定 OT 缩放**；DVGP 的动态 OT_loss_scale 已被移除，因此与 E0 是不同的实验条件，不能仅凭 E0 vs F2 就下结论。

1. 第一组 F1/F2/F3/F4 × seed1，固定 PMME_TRANSFER_OT_SCALE=0.01（保守的初始尺度；**不是**经证明与 E0 等效的值）。
2. 对 F1/F2 × seed1 补充 scale=1.0 的**强度敏感性**比较。同一模型变体/不同缩放写入不同工作目录，不混日志。
3. 检查每次记录的 `task_utility_gradient_diagnostic`：在 **projection1** 共享层计算 `||g_OT||/||g_target||` 与方向余弦。不能用 projection2，因为 OT 不经过该层参数。
4. 若 OT/目标梯度比例极端，调整固定缩放并**同时重跑 F1 与 F2**，避免单独为某一方法调参造成不公平。
5. 方法选择依据：相同 seed 配对比较目标支持集选出的 Stage-I/Stage-II 最佳检查点对应的 10/60/120/180 min MAE/RMSE；最终三 seed 平均 ± 标准差、F1–F4 相互比较。测试集不用于选模；检查日志的权重 ESS、余弦正比例、区间可靠性。
6. F2 若有效→研究“真正 OT 梯度对目标损失的边际收益”以形成区别于单纯梯度余弦的设计；F3 有效→验证区间偏好与共享/区间 OT 的关系；F4 有效→做目标时间窗多批次可靠性而不是复制 GradVac。无效先分析代理偏差、OT 强度和抽样，不盲目叠模块。

## 代码改动范围与验证状态

- `model/Meta_Models/PMME.py`：F1–F4 权重接口，固定可调 `PMME_TRANSFER_OT_SCALE`，在 `projection1.weight` 上每 25 epoch 做 OT/目标梯度量级和夹角诊断（只读，无额外优化步骤）。
- `model/Meta_Models/transfer_utility.py`：原 F1–F4 解析梯度代理、权重计算不变。
- `run_transfer_experiments.sh`：scale×seed×variant 隔离，记录配置。
- 官方 `train_DA_OT.py`、`PMME_2ndStage.py`、SOFTS、配置不变。
- GPU 完整训练尚未执行；需在服务器先执行 Python CPU 测试，再进行 seed1 的顺序试跑。
