# MTPB 有机融合 v2：替换源域元更新规则，不叠加外挂修正

日期：2026-11-09；基线：研究仓库 main
\`a4f72961cb26e8204b9b12587207c0998a0d04c9\`。
新算法位于独立 iteration 分支；未经过完整 MTPB 四城市运行，不宣称精度提升。

## 上游基线与实际替换位置

MTPB (Zhanyu Liu, Guanjie Zheng, Yanwei Yu), ACM TKDD 19(4), 2025,
DOI https://doi.org/10.1145/3727622 。
官方仓库 https://github.com/zhyliu00/MTPB ，固定源码提交
\`5676a38a34bf3aebbd83cef3cb0d617a9a153b15\` 。

官方 \`model/Meta_Models/rep_model_final.py\` 中：
- \`PatchFSL.forward\`：多尺度模式库检索、基于模式的图重建、GWN
  短期特征、长期分支、特征融合与预测输出；
- \`STRep.meta_train_revise\`：源任务 support 内更新、query
  损失及最终参数更新；
- \`STRep.finetuning\`：目标城市适配与评估。

**只替换第二项：源域 meta-train 的参数更新逻辑。**
\`integrated\` 模式完全不调用 \`STRep.meta_train_revise\` ，不再出现
「先 MTPB 原生更新、再进行额外梯度补偿」。
保持完整的 \`PatchFSL\`、Graph WaveNet、模式库、预测头、
目标微调 \`finetuning\` 与 288→36 时域协议。原生 MTPB
\`native\` 模式仅作为控制组保留，不改写作者源码。

本项目先前的 \`iteration/mtpb-native-288x36-gradient-adapter\`
采用的「原更新 + delta_g 后置补丁」在本版本被 **完全放弃**；
旧尝试未运行四城市实证，因此不是已证明失败的效果实验。

## 核心机制（一个替代模块）

对每个城市的 support/query 执行 K=2 步 first-order
support adaptation，得到 query 元梯度：

\[
\theta_c^{(k+1)}=\theta_c^{(k)}-\eta_{\rm inner}
\nabla L_{S_c}(\theta_c^{(k)}),\qquad
g_c=\frac1K\sum_{k=1}^{K}\nabla L_{Q_c}(\theta_c^{(k)}) .
\]

这是 **一阶元梯度近似**，不等同于 MTPB 作者原代码的二阶 MAML，
也不声称复制其完全相同的优化轨迹。

三个来源城市：METR-LA、成都、深圳；
目标 PEMS-BAY 只取第 1–3 天有标签训练窗口。
PGrad 顺序虚拟优化和源城市关系学习构成参考方向 \(p\)，
目标梯度 \(g_T\) 指导源方向过滤。
前述模块的 PGrad/rotation/fusion 代码保持可复用，未另堆新模块。

令 \(w_c\) 是 source interaction 给定的非负城市权重，\(v_c\)
是原 residual fusion 的非负权重，\(R(g_c,g_T,p,\phi_c)\)
是目标正交空间旋转结果。更新梯度为

\[
g_{\parallel}
=\sum_c w_c\left[\frac{\langle g_c,g_T\rangle}
{\|g_T\|^2}\right]_+g_T,\quad
g_{\perp}=\lambda\sum_c v_c R(g_c,g_T,p,\phi_c),
\]

\[
\boxed{\quad\theta\gets \theta-
\eta_{\rm meta}\operatorname{ClipNorm}_{10}
(g_{\parallel}+g_{\perp})\quad}
\]

目标梯度用于确定方向与融合，不会在末尾另外执行一次目标 SGD。
角度控制器仍用目标**训练窗口**的独立查询批次进行一阶更新。
如果所有可用的源方向都被拒绝，梯度可以为零；
需通过日志确认是否导致训练停滞。

模型的核心创新位置是 **MTPB 源任务元梯度形成和更新**，
而不是 MTPB 预测架构或新道路拓扑。

## 输入长度和数据隔离

- 官方 \`configs/config.yaml\`：history=288，prediction=36。
- MTPB 的 \`datasets.py\` 已将 10 分钟采样的成都和深圳数据插值为
  5 分钟网格。跨城统一 288→36，而**不是**沿用旧 TSJT 6→6 / 12→12。
- 目标梯度数据仅来自 \`target_maml\` 的前三天；
  \`test\` 数据只进入原生最终评价 \`STRep.finetuning\`。
- 原生 MTPB 的 \`source_train\` 默认 \`add_target=True\`，
  这里对 native/integrated 两模式均设 \`add_target=False\`，
  使源城市列表明确仅为 LA/成都/深圳。
  因而与论文作者默认源域训练协议不完全相同。
- 上游 MTPB 在切分之前使用全量目标速度计算均值/标准差，
  具有测试期信息风险。当前 runner 对 native/integrated
  两模式均改为仅用前三天目标速度计算标准化统计量；
  真实速度 y 保持原始尺度。这是数据隔离修复而不是额外算法创新。
- support/query 抽样索引不重合，但滑窗在物理时间上可能重叠；
  下一轮应实验严格时间块隔离对角度稳定性的影响。
- 原模型仍使用自带的图重建/邻接矩阵输入；道路拓扑不作为
  我们新增的创新或依赖，不增添 POI、额外数据、LLM。

## 运行

外部先单独下载并固定上游作者仓库，不向作者仓库提交代码：

\`\`\`bash
git clone https://github.com/zhyliu00/MTPB /path/to/MTPB
cd /path/to/MTPB
git checkout 5676a38a34bf3aebbd83cef3cb0d617a9a153b15
# 按作者 README 准备 data/；pattern/ 已由官方仓库提供
cd /path/to/TSJT-SOFTS-Research
pip install torch numpy PyYAML pytest
PYTHONPATH=src pytest -q tests/test_mtpb_integrated_meta.py tests/test_mtpb_train_only_norm.py

# 无数据时仅运行参数解析
PYTHONPATH=src python scripts/run_mtpb_integrated.py --help

# 有完整真实数据+依赖后，先各进行 1 次源域元更新
PYTHONPATH=src python scripts/run_mtpb_integrated.py --upstream /path/to/MTPB \
  --mode native --epochs 1 --skip-finetune --output outputs/mtpb_native_smoke
PYTHONPATH=src python scripts/run_mtpb_integrated.py --upstream /path/to/MTPB \
  --mode integrated --epochs 1 --order-count 1 --skip-finetune \
  --output outputs/mtpb_integrated_smoke

# 有计算资源并明确启动正式实验时（暂未执行）
PYTHONPATH=src python scripts/run_mtpb_integrated.py --upstream /path/to/MTPB \
  --mode native --epochs 50 --finetune-epochs 300 \
  --output outputs/mtpb_native_seed7
PYTHONPATH=src python scripts/run_mtpb_integrated.py --upstream /path/to/MTPB \
  --mode integrated --epochs 50 --finetune-epochs 300 \
  --output outputs/mtpb_integrated_seed7
\`\`\`

官方 README 所用 PyTorch 1.10 与我们 \`torch.func.functional_call\`
要求的 PyTorch 2.x 不兼容，需要另外确认 PyG 和官方模型依赖版本。
\`upstream_compat.py\` 仅修复作者原 GWN Conv1d 接收四维张量的兼容问题；
两组同时修，不以其作为性能提升原因。

## 证据、限制及下一轮

- 本轮在本地 CPU 用 PyTorch 2.10.0 的**模拟 MTPB 接口**
  执行单测；这只说明逻辑连接正确，不证明官方模型可前向。
- 本轮环境不能联网 clone GitHub 原始源码/数据，未运行四城市，
  无 PEMS-BAY MAE、RMSE、训练曲线、pretrained checkpoint 或
  可采信的梯度统计结果。
- 新实现每轮单独处理三个源域 support/query、目标训练
  support/query 及 PGrad 虚拟轨迹。与原 MTPB 虽保持 50 个
  元训练 epoch，但 FLOPs/前后向次数并非天然相同。
  正式实验须记录每轮耗时及总训练预算，做 compute-matched 对照。
- 下一步优先用原版完整 \`PatchFSL\` 实测前向、反向、参数更新和
  \`torch.func\` 兼容性；确认 pred std、Persistence、旋转触发
  次数、源目标余弦、融合非零贡献及更新范数。
  随后固定 50/300 epoch 与同一源域采样协议比较 MAE/RMSE。

## 论文核验（方法参考不是效果背书）

1. MTPB, ACM TKDD 2025, doi:10.1145/3727622,
   https://github.com/zhyliu00/MTPB ，原生任务元适配及模式检索。
2. *Drawing Informative Gradients from Sources* (TSJT),
   AAAI 2025, doi:10.1609/aaai.v39i1.32102，目标引导源梯度
   过滤；原论文不是 MTPB 的直接训练验证。
3. *PGrad: Learning Principal Gradients for Domain Generalization*,
   ICLR 2023, https://github.com/QData/PGrad ，源域顺序虚拟
   训练及主方向；此处迁移至 MTPB 属于研究假设。
4. *PiKE: Adaptive Data Mixing for Large-Scale Multi-Task Learning
   Under Low Gradient Conflicts*, NeurIPS 2025,
   https://papers.nips.cc/paper_files/paper/2025/hash/f7a94134f1c726796c6f81fb946e489d-Abstract-Conference.html 。
   正向任务交互可能有价值，但本轮**没有**移植 PiKE 采样算法。
