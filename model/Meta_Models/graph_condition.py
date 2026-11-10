"""
GATP Graph Conditioning Module（移植自官方仓库 Jatin-Sharma-11/GATP）。

官方组件（arch/node_stid_graph_conditioned.py）：
    GraphConvLayer          —— Graph WaveNet 风格 diffusion 图卷积：
                               x, x@A, x@A@A (order-2) 拼接后 Conv2d(1x1) 融合
    GraphConditioningBlock  —— gcn(x) + x (residual) -> LayerNorm -> MLP -> + -> LayerNorm

关键说明（与“gated preconditioning”称谓的差异）：
    GATP 官方实现里没有显式的 sigmoid gate。所谓“门控/条件化”是通过
    residual connection + LayerNorm 隐式实现的：图卷积输出作为“图结构修正项”，
    与原始 temporal 表示做加法残差，从而控制拓扑信息注入强度。
    本移植严格复用官方这一 residual 机制，不额外发明 gate。

为适配 PMME，做的最小改动：
    1. 输入从官方 [B, C, N, 1] 改为 PMME 的 rep [B, N, C]（C=128），
       模块内部统一用 [B, N, C]，einsum 沿 N 轴做邻域传播。
    2. adjacency 用 PMME 现有的归一化矩阵（get_normalized_adj 得到的 A_wave，
       即 D^-1/2 (A+I) D^-1/2），不再用官方的 doubletransition。
    3. 关闭 adaptive adjacency（官方 nodevec1/nodevec2 依赖固定 num_nodes，
       而 PMME 各城市 N 不同且共享同一套参数，故关闭）。
    4. 去掉官方 NodeSTID 的 backbone/regression，只保留 GraphConvLayer 与
       GraphConditioningBlock 两个核心组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Diffusion-style graph convolution（移植自 GATP 官方 GraphConvLayer）。

    官方实现以 [B, C, N, 1]，本移植以 [B, N, C]（PMRE rep 格式），
    语义等价：对每个通道沿 N 做邻域扩散（self + order-2 hops），再 1x1 卷积融合。
    """

    def __init__(self, c_in, c_out, support_len=1, order=2, dropout=0.1):
        super().__init__()
        self.order = order
        self.support_len = support_len
        total_c_in = c_in * (order * support_len + 1)
        # 官方用 Conv2d(1,1)；这里 [B,N,C] 输入用 Conv1d 沿通道融合等价
        self.fc = nn.Conv1d(total_c_in, c_out, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, supports):
        """
        Args:
            x        : [B, N, C]
            supports : list of [N, N] adjacency matrices（PMME 归一化后的 A_wave）
        Returns:
            [B, N, c_out]
        """
        # 转 [B, C, N] 便于 einsum
        x = x.permute(0, 2, 1)                 # [B, C, N]
        out = [x]                              # self
        for a in supports:
            a = a.to(x.device)
            # 一阶邻域传播
            x1 = torch.einsum('bcn,nm->bcm', x, a)      # [B, C, N]
            out.append(x1)
            # 高阶扩散
            for _ in range(2, self.order + 1):
                x2 = torch.einsum('bcn,nm->bcm', x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)              # [B, C*(order*support_len+1), N]
        h = self.fc(h)                         # [B, c_out, N]
        h = self.dropout(h)
        return h.permute(0, 2, 1)              # [B, N, c_out]


class GraphConditioningBlock(nn.Module):
    """
    GATP 官方 GraphConditioningBlock：图卷积 + residual + LayerNorm + MLP + residual。

    官方结构（保持不变，仅维度 [B,C,N,1] -> [B,N,C]）：
        residual = x
        x = gcn(x, supports)
        x = x + residual
        x = LayerNorm(x)
        residual2 = x
        x = mlp(x)
        x = x + residual2
        x = LayerNorm(x)
    """

    def __init__(self, hidden_dim, support_len=1, order=2, dropout=0.1):
        super().__init__()
        self.gcn = GraphConvLayer(hidden_dim, hidden_dim, support_len, order, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, supports):
        """
        Args:
            x        : [B, N, C]
            supports : list of [N, N]
        Returns:
            [B, N, C]
        """
        # graph conv + residual
        residual = x
        x = self.gcn(x, supports)
        x = x + residual
        x = self.norm(x)
        # MLP + residual
        residual2 = x
        x = self.mlp(x)
        x = x + residual2
        x = self.norm2(x)
        return x


class GraphCondition(nn.Module):
    """
    PMME 接口层：SOFTS rep [B, N, 128] -> Graph Conditioning -> H^topo [B, N, 128]。

    所有城市共享同一套参数，邻接矩阵 A 由调用方传入各自城市的归一化矩阵。
    """

    def __init__(self, hidden_dim=128, num_layers=2, order=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.blocks = nn.ModuleList([
            GraphConditioningBlock(hidden_dim, support_len=1, order=order, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, h_temp, adj):
        """
        Args:
            h_temp : [B, N, C]  SOFTS 输出 rep（H^temp）
            adj    : [N, N]     该城市归一化后的邻接矩阵（A_wave）
        Returns:
            h_topo : [B, N, C]  H^topo，维度与输入一致
        """
        supports = [adj]
        x = h_temp
        for block in self.blocks:
            x = block(x, supports)
        return x
