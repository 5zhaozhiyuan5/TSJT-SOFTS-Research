"""
Topology-aware Pattern Matching 拓扑模块（v2：structural role + 显式 topology cost）。

本文件提供：
1. compute_role_descriptor   —— 由原始 matrix.npy 构造 multi-scale structural role 描述子（纯 numpy，可缓存）
2. RoleEncoder               —— 可选的共享参数 MLP（仅供消融；主模型使用固定描述子）
3. role_edge_index           —— 由原始矩阵恢复有效邻接（供 role descriptor 的 k-hop 统计使用）

设计原则（与 v1 不同）：
- 不再用 GraphSAGE + 邻接重构预训练（那学到的是"谁相邻"，不是跨城市可比的"结构角色"）。
- role 描述子只描述节点的拓扑角色，不标识节点 ID。
- 所有城市必须用同一套特征定义和归一化，保证 Source/Target 可比。
- 主模型直接使用固定描述子，而不训练 RoleEncoder。这样可避免 OT 目标把一个无监督
  RoleEncoder 压缩为常量、从而丢失拓扑信息的退化解。
- 描述子只提供节点级结构角色；它由 PMME.py/PMME_2ndStage.py 分别接入
  DVGP、Sinkhorn OT 和残差记忆，绝不修改 DVGP 的质量向量归一化规则。

构图规则（重要）：
- 原 matrix.npy 中 metr-la/pems-bay 是有向图（A_ij != A_ji 大量存在），chengdu/shenzhen 基本对称。
- 权重最小 >= 0.1，不存在微小非零值，因此 A > 0 不会造成近似全连接（density 最大仅 0.04）。
- 对于 k-hop 统计，我们按**无向化**处理：A_und = (A > 0) | (A > 0).T，即"存在任一向连接即视为一条无向边"，
  这与道路网络的物理意义（路段双向连通）一致，也避免把有向权重不对称误当成结构差异。
- 对有向图，保留 in/out degree 与 in/out strength；对局部团簇和 k-hop 统计，
  使用无向化图以描述道路在物理网络中的结构角色。

feat_dim（结构描述子维度）= 17（见 F_DIM 注释）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 构图工具
# ----------------------------------------------------------------------------
def to_undirected_bin(A):
    """原始邻接 -> 无向二值邻接：任一向存在连接即视为无向边。"""
    A_bin = (np.asarray(A) > 0).astype(np.float32)
    A_bin = ((A_bin + A_bin.T) > 0).astype(np.float32)
    # matrix.npy 中不应把自环当作道路结构；显式移除以避免污染 degree/k-core。
    np.fill_diagonal(A_bin, 0.0)
    return A_bin


def role_edge_index(A, device=None):
    """
    由原始矩阵得到供 k-hop 统计 / 可选 GNN 使用的无向二值 edge_index。
    这里采用无向化：任一向连接即视为边，去掉自环。
    """
    A_u = to_undirected_bin(A)
    np.fill_diagonal(A_u, 0.0)
    src, dst = np.nonzero(A_u)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    if device is not None:
        edge_index = edge_index.to(device)
    return edge_index


def _zscore_features(feats):
    """逐特征（按城市独立）z-score 归一化。"""
    mu = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-8
    return (feats - mu) / std


# ----------------------------------------------------------------------------
# structural role descriptor
# ----------------------------------------------------------------------------
def page_rank(Abin, alpha=0.85, max_iter=100, tol=1e-8):
    N = Abin.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.float32)
    out_deg = Abin.sum(axis=1)
    dangling = out_deg == 0
    out_deg[dangling] = 1.0
    M = Abin / out_deg[:, None]
    # Preserve probability mass at isolated sensors instead of silently
    # dropping it.  This is the standard dangling-node PageRank treatment.
    M[dangling] = 1.0 / N
    pr = np.full(N, 1.0 / N, dtype=np.float64)
    for _ in range(max_iter):
        new_pr = (1 - alpha) / N + alpha * (M.T @ pr)
        if np.abs(new_pr - pr).sum() < tol:
            pr = new_pr
            break
        pr = new_pr
    return pr.astype(np.float32)


def clustering_coefficient(A):
    """局部聚类系数（无向二值图）。"""
    N = A.shape[0]
    A2 = (A @ A) * A
    triangles = A2.sum(axis=1)
    k = A.sum(axis=1)
    denom = k * (k - 1)
    cc = np.zeros(N, dtype=np.float32)
    mask = denom > 0
    cc[mask] = triangles[mask] / denom[mask]
    return cc


def k_core(A):
    """k-core decomposition（Batagelj-Zaversnik），返回每个节点的 core number。"""
    N = A.shape[0]
    deg = A.sum(axis=1).astype(np.int64)
    neighs = [np.where(A[i] > 0)[0].tolist() for i in range(N)]
    core = np.zeros(N, dtype=np.int64)
    max_deg = int(deg.max()) if N > 0 else 0
    verts = [[] for _ in range(max_deg + 1)]
    for i in range(N):
        verts[int(deg[i])].append(i)
    bin_pos = [0] * (max_deg + 1)
    cnt = 0
    for d in range(max_deg + 1):
        bin_pos[d] = cnt
        cnt += len(verts[d])
    pos = [0] * N
    order = [0] * N
    for d in range(max_deg + 1):
        for idx, v in enumerate(verts[d]):
            pos[v] = bin_pos[d] + idx
            order[pos[v]] = v
    deg_cur = deg.copy()
    for i in range(N):
        v = order[i]
        core[v] = deg_cur[v]
        for u in neighs[v]:
            if deg_cur[u] > deg_cur[v]:
                du = deg_cur[u]
                pu = pos[u]
                pw = bin_pos[du]
                w = order[pw]
                if u != w:
                    pos[u], pos[w] = pw, pu
                    order[pu], order[pw] = w, u
                bin_pos[du] += 1
                deg_cur[u] -= 1
    return core.astype(np.float32)


def compute_role_descriptor(A):
    """
    由原始邻接矩阵 A 构造 multi-scale structural role 描述子 [N, F_DIM]。

    特征清单（F_DIM = 17）：
      0  undirected_degree          （无向二值图）
      1  in_degree                  （原始有向二值图）
      2  out_degree                 （原始有向二值图）
      3  in_strength                （原始有向加权图）
      4  out_strength               （原始有向加权图）
      5  pagerank                   （无向二值图）
      6  clustering                 （无向二值图）
      7  kcore                      （无向二值图）
      8  neighbor_avg_degree        （无向二值图）
      9  2hop_degree_mean           （2-hop 邻居的度均值）
      10 2hop_degree_std            （2-hop 邻居的度标准差）
      11 2hop_degree_max            （2-hop 邻居的度最大）
      12 2hop_neighborhood_size     （2-hop 内不重复节点数）
      13 3hop_neighborhood_size     （3-hop 内不重复节点数）
      14 egonet_size                （1-hop 自我网络节点数 = degree+1）
      15 egonet_density             （1-hop 自我网络内边密度）
      16 egonet_triangles           （1-hop 自我网络三角形数）

    跨城市可比性：最后逐特征 z-score 归一化（在城市内部）。
    """
    A = np.asarray(A, dtype=np.float32).copy()
    N = A.shape[0]
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"adjacency matrix must be square, got {A.shape}")
    # Negative similarities have no road-connectivity interpretation here.
    A = np.maximum(A, 0.0)
    np.fill_diagonal(A, 0.0)
    A_dir = (A > 0).astype(np.float32)
    A_u = to_undirected_bin(A)

    degree = A_u.sum(axis=1)
    in_degree = A_dir.sum(axis=0)
    out_degree = A_dir.sum(axis=1)
    in_strength = A.sum(axis=0)
    out_strength = A.sum(axis=1)
    pr = page_rank(A_u)
    cc = clustering_coefficient(A_u)
    kc = k_core(A_u)

    eps = 1e-8
    neighbor_avg_degree = (A_u @ degree) / np.maximum(degree, eps)

    # k-hop 邻居集合
    # A_u 是无向二值邻接，A_u^k 的 (i,j) 入口 >0 表示 i 到 j 存在 <=k 步路径
    A2 = (np.matmul(A_u, A_u) > 0).astype(np.float32)          # 2-hop 可达（含自身/邻居）
    A3 = (np.matmul(A2, A_u) > 0).astype(np.float32)           # 3-hop 可达

    # 2-hop 邻居（排除自身与直接邻居）：A2 \ (A_u ∪ I)
    I = np.eye(N, dtype=np.float32)
    A2_new = ((A2 > 0) & ~((A_u > 0) | (I > 0))).astype(np.float32)
    # 3-hop 邻居（排除 <=2 hop 已覆盖）
    A3_new = ((A3 > 0) & ~((A2 > 0) | (I > 0))).astype(np.float32)

    two_hop_deg_mean = np.zeros(N, dtype=np.float32)
    two_hop_deg_std = np.zeros(N, dtype=np.float32)
    two_hop_deg_max = np.zeros(N, dtype=np.float32)
    two_hop_nsize = np.zeros(N, dtype=np.float32)
    three_hop_nsize = np.zeros(N, dtype=np.float32)
    for i in range(N):
        nei2 = np.where(A2_new[i] > 0)[0]
        if nei2.size > 0:
            d2 = degree[nei2]
            two_hop_deg_mean[i] = d2.mean()
            two_hop_deg_std[i] = d2.std()
            two_hop_deg_max[i] = d2.max()
            two_hop_nsize[i] = nei2.size
        nei3 = np.where(A3_new[i] > 0)[0]
        three_hop_nsize[i] = nei3.size

    # egonet（1-hop 自我网络）统计
    egonet_size = degree + 1.0
    egonet_triangles = ((A_u @ A_u) * A_u).sum(axis=1) / 2.0  # 邻域内三角形数
    egonet_density = np.zeros(N, dtype=np.float32)
    for i in range(N):
        k = int(degree[i])
        if k > 1:
            # 本节点 + 邻居的子图边数（不含自环）
            nb = np.where(A_u[i] > 0)[0]
            if nb.size > 0:
                sub = A_u[np.ix_(nb, nb)]
                edges_in_egonet = sub.sum() / 2.0
                egonet_density[i] = 2.0 * edges_in_egonet / (k * (k - 1))

    F_DIM = 17
    feats = np.stack([
        degree,
        in_degree,
        out_degree,
        in_strength,
        out_strength,
        pr,
        cc,
        kc,
        neighbor_avg_degree,
        two_hop_deg_mean,
        two_hop_deg_std,
        two_hop_deg_max,
        two_hop_nsize,
        three_hop_nsize,
        egonet_size,
        egonet_density,
        egonet_triangles,
    ], axis=1).astype(np.float32)

    return _zscore_features(feats)


F_DIM = 17


# ----------------------------------------------------------------------------
# RoleEncoder：共享 MLP，跨城市可比
# ----------------------------------------------------------------------------
class RoleEncoder(nn.Module):
    """
    把 structural role 描述子 [N, F_DIM] 映射为 E_role [N, d_role]（L2 归一化）。

    共享参数（所有城市用同一个 encoder），保证 Source/Target role 在同一个语义空间可直接比较。
    采用 GraphWave 风格的思路：用 spectral / 结构描述子替代 node id，用一个小 MLP 压缩到低维。
    """

    def __init__(self, feat_dim=F_DIM, d_role=32, hidden=128):
        super(RoleEncoder, self).__init__()
        self.feat_dim = feat_dim
        self.d_role = d_role
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, d_role),
        )

    def forward(self, X_desc):
        """
        X_desc : [N, F_DIM]
        返回 E_role : [N, d_role]，已 L2 归一化。
        """
        out = self.mlp(X_desc)
        return F.normalize(out, p=2, dim=-1, eps=1e-8)
