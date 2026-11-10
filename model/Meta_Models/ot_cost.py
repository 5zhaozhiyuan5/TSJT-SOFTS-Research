"""
OT cost 构造 + 手写 Sinkhorn（显式 cost matrix）。

核心公式（v2）：
    C^{PM}_{ij}   = || [g(x_i^s), eta*y_i^s] - [g(x_j^t), eta*y_j^t] ||_1     （原 PMME pattern cost）
    C^{topo}_{ij} = 1 - cos(e_i^{role}, e_j^{role})                              （结构角色差异）
    G^{pattern}_{ij} = exp( - ||y_i^s - y_j^t||_2^2 / tau_p )                    （时间 pattern 门控）
    C^new = (C^{PM} / mean(C^{PM})) + lambda_topo * G^{pattern} * (C^{topo} / mean(C^{topo}))

然后 C^new 作为显式 cost matrix 送入熵正则 Sinkhorn：

    solve  P = argmin_{P 1 = alpha, P^T 1 = beta}  <P, C^new> + eps * <P, log P - 1>

全部可微分（对 alpha/beta/role embedding 与 C^{PM} 均可反向传播）。
"""

import torch


# ----------------------------------------------------------------------------
# cost 构造
# ----------------------------------------------------------------------------
def pattern_gate(y_src, y_tgt, tau_p):
    """
    时间 pattern 相似性门控矩阵 G^pattern ∈ [Ms, Mt]。

    参数：
        y_src : [Ms, pred]  源节点未来多步标签
        y_tgt : [Mt, pred]  目标节点未来多步标签
        tau_p : temperature
    返回：
        G : [Ms, Mt]，取值 (0, 1]，pattern 越像越接近 1。
    """
    # ||y_i - y_j||^2 = |y_i|^2 + |y_j|^2 - 2<y_i, y_j>
    sq_src = (y_src ** 2).sum(dim=1, keepdim=True)
    sq_tgt = (y_tgt ** 2).sum(dim=1, keepdim=True)
    d2 = sq_src + sq_tgt.t() - 2.0 * (y_src @ y_tgt.t())
    d2 = torch.clamp(d2, min=0.0)
    return torch.exp(-d2 / tau_p)


def pm_cost(x_src, x_tgt, p=1.0, chunk_src=512):
    """
    原 PMME 的 ground cost：C^{PM}_{ij} = ||x_i - x_j||_p^p（x 已含 rep 与缩放后的 label）。

    参数：
        x_src : [Ms, d_pm]  (=[rep, eta*y] 拼接后的 source 表示)
        x_tgt : [Mt, d_pm]
    返回：
        C : [Ms, Mt]

    实现：按 source 分块计算，避免 materialize [Ms, Mt, d_pm] 导致 OOM。
          （Ms=M*N_src、Mt=M*N_tgt 时可达 M^2*N_src*N_tgt*d，N=627、d=164、M=16 时约 21GiB）
    """
    Ms = x_src.shape[0]
    C = torch.empty((Ms, x_tgt.shape[0]), device=x_src.device, dtype=x_src.dtype)
    for s in range(0, Ms, chunk_src):
        diff = x_src[s:s + chunk_src, None, :] - x_tgt[None, :, :]  # [chunk, Mt, d_pm]
        if p == 1.0:
            C[s:s + chunk_src] = diff.abs().sum(dim=-1)             # L1
        else:
            C[s:s + chunk_src] = (diff ** 2).sum(dim=-1) ** (p / 2.0)
    return C


def topo_cost(E_src, E_tgt):
    """
    结构角色差异矩阵：C^{topo}_{ij} = 1 - cos(e_i, e_j)。

    参数：
        E_src : [Ms, d_role]  （已按 batch 展开到样本级，行序 b*N+n 与 rep/y 对齐）
        E_tgt : [Mt, d_role]
    返回：
        C : [Ms, Mt]
    """
    E_src = E_src / (E_src.norm(dim=1, keepdim=True) + 1e-8)
    E_tgt = E_tgt / (E_tgt.norm(dim=1, keepdim=True) + 1e-8)
    cos = E_src @ E_tgt.t()
    return 1.0 - cos


def build_ot_cost(x_src, x_tgt, E_src, E_tgt, y_src, y_tgt,
                  tau_p, lambda_topo):
    """
    构造最终 OT cost 矩阵 C^new。

    参数：
        x_src, x_tgt : [Ms, d_pm] / [Mt, d_pm]   pattern 表示（rep + eta*y）
        E_src, E_tgt : [Ms, d_role] / [Mt, d_role]  role embedding（样本级对齐）
        y_src, y_tgt : [Ms, pred] / [Mt, pred]   原始 label（用于 pattern gate，未缩放）
        tau_p, lambda_topo : 超参
    返回：
        C_new : [Ms, Mt]

    内存：只返回 C_new；C_pm/C_topo/G 均为临时量，随函数返回即释放，避免
          同时持有多个 [Ms, Mt] 大矩阵。
    """
    eps = 1e-8
    C_pm = pm_cost(x_src, x_tgt, p=1.0)                 # [Ms, Mt]
    C_pm = C_pm / (C_pm.mean() + eps)
    C_topo = topo_cost(E_src, E_tgt)                     # [Ms, Mt]
    C_topo = C_topo / (C_topo.mean() + eps)
    G = pattern_gate(y_src, y_tgt, tau_p)                # [Ms, Mt]
    return C_pm + lambda_topo * G * C_topo


# ----------------------------------------------------------------------------
# 手写熵正则 Sinkhorn（显式 cost matrix，可微分）
# ----------------------------------------------------------------------------
def sinkhorn_knopp(C, alpha, beta, eps, num_iters=100, tol=1e-6, log=False):
    """
    稳定 Sinkhorn（对数域，数值稳定）。

    求解熵正则 OT：
        min_{P}  <P, C> + eps * <P, log P - 1>
        s.t.     P 1 = alpha,   P^T 1 = beta

    参数：
        C     : [Ms, Mt]  显式 cost 矩阵
        alpha : [Ms]      源边际（归一化后求和=1）
        beta  : [Mt]      目标边际
        eps   : 熵正则系数（>0）
    返回：
        ot_cost（scalar，可微分）；若 log=True 额外返回 plan P 与对偶势 f,g。
    """
    return _sinkhorn_log(C, alpha, beta, eps, num_iters, tol, log)


def _sinkhorn_log(C, alpha, beta, eps, num_iters=50, tol=1e-6, log=False):
    """对数域 Sinkhorn（数值稳定）。返回 OT cost 与（可选）transport plan。

    内存关键点：通过自定义 autograd.Function 求解 plan P（forward 全程 no_grad），
    backward 只返回 ∂ot_cost/∂C = P。图中仅一个算子节点，不在 autograd 图中
    保存任何 [Ms,Mt] 中间量，避免 retain_graph 训练下的图内存累积（OOM 根因）。
    """
    return SinkhornKnopp.apply(C, alpha, beta, eps, num_iters, tol, log)


class SinkhornKnopp(torch.autograd.Function):
    """自定义 Sinkhorn-Knopp（熵正则 OT）算子：前向解最优 plan，反向给解析梯度 ∂loss/∂C = P。

    采用标准 Sinkhorn-Knopp 缩放（对数域），迭代收敛到严格满足双边际的 plan：
        P = exp((f_i + g_j - C_ij)/eps)，其中 f/g 使行和=alpha、列和=beta。
    收敛后 P 即对偶最优解，∂<P, C>/∂C = P 为严格解析梯度。
    """

    @staticmethod
    def forward(ctx, C, alpha, beta, eps, num_iters, tol, log):
        Ms, Mt = C.shape
        with torch.no_grad():
            log_alpha = torch.log(alpha + 1e-12)
            log_beta = torch.log(beta + 1e-12)
            logu = torch.zeros(Ms, device=C.device, dtype=C.dtype)
            logv = torch.zeros(Mt, device=C.device, dtype=C.dtype)
            K = (-C.detach()) / eps   # log kernel

            for _ in range(int(num_iters)):
                logu_prev = logu
                logu = log_alpha - torch.logsumexp(K + logv[None, :], dim=1)
                logv = log_beta - torch.logsumexp(K + logu[:, None], dim=0)
                if torch.abs(logu - logu_prev).max() < tol:
                    break

            P = torch.exp(logu[:, None] + logv[None, :] + K)   # 行和≈alpha、列和≈beta

        ot_cost = (P * C).sum()
        ctx.save_for_backward(P)
        return ot_cost

    @staticmethod
    def backward(ctx, grad_output):
        P = ctx.saved_tensors[0]
        # ∂ <P, C> / ∂C = P（最优 plan，对偶强对偶下为严格梯度）
        return grad_output * P, None, None, None, None, None, None


def _sinkhorn_log_legacy(C, alpha, beta, eps, num_iters=50, tol=1e-6, log=False):
    """（保留的纯 Python 实现，供对照/调试用。注意：此版会在 autograd 图中积累 [Ms,Mt] 中间量。）"""
    Ms, Mt = C.shape
    log_alpha = torch.log(alpha + 1e-12)
    log_beta = torch.log(beta + 1e-12)
    with torch.no_grad():
        f = torch.zeros(Ms, device=C.device, dtype=C.dtype)
        g = torch.zeros(Mt, device=C.device, dtype=C.dtype)
        Cno = C.detach()
        for _ in range(num_iters):
            logv_prev = g
            f = eps * (log_alpha - torch.logsumexp((-Cno / eps) + g[None, :], dim=1))
            g = eps * (log_beta - torch.logsumexp((-Cno / eps) + f[:, None], dim=0))
            if torch.abs(g - logv_prev).max() < tol:
                break
        logP = (f[:, None] + g[None, :] - Cno) / eps
        P = torch.exp(logP)
        row_sum = P.sum(dim=1, keepdim=True) + 1e-12
        P = P / row_sum * alpha[:, None]
    ot_cost = (P * C).sum()
    if log:
        return ot_cost, P.detach(), f, g
    return ot_cost
