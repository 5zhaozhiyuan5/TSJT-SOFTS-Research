import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_initial_bandwidth_full(xs: torch.Tensor, xt: torch.Tensor, device: torch.device = None):
    """
    Compute initial RBF kernel bandwidth (median heuristic) for two entire domains 
    without subsampling.

    Args:
        xs (torch.Tensor): Source domain features, shape (N_s, d).
        xt (torch.Tensor): Target domain features, shape (N_t, d).
        device (torch.device, optional): Device to perform computation on.
    
    Returns:
        sigma_median (float): (median_source_offdiag + median_target_offdiag) / 2
    """
    if device is None:
        device = xs.device

    # Move data to device
    xs = xs.to(device)
    xt = xt.to(device)

    # Efficient pairwise Euclidean distances
    def pairwise_distances(A: torch.Tensor):
        sq = (A * A).sum(dim=1, keepdim=True)
        D2 = sq + sq.t() - 2 * (A @ A.t())
        D2 = torch.clamp(D2, min=0.0)
        return torch.sqrt(D2)

    d_s = pairwise_distances(xs)  # (N_s, N_s)
    d_t = pairwise_distances(xt)  # (N_t, N_t)

    # Exclude the zero-distance diagonal entries
    mask_s = ~torch.eye(d_s.size(0), dtype=torch.bool, device=device)
    mask_t = ~torch.eye(d_t.size(0), dtype=torch.bool, device=device)
    offdiag_s = d_s[mask_s]
    offdiag_t = d_t[mask_t]

    # Median heuristic
    median_s = offdiag_s.median().item()
    median_t = offdiag_t.median().item()

    # Final σ
    sigma_median = (median_s + median_t) / 2

    return sigma_median


def rbf_kernel(x, y=None, sigma=1.0):
    """
    Compute RBF kernel matrix between x and y (if provided), otherwise x with itself.
    Args:
        x: (n, d)
        y: (m, d) or None
        sigma: bandwidth
    Returns:
        Kernel matrix of shape (n, m) or (n, n) if y is None.
    """
    if y is None:
        y = x
    # Squared norms
    x_norm = (x * x).sum(dim=1, keepdim=True)  # (n,1)
    y_norm = (y * y).sum(dim=1, keepdim=True)  # (m,1)
    # Compute squared distances
    D2 = x_norm + y_norm.t() - 2 * (x @ y.t())
    D2 = torch.clamp(D2, min=0.0)
    return torch.exp(-D2 / (2 * sigma ** 2))


class CMMDLoss(nn.Module):
    """
    Conditional Maximum Mean Discrepancy (CMMD) Loss for regression DA.
    Approximates integral over y of ||μ_S(x|y) - μ_T(x|y)||_H^2.
    """
    def __init__(self, sigma_x=1.0, sigma_y=1.0):
        super().__init__()
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def forward(self, xs, ys, xt, yt):
        """
        xs, ys: source features and labels, shapes (ns, d), (ns,)
        xt, yt: target features and labels, shapes (nt, d), (nt,)
        Returns: scalar CMMD^2 loss
        """
        # Compute kernels
        Kx_ss = rbf_kernel(xs, None, self.sigma_x)        # (ns, ns)
        Ky_ss = rbf_kernel(ys, None, self.sigma_y)  # (ns, ns)
        Kx_tt = rbf_kernel(xt, None, self.sigma_x)        # (nt, nt)
        Ky_tt = rbf_kernel(yt, None, self.sigma_y)  # (nt, nt)
        Kx_st = rbf_kernel(xs, xt, self.sigma_x)          # (ns, nt)
        Ky_st = rbf_kernel(ys, yt, self.sigma_y)  # (ns, nt)

        # Weighted MMD components
        m_ss = (Kx_ss * Ky_ss).mean()
        m_tt = (Kx_tt * Ky_tt).mean()
        m_st = (Kx_st * Ky_st).mean()

        # CMMD^2 = E[kx*ky]_SS + E[kx*ky]_TT - 2 E[kx*ky]_ST
        loss = m_ss + m_tt - 2 * m_st
        return loss

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss with RBF kernel.
    Computes squared MMD between two distributions p and q given samples xs, xt.
    """
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """
        xs: source samples, shape (ns, d)
        xt: target samples, shape (nt, d)
        Returns: scalar MMD^2 loss
        """
        K_xx = rbf_kernel(xs, None, self.sigma)   # (ns, ns)
        K_tt = rbf_kernel(xt, None, self.sigma)   # (nt, nt)
        K_xt = rbf_kernel(xs, xt,   self.sigma)   # (ns, nt)

        m_xx = K_xx.mean()    # E_{x,x'∼p}[k(x,x')]
        m_tt = K_tt.mean()    # E_{y,y'∼q}[k(y,y')]
        m_xt = K_xt.mean()    # E_{x∼p,y∼q}[k(x,y)]

        loss = m_xx + m_tt - 2 * m_xt
        return loss


class CORALLoss(nn.Module):
    """
    Correlation Alignment (CORAL) Loss.
    Aligns second-order statistics (covariances) of source and target features.
    """
    def __init__(self):
        super().__init__()

    def forward(self, xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """
        xs: source features, shape (ns, d)
        xt: target features, shape (nt, d)
        Returns: scalar CORAL loss
        """
        # 1. Center the features
        xs_centered = xs - xs.mean(dim=0, keepdim=True)
        xt_centered = xt - xt.mean(dim=0, keepdim=True)

        # 2. Compute covariance matrices
        # Using unbiased estimator (divide by N-1)
        ns = xs.size(0)
        nt = xt.size(0)
        # (d × ns) @ (ns × d) -> (d, d)
        cov_xs = xs_centered.t().matmul(xs_centered) / (ns - 1)
        cov_xt = xt_centered.t().matmul(xt_centered) / (nt - 1)

        # 3. Compute Frobenius norm squared between covariances
        d = xs.size(1)
        diff = cov_xs - cov_xt
        loss = torch.norm(diff, p='fro')**2 / (4 * d * d)

        return loss
        
