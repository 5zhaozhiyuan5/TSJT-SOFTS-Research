"""Read-only PMME Stage-I gradient decomposition; never calls optimizer.step.

Use on the same training batch and parameter list as PMME Stage-I. The target
loss must contain ONLY the authorized ~3-day target training labels.
"""
import torch


def _flatten(loss, params, retain_graph=True):
    if loss is None:
        return torch.zeros(sum(p.numel() for p in params), device=params[0].device)
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                      for p, g in zip(params, grads)])


def _cos(a, b, eps=1e-12):
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom <= eps:
        return float('nan')
    return float((torch.dot(a, b) / denom).item())


def audit_stage1_gradients(source_loss, alignment_loss, target_loss, params):
    """Return CPU scalar diagnostics without changing parameter.grad.

    source_loss is the sum of source-city prediction losses; alignment_loss
    is the sum of the three OT losses (or zero/None when disabled).
    """
    params = [p for p in params if p.requires_grad]
    if not params:
        raise ValueError('No trainable parameters')
    gs = _flatten(source_loss, params)
    ga = _flatten(alignment_loss, params)
    gt = _flatten(target_loss, params)
    combined = gs + ga
    return {
        'source_target_cos': _cos(gs, gt),
        'ot_target_cos': _cos(ga, gt),
        'combined_target_cos': _cos(combined, gt),
        'source_norm': float(gs.norm().item()),
        'ot_norm': float(ga.norm().item()),
        'target_norm': float(gt.norm().item()),
        'combined_norm': float(combined.norm().item()),
        'source_ot_dot': float(torch.dot(gs, ga).item()),
    }
