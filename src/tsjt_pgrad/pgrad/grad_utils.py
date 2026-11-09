"""Safe shared-backbone gradient geometry and centered trajectory PCA."""
from __future__ import annotations
from collections.abc import Sequence
import torch

Gradients = Sequence[torch.Tensor | None]

def flatten_gradients(parameters: Sequence[torch.Tensor], gradients: Gradients, *, zeros_for_none: bool = True) -> torch.Tensor:
    if len(parameters) != len(gradients):
        raise ValueError(f"gradient/parameter length mismatch: {len(gradients)} vs {len(parameters)}")
    parts = []
    for p, g in zip(parameters, gradients):
        if g is None:
            if not zeros_for_none:
                continue
            parts.append(torch.zeros_like(p).reshape(-1))
        else:
            if g.shape != p.shape:
                raise ValueError(f"gradient shape {g.shape} != parameter shape {p.shape}")
            parts.append(g.reshape(-1))
    if not parts:
        raise ValueError("empty gradient list")
    return torch.cat(parts).unsqueeze(0)

def unflatten_gradients(parameters: Sequence[torch.Tensor], flat: torch.Tensor) -> tuple[torch.Tensor, ...]:
    v = flat.reshape(-1)
    if v.numel() != sum(p.numel() for p in parameters):
        raise ValueError("incorrect flattened gradient length")
    off = 0
    result = []
    for p in parameters:
        result.append(v[off:off+p.numel()].reshape(p.shape))
        off += p.numel()
    return tuple(result)

def global_norm(gradients: Gradients) -> float:
    return sum(float(g.detach().square().sum()) for g in gradients if g is not None)**0.5

@torch.no_grad()
def principal_directions_gram(trajectory: Sequence[torch.Tensor], num_components: int, *, eps: float=1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Centered PCA on trajectory *displacements* (not raw parameters), using m x m Gram.

    Returns rows of orthonormal right singular vectors and their eigenvalues.
    Empty-rank inputs return [0,D] and [0], rather than fabricated directions.
    """
    if not trajectory or num_components < 1:
        raise ValueError("trajectory and num_components must be nonempty")
    matrix = torch.cat([x.reshape(1,-1).to(torch.float64) for x in trajectory], dim=0)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    # Input rows are displacements so centering does not erase tiny changes in huge weights.
    gram = centered @ centered.t()
    vals, vectors = torch.linalg.eigh(gram)
    vals = vals.clamp_min(0)
    order = torch.argsort(vals, descending=True)
    maxval = vals[order[0]]
    valid = int((vals[order] > torch.maximum(maxval * 1e-9, torch.tensor(eps, device=vals.device))).sum())
    k = min(int(num_components), valid)
    if k == 0:
        return centered.new_zeros((0, centered.shape[1])).to(trajectory[0].dtype), vals.new_zeros((0,)).to(trajectory[0].dtype)
    u = vectors[:, order[:k]]
    directions = (u.T @ centered) / torch.sqrt(vals[order[:k]]).unsqueeze(1)
    directions = torch.nn.functional.normalize(directions, dim=1)
    return directions.to(trajectory[0].dtype), vals[order[:k]].to(trajectory[0].dtype)
