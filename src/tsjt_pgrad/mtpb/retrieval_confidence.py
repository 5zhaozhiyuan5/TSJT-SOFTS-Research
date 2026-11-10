"""Read official MTPB pattern retrieval uncertainty without changing its forward.

Official PatternEncoder_patternkeyv2 retrieves over K prototypes separately at
1/3/6/12/24-hour scales. A pre-hook reads its actual input X and recomputes
the same query logits; no target test labels or extra topology are used.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
import torch
import torch.nn.functional as F


def retrieval_confidence(encoder, x):
    """Return normalized retrieval certainty in [0,1] and per-scale values."""
    if x.ndim != 4:
        raise ValueError('MTPB pattern input must be [B,N,L,D]')
    values = {}
    with torch.no_grad():
        for hour in encoder.hour_list:
            h = int(hour)
            qry = encoder.Qnet_dict[hour](x.detach())
            b,n,l,d = qry.shape
            if l % h:
                raise ValueError(f'pattern scale {h} does not divide length {l}')
            qry = qry.reshape(b,n,l//h,d*h)
            logits = torch.einsum('dk,bnld->bnlk', encoder.qry_mat_dict[hour], qry)
            k = logits.shape[-1]
            if k < 2:
                raise ValueError('pattern bank must have >=2 prototypes per scale')
            p = F.softmax(logits, dim=-1)
            entropy = -(p * p.clamp_min(1e-30).log()).sum(-1).mean()
            confidence = (1.0 - entropy / math.log(k)).clamp(0,1)
            if not torch.isfinite(confidence):
                raise FloatingPointError('nonfinite pattern retrieval confidence')
            values[hour] = float(confidence)
    if not values:
        raise ValueError('empty MTPB pattern scale list')
    return sum(values.values()) / len(values), values


@contextmanager
def capture_retrieval_confidence(model):
    """Capture one encoder call during the existing target gradient forward."""
    if not hasattr(model, 'model_list') or len(model.model_list) < 1:
        raise ValueError('MTPB model_list[0] pattern encoder is required')
    encoder = model.model_list[0]
    if not all(hasattr(encoder, key) for key in ('hour_list','Qnet_dict','qry_mat_dict')):
        raise ValueError('expected official MTPB PatternEncoder_patternkeyv2')
    record = []
    def hook(module, inputs):
        if len(inputs) != 1:
            raise ValueError('unexpected MTPB pattern encoder inputs')
        record.append(retrieval_confidence(module, inputs[0]))
    handle = encoder.register_forward_pre_hook(hook)
    try:
        yield record
    finally:
        handle.remove()
