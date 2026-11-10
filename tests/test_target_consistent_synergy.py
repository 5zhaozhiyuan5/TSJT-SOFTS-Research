"""Synthetic falsification tests for target-consistent source synergy.

No real-city MAE/RMSE claims. Compare lambda=0 against lambda=1 with identical
PGrad references, disabled EMA, and zero conflict/redundancy coefficients.
"""
import pytest
import torch
from torch import nn
from tsjt_pgrad.pgrad.source_interaction import combine_source_interaction


def _case():
    p = tuple(nn.Linear(4, 1, bias=False).parameters())
    # First two cities share a direction opposed to target; third aligns.
    source = [
        (torch.tensor([[-1., 0., 0., 0.]]),),
        (torch.tensor([[-1., 0., 0., 0.]]),),
        (torch.tensor([[1., 0., 0., 0.]]),),
    ]
    target = (torch.tensor([[1., 0., 0., 0.]]),)
    principal = [(torch.tensor([[0., 1., 0., 0.]]),)]
    return p, source, target, principal


def test_conflicting_pair_loses_positive_synergy():
    p, s, t, ref = _case()
    opts = dict(target_strength=0., conflict_strength=0.,
                redundancy_strength=0., redundancy_threshold=1.,
                uniform_mix=0., weight_ema=0., temperature=.35)
    _, baseline = combine_source_interaction(p, s, t, ref,
        target_consistent_synergy=0., **opts)
    _, gated = combine_source_interaction(p, s, t, ref,
        target_consistent_synergy=1., **opts)
    assert baseline['source_positive_synergy'][0] == pytest.approx(.5)
    assert gated['source_positive_synergy'][0] == pytest.approx(0.)
    assert gated['source_positive_synergy'][1] == pytest.approx(0.)
    assert gated['source_weights'][2] > baseline['source_weights'][2]
    assert sum(gated['source_weights']) == pytest.approx(1.)


def test_default_is_bitwise_equal_to_explicit_disabled_gate():
    p, s, t, ref = _case()
    a, am = combine_source_interaction(p, s, t, ref, weight_ema=0.)
    b, bm = combine_source_interaction(p, s, t, ref,
        target_consistent_synergy=0., weight_ema=0.)
    torch.testing.assert_close(a[0], b[0], rtol=0., atol=0.)
    assert am['source_weights'] == bm['source_weights']


def test_invalid_gate_rejected():
    p, s, t, ref = _case()
    for value in (-.01, 1.01):
        with pytest.raises(ValueError, match='target_consistent_synergy'):
            combine_source_interaction(p, s, t, ref,
                target_consistent_synergy=value)
