import pytest
import torch

from model.Meta_Models.pattern_moe import _fixed_router_gate


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("uniform", [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        ("expert1", [1.0, 0.0, 0.0]),
        ("expert2", [0.0, 1.0, 0.0]),
        ("expert3", [0.0, 0.0, 1.0]),
    ],
)
def test_fixed_router_gate_values_and_shape(mode, expected):
    gate = _fixed_router_gate(
        mode,
        source_count=5,
        target_count=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert gate.shape == (5, 7, 3)
    expected_tensor = torch.tensor(expected, dtype=torch.float32).view(1, 1, 3)
    assert torch.allclose(gate, expected_tensor.expand_as(gate))
    assert torch.allclose(gate.sum(dim=-1), torch.ones(5, 7))


def test_fixed_router_gate_rejects_learned_mode():
    with pytest.raises(ValueError):
        _fixed_router_gate(
            "learned",
            source_count=2,
            target_count=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
