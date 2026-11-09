"""Fix official MTPB GWN's Conv1d-on-4D incompatibility, without changing weights.

reconstruction.BatchA_patch_gwnet declares nn.Conv1d with 2D kernels and
feeds [B,C,N,T] into those layers. Convert to Conv2d with identical tensors.
Apply the same compatibility repair to native and target-guided runs.
"""
from torch import nn


def repair_mtpb_gwnet_convs(native_rep):
    gwn = native_rep.model.model_list[1]
    count = 0
    for name in ('gate_convs', 'residual_convs', 'skip_convs'):
        layers = getattr(gwn, name)
        for idx, old in enumerate(layers):
            if not isinstance(old, nn.Conv1d):
                continue
            if len(old.kernel_size) != 2:
                raise ValueError(
                    f'Unexpected upstream {name}[{idx}] kernel: {old.kernel_size}')
            new = nn.Conv2d(
                old.in_channels, old.out_channels,
                kernel_size=old.kernel_size,
                stride=(1, old.stride[-1]),
                padding=(0, old.padding[-1]),
                dilation=(1, old.dilation[-1]),
                groups=old.groups, bias=old.bias is not None,
            ).to(device=old.weight.device, dtype=old.weight.dtype)
            new.load_state_dict(old.state_dict(), strict=True)
            layers[idx] = new
            count += 1
    return count
