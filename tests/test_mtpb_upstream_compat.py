"""Exact tensor-shape regression for a known bug in official MTPB GWN."""
from types import SimpleNamespace
import torch

from tsjt_pgrad.mtpb.upstream_compat import repair_mtpb_gwnet_convs


def test_upstream_gwnet_conv1d_to_conv2d_keeps_weights_and_runs_4d():
    class GWN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_convs=torch.nn.ModuleList([
                torch.nn.Conv1d(3,4,kernel_size=(1,2))])
            self.residual_convs=torch.nn.ModuleList([
                torch.nn.Conv1d(3,3,kernel_size=(1,1))])
            self.skip_convs=torch.nn.ModuleList([
                torch.nn.Conv1d(3,5,kernel_size=(1,1))])
    model=torch.nn.Module()
    model.model_list=torch.nn.ModuleList([torch.nn.Identity(),GWN()])
    rep=SimpleNamespace(model=model)
    original=[c.weight.detach().clone() for name in
              ('gate_convs','residual_convs','skip_convs')
              for c in getattr(model.model_list[1],name)]
    assert repair_mtpb_gwnet_convs(rep)==3
    repaired=[c for name in ('gate_convs','residual_convs','skip_convs')
              for c in getattr(model.model_list[1],name)]
    assert all(isinstance(c,torch.nn.Conv2d) for c in repaired)
    assert all(torch.equal(a,c.weight) for a,c in zip(original,repaired))
    assert repaired[0](torch.randn(2,3,4,8)).shape==(2,4,4,7)
    assert repair_mtpb_gwnet_convs(rep)==0
