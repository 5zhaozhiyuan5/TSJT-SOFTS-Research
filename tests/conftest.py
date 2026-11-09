"""Standalone core tests: the original tsjt_c1/tsjt_paper repositories are not attached."""
from __future__ import annotations
import sys
import types
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src'

def fakepkg(name,path=None):
    if name not in sys.modules:
        m=types.ModuleType(name)
        m.__path__=([str(path)] if path else [])
        sys.modules[name]=m

fakepkg('tsjt_pgrad',SRC/'tsjt_pgrad')
fakepkg('tsjt_pgrad.pgrad',SRC/'tsjt_pgrad'/'pgrad')
fakepkg('tsjt_pgrad.training',SRC/'tsjt_pgrad'/'training')
fakepkg('tsjt_pgrad.models',SRC/'tsjt_pgrad'/'models')

# Only for testing the composition logic without the unavailable real TSJT backbone.
fakepkg('tsjt_c1')
fakepkg('tsjt_c1.training')
engine=types.ModuleType('tsjt_c1.training.engine')
sys.modules['tsjt_c1.training.engine']=engine
sys.modules['tsjt_c1.training'].engine=engine
stub=types.ModuleType('tsjt_c1.training.tsb')
def filter_source_gradient(t,s):
    dot=(t*s).sum()
    if dot < 0 and t.square().sum()>0:
        return s-dot/t.square().sum()*t,{'dot':dot}
    return s,{'dot':dot}
stub.filter_source_gradient=filter_source_gradient
sys.modules['tsjt_c1.training.tsb']=stub

class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor([0.,0.,0.]))
    def forward(self,x,adjacency,horizon=None):
        return (x*self.weight).sum(dim=-1,keepdim=True)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        from tsjt_pgrad.pgrad.rotation import AngleController
        self.backbone=ToyBackbone()
        self.angle_controller=AngleController()
    def forward(self,x,adjacency,horizon=None):
        return self.backbone(x,adjacency,horizon=horizon)

# Do not override the real Graph WaveNet TSJTPGrad module in sys.modules.
# ToyModel remains available to legacy tests through this conftest namespace.
