"""Graph WaveNet traffic forecaster for a *shared*, variable-node-count TSJT model.

Based on Wu et al., Graph WaveNet (IJCAI 2019): gated dilated temporal
convolutions, skip/residual connections and multi-order diffusion graph conv.
Reference implementation: https://github.com/nnzhan/Graph-WaveNet

Cross-city adaptation (NOT in the original paper): derive adaptive adjacency
from each batch's *historical node signals* using shared query/key networks,
rather than city-specific N x embedding tables. No POI, map or test labels.
Optional supplied adjacency is a conventional backbone input, not an innovation.
All four cities use exactly the same trainable weights even for N=207/524/627/325.
Six 10-min observations are aligned to twelve 5-min observations using history
only, and horizons 6 read off +10,...,+60min of the canonical 12 outputs.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
from torch.nn import functional as F


def _align_history(x: torch.Tensor, canonical_steps: int) -> torch.Tensor:
    steps = x.shape[1]
    if steps == canonical_steps:
        return x
    ratio = canonical_steps / steps
    loc = ((torch.arange(canonical_steps, device=x.device, dtype=x.dtype)
           - (canonical_steps-1)) / ratio + (steps-1)).clamp(0, steps-1)
    left = loc.floor().long()
    right = (left+1).clamp(max=steps-1)
    w = (loc-left).view(1, canonical_steps, 1, 1)
    return x.index_select(1,left)*(1-w) + x.index_select(1,right)*w


class _DiffusionGCN(nn.Module):
    """Graph WaveNet's K-hop message passing followed by a 1x1 projection."""
    def __init__(self, channels: int, out_channels: int, order: int, max_supports: int, dropout: float):
        super().__init__()
        self.order = order
        self.project = nn.Conv2d((1 + order*max_supports)*channels, out_channels, kernel_size=1)
        self.dropout = dropout
        self.max_supports = max_supports

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        if len(supports) != self.max_supports:
            raise ValueError(f'expected {self.max_supports} graph supports, got {len(supports)}')
        outputs = [x]
        for adj in supports:
            h = x
            for _ in range(self.order):
                h = torch.einsum('bcnt,nm->bcmt',h,adj)
                outputs.append(h)
        return F.dropout(self.project(torch.cat(outputs,dim=1)), p=self.dropout, training=self.training)


class GraphWaveNetBackbone(nn.Module):
    """Shared Graph WaveNet adapter: [B,T,N,F] -> [B,H,N,1]."""
    def __init__(self, *, input_dim: int = 3, canonical_steps: int = 12,
                 supported_horizons: Sequence[int] = (6,12),
                 residual_channels: int = 32, dilation_channels: int = 32,
                 skip_channels: int = 64, end_channels: int = 128,
                 blocks: int = 2, layers: int = 2, kernel_size: int = 2,
                 diffusion_order: int = 2, dropout: float = 0.1,
                 use_reverse_support: bool = True,
                 use_knowledge_graph: bool = False, adaptive_dim: int = 16):
        super().__init__()
        self.input_dim = int(input_dim)
        self.canonical_steps = int(canonical_steps)
        self.supported_horizons = tuple(int(h) for h in supported_horizons)
        self.default_horizon = self.canonical_steps
        if self.canonical_steps < 1 or len(set(self.supported_horizons)) != len(self.supported_horizons):
            raise ValueError('invalid history/horizon resolution')
        if not self.supported_horizons or any(h<=0 or self.canonical_steps%h for h in self.supported_horizons):
            raise ValueError('horizons must divide canonical_steps')
        if min(residual_channels,dilation_channels,skip_channels,end_channels,adaptive_dim) <= 0:
            raise ValueError('channel sizes must be positive')
        if self.input_dim < 1 or blocks < 1 or layers < 1 or kernel_size < 2 or diffusion_order < 1:
            raise ValueError('invalid network dimensions')
        self.use_reverse_support = bool(use_reverse_support)
        self.use_knowledge_graph = bool(use_knowledge_graph)
        self.adaptive_dim = adaptive_dim
        # Original GWN uses N-specific nodevec1/nodevec2, which cannot be shared
        # across different city sizes. Shared projections of historical signals
        # produce one N x N matrix independent of any city-specific embeddings.
        self.node_q = nn.Linear(4, adaptive_dim)
        self.node_k = nn.Linear(4, adaptive_dim)
        self.start_conv = nn.Conv2d(self.input_dim,residual_channels,kernel_size=1)
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        supports_count = (2 if self.use_reverse_support else 1) + (2 if self.use_knowledge_graph and self.use_reverse_support else 1 if self.use_knowledge_graph else 0)
        self.receptive_field = 1
        for _ in range(blocks):
            dilation = 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dilation)))
                self.gate_convs.append(nn.Conv2d(residual_channels,dilation_channels,(1,kernel_size),dilation=(1,dilation)))
                self.skip_convs.append(nn.Conv2d(residual_channels,skip_channels,1))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.gconv.append(_DiffusionGCN(dilation_channels,residual_channels,diffusion_order,supports_count,dropout))
                self.receptive_field += (kernel_size-1)*dilation
                dilation *= 2
        self.end_conv_1 = nn.Conv2d(skip_channels,end_channels,1)
        self.end_conv_2 = nn.Conv2d(end_channels,self.canonical_steps,1)

    @staticmethod
    def _stochastic_adjacency(a: torch.Tensor) -> torch.Tensor:
        a = a.clamp_min(0.)
        eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
        a = a + eye  # safe even for supplied all-zero adjacency
        return a / a.sum(-1,keepdim=True).clamp_min(1e-8)

    def _supports(self, historical: torch.Tensor, adjacency: torch.Tensor | None) -> list[torch.Tensor]:
        speed = historical[...,0]  # [B,T,N] train-only normalized speed
        first, last = speed[:,0],speed[:,-1]
        moments = torch.stack((speed.mean(1),speed.std(1,unbiased=False),last,last-first),dim=-1)
        moments = moments.mean(0)  # aggregate over batch; Q/K remain differentiable
        q = self.node_q(moments)
        k = self.node_k(moments)
        a = torch.softmax(torch.relu(q @ k.T) / math.sqrt(self.adaptive_dim),dim=-1)
        supports = [a]
        if self.use_reverse_support:
            supports.append(self._stochastic_adjacency(a.T))
        if self.use_knowledge_graph:
            if adjacency is None:
                raise ValueError('use_knowledge_graph=True requires supplied adjacency')
            if adjacency.ndim != 2 or tuple(adjacency.shape) != (historical.shape[2],historical.shape[2]):
                raise ValueError('adjacency must have shape [N,N]')
            adj = adjacency.to(dtype=a.dtype,device=a.device)
            if not torch.isfinite(adj).all():
                raise ValueError('non-finite adjacency')
            supports.append(self._stochastic_adjacency(adj))
            if self.use_reverse_support:
                supports.append(self._stochastic_adjacency(adj.T))
        return supports

    def align_history(self,x:torch.Tensor)->torch.Tensor:
        if x.shape[1] not in self.supported_horizons:
            raise ValueError(f'unsupported history {x.shape[1]}')
        return _align_history(x,self.canonical_steps)

    def forward(self,x:torch.Tensor,adjacency:torch.Tensor|None=None,
                horizon:int|None=None,return_aux:bool=False):
        if x.ndim!=4 or x.shape[-1]<self.input_dim:
            raise ValueError(f'expect [B,T,N,F>=input_dim], got {tuple(x.shape)}')
        if not torch.isfinite(x).all():
            raise ValueError('non-finite traffic inputs')
        steps = x.shape[1]
        horizon = steps if horizon is None else int(horizon)
        if horizon not in self.supported_horizons:
            raise ValueError(f'unsupported horizon {horizon}')
        aligned = self.align_history(x)
        supports = self._supports(aligned,adjacency)
        h = aligned[...,:self.input_dim].permute(0,3,2,1).contiguous()
        if h.size(-1)<self.receptive_field:
            h=F.pad(h,(self.receptive_field-h.size(-1),0))
        h=self.start_conv(h)
        skip=None
        for filt,gate,sk,bn,gcn in zip(self.filter_convs,self.gate_convs,
                                        self.skip_convs,self.bn,self.gconv):
            old=h
            h=torch.tanh(filt(h))*torch.sigmoid(gate(h))
            h=gcn(h,supports)
            # Read from the post-GCN residual path so even the final
            # graph and normalization layer contribute to forecast gradients.
            h=bn(h+old[...,-h.size(-1):])
            s=sk(h)
            skip=s if skip is None else skip[...,-s.size(-1):]+s
        out=self.end_conv_2(F.relu(self.end_conv_1(F.relu(skip))))
        # If input history exceeds receptive field, use the final forecast location.
        out=out[...,-1].contiguous()  # B,12,N
        if horizon!=self.canonical_steps:
            stride=self.canonical_steps//horizon
            out=out[:,stride-1::stride]
        prediction=out.unsqueeze(-1)
        return {'prediction':prediction} if return_aux else prediction


__all__=['GraphWaveNetBackbone']
