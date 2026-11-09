"""Gradient-only integration with the official MTPB PatchFSL/STRep 288->36 model.

Author's code and traffic pattern banks remain external dependencies:
https://github.com/zhyliu00/MTPB @ 5676a38a34bf3aebbd83cef3cb0d617a9a153b15.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch.func import functional_call

from tsjt_pgrad.pgrad.fusion import residual_matching_fusion
from tsjt_pgrad.pgrad.rotation import AngleController, rotate_source_gradient
from tsjt_pgrad.pgrad.source_interaction import source_interaction_principal_gradient


@dataclass
class MTPBGradientConfig:
    virtual_lr: float = 1e-4
    correction_lr: float = 5e-4
    correction_cap: float = 0.2
    order_count: int = 3
    fusion_lambda: float = 1.0
    fusion_mu_w: float = 1e-3
    angle_lr: float = 1e-3
    angle_meta_step: float = 1e-5


class MTPBGradientAdapter:
    """Add target-guided source-gradient correction after a native MTPB meta-step.

    Original MTPB pattern encoder, graph reconstruction, GWN, long-term branch,
    forecasting head, 288->36 task, and target finetuning remain upstream.
    Only source and first-three-day target-training gradients are consulted.
    """
    def __init__(self, rep, config: MTPBGradientConfig | None = None):
        self.rep = rep
        self.cfg = config or MTPBGradientConfig()
        self.angle_controller = AngleController().to(rep.device)
        self.angle_optimizer = torch.optim.Adam(
            self.angle_controller.parameters(), lr=self.cfg.angle_lr)
        self.previous_weights = None
        self.last_diagnostics = {}

    @staticmethod
    def _sample(dataset, city: str, *, exclude=None):
        x = dataset.x_list[city]
        total = len(x)
        batch_size = int(dataset.task_args['batch_size'])
        eligible = torch.arange(total)
        if exclude is not None:
            eligible = eligible[~torch.isin(eligible, exclude.cpu())]
        if len(eligible) < batch_size:
            raise ValueError(f"insufficient disjoint train windows for {city}")
        idx = eligible[torch.randperm(len(eligible))[:batch_size]]
        data = SimpleNamespace(
            x=x[idx].float(), y=dataset.y_list[city][idx].float(),
            means=dataset.means_list[city], stds=dataset.stds_list[city],
            data_name=city)
        adjacency = dataset.A_list[city].float()
        return (data, adjacency.unsqueeze(0).expand(batch_size, -1, -1)), idx

    @staticmethod
    def _loss(model, pair, *, overrides=None):
        data, adjacency = pair
        if overrides is None:
            out, y, _ = model(data, adjacency)
        else:
            buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
            out, y, _ = functional_call(model, (overrides, buffers), (data, adjacency))
        if out.shape != y.shape or out.shape[-1] != 36:
            raise ValueError(
                f"MTPB must predict 36 steps; got {tuple(out.shape)} vs {tuple(y.shape)}")
        return torch.nn.functional.mse_loss(out, y)

    def train_step(self, support, support_adj, query, query_adj,
                   source_dataset, target_dataset):
        model = self.rep.model
        named = OrderedDict((n, p) for n, p in model.named_parameters() if p.requires_grad)
        params = list(named.values())
        if not params:
            raise ValueError('native MTPB has no trainable parameters')
        start = [p.detach().clone() for p in params]
        original_buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        sources = [str(c) for c in source_dataset.data_list]
        target = str(target_dataset.data_list[0])
        if len(sources) < 2 or target in sources:
            raise ValueError('source dataset must exclude the target city')
        source_pairs = [self._sample(source_dataset, c)[0] for c in sources]
        target_pair, target_idx = self._sample(target_dataset, target)
        query_pair, query_idx = self._sample(
            target_dataset, target, exclude=target_idx)
        model.zero_grad(set_to_none=True)

        def grad(pair, overrides=None):
            loss = self._loss(model, pair, overrides=overrides)
            p = tuple(overrides.values()) if overrides is not None else tuple(params)
            g = torch.autograd.grad(loss, p, allow_unused=True)
            return tuple(torch.zeros_like(w) if gi is None else gi.detach()
                         for w, gi in zip(p, g))

        target_grad = grad(target_pair)
        query_grad = grad(query_pair)
        source_grads = [grad(pair) for pair in source_pairs]

        def virtual_callback(city, virtual_params):
            return grad(source_pairs[city], virtual_params)

        principal, interaction = source_interaction_principal_gradient(
            model, virtual_callback, params, source_grads, target_grad,
            virtual_lr=self.cfg.virtual_lr, order_count=self.cfg.order_count,
            previous_weights=self.previous_weights)
        self.previous_weights = interaction['source_weights']
        rotations = []
        rotated_count = 0
        for s in source_grads:
            direction, info = rotate_source_gradient(
                s, target_grad, principal, params, self.angle_controller)
            rotations.append(direction)
            rotated_count += int(info['rotated'])
        source, fusion = residual_matching_fusion(
            target_grad, rotations, params, target_query_gradients=query_grad,
            lam=self.cfg.fusion_lambda, mu_w=self.cfg.fusion_mu_w,
            return_source_only=True)

        # Angle meta-gradient must remain connected to the rotation MLP.
        # The fusion solver is deliberately non-differentiable: hold its
        # nonnegative weights fixed and rebuild the weighted rotated vectors.
        angle_meta_loss = None
        if rotated_count and any(float(w) > 0 for w in fusion['weights']):
            differentiable_rotations = []
            for g in source_grads:
                rotated, _, _ = rotate_source_gradient(
                    g, target_grad, principal, params, self.angle_controller,
                    return_phi_tensor=True)
                differentiable_rotations.append(rotated)
            weights = [float(w) for w in fusion['weights']]
            weighted = tuple(
                self.cfg.fusion_lambda * sum(
                    w * direction[i] for w, direction in zip(weights, differentiable_rotations))
                for i in range(len(params)))
            virtual = OrderedDict(
                (name, p.detach() - self.cfg.angle_meta_step * (g_t + g_s))
                for (name, p), g_t, g_s in zip(named.items(), target_grad, weighted))
            query_loss = self._loss(model, query_pair, overrides=virtual)
            angle_grads = torch.autograd.grad(
                query_loss, tuple(self.angle_controller.parameters()),
                allow_unused=True)
            if any(g is not None for g in angle_grads):
                self.angle_optimizer.zero_grad(set_to_none=True)
                for p, g in zip(self.angle_controller.parameters(), angle_grads):
                    p.grad = g
                self.angle_optimizer.step()
            angle_meta_loss = float(query_loss.detach())

        # Eliminate diagnostic-forward BN state changes before native MTPB.
        with torch.no_grad():
            for p, v in zip(params, start):
                p.copy_(v)
            for name, buffer in model.named_buffers():
                buffer.copy_(original_buffers[name])
        native_metrics = self.rep.meta_train_revise(
            support, support_adj, query, query_adj)
        with torch.no_grad():
            native_delta_norm = torch.sqrt(sum(
                ((p-v)**2).sum() for p,v in zip(params,start)))
            raw_source = tuple(
                sum(g[i] for g in source_grads)/len(source_grads)
                for i in range(len(params)))
            correction = tuple(s.detach()-base for s,base in zip(source,raw_source))
            correction_norm = torch.sqrt(sum((g**2).sum() for g in correction))
            scale = 0.0
            if float(native_delta_norm)>0 and float(correction_norm)>1e-10:
                scale = min(
                    self.cfg.correction_lr,
                    self.cfg.correction_cap*float(native_delta_norm/correction_norm))
                for p,delta in zip(params,correction):
                    p.add_(delta,alpha=-scale)
        self.last_diagnostics = {
            'source_cities':sources, 'target_city':target,
            'history':288, 'horizon':36,
            'source_target_cosine':interaction['source_target_cosine'],
            'pairwise_cosine':interaction['pairwise_cosine'],
            'source_weights':interaction['source_weights'],
            'rotation_trigger_count':rotated_count,
            'fusion_weights':fusion['weights'],
            'native_step_norm':float(native_delta_norm),
            'correction_norm':float(correction_norm),
            'correction_scale':scale,
            'angle_meta_train_query_loss':angle_meta_loss,
            'target_support_indices':target_idx.tolist(),
            'target_query_indices':query_idx.tolist()}
        return native_metrics
