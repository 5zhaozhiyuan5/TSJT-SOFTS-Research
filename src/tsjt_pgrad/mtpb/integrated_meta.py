"""Replace original MTPB source meta-update with target-guided source gradients.

Keeps upstream PatchFSL pattern bank, graph reconstruction, Graph WaveNet,
long-term features, 288->36 forecast head, and target-city finetuning intact.
Upstream revision: zhyliu00/MTPB @ 5676a38a34bf3aebbd83cef3cb0d617a9a153b15.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.func import functional_call

from tsjt_pgrad.pgrad.grad_utils import flatten_gradients
from tsjt_pgrad.pgrad.rotation import AngleController, rotate_source_gradient
from tsjt_pgrad.pgrad.source_interaction import source_interaction_principal_gradient
from tsjt_pgrad.pgrad.fusion import residual_matching_fusion


@dataclass(frozen=True)
class IntegratedMetaConfig:
    inner_lr: float = 5e-4
    inner_steps: int = 2
    meta_lr: float = 5e-4
    virtual_lr: float = 1e-4
    order_count: int = 3
    grad_clip_norm: float = 10.0
    fusion_lambda: float = 1.0
    fusion_mu_w: float = 1e-3
    angle_lr: float = 1e-3
    angle_meta_step: float = 1e-5


class TargetGuidedMTPBMetaUpdate:
    """Directly replace STRep.meta_train_revise with source-gradient meta-update.

    First-order support/query adaptation is retained. The native updater is
    NOT called and no separate after-the-fact delta correction is applied.
    """
    def __init__(self, rep, cfg: IntegratedMetaConfig | None = None):
        self.rep = rep
        self.cfg = cfg or IntegratedMetaConfig()
        if self.cfg.inner_steps < 1 or self.cfg.order_count < 1:
            raise ValueError('inner_steps and order_count must be positive')
        if min(self.cfg.inner_lr, self.cfg.meta_lr, self.cfg.virtual_lr) <= 0:
            raise ValueError('training learning rates must be positive')
        self.angle_controller = AngleController().to(rep.device)
        self.angle_optimizer = torch.optim.Adam(
            self.angle_controller.parameters(), lr=self.cfg.angle_lr)
        self.previous_weights = None
        self.last_diagnostics = {}

    @staticmethod
    def sample(dataset, city, *, exclude=None):
        available = torch.arange(len(dataset.x_list[city]))
        if exclude is not None:
            available = available[~torch.isin(available, exclude.cpu())]
        batch = int(dataset.task_args['batch_size'])
        if len(available) < batch:
            raise ValueError(f'not enough distinct training indices in {city}')
        indices = available[torch.randperm(len(available))[:batch]]
        data = SimpleNamespace(
            x=dataset.x_list[city][indices].float(),
            y=dataset.y_list[city][indices].float(),
            means=dataset.means_list[city],
            stds=dataset.stds_list[city],
            data_name=city,
        )
        adjacency = dataset.A_list[city].float().unsqueeze(0).expand(batch, -1, -1)
        return (data, adjacency), indices

    @staticmethod
    def loss(model, pair, state):
        data, adjacency = pair
        buffers = {name: value.detach().clone() for name,value in model.named_buffers()}
        pred, truth, _ = functional_call(model, (state, buffers), (data, adjacency))
        if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != 36:
            raise ValueError(
                f'expected official MTPB [B,N,36], got {tuple(pred.shape)} / {tuple(truth.shape)}')
        return F.mse_loss(pred, truth)

    def task_gradient(self, model, support, query, initial):
        """First-order support/query meta gradient at the supplied parameter state."""
        state = OrderedDict((name, value.detach().clone().requires_grad_(True))
                            for name,value in initial.items())
        result = [torch.zeros_like(v) for v in initial.values()]
        losses = []
        for _ in range(self.cfg.inner_steps):
            support_loss = self.loss(model, support, state)
            gs = torch.autograd.grad(
                support_loss, tuple(state.values()), allow_unused=True)
            state = OrderedDict((
                name, (weight - self.cfg.inner_lr * (
                    g.detach() if g is not None else torch.zeros_like(weight)))
                    .detach().requires_grad_(True)
            ) for (name,weight),g in zip(state.items(),gs))
            query_loss = self.loss(model, query, state)
            gq = torch.autograd.grad(
                query_loss, tuple(state.values()), allow_unused=True)
            for i,g in enumerate(gq):
                if g is not None:
                    result[i] += g.detach()/self.cfg.inner_steps
            losses.append(float(query_loss.detach()))
        if not all(bool(torch.isfinite(g).all()) for g in result):
            raise FloatingPointError('non-finite MTPB task gradient')
        return tuple(result), sum(losses)/len(losses)

    def step(self, source_dataset, target_train_dataset):
        model = self.rep.model
        sources = [str(c) for c in source_dataset.data_list]
        target = str(target_train_dataset.data_list[0])
        if len(sources) != 3 or len(set(sources)) != 3 or target in sources:
            raise ValueError('expected exactly 3 distinct sources excluding the target')
        named = OrderedDict(
            (name,p) for name,p in model.named_parameters() if p.requires_grad)
        params = tuple(named.values())
        if not params:
            raise ValueError('no trainable MTPB parameters')
        initial = OrderedDict((n,p.detach()) for n,p in named.items())

        source_tasks = []
        for city in sources:
            s,idx = self.sample(source_dataset,city)
            q,_ = self.sample(source_dataset,city,exclude=idx)
            source_tasks.append((s,q))
        t_support,t_idx = self.sample(target_train_dataset,target)
        t_query,q_idx = self.sample(target_train_dataset,target,exclude=t_idx)
        target_query,_ = self.sample(target_train_dataset,target,
            exclude=torch.cat((t_idx,q_idx)))

        source_results = [
            self.task_gradient(model,s,q,initial) for s,q in source_tasks]
        source_grads = [r[0] for r in source_results]
        target_grad,_ = self.task_gradient(
            model,t_support,t_query,initial)
        target_query_grad,_ = self.task_gradient(
            model,t_query,target_query,initial)

        def source_callback(city,virtual_state):
            return self.task_gradient(
                model,*source_tasks[city],virtual_state)[0]

        principal,interaction = source_interaction_principal_gradient(
            model,source_callback,params,source_grads,target_grad,
            virtual_lr=self.cfg.virtual_lr,order_count=self.cfg.order_count,
            previous_weights=self.previous_weights)
        self.previous_weights = interaction['source_weights']

        rotated = []
        rotation_flags = []
        for g in source_grads:
            direction,details = rotate_source_gradient(
                g,target_grad,principal,params,self.angle_controller)
            rotated.append(direction)
            rotation_flags.append(bool(details['rotated']))
        orthogonal,fusion = residual_matching_fusion(
            target_grad,rotated,params,
            target_query_gradients=target_query_grad,
            lam=self.cfg.fusion_lambda,mu_w=self.cfg.fusion_mu_w,
            return_source_only=True)

        # The source update is a target-aligned parallel component PLUS the
        # residual-matched, target-orthogonal rotated component. Neither
        # original MTPB's meta update nor post-hoc gradient corrections remain.
        target_flat = flatten_gradients(params,target_grad).reshape(-1)
        target_norm2 = target_flat.square().sum()
        positive_parallel = [torch.zeros_like(p) for p in params]
        if float(target_norm2) > 1e-12:
            for g,weight in zip(source_grads,interaction['source_weights']):
                gf = flatten_gradients(params,g).reshape(-1)
                coeff = ((gf @ target_flat)/target_norm2).clamp_min(0).detach()
                for i in range(len(params)):
                    positive_parallel[i] += float(weight)*coeff*target_grad[i]
        source_direction = tuple(
            (parallel+orth).detach()
            for parallel,orth in zip(positive_parallel,orthogonal))

        # Non-differentiable fusion QP supplies fixed coefficients. Rebuild
        # rotated directions to propagate target-training query loss through
        # the angle controller, without differentiating the QP itself.
        angle_query_loss = None
        if any(rotation_flags) and any(float(w)>0 for w in fusion['weights']):
            differentiable = []
            for g in source_grads:
                ri,_,_ = rotate_source_gradient(
                    g,target_grad,principal,params,self.angle_controller,
                    return_phi_tensor=True)
                differentiable.append(ri)
            fused_diff = tuple(
                self.cfg.fusion_lambda*sum(
                    float(w)*d[i] for w,d in zip(fusion['weights'],differentiable))
                for i in range(len(params)))
            virtual = OrderedDict(
                (name,p.detach()-self.cfg.angle_meta_step*(gp+go))
                for (name,p),gp,go in zip(named.items(),positive_parallel,fused_diff))
            query_loss = self.loss(model,target_query,virtual)
            phi_grad = torch.autograd.grad(
                query_loss,tuple(self.angle_controller.parameters()),
                allow_unused=True)
            if any(g is not None for g in phi_grad):
                self.angle_optimizer.zero_grad(set_to_none=True)
                for p,g in zip(self.angle_controller.parameters(),phi_grad):
                    p.grad = g
                self.angle_optimizer.step()
            angle_query_loss = float(query_loss.detach())

        gradient_norm = torch.sqrt(sum(
            g.square().sum() for g in source_direction))
        clip = min(1.,self.cfg.grad_clip_norm/(float(gradient_norm)+1e-12))
        with torch.no_grad():
            for p,g in zip(params,source_direction):
                p.add_(g,alpha=-self.cfg.meta_lr*clip)
        if hasattr(self.rep,'current_epoch'):
            self.rep.current_epoch += 1

        self.last_diagnostics = {
            'source_cities':sources,'target_city':target,'history':288,'horizon':36,
            'source_task_query_mse':[r[1] for r in source_results],
            'source_weights':interaction['source_weights'],
            'source_target_cosine':interaction['source_target_cosine'],
            'pairwise_cosine':interaction['pairwise_cosine'],
            'principal_norm':interaction['principal_norm'],
            'rotation_triggers':int(sum(rotation_flags)),
            'fusion_weights':fusion['weights'],
            'gradient_norm':float(gradient_norm),'clip_factor':clip,
            'realized_update_norm':float(gradient_norm)*self.cfg.meta_lr*clip,
            'angle_query_mse':angle_query_loss,'target_train_only':True,
        }
        return self.last_diagnostics
