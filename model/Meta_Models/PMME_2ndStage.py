import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import Parameter

import time
from copy import deepcopy
from torch.nn.utils import weight_norm
import math
import tqdm

from block import *
import sys
from pathlib import Path
sys.path.append('../../')
from utils import *
import datetime

from geomloss.samples_loss import SamplesLoss 
from torch.utils.data import TensorDataset, DataLoader
#from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.likelihoods import BernoulliLikelihood, LaplaceLikelihood

from SOFTS.SOFTS_new import Model as softs
from iTransformer.iTransformer_new import Model as iTransformer
from dynamic_pattern import (
    CrossFittedPrototypeResidualMemory,
    DynamicPatternMoE,
    ExpertResidualMemory,
    PatternPrototypeResidualMemory,
)

def log_verbose(args, msg):
    print(msg)
    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
        f.write(str(msg) + "\n")

class PatchFSL(nn.Module):
    """
    Full PatchFSL Model
    """
    def __init__(self, data_args, model_args, task_args, memory_size_args, temperature_args, model_name, PatchFSL_cfg, args, model='GWN'):
        super(PatchFSL,self).__init__()

        self.data_args, self.model_args, self.task_args, self.memory_size_args, self.temperature_args, self.model_name, self.PatchFSL_cfg = data_args, model_args, task_args, memory_size_args, temperature_args, model_name, PatchFSL_cfg
        self.args = args
        
        num_model = 1
        #########
        
        if model_name == 'SOFTS':
            self.forecasting_model = softs()
        if model_name == 'iTransformer':
            self.forecasting_model = iTransformer()
        elif model_name in ['MLP']:  
            self.forecasting_model = MLP(288, 36)

        self.dynamic_moe = DynamicPatternMoE(
            d_model=128,
            pred_len=36,
            num_experts=getattr(args, 'num_experts', 4),
            top_k=getattr(args, 'top_k', 2),
            bottleneck=getattr(args, 'expert_bottleneck', 16),
            router_hidden=getattr(args, 'router_hidden', 64),
        )
        self.last_router_probs = None
        self.last_router_aux = None
        self.last_memory_correction = None
        self.last_base_normalized = None
        self.last_target_normalized = None
        self._stage1_frozen = False
        self.memory_mode = getattr(args, 'memory_mode', 'learned')
        self.memory = None 
        
        if self.temperature_args>0:
            if self.memory_mode == 'prototype':
                self.expert_memory = CrossFittedPrototypeResidualMemory(
                    num_experts=getattr(args, 'num_experts', 4),
                    d_model=128,
                    pred_len=36,
                    slots_per_expert=getattr(args, 'memory_slots_per_expert', 512),
                    retrieval_top_k=getattr(args, 'memory_top_k', 16),
                    temperature=getattr(args, 'prototype_memory_temperature', 0.10),
                    residual_clip=getattr(args, 'prototype_residual_clip', 3.0),
                )
                print(
                    'cross-fitted prototype memory: {} experts x {} slots'.format(
                        self.expert_memory.num_experts,
                        self.expert_memory.slots_per_expert,
                    )
                )
            elif self.memory_mode == 'pattern_proto':
                self.expert_memory = PatternPrototypeResidualMemory(
                    num_experts=getattr(args, 'num_experts', 4),
                    d_model=128,
                    pred_len=36,
                    residual_gate_init=getattr(args, 'residual_gate_init', 0.0),
                    correction_clip=getattr(args, 'memory_correction_clip', 0.5),
                    horizon_decay=getattr(args, 'pattern_proto_horizon_decay', 0.45),
                )
                print(
                    'pattern-prototype residual memory: {} experts'.format(
                        self.expert_memory.num_experts
                    )
                )
            elif self.memory_mode == 'original':
                # PAPER-ORIGINAL PMME memory enhancement: a unified learnable bank
                # of MemSize x 128 slots, retrieved by attention over the Stage-I
                # (dynamic-pattern) enhanced representation, producing an additive
                # bias via bias_mlp.  gating_mlp defined but, matching the paper,
                # not wired into the forward path.
                self.memory = nn.Parameter(torch.randn(memory_size_args, 128))
                print('self.memory.shape (paper original)', self.memory.shape)
                D, D_out, H_b = 128, 36, 256
                self.bias_mlp = nn.Sequential(
                    nn.Linear(D, D_out),
                )
                self.gating_mlp = nn.Sequential(
                    nn.Linear(memory_size_args, H_b),
                    nn.GELU(),
                    nn.Linear(H_b, H_b),
                    nn.GELU(),
                    nn.Linear(H_b, 1),
                    nn.Sigmoid()
                )
                self.Wq = nn.Linear(D, D, bias=False)
                self.Wk = nn.Linear(D, D, bias=False)
                self.Wv = nn.Linear(D, D, bias=False)
            elif self.memory_mode == 'route_bias':
                self.memory = nn.Parameter(torch.randn(memory_size_args, 128))
                print('self.memory.shape (route bias)', self.memory.shape)
                D, D_out = 128, 36
                self.num_experts = getattr(args, 'num_experts', 4)
                self.Wq = nn.Linear(D, D, bias=False)
                self.Wk = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(self.num_experts)])
                self.Wv = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(self.num_experts)])
                self.bias_mlp = nn.ModuleList([nn.Linear(D, D_out) for _ in range(self.num_experts)])
                self.gating_mlp = None
            else:
                self.expert_memory = ExpertResidualMemory(
                    reference_slots=memory_size_args,
                    num_experts=getattr(args, 'num_experts', 4),
                    d_model=128,
                    pred_len=36,
                    slots_per_expert=getattr(args, 'memory_slots_per_expert', 512),
                    retrieval_top_k=getattr(args, 'memory_top_k', 32),
                    residual_gate_init=getattr(args, 'residual_gate_init', 0.0),
                    correction_clip=getattr(args, 'memory_correction_clip', 0.5),
                    horizon_decay=getattr(args, 'pattern_proto_horizon_decay', 0.0),
                )
                print(
                    'expert memory: {} experts x {} slots ({} total)'.format(
                        self.expert_memory.num_experts,
                        self.expert_memory.slots_per_expert,
                        self.expert_memory.total_slots,
                    )
                )

    def freeze_stage1(self):
        """Freeze the predictor and the exact Stage-I dynamic semantics."""
        for module in (self.forecasting_model, self.dynamic_moe):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._stage1_frozen = True
            
    
    def forward(self, data_i, stage = 'train'):  
        
        if(stage == 'train'):
            
            self.forecasting_model.train(not self._stage1_frozen)
            self.dynamic_moe.train(not self._stage1_frozen)
            if self.temperature_args>0:
                if self.memory_mode == 'original':
                    self.bias_mlp.train()
                    self.gating_mlp.train()
                    self.Wq.train()
                    self.Wk.train()
                    self.Wv.train()
                elif self.memory_mode == 'route_bias':
                    self.Wq.train()
                    for m in self.Wk: m.train()
                    for m in self.Wv: m.train()
                    for m in self.bias_mlp: m.train()
                else:
                    self.expert_memory.train()
        else:
            
            self.forecasting_model.eval()
            self.dynamic_moe.eval()
            if self.temperature_args>0:
                if self.memory_mode == 'original':
                    self.bias_mlp.eval()
                    self.gating_mlp.eval()
                    self.Wq.eval()
                    self.Wk.eval()
                    self.Wv.eval()
                elif self.memory_mode == 'route_bias':
                    self.Wq.eval()
                    for m in self.Wk: m.eval()
                    for m in self.Wv: m.eval()
                    for m in self.bias_mlp: m.eval()
                else:
                    self.expert_memory.eval()
            
        x, y, means, stds = data_i.x, data_i.y, data_i.means, data_i.stds
        # print("x shape is : {}, y shape is : {}".format(x.shape, y.shape))
        x, y, means, stds = torch.tensor(x).to(self.PatchFSL_cfg['device']), torch.tensor(y).to(self.PatchFSL_cfg['device']),torch.tensor(means).to(self.PatchFSL_cfg['device']),torch.tensor(stds).to(self.PatchFSL_cfg['device']) #,torch.tensor(A,dtype=torch.float32).to(self.PatchFSL_cfg['device'])
        # remember that the input of TSFormer is [B, N, 2, L]
        x = x.permute(0,1,3,2)

        # ############

        x = x[:,:,0, -288:]
        self.eps = 1e-9
        self.RevIn_mean = x.mean(dim=2, keepdim=True).detach()         # [B,D,1]
        var = x.var(dim=2, keepdim=True, unbiased=False).detach()                     
        self.RevIn_std  = torch.sqrt(var + self.eps)      
        x = (x - self.RevIn_mean) / self.RevIn_std
        x_node = x
        # y = (y - self.RevIn_mean) / self.RevIn_std 

        x = x.permute(0,2,1) # # [B, D, T] -> [B, T, D]

        if self.model_name in ['MLP']:
            x = x.permute(0,2,1)
        rep, out = self.forecasting_model(x)
        rep, route, router_aux = self.dynamic_moe(rep, x_node)
        self.last_router_probs = route
        self.last_router_aux = router_aux

        out = out.detach()
        rep = rep.detach()
        route = route.detach()

        y_unscale = norm(y, means, stds)
        y_unscale = (y_unscale - self.RevIn_mean) / self.RevIn_std
        self.last_base_normalized = out
        self.last_target_normalized = y_unscale

        B, N, D = rep.shape
        
        normed_memory = None
        self.last_memory_correction = torch.zeros_like(out)
        if self.temperature_args>0:
            if self.memory_mode == 'original':
                # PAPER-ORIGINAL retrieval: attention over the unified bank,
                # additive bias via bias_mlp (gating_mlp unused, as in paper).
                normed_memory = self.memory
                r = rep.view(B * N, D)                # [B*N, d]
                q = self.Wq(r)                        # [B*N, d]
                K = self.Wk(self.memory)              # [MemSize, d]
                V = self.Wv(self.memory)              # [MemSize, d]
                scores = torch.matmul(q, K.t())       # [B*N, MemSize]
                scores = scores / (D ** 0.5)
                alpha = torch.softmax(scores, dim=1)  # [B*N, MemSize]
                retrieved = torch.matmul(alpha, V)    # [B*N, d]
                delta_b = self.bias_mlp(retrieved)    # [B*N, D_out]
                delta_b = delta_b.view(B, N, -1)
                self.last_memory_correction = delta_b
                out = out + delta_b
            elif self.memory_mode == 'route_bias':
                normed_memory = self.memory
                r = rep.view(B * N, D)                # [B*N, d]
                q = self.Wq(r)                        # [B*N, d]
                route_flat = route.view(B * N, self.num_experts)
                delta_b = r.new_zeros(B * N, 36)
                for e in range(self.num_experts):
                    K = self.Wk[e](self.memory)
                    V = self.Wv[e](self.memory)
                    scores = torch.matmul(q, K.t()) / (D ** 0.5)
                    alpha = torch.softmax(scores, dim=1)
                    retrieved = torch.matmul(alpha, V)
                    delta_e = self.bias_mlp[e](retrieved)
                    delta_b = delta_b + route_flat[:, e:e+1] * delta_e
                delta_b = delta_b.view(B, N, -1)
                self.last_memory_correction = delta_b
                out = out + delta_b
            else:
                normed_memory = self.expert_memory.keys
                self.last_memory_correction = self.expert_memory(rep, route)
                out = out + self.last_memory_correction
            
        out = out * self.RevIn_std + self.RevIn_mean
        out = unnorm(out, means, stds)

        return out, y, rep, y_unscale, normed_memory

    @torch.no_grad()
    def fit_target_memory(self, data_i):
        if self.temperature_args <= 0 or self.memory_mode != 'prototype':
            raise RuntimeError('fit_target_memory requires prototype memory mode')
        self.expert_memory.ready.fill_(False)
        _, _, representation, _, _ = self.forward(data_i, stage='test')
        residual = self.last_target_normalized - self.last_base_normalized
        return self.expert_memory.fit(
            representation,
            self.last_router_probs,
            residual,
            mae_weight=self.RevIn_std.expand_as(residual),
            folds=getattr(self.args, 'prototype_memory_folds', 3),
            ridge=getattr(self.args, 'prototype_memory_ridge', 0.10),
            gate_prior=getattr(self.args, 'prototype_gate_prior', 0.25),
            gate_max=getattr(self.args, 'prototype_gate_max', 1.25),
        )

    @torch.no_grad()
    def collect_source_residual(self, data_i):
        """Run the frozen Stage-I model on one source batch and return the
        flattened (representation, route, residual, mae_weight) tuple used to
        build the expert residual memory.  ``residual`` is the Stage-I forecast
        error in the RevIN-normalised space; its source-city scale is carried by
        ``mae_weight`` so the gate calibration minimises true MAE.
        """
        _, _, representation, _, _ = self.forward(data_i, stage='test')
        residual = self.last_target_normalized - self.last_base_normalized
        rep = representation.detach().reshape(-1, representation.shape[-1])
        route = self.last_router_probs.detach().reshape(-1, self.last_router_probs.shape[-1])
        res = residual.detach().reshape(-1, residual.shape[-1])
        weight = self.RevIn_std.detach().expand_as(residual).reshape(-1, residual.shape[-1])
        return rep, route, res, weight

    @torch.no_grad()
    @torch.no_grad()
    def finalize_source_seed(self):
        """Re-scale the source-seeded learned bank after ``fit_source_memory``."""
        mem = self.expert_memory
        if hasattr(mem, 'finalize_source_seed'):
            return mem.finalize_source_seed(
                getattr(self.args, 'memory_correction_clip', 0.5)
            )

    def fit_source_memory(self, source_batches):
        """Build the expert residual memory from *source-city* Stage-I
        residuals rather than the few target days (the latter are too scarce and
        overfit, which is why the gates previously collapsed to zero).

        This matches the original PMME Residual-Memory semantics: learn the
        systematic forecast errors of the backbone on the data-rich source
        cities, partitioned by the dynamic pattern (expert route), so that the
        same error modes can be corrected on the target city.
        """
        if self.temperature_args <= 0:
            raise RuntimeError('fit_source_memory requires a residual memory')
        if hasattr(self.expert_memory, 'ready'):
            self.expert_memory.ready.fill_(False)

        rep_parts, route_parts, res_parts, w_parts = [], [], [], []
        for data_i in source_batches:
            rep, route, res, w = self.collect_source_residual(data_i)
            rep_parts.append(rep)
            route_parts.append(route)
            res_parts.append(res)
            w_parts.append(w)

        representation = torch.cat(rep_parts, dim=0)
        route_all = torch.cat(route_parts, dim=0)
        residual = torch.cat(res_parts, dim=0)
        weight = torch.cat(w_parts, dim=0)

        # Cap the number of memory-building samples to keep gate calibration
        # fast and to avoid one huge city dominating the bank.
        cap = getattr(self.args, 'source_memory_cap', 200000)
        if representation.shape[0] > cap:
            idx = torch.randperm(representation.shape[0], device=representation.device)[:cap]
            representation = representation[idx]
            route_all = route_all[idx]
            residual = residual[idx]
            weight = weight[idx]

        # Learned mode: seed the learned bank with source-city residuals
        # partitioned by dynamic pattern; the few target days then only
        # fine-tune the fixed bank.  This is the "pattern-consistent,
        # transferable residual" core of the innovation.
        if self.memory_mode != 'prototype':
            return self.expert_memory.seed_from_source(
                representation, route_all, residual, weight
            )

        # ``CrossFittedPrototypeResidualMemory.fit`` computes folds over the
        # *leading* dimension, so reshape the flattened samples into [folds, -1,
        # ...] to get a meaningful K-fold split even though the source cities'
        # node counts differ.
        n_folds = getattr(self.args, 'prototype_memory_folds', 3)
        m = representation.shape[0]
        per_fold = max(1, m // n_folds)
        m_clip = per_fold * n_folds
        d = representation.shape[-1]
        k = route_all.shape[-1]
        p = residual.shape[-1]

        representation = representation[:m_clip].view(n_folds, per_fold, d)
        route_all = route_all[:m_clip].view(n_folds, per_fold, k)
        residual = residual[:m_clip].view(n_folds, per_fold, p)
        weight = weight[:m_clip].view(n_folds, per_fold, p)

        return self.expert_memory.fit(
            representation,
            route_all,
            residual,
            mae_weight=weight,
            folds=n_folds,
            ridge=getattr(self.args, 'prototype_memory_ridge', 0.10),
            gate_prior=getattr(self.args, 'prototype_gate_prior', 0.25),
            gate_max=getattr(self.args, 'prototype_gate_max', 1.25),
        )

class STRep(nn.Module):
    """
    Reptile-based Few-shot learning architecture for STGNN
    """
    def __init__(self, classifier_args, OTepoch_args, OTcoef_args, xySquaredRatio_args, memory_size_args, temperature_args, model_name, data_args, task_args, model_args,PatchFSL_cfg, args):
        super(STRep, self).__init__()
        self.classifier_args = classifier_args
        self.OTepoch_args = OTepoch_args
        self.OTcoef_args = OTcoef_args
        self.xySquaredRatio_args = xySquaredRatio_args
        self.temperature_args = temperature_args # 0: only node 原来的multi_scale
        self.memory_size_args = memory_size_args # 0: no reweighting; 1: target few reweighting; 2: target total estimated reweighting
        self.model_name = model_name
        
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        self.PatchFSL_cfg = PatchFSL_cfg
        self.args = args

        self.lr = model_args['STnet']['lr']
        self.model_name = args.STmodel
        self.device = PatchFSL_cfg['device']
        self.current_epoch = 0

        self.model = PatchFSL(data_args, model_args, task_args, memory_size_args, temperature_args, model_name, PatchFSL_cfg, args).to(self.device)
         
        # print(self.model)
        log_verbose(args,"model params: {}".format(count_parameters(self.model)))

        self.main_optim = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.loss_criterion = nn.SmoothL1Loss(reduction='mean', beta=getattr(args, 'smooth_l1_beta', 5e-4)) 
        # Stage-II horizon-adaptive regression weight (independent of Stage-I).
        # Weights long horizons more so the bias_mlp residual finetune does not
        # under-fit the 180-min lead time.  gamma=0 reproduces the flat loss.
        self._stage2_horizon_gamma = getattr(args, 'stage2_horizon_gamma', 0.0)
        _pred_num = int(task_args['source']['pred_num'])
        _h = torch.arange(_pred_num, dtype=torch.float32, device=self.device)
        self.stage2_horizon_weights = 1.0 + self._stage2_horizon_gamma * (_h / max(_pred_num - 1, 1))

    def _weighted_horizon_loss(self, out, y):
        if self._stage2_horizon_gamma <= 0:
            return self.loss_criterion(out, y)
        w = self.stage2_horizon_weights.to(out.device)
        diff = out - y
        beta = getattr(self.args, 'smooth_l1_beta', 5e-4)
        abs_diff = diff.abs()
        l2 = 0.5 * diff.pow(2) / beta
        l1 = abs_diff - 0.5 * beta
        per_element = torch.where(abs_diff < beta, l2, l1)
        return (per_element * w.view(1, 1, -1)).mean()
        if self.temperature_args>0:
            self.main_optim = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)

    def freeze_stage1(self):
        self.model.freeze_stage1()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.main_optim = (
            optim.AdamW(trainable, lr=self.lr, weight_decay=1e-2)
            if trainable else None
        )

    def fit_target_memory(self, data_i):
        return self.model.fit_target_memory(data_i)

    def fit_source_memory(self, source_batches):
        return self.model.fit_source_memory(source_batches)

    def finalize_source_seed(self):
        return self.model.finalize_source_seed()
    
    def meta_train_revise(self, data, matrix, epoch, target=False): # matrix没有用到
        
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        
        task_losses = []
        rep_source = []
        rep_target = []
        y_source = []
        y_target = []

        target_task_loss = 0
        correction_penalties = []
        #origin_normed_memory = 0
        

        for i in range(len(data)):  # self.task_num):
            task_loss = 0
            #  for k in range(1):
            batch_size, node_num, seq_len, _ = data[i].x.shape
            out, y, rep, y_unscale, normed_memory = self.model(data[i], stage='train') 
            if self.temperature_args > 0 and self.model.memory_mode not in ('original', 'route_bias'):
                correction_penalties.append(self.model.last_memory_correction.square().mean())
            
            if epoch%25==0:
                print(out.shape, rep.shape)
            
            if i == 0:
                split_indices = [15, 15, rep.shape[0] - 30]
                rep_split = torch.split(rep, split_indices, dim=0)
                y_unscale_split = torch.split(y_unscale, split_indices, dim=0)
                
                rep_target.extend(rep_split)
                y_target.extend(y_unscale_split)
                
            else:
                rep_source.append(rep)
                y_source.append(y_unscale) #y_unscale
                
            if True: 
                if i>0: #True:  
                    
                    loss = self._weighted_horizon_loss(out, y)
    
                    task_loss = loss
    
                    # if(k == 0):
                    MSE,RMSE,MAE,MAPE = calc_metric(out, y)
                    if epoch%25==0:
                        print('MAPE',MAPE)
                    total_mse.append(MSE.cpu().detach().numpy())
                    total_rmse.append(RMSE.cpu().detach().numpy())
                    total_mae.append(MAE.cpu().detach().numpy())
                    total_mape.append(MAPE.cpu().detach().numpy())
    
                    #if i>0:
                    task_losses.append(task_loss)

                #elif target == True:
                if i ==0 :
                    loss = self._weighted_horizon_loss(out, y)
                    target_task_loss = loss
    
        
        OT_losses = []
        y_norm_sources = []
        y_norm_targets = []
        
        OT_loss_scale = 1
        model_loss = 0
        
        task_loss = sum(task_losses)
        if self.temperature_args > 0 and self.model.memory_mode not in ('original', 'route_bias'):
            task_loss = task_loss + getattr(self.args, 'memory_confidence_coef', 1e-3) * \
                self.model.expert_memory.confidence_regularisation()
            # Bounded-correction shrinkage.  The correction already passed a
            # tanh saturation, so its magnitude lives in [-clip, clip].  We add
            # a *soft-threshold* penalty that is zero for small (mild) nudges
            # and grows only once a correction approaches the saturation cap.
            # This specifically tames the outlier over-shoot that was inflating
            # the 180-min RMSE, WITHOUT the earlier mistake of penalising every
            # non-zero correction (which had shut the memory down entirely).
            clip = getattr(self.args, 'memory_correction_clip', 0.5)
            threshold = 0.6 * clip
            corrections = [c for c in correction_penalties if c is not None]
            if corrections:
                over = torch.stack([
                    torch.relu(c.abs() - threshold) for c in corrections
                ])
                overshoot_penalty = over.pow(2).mean()
                task_loss = task_loss + getattr(
                    self.args, 'memory_correction_coef', 1e-2
                ) * overshoot_penalty

        def pcgrad_one_sided(grad_a, grad_b, eps=1e-8):
            
            dot = torch.dot(grad_a, grad_b)
            if dot < 0:
                proj_coeff = dot / (grad_b.norm()**2 + eps)
                grad_a = grad_a - proj_coeff * grad_b
            return grad_a
        
        def flatten_and_pad_grads(params):
            
            flat_grads = []
            mapping = []  # list of (param, start, length)
            offset = 0
            for p in params:
                numel = p.numel()
                if p.grad is None:
                    g = torch.zeros(numel, device=p.device, dtype=p.dtype)
                else:
                    g = p.grad.detach().view(-1).clone()
                flat_grads.append(g)
                mapping.append((p, offset, numel))
                offset += numel
            return torch.cat(flat_grads, dim=0), mapping
        
        def write_back_flat_grads(flat_grads, mapping):
            for p, start, numel in mapping:
                sub = flat_grads[start:start+numel].view_as(p)
            
                p.grad = sub.clone()
        self.main_optim.zero_grad()
        task_loss.backward(retain_graph=True)
        
        flat_A, mapping = flatten_and_pad_grads(self.model.parameters())
        self.main_optim.zero_grad()
        target_task_loss.backward(retain_graph=True)
        
        flat_B, _ = flatten_and_pad_grads(self.model.parameters())
        flat_proj_A = pcgrad_one_sided(flat_A, flat_B)
        if self.model.memory_mode in ('original', 'route_bias'):
            write_back_flat_grads(flat_proj_A, mapping)
        else:
            # Original PMME only removes source gradients conflicting with the
            # target few-shot gradient.  For expert memories this left all values
            # source-dominated and empirically worsened every METR-LA horizon.
            # Retain PCGrad, but add a controlled target residual component.
            target_weight = getattr(self.args, 'target_memory_weight', 0.25)
            write_back_flat_grads(flat_proj_A + target_weight * flat_B, mapping)
        self.main_optim.step()
        self.main_optim.zero_grad()


        model_loss = task_loss
        
        if epoch%25==0:
            print('len(task_losses), len(OT_losses)', len(task_losses), len(OT_losses))
        
        return model_loss.detach().cpu().numpy(),np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)
        
        
    def forward(self, data, matrix):
        out, meta_graph = self.model(data)
        return out, meta_graph

    def test_batch(self,start,end,source_dataset,stage = "test"):  
        total_loss = []
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        print('start,end', start,end)
        
        print(type(source_dataset))
        try:
            print(len(source_dataset)) # 265
        except:
            print(source_dataset.shape[0])
        
        with torch.no_grad():
            for idx in range(start,end):
                data_i, A = source_dataset[idx]
                B, N, D, L = data_i.x.shape
                if B==0:
                    break
                    
                out, y, rep, y_unscale, normed_memory = self.model(data_i, stage='test') # KL=0
            
                MSE,RMSE,MAE,MAPE = calc_metric(out, y, stage='test')
                total_mse.append((MSE.cpu().detach().numpy(), len(out)))
                total_rmse.append((RMSE.cpu().detach().numpy(), len(out)))
                total_mae.append((MAE.cpu().detach().numpy(), len(out)))
                total_mape.append((MAPE.cpu().detach().numpy(), len(out)))
        return total_mse,total_rmse, total_mae, total_mape, total_loss
    
    
    def evaluation(self, finetune_dataset, test_dataset, target_epochs):
        curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")

        print("[INFO] Enter validation phase")
        
        total_mse_horizon,total_rmse_horizon, total_mae_horizon, total_mape_horizon, total_loss = self.test_batch(0, 1, finetune_dataset, "test") 
    
        validation_score = []
        
        for i in range(self.task_args['source']['pred_num']):
            total_len = 0
            
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            for j in range(len(total_mse_horizon)):

                total_len += total_mse_horizon[j][1]
                total_mse.append(total_mse_horizon[j][0][i]*total_mse_horizon[j][1])
                total_mae.append(total_mae_horizon[j][0][i]*total_mae_horizon[j][1])
                total_mape.append(total_mape_horizon[j][0][i]*total_mape_horizon[j][1])
            
            total_mae = sum(total_mae)/total_len
            total_mse = sum(total_mse)/total_len
            total_rmse = np.sqrt(total_mse)
            total_mape = sum(total_mape)/total_len
            
            validation_score.append(total_mae)

            #print('total_len', total_len)

            print('Horizon {} : Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(i, total_mse, total_rmse, total_mae, total_mape))
            
        
        print("[INFO] Enter test phase")
        length = test_dataset.__len__()
        # self.model = copy.deepcopy(best_model)
        print('length', length) # 265
        total_mse_horizon,total_rmse_horizon, total_mae_horizon, total_mape_horizon, total_loss = self.test_batch(0,length+1, test_dataset,"test")   

        testing_score = []
        
        #total_len = 0
        
        for i in range(self.task_args['source']['pred_num']):
            total_len = 0
            
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            for j in range(len(total_mse_horizon)):

                total_len += total_mse_horizon[j][1]
                total_mse.append(total_mse_horizon[j][0][i]*total_mse_horizon[j][1])
                total_mae.append(total_mae_horizon[j][0][i]*total_mae_horizon[j][1])
                total_mape.append(total_mape_horizon[j][0][i]*total_mape_horizon[j][1])

            total_mae = sum(total_mae)/total_len
            total_mse = sum(total_mse)/total_len
            total_rmse = np.sqrt(total_mse)
            total_mape = sum(total_mape)/total_len
            
            testing_score.append(total_mae)
            #print('total_len', total_len) total_len 2782 = 16*173+14

            print('Horizon {} : Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(i, total_mse, total_rmse, total_mae, total_mape))
            #print('Horizon {} : Unnormed MSE : {:.5f}, RMSE : {:.5f}, 

        return validation_score, testing_score


    
    def valid_evaluation(self, num_fold, Kth_fold, finetune_dataset, test_dataset, target_epochs):
        curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")
        
        print("[INFO] Enter validation phase")
        
        total_mse_horizon,total_rmse_horizon, total_mae_horizon, total_mape_horizon, total_loss = self.valid_batch(num_fold, Kth_fold, 0, 1, finetune_dataset, "test")
    
        validation_score = []
        
        for i in range(self.task_args['source']['pred_num']):
            total_len = 0
            
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            for j in range(len(total_mse_horizon)):

                total_len += total_mse_horizon[j][1]
                # print(total_mse_horizon[j][1], total_mae_horizon[j][1], total_mape_horizon[j][1]) 46
                total_mse.append(total_mse_horizon[j][0][i]*total_mse_horizon[j][1])
                total_mae.append(total_mae_horizon[j][0][i]*total_mae_horizon[j][1])
                total_mape.append(total_mape_horizon[j][0][i]*total_mape_horizon[j][1])
                
            total_mae = sum(total_mae)/total_len
            total_mse = sum(total_mse)/total_len
            total_rmse = np.sqrt(total_mse)
            total_mape = sum(total_mape)/total_len
            
            validation_score.append(total_mae)

            print('Horizon {} : Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(i, total_mse, total_rmse, total_mae, total_mape))

            
        return validation_score
    
    def test_evaluation(self, finetune_dataset, test_dataset, target_epochs):
        curr_time = datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S")
        
        print("[INFO] Enter test phase")
        length = test_dataset.__len__()
        # self.model = copy.deepcopy(best_model)
        print('length', length) # 265
        total_mse_horizon,total_rmse_horizon, total_mae_horizon, total_mape_horizon, total_loss = self.test_batch(0,length+1, test_dataset,"test")      
        testing_score = []
        
        for i in range(self.task_args['source']['pred_num']):
            total_len = 0
            
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            for j in range(len(total_mse_horizon)):

                total_len += total_mse_horizon[j][1]
                total_mse.append(total_mse_horizon[j][0][i]*total_mse_horizon[j][1])
                total_mae.append(total_mae_horizon[j][0][i]*total_mae_horizon[j][1])
                total_mape.append(total_mape_horizon[j][0][i]*total_mape_horizon[j][1])
                

            total_mae = sum(total_mae)/total_len
            total_mse = sum(total_mse)/total_len
            total_rmse = np.sqrt(total_mse)
            total_mape = sum(total_mape)/total_len
            
            testing_score.append(total_mae)

            print('Horizon {} : Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(i, total_mse, total_rmse, total_mae, total_mape))

        return testing_score
    
