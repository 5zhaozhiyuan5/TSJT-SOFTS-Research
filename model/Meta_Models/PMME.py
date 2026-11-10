import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import Parameter

import time
from copy import deepcopy
from torch.nn.utils import weight_norm
import math

import sys
from pathlib import Path
sys.path.append('../../')
from utils import *
import datetime
from block import *

from geomloss.samples_loss import SamplesLoss 
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.likelihoods import BernoulliLikelihood, LaplaceLikelihood

from SOFTS.SOFTS_new import Model as softs
from iTransformer.iTransformer_new import Model as iTransformer

from DA_tools import *
from dynamic_pattern import DynamicPatternMoE, prototype_partial_alignment

def log_verbose(args, msg):
    print(msg)
    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
        f.write(str(msg) + "\n")

class PatchFSL(nn.Module):
    """
    Full PatchFSL Model
    """
    def __init__(self, data_args, model_args, task_args, memory_size_args, projection_args, model_name, PatchFSL_cfg, args, model='GWN'):
        super(PatchFSL,self).__init__()

        self.data_args, self.model_args, self.task_args, self.memory_size_args, self.projection_args, self.model_name, self.PatchFSL_cfg = data_args, model_args, task_args, memory_size_args, projection_args, model_name, PatchFSL_cfg
        
        num_model = 1
        #########
        #print('model_name', model_name)
        
        self.model_name = model_name
        if model_name == 'SOFTS':
            self.forecasting_model = softs()
        elif model_name == 'iTransformer':
            self.forecasting_model = iTransformer()
        elif model_name in ['MLP', 'MLP_noRevIn']:
            self.forecasting_model = MLP(288, 36)

        self.dynamic_moe = DynamicPatternMoE(
            d_model=128,
            pred_len=36,
            num_experts=getattr(args, 'num_experts', 4),
            top_k=getattr(args, 'top_k', 2),
            bottleneck=getattr(args, 'expert_bottleneck', 16),
            router_hidden=getattr(args, 'router_hidden', 64),
            fusion_mode=getattr(args, 'moe_fusion_mode', 'add'),
            gate_coef=getattr(args, 'moe_gate_coef', 0.0),
        )
        self.last_router_probs = None
        self.last_router_aux = None
        self.last_router_dense = None
        self.memory = None 
        
    def forward(self, data_i, stage = 'train'):  
        if(stage == 'train'):
            
            self.forecasting_model.train()
        else: 
            self.forecasting_model.eval()
           
        x, y, means, stds = data_i.x, data_i.y, data_i.means, data_i.stds
        # print("x shape is : {}, y shape is : {}".format(x.shape, y.shape))
        x, y, means, stds = torch.tensor(x).to(self.PatchFSL_cfg['device']), torch.tensor(y).to(self.PatchFSL_cfg['device']),torch.tensor(means).to(self.PatchFSL_cfg['device']),torch.tensor(stds).to(self.PatchFSL_cfg['device']) #,torch.tensor(A,dtype=torch.float32).to(self.PatchFSL_cfg['device'])
        # remember that the input of TSFormer is [B, N, 2, L]
        x = x.permute(0,1,3,2)

        # ############

        x = x[:,:,0, -288:]

        if self.model_name == 'MLP_noRevIn':
            #x = x.permute(0,2,1)
            #print('x.shape', x.shape)
            x_node = x
            rep, out = self.forecasting_model(x)
            out = unnorm(out, means, stds)
    
            #y_unscale = y * self.RevIn_std + self.RevIn_mean
            y_unscale = norm(y, means, stds)
            normed_memory = None
            
        else:
            self.eps = 1e-9
            self.RevIn_mean = x.mean(dim=2, keepdim=True).detach()         # [B,D,1]
            var = x.var(dim=2, keepdim=True, unbiased=False).detach()                     
            self.RevIn_std  = torch.sqrt(var + self.eps)      
    
            x = (x - self.RevIn_mean) / self.RevIn_std
            x_node = x
            x = x.permute(0,2,1) # # [B, D, T] -> [B, T, D]
            if self.model_name == 'MLP':
                x = x.permute(0,2,1)
            
            rep, out = self.forecasting_model(x)

            normed_memory = None

        rep, route, router_aux = self.dynamic_moe(rep, x_node)
        # The MoE only refines the representation used for cross-city OT
        # alignment and residual compensation; the backbone forecast `out` is
        # intentionally left unchanged so the dynamic-pattern expert never
        # steals the regression task from the (frozen-in-Stage-II) backbone.
        self.last_router_probs = route
        self.last_router_aux = router_aux
        self.last_router_dense = router_aux['dense_prob']
    
        if self.model_name != 'MLP_noRevIn':
            out = out * self.RevIn_std + self.RevIn_mean
            out = unnorm(out, means, stds)
    
            #y_unscale = y * self.RevIn_std + self.RevIn_mean
            y_unscale = norm(y, means, stds)
            y_unscale = (y_unscale - self.RevIn_mean) / self.RevIn_std 
        
        return out, y, rep, y_unscale, normed_memory

class STRep(nn.Module):
    """
    Reptile-based Few-shot learning architecture for STGNN
    """
    def __init__(self, classifier_args, DAepoch_args, DAcoef_args, xySquaredRatio_args, memory_size_args, projection_args, model_name, DA, data_args, task_args, model_args,PatchFSL_cfg, args):
        super(STRep, self).__init__()
        self.classifier_args = classifier_args
        self.DAepoch_args = DAepoch_args
        self.DAcoef_args = DAcoef_args
        self.xySquaredRatio_args = xySquaredRatio_args
        self.projection_args = projection_args 
        self.memory_size_args = memory_size_args 
        
        self.model_name = model_name
        self.DA = DA
        
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        self.PatchFSL_cfg = PatchFSL_cfg
        self.args = args

        self.lr = model_args['STnet']['lr']
        self.device = PatchFSL_cfg['device']
        self.current_epoch = 0

        self.model = PatchFSL(data_args, model_args, task_args, memory_size_args, projection_args, model_name, PatchFSL_cfg, args).to(self.device)
        
        for name, params in self.model.named_parameters():
            log_verbose(args,"{} : {}, require_grads : {}".format(name, params.shape,params.requires_grad))
        
        # print(self.model)
        log_verbose(args,"model params: {}".format(count_parameters(self.model)))

        self.main_optim = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        
        self.loss_criterion = nn.SmoothL1Loss(reduction='mean', beta=5e-4)

        # Horizon-adaptive regression weight.  The 180-min (long-range) horizon
        # is the one place where the transfer lags the paper baseline: a flat
        # SmoothL1 treats all 36 horizons equally, so the easy short horizons
        # soak up the gradient while the hard long horizons are under-fitted.
        # We re-weight the per-horizon loss to increase with the forecast lead
        # time (cf. horizon-progressive / curriculum weighting in long-horizon
        # time-series forecasting).  ``horizon_weight_gamma`` controls how fast
        # the weight grows; 0 reproduces the original flat loss.
        pred_num = int(task_args['source']['pred_num'])
        self._horizon_weight_gamma = getattr(args, 'stage1_horizon_gamma', getattr(args, 'horizon_weight_gamma', 0.0))
        _h = torch.arange(pred_num, dtype=torch.float32, device=self.device)
        self.horizon_weights = 1.0 + self._horizon_weight_gamma * (_h / max(pred_num - 1, 1))

        input_dim = 36
        
        num_inducing = 256
        inducing_points1 = torch.randn(num_inducing, input_dim).to(self.device) 
        inducing_points2 = torch.randn(num_inducing, input_dim).to(self.device) 
        inducing_points3 = torch.randn(num_inducing, input_dim).to(self.device) 
        
    
        self.gp1 = GPClassificationModel(input_dim, inducing_points1).to(self.device)
        self.gp2 = GPClassificationModel(input_dim, inducing_points2).to(self.device)
        self.gp3 = GPClassificationModel(input_dim, inducing_points3).to(self.device)
        
        self.likelihood1 = BernoulliLikelihood().to(self.device)
        self.likelihood2 = BernoulliLikelihood().to(self.device)
        self.likelihood3 = BernoulliLikelihood().to(self.device)
        
        self.optimizer1 = torch.optim.Adam([
            {'params': self.gp1.parameters()},
            {'params': self.likelihood1.parameters()},
        ], lr=0.01)
        
        self.optimizer2 = torch.optim.Adam([
            {'params': self.gp2.parameters()},
            {'params': self.likelihood2.parameters()},
        ], lr=0.01)
        
        self.optimizer3 = torch.optim.Adam([
            {'params': self.gp3.parameters()},
            {'params': self.likelihood3.parameters()},
        ], lr=0.01)
        
    
    def _weighted_horizon_loss(self, out, y):
        """Horizon-adaptive SmoothL1 regression loss.

        ``out``/``y`` are [B, N, pred_len]; the last axis is the forecast
        horizon.  When ``horizon_weight_gamma`` is 0 the per-horizon weights are
        all 1 and this reduces to the original flat loss.  Otherwise long-range
        horizons receive a proportionally larger weight so the backbone does not
        under-fit the 180-min lead time.
        """
        if self._horizon_weight_gamma <= 0:
            return self.loss_criterion(out, y)
        w = self.horizon_weights.to(out.device)
        diff = out - y
        # SmoothL1 elementwise, beta matches loss_criterion.
        beta = 5e-4
        abs_diff = diff.abs()
        l2 = 0.5 * diff.pow(2) / beta
        l1 = abs_diff - 0.5 * beta
        per_element = torch.where(abs_diff < beta, l2, l1)
        weighted = per_element * w.view(1, 1, -1)
        return weighted.mean()

    def meta_train_revise(self, data, matrix, epoch):
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        
        task_losses = []
        rep_source = []
        rep_target = []
        y_source = []
        y_target = []
        route_source = []
        route_target = []
        route_source_dense = []
        route_target_dense = []
        router_regularisers = []

        target_task_loss = 0
        for i in range(len(data)):  
            task_loss = 0
            batch_size, node_num, seq_len, _ = data[i].x.shape
            
            out, y, rep, y_unscale, normed_memory = self.model(data[i], stage='train') 
            route = self.model.last_router_probs
            route_dense = self.model.last_router_dense
            aux = self.model.last_router_aux
            router_regularisers.append(
                getattr(self.args, 'balance_coef', 0.01) * aux['balance']
                + getattr(self.args, 'diversity_coef', 0.01) * aux['diversity']
                + getattr(self.args, 'adapter_l2_coef', 0.0) * aux['adapter_l2']
            )
            
            if epoch%25==0:
                print(out.shape, rep.shape)
            
            if i == 0:
                
                split_indices = [15, 15, rep.shape[0] - 30]
                rep_split = torch.split(rep, split_indices, dim=0)
                y_unscale_split = torch.split(y_unscale, split_indices, dim=0)
                
                rep_target.extend(rep_split)
                y_target.extend(y_unscale_split)
                route_target.extend(torch.split(route, split_indices, dim=0))
                route_target_dense.extend(torch.split(route_dense, split_indices, dim=0))
                
            else:
                rep_source.append(rep)
                y_source.append(y_unscale)
                route_source.append(route)
                route_source_dense.append(route_dense)
                
            
            if i>0:
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

                if i>0:
                    task_losses.append(task_loss)

            if i ==0 :
                loss = self._weighted_horizon_loss(out, y)

                target_task_loss = loss
            
            
        model_loss = 0 

        DA_loss = 0

        shape0_list = [207, 524, 627, 325]
        print(rep_target[0].shape[1]) # 325
        shape0_list.remove(rep_target[0].shape[1])
        print(shape0_list)

        DA_losses = []
        y_norm_sources = []
        y_norm_targets = []
        
        OT_loss_scale = 1   
        ratio1, ratio2 = 10, 10
        
        B = 16
        OT_loss = 0.0
        OT_loss_ratio = 1.0  
        criterion = SamplesLoss(loss="sinkhorn", p=1, backend="tensorized", debias=False, blur=0.001) # p=1

        for index in range(len(rep_source)):      

            size1, size2 = rep_source[index].shape[0]*rep_source[index].shape[1], rep_target[index].shape[0]*rep_target[index].shape[1]
            # print(size1, size2)
            labels1, labels2 = torch.zeros(size1, device=self.device), torch.ones(size2, device=self.device) 
            labels = torch.cat((labels1, labels2), dim=0)

            x1,y1 = rep_source[index], y_source[index] 
            x2,y2 = rep_target[index], y_target[index]
            if getattr(self.args, 'expert_guide_dense', False):
                # Soft (non-sparse) routing for expert-guided OT: keep the full
                # per-expert distribution so mid-horizon transition dynamics
                # are not over-committed to a single expert.  Optional temperature
                # sharpens (T<1) or softens (T>1) the distribution.
                temp = getattr(self.args, 'expert_guide_temp', 1.0)
                rd1 = route_source_dense[index].clamp_min(1e-8)
                rd2 = route_target_dense[index].clamp_min(1e-8)
                r1 = (rd1 / rd1.sum(-1, keepdim=True).clamp_min(1e-8)) ** (1.0 / max(temp, 1e-6))
                r2 = (rd2 / rd2.sum(-1, keepdim=True).clamp_min(1e-8)) ** (1.0 / max(temp, 1e-6))
                r1 = r1 / r1.sum(-1, keepdim=True).clamp_min(1e-8)
                r2 = r2 / r2.sum(-1, keepdim=True).clamp_min(1e-8)
            else:
                r1, r2 = route_source[index], route_target[index]
            shape0 = x1.shape[1]
                    
            y_norm_sources.append(y1)
            y_norm_targets.append(y2)

            x1 = x1.view(-1, x1.size(-1))
            x2 = x2.view(-1, x2.size(-1))
            y1 = y1.view(-1, y1.size(-1))
            y2 = y2.view(-1, y2.size(-1))
            r1 = r1.view(-1, r1.size(-1))
            r2 = r2.view(-1, r2.size(-1))
            

            if self.classifier_args>1e-6:
                y = torch.cat((y1, y2), dim=0).detach()
                if epoch%25==0:
                    print('optimizing now')

                if shape0 == shape0_list[0]:
                    self.elbo1 = VariationalELBO(self.likelihood1, self.gp1, num_data=y.shape[0])
                    self.optimizer1.zero_grad()
                    outputs = self.gp1(y)
                    probs = self.likelihood1(outputs).probs
                    loss = -self.elbo1(outputs, labels.long())
                    loss.backward()
                    self.optimizer1.step()
                    
                elif shape0 == shape0_list[1]:
                    self.elbo2 = VariationalELBO(self.likelihood2, self.gp2, num_data=y.shape[0])
                    self.optimizer2.zero_grad()
                    outputs = self.gp2(y)
                    probs = self.likelihood2(outputs).probs
                    loss = -self.elbo2(outputs, labels.long())
                    loss.backward()
                    self.optimizer2.step()
                    
                elif shape0 == shape0_list[2]:
                    self.elbo3 = VariationalELBO(self.likelihood3, self.gp3, num_data=y.shape[0])
                    self.optimizer3.zero_grad()
                    outputs = self.gp3(y)
                    probs = self.likelihood3(outputs).probs
                    loss = -self.elbo3(outputs, labels.long())
                    loss.backward()
                    self.optimizer3.step()
               
                l1 = x1.shape[0]
                prob1, prob2 = probs[:l1], 1-probs[l1:] 
                
                prob1_true = 1-prob1 
                prob2_true = 1-prob2 
                
                mask1 = prob1_true > 0.5
                
                count1 = torch.sum(mask1)
                
                mask2 = prob2_true > 0.5
                count2 = torch.sum(mask2)
                
                acc = (count1+count2)/(len(prob1)+len(prob2))
                if epoch%25==0:
                    print(f'Classifier_Accuracy_update: {acc:.4f}',len(prob1),len(prob2))
                    print('error_prob_mean', prob1.mean(), prob2.mean()) 
                    
                thresh = self.classifier_args 
                if getattr(self.args, 'dvgp_soft_selector', False):
                    # ---- CDTrans/FixMatch-style soft + adaptive confidence selector ----
                    # 1) Temperature de-saturation: power-like sharpening that tightens
                    #    the margin between shared/private patterns WITHOUT collapsing the
                    #    probs toward the over-confident extremes that kill OT_loss_scale.
                    T = max(getattr(self.args, 'dvgp_temp', 1.0), 1e-6)
                    # Temperature de-saturation keeps the ORIGINAL [0,1] scale
                    # (no sum-normalisation) so OT_loss_scale = sqrt(prob1.mean
                    # * prob2.mean) stays well-posed; normalising to sum=1 would
                    # shrink probs to ~1/N and drive OT_loss_scale to 0 -> NaN
                    # in Sinkhorn.
                    prob1_t = prob1 ** (1.0 / T)
                    prob2_t = prob2 ** (1.0 / T)
                    # 2) Adaptive threshold: curriculum from start -> end over training,
                    #    loose early (keep most samples) to strict late (keep confident).
                    th_start = getattr(self.args, 'dvgp_thresh_start', 0.05)
                    th_end = getattr(self.args, 'dvgp_thresh_end', 0.3)
                    # Per-city adaptation (route A): chengdu_m (524 nodes) has DVGP
                    # accuracy ~0.98, so its prob1 collapses and ratio1~0.008 late in
                    # training, over-excluding the source city that carries the richest
                    # LONG-horizon dynamics (180min depends on it).  Give chengdu a lower
                    # threshold end so more of its samples survive, keeping ratio1 in a
                    # healthy band.  scale=1.0 reproduces v58 exactly.
                    if int(shape0) == 524:
                        chengdu_scale = getattr(self.args, 'dvgp_chengdu_scale', 1.0)
                        th_end = min(1.0, th_end * chengdu_scale)
                    ramp = min(1.0, max(0.0, epoch / float(self.DAepoch_args)))
                    thresh_t = th_start + (th_end - th_start) * ramp
                    # 3) Soft gating (no hard zero): confidence minus threshold, clipped.
                    #    A sample barely below threshold is smoothly damped instead of
                    #    discarded, preserving its (small) contribution to OT mass.
                    def soft_gate(x, t):
                        g = (x - t).clamp(min=0.0)
                        g = g / g.max().clamp_min(1e-8)
                        return g
                    w1 = soft_gate(prob1_t, thresh_t)
                    w2 = soft_gate(prob2_t, thresh_t)
                    count1 = (prob1 > thresh_t).sum().detach().cpu().numpy()
                    count2 = (prob2 > thresh_t).sum().detach().cpu().numpy()
                    ratio1 = count1/len(prob1)
                    ratio2 = count2/len(prob2)
                else:
                    thresh_t = thresh
                    mask1 = prob1 > thresh
                    mask2 = prob2 > thresh
                    count1 = torch.sum(mask1).detach().cpu().numpy()
                    count2 = torch.sum(mask2).detach().cpu().numpy()
                    ratio1 = count1/len(mask1)
                    ratio2 = count2/len(mask2)
                    w1 = None; w2 = None
                
                if ratio1>0.00 and ratio2>0.00:
                    if epoch%25==0:
                        print('ratio1, ratio2', ratio1, ratio2)
                    def threshold(x):   
                        return torch.where(x > thresh, x, torch.tensor(0.0, device=x.device))
                    
                    if getattr(self.args, 'dvgp_soft_selector', False):
                        prob1, prob2 = w1.detach(), w2.detach()
                    else:
                        prob1, prob2 = threshold(prob1).detach(), threshold(prob2).detach() 
                    alpha = (prob1/torch.sum(prob1)).detach()
                    beta = (prob2/torch.sum(prob2)).detach()
                    
                   
                    OT_loss_scale = torch.sqrt(((prob1.mean())*(prob2.mean())).detach())
                    _olf = getattr(self.args, 'ot_loss_scale_floor', 0.0)
                    if _olf > 0.0:
                        # DVGP becomes over-confident late in training (ratio1
                        # collapses toward ~0.008), driving OT_loss_scale down to
                        # ~0.014 and effectively switching the DA alignment off.
                        # A hard floor keeps the alignment active while preserving
                        # the confidence weighting above the floor.  floor=0 (v31)
                        # reproduces the original behaviour.
                        OT_loss_scale = OT_loss_scale.clamp_min(_olf)
                    
                    if epoch%25==0:
                        print('OT_loss_scale', OT_loss_scale)
            
            else:
                alpha = torch.ones(size1, device=self.device)/size1
                beta = torch.ones(size2, device=self.device)/size2
                OT_loss_scale = 1

            if self.DA == 'OT':
                if (epoch>=self.DAepoch_args): 
                    warmup = min(1.0, max(0.0, (epoch - self.DAepoch_args) / float(getattr(self.args, 'DAwarmup', 1500))))
                    x_y_scale = math.sqrt(self.xySquaredRatio_args)
                    y1, y2 = y1*x_y_scale, y2*x_y_scale

                    if (ratio1>0.00 and ratio2>0.00):
                        prototype_loss, prototype_plan, mapped_r1, route_entropy_loss = prototype_partial_alignment(
                            x1, y1, r1, alpha,
                            x2, y2, r2, beta,
                            transported_mass=getattr(self.args, 'partial_mass', 0.8),
                            epsilon=getattr(self.args, 'proto_epsilon', 0.05),
                            iterations=getattr(self.args, 'proto_iters', 50),
                            label_weight=getattr(self.args, 'proto_label_weight', 1.0),
                            mode_anchor=self.model.dynamic_moe.mode_anchor,
                            route_weight=getattr(self.args, 'route_cost_weight', 0.0),
                        )
                        dynamic_scale = math.sqrt(getattr(self.args, 'dynamic_cost_weight', 1.0))
                        expert_guide_scale = getattr(self.args, 'expert_guide_scale', 0.0)
                        if getattr(self.args, 'expert_guide_warmup', False):
                            # Ramp the guidance up with the DA warmup so the
                            # expert-guided transport does not suddenly jerk the
                            # mid-horizon dynamics at DA activation.
                            expert_guide_scale = expert_guide_scale * warmup
                        route_ent_scale = getattr(self.args, 'route_entropy_guide_scale', 0.0)
                        if expert_guide_scale > 0.0:
                            # Expert-guided OT: append the raw per-sample routing
                            # distribution so the sample-level Sinkhorn matches
                            # source/target samples of the SAME dynamic type first
                            # (cross-type transport becomes geometrically more
                            # expensive).  scale=0 reproduces the original.
                            x = torch.cat((x1, y1, mapped_r1 * dynamic_scale, r1 * expert_guide_scale), dim=-1)
                            y = torch.cat((x2, y2, r2 * dynamic_scale, r2 * expert_guide_scale), dim=-1) # y: target
                        else:
                            x = torch.cat((x1, y1, mapped_r1 * dynamic_scale), dim=-1)
                            y = torch.cat((x2, y2, r2 * dynamic_scale), dim=-1) # y: target
                        # Route-uncertainty guide: append the per-sample routing
                        # entropy as an extra geometry dimension.  A sample whose
                        # sparse route is sharp (low entropy) is confidently one
                        # dynamic pattern; a high-entropy sample is ambiguous.
                        # Placing entropy in the transport geometry lets the
                        # Sinkhorn distinguish "clear" vs "ambiguous" samples so
                        # ambiguous few-shot target samples are not forcibly
                        # mis-matched (which is what injects the systematic MAE
                        # bias).  route_entropy_guide_scale=0 (v31) skips this.
                        if route_ent_scale > 0.0:
                            ent1 = -(r1.clamp_min(1e-8) * r1.clamp_min(1e-8).log()).sum(-1, keepdim=True)
                            ent2 = -(r2.clamp_min(1e-8) * r2.clamp_min(1e-8).log()).sum(-1, keepdim=True)
                            x = torch.cat((x, ent1 * route_ent_scale), dim=-1)
                            y = torch.cat((y, ent2 * route_ent_scale), dim=-1)
                        sample_ot = torch.sum(criterion(alpha, x, beta, y))
                        OT_loss = (
                            sample_ot
                            + getattr(self.args, 'prototype_loss_coef', 0.1) * prototype_loss
                            + getattr(self.args, 'route_entropy_coef', 0.0) * route_entropy_loss
                        ) * self.DAcoef_args * warmup * OT_loss_scale
                    else:
                        OT_loss = 0
                    DA_losses.append(OT_loss)

            elif self.DA == 'CMMD':
                if (epoch>=self.DAepoch_args): 
                    sigma_x = compute_initial_bandwidth_full(x1, x2)
                    sigma_y = compute_initial_bandwidth_full(y1, y2)
            
                    cmmd = CMMDLoss(sigma_x=sigma_x, sigma_y=sigma_y)
                    DA_losses.append(cmmd(x1, y1, x2, y2)*self.DAcoef_args)
    
    
            elif self.DA == 'MMD':
                if (epoch>=self.DAepoch_args):
                    
                    sigma = compute_initial_bandwidth_full(x1, x2)
                    mmd = MMDLoss(sigma=sigma)
                    DA_losses.append(mmd(x1, x2)*self.DAcoef_args)
                
    
            elif self.DA == 'CORAL':
                if (epoch>=self.DAepoch_args):
                    coral = CORALLoss()
                    DA_losses.append(coral(x1, x2)*self.DAcoef_args)
            
                    
        task_loss = task_losses[0] + task_losses[1] + task_losses[2]
        if len(DA_losses) == 0:
            DA_loss = 0
        else:
            DA_loss = DA_losses[0] + DA_losses[1] + DA_losses[2]         
        
        router_loss = sum(router_regularisers) / max(len(router_regularisers), 1)
        model_loss += (task_loss + DA_loss + router_loss)

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

        print('self.projection_args', self.projection_args)


        if (self.projection_args == 0):
            self.main_optim.zero_grad()
            (model_loss).backward() #(task_loss + OT_loss).backward()
            self.main_optim.step()
        else:            
            
            self.main_optim.zero_grad()
            model_loss.backward(retain_graph=True)
            
            flat_A, mapping = flatten_and_pad_grads(self.model.parameters())
            
            self.main_optim.zero_grad()
            target_task_loss.backward(retain_graph=True)
            
            flat_B, _ = flatten_and_pad_grads(self.model.parameters())
            
            flat_proj_A = pcgrad_one_sided(flat_A, flat_B)
            
            write_back_flat_grads(flat_proj_A, mapping)
            
            self.main_optim.step()
            self.main_optim.zero_grad()
        
            self.current_epoch += 1
        
            
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
                    
                out, y, rep, y_unscale, normed_memory = self.model(data_i, stage='test') 
                
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

            print('Horizon {} : Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(i, total_mse, total_rmse, total_mae, total_mape))
            
        
        print("[INFO] Enter test phase")
        length = test_dataset.__len__()
       
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
    
