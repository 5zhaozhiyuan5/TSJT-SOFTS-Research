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
from horizon_transfer import ContinuousHorizonTransferAlignment, build_future_change_descriptors
from pattern_moe import HorizonPatternMoEAlignment
from experiment_diagnostics import ExperimentDiagnostics

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
    
            x = x.permute(0,2,1) # # [B, D, T] -> [B, T, D]
            if self.model_name == 'MLP':
                x = x.permute(0,2,1)
            
            rep, out = self.forecasting_model(x) 
    
            normed_memory = None
            
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
    def __init__(self, classifier_args, DAepoch_args, DAcoef_args, xySquaredRatio_args, memory_size_args, projection_args, model_name, DA, data_args, task_args, model_args,PatchFSL_cfg, args, pattern_bank=None):
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
        self.diagnostics = getattr(args, "diagnostics", None)
        if self.diagnostics is None:
            self.diagnostics = ExperimentDiagnostics(
                getattr(args, "save_dir", "."),
                metadata={
                    "seed": getattr(args, "seed", None),
                    "target_dataset": getattr(args, "test_dataset", None),
                    "source_datasets": getattr(args, "data_list", None),
                    "model_name": model_name,
                    "alignment": DA,
                    "diagnostic_scope": "Stage-I training and evaluation",
                },
            )

        self.lr = model_args['STnet']['lr']
        self.device = PatchFSL_cfg['device']
        self.current_epoch = 0

        self.model = PatchFSL(data_args, model_args, task_args, memory_size_args, projection_args, model_name, PatchFSL_cfg, args).to(self.device)

        # Stage-I only: Continuous Horizon-conditioned Transfer Alignment (CHTA).
        # This module consumes the existing backbone representation; SOFTS itself
        # and the official Stage-II implementation are not modified.
        chta_cfg = model_args.get('chta', {})
        self.chta_enabled = bool(chta_cfg.get('enabled', False))
        self.chta_loss_weight = float(chta_cfg.get('loss_weight', 0.1))
        self.chta_source_temperature = float(chta_cfg.get('source_temperature', 0.25))
        self.chta_smoothness_weight = float(chta_cfg.get('smoothness_weight', 1e-4))
        self.chta_target_loss_weight = float(chta_cfg.get('target_loss_weight', 0.1))
        self.chta = None
        if self.chta_enabled:
            self.chta = ContinuousHorizonTransferAlignment(
                input_dim=int(chta_cfg.get('input_dim', 128)),
                horizon_steps=int(chta_cfg.get('horizon_steps', task_args.get('source', {}).get('pred_num', 36))),
                minutes_per_step=float(chta_cfg.get('minutes_per_step', 5.0)),
                anchor_steps=tuple(chta_cfg.get('anchor_steps', [2, 12, 24, 36])),
                time_dim=int(chta_cfg.get('time_dim', 16)),
                hidden_dim=int(chta_cfg.get('hidden_dim', 64)),
                relation_dim=int(chta_cfg.get('relation_dim', 32)),
                epsilon=float(chta_cfg.get('epsilon', 0.08)),
                sinkhorn_iterations=int(chta_cfg.get('sinkhorn_iterations', 20)),
                max_samples=int(chta_cfg.get('max_samples', 128)),
                future_weight=float(chta_cfg.get('future_weight', 1.0)),
                prior_kl_weight=float(chta_cfg.get('prior_kl_weight', 1.0)),
            ).to(self.device)

        # MTPB-inspired multi-scale pattern bank + horizon-conditioned MoE.
        # It is a Stage-I alignment module only; the SOFTS forward interface and
        # PMME_2ndStage.py remain untouched.
        pattern_cfg = model_args.get('pattern_moe', {})
        self.pattern_moe_enabled = bool(pattern_cfg.get('enabled', False))
        if self.pattern_moe_enabled and self.chta_enabled:
            raise ValueError("Enable either pattern_moe or legacy chta, not both.")
        if self.pattern_moe_enabled and self.DA != 'OT':
            raise ValueError("pattern_moe currently requires DA='OT'.")
        self.pattern_moe = None
        self.pattern_moe_loss_weight = float(pattern_cfg.get('loss_weight', 1.0))
        self.pattern_moe_source_temperature = float(pattern_cfg.get('source_temperature', 0.25))
        self.pattern_moe_smoothness_weight = float(pattern_cfg.get('smoothness_weight', 1e-4))
        # Keep FiLM horizon-conditioning smoothness, but independently control
        # router smoothness so the ablation can remove only cross-horizon gate coupling.
        self.pattern_moe_router_smoothness_weight = float(
            pattern_cfg.get('router_smoothness_weight', 0.0)
        )
        self.pattern_moe_target_loss_weight = float(pattern_cfg.get('target_loss_weight', 0.1))
        self.pattern_moe_router_balance_weight = float(pattern_cfg.get('router_balance_weight', 0.1))
        if self.pattern_moe_enabled:
            if pattern_bank is None:
                raise ValueError(
                    "pattern_moe is enabled but no source-only pattern_bank was supplied. "
                    "Build it from source_dataset.x_list in train_DA_OT.py."
                )
            self.pattern_moe = HorizonPatternMoEAlignment(
                pattern_bank=pattern_bank,
                input_dim=int(pattern_cfg.get('input_dim', 128)),
                horizon_steps=int(pattern_cfg.get('horizon_steps', task_args.get('source', {}).get('pred_num', 36))),
                minutes_per_step=float(pattern_cfg.get('minutes_per_step', 5.0)),
                anchor_steps=tuple(pattern_cfg.get('anchor_steps', [2, 12, 24, 36])),
                scale_steps=tuple(pattern_cfg.get('scale_steps', [12, 36, 72, 144, 288])),
                time_dim=int(pattern_cfg.get('time_dim', 16)),
                hidden_dim=int(pattern_cfg.get('hidden_dim', 64)),
                relation_dim=int(pattern_cfg.get('relation_dim', 32)),
                epsilon=float(pattern_cfg.get('epsilon', 0.08)),
                sinkhorn_iterations=int(pattern_cfg.get('sinkhorn_iterations', 20)),
                max_samples=int(pattern_cfg.get('max_samples', 128)),
                pattern_temperature=float(pattern_cfg.get('pattern_temperature', 0.15)),
                router_temperature=float(pattern_cfg.get('router_temperature', 1.0)),
                future_weight=float(pattern_cfg.get('future_weight', 1.0)),
                pattern_weight=float(pattern_cfg.get('pattern_weight', 0.5)),
                prior_kl_weight=float(pattern_cfg.get('prior_kl_weight', 1.0)),
                router_min_usage=float(pattern_cfg.get('router_min_usage', 0.05)),
                router_mode=str(pattern_cfg.get('router_mode', 'learned')),
            ).to(self.device)

        self.optimized_parameters = list(self.model.parameters())
        if self.chta is not None:
            self.optimized_parameters.extend(list(self.chta.parameters()))
        if self.pattern_moe is not None:
            self.optimized_parameters.extend(list(self.pattern_moe.parameters()))

        for name, params in self.model.named_parameters():
            log_verbose(args,"{} : {}, require_grads : {}".format(name, params.shape,params.requires_grad))
        
        # print(self.model)
        log_verbose(args,"model params: {}".format(count_parameters(self.model)))

        if self.chta is not None:
            for name, params in self.chta.named_parameters():
                log_verbose(args, "chta.{} : {}, require_grads : {}".format(name, params.shape, params.requires_grad))
        if self.pattern_moe is not None:
            for name, params in self.pattern_moe.named_parameters():
                log_verbose(args, "pattern_moe.{} : {}, require_grads : {}".format(name, params.shape, params.requires_grad))
            log_verbose(args, "pattern bank shape: {}".format(tuple(self.pattern_moe.pattern_prototypes.shape)))
            self.diagnostics.log_pattern_bank(
                self.pattern_moe.pattern_prototypes,
                self.pattern_moe.scale_steps,
                city_names=str(getattr(args, "data_list", "")).split("_"),
            )
        self.main_optim = optim.AdamW(self.optimized_parameters, lr=self.lr, weight_decay=1e-2)
        
        self.loss_criterion = nn.SmoothL1Loss(reduction='mean', beta=5e-4) 
        
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
        
    
    
    def meta_train_revise(self, data, matrix, epoch): 
        epoch_started = time.perf_counter()
        
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        train_horizon_metrics = []
        
        task_losses = []
        rep_source = []
        rep_target = []
        y_source = []
        y_target = []
        source_last_history = []
        target_last_history = []
        source_history_windows = []
        target_history_windows = []

        target_task_loss = 0
        for i in range(len(data)):  
            task_loss = 0
            batch_size, node_num, seq_len, _ = data[i].x.shape
            
            out, y, rep, y_unscale, normed_memory = self.model(data[i], stage='train')

            # Reproduce the backbone wrapper's RevIN transform for the final
            # observed history value. This does not alter or wrap the SOFTS model.
            raw_history = torch.as_tensor(data[i].x, device=self.device, dtype=rep.dtype)
            raw_history = raw_history.permute(0, 1, 3, 2)[:, :, 0, -288:]
            if self.model_name == 'MLP_noRevIn':
                last_history = raw_history[:, :, -1]
            else:
                history_mean = raw_history.mean(dim=2, keepdim=True).detach()
                history_std = torch.sqrt(
                    raw_history.var(dim=2, keepdim=True, unbiased=False).detach() + 1e-9
                )
                last_history = ((raw_history[:, :, -1:] - history_mean) / history_std).squeeze(-1)

            if epoch%25==0:
                print(out.shape, rep.shape)
            
            if i == 0:
                
                split_indices = [15, 15, rep.shape[0] - 30]
                rep_split = torch.split(rep, split_indices, dim=0)
                y_unscale_split = torch.split(y_unscale, split_indices, dim=0)
                
                rep_target.extend(rep_split)
                y_target.extend(y_unscale_split)
                target_last_history.extend(torch.split(last_history, split_indices, dim=0))
                target_history_windows.extend(torch.split(raw_history, split_indices, dim=0))

            else:
                rep_source.append(rep)
                y_source.append(y_unscale)
                source_last_history.append(last_history)
                source_history_windows.append(raw_history)
                
            
            if i>0:   
                loss = self.loss_criterion(out, y)

                task_loss = loss

                # if(k == 0):
                MSE,RMSE,MAE,MAPE = calc_metric(out, y)
                if epoch%25==0:
                    print('MAPE',MAPE)
                total_mse.append(MSE.cpu().detach().numpy())
                total_rmse.append(RMSE.cpu().detach().numpy())
                total_mae.append(MAE.cpu().detach().numpy())
                total_mape.append(MAPE.cpu().detach().numpy())
                with torch.no_grad():
                    difference = out.detach() - y.detach()
                    per_horizon_mse = difference.square().mean(dim=(0, 1))
                    per_horizon_mae = difference.abs().mean(dim=(0, 1))
                    per_horizon_rmse = per_horizon_mse.sqrt()
                    per_horizon_mape = (
                        difference.abs() / y.detach().abs().clamp_min(1e-6)
                    ).mean(dim=(0, 1))
                    train_horizon_metrics.append((
                        per_horizon_mae.cpu().numpy(),
                        per_horizon_rmse.cpu().numpy(),
                        per_horizon_mse.cpu().numpy(),
                        per_horizon_mape.cpu().numpy(),
                    ))

                if i>0:
                    task_losses.append(task_loss)

            if i ==0 :
                loss = self.loss_criterion(out, y)
    
                target_task_loss = loss
            
            
        model_loss = 0 

        DA_loss = 0

        shape0_list = [207, 524, 627, 325]
        print(rep_target[0].shape[1]) # 325
        shape0_list.remove(rep_target[0].shape[1])
        print(shape0_list)

        DA_losses = []
        CHTA_horizon_losses = []
        CHTA_source_scores = []
        CHTA_diagnostics = []
        PatternMoE_horizon_losses = []
        PatternMoE_source_scores = []
        PatternMoE_router_smoothness_losses = []
        PatternMoE_router_balance_losses = []
        PatternMoE_diagnostics = []
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
            shape0 = x1.shape[1]

            source_horizon_desc, target_horizon_desc = None, None
            alignment_module = self.pattern_moe if self.pattern_moe is not None else self.chta
            if alignment_module is not None and self.DA == 'OT' and epoch >= self.DAepoch_args:
                anchor_steps = alignment_module.anchor_steps
                source_horizon_desc = build_future_change_descriptors(
                    y1, source_last_history[index], anchor_steps
                ).reshape(-1, len(anchor_steps), 4)
                target_horizon_desc = build_future_change_descriptors(
                    y2, target_last_history[index], anchor_steps
                ).reshape(-1, len(anchor_steps), 4)

            y_norm_sources.append(y1)
            y_norm_targets.append(y2)

            x1 = x1.view(-1, x1.size(-1))
            x2 = x2.view(-1, x2.size(-1))
            y1 = y1.view(-1, y1.size(-1))
            y2 = y2.view(-1, y2.size(-1))
            

            base_prob1 = torch.full((size1,), 1.0 / max(size1, 1), device=self.device)
            base_prob2 = torch.full((size2,), 1.0 / max(size2, 1), device=self.device)
            has_overlap = True

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
                mask1 = prob1 > thresh
                mask2 = prob2 > thresh
                count1 = torch.sum(mask1).detach().cpu().numpy()
                count2 = torch.sum(mask2).detach().cpu().numpy()
                ratio1 = count1/len(mask1)
                ratio2 = count2/len(mask2)
                
                if ratio1>0.00 and ratio2>0.00:
                    if epoch%25==0:
                        print('ratio1, ratio2', ratio1, ratio2)
                    def threshold(x):
                        return torch.where(x > thresh, x, torch.tensor(0.0, device=x.device))

                    base_prob1 = threshold(prob1).detach()
                    base_prob2 = threshold(prob2).detach()
                    alpha = (base_prob1 / torch.sum(base_prob1)).detach()
                    beta = (base_prob2 / torch.sum(base_prob2)).detach()
                    OT_loss_scale = torch.sqrt((base_prob1.mean() * base_prob2.mean()).detach())
                    if epoch%25==0:
                        print('OT_loss_scale', OT_loss_scale)
                else:
                    # Keep the batch numerically valid, but do not force an OT
                    # update when DVGP finds no overlapping samples.
                    has_overlap = False
                    alpha = torch.ones(size1, device=self.device) / max(size1, 1)
                    beta = torch.ones(size2, device=self.device) / max(size2, 1)
                    OT_loss_scale = 0.0

            else:
                alpha = torch.ones(size1, device=self.device) / max(size1, 1)
                beta = torch.ones(size2, device=self.device) / max(size2, 1)
                base_prob1 = alpha.detach()
                base_prob2 = beta.detach()
                OT_loss_scale = 1.0

            if self.DA == 'OT':
                if (epoch>=self.DAepoch_args):
                    x_y_scale = math.sqrt(self.xySquaredRatio_args)
                    y1, y2 = y1*x_y_scale, y2*x_y_scale

                    if self.pattern_moe is not None:
                        # The horizon-conditioned MoE-OT replaces the old
                        # full-sequence GeomLoss OT as the Stage-I main alignment.
                        # DVGP still supplies the base sample reliability weights.
                        if (
                            has_overlap
                            and source_horizon_desc is not None
                            and target_horizon_desc is not None
                        ):
                            source_h_marginals, target_h_marginals, source_h_scores = (
                                self.pattern_moe.horizon_marginals(
                                    base_prob1,
                                    base_prob2,
                                    source_horizon_desc,
                                    target_horizon_desc,
                                )
                            )
                            source_history = source_history_windows[index].reshape(
                                -1, source_history_windows[index].shape[-1]
                            )
                            target_history = target_history_windows[index].reshape(
                                -1, target_history_windows[index].shape[-1]
                            )
                            _, conditional_stats = self.pattern_moe.compute_alignment_loss(
                                x1,
                                x2,
                                source_history,
                                target_history,
                                source_horizon_desc,
                                target_horizon_desc,
                                source_h_marginals,
                                target_h_marginals,
                                base_prob1,
                                base_prob2,
                            )
                            PatternMoE_horizon_losses.append(
                                conditional_stats["horizon_losses"] * OT_loss_scale
                            )
                            PatternMoE_source_scores.append(source_h_scores)
                            PatternMoE_router_smoothness_losses.append(
                                conditional_stats["router_smoothness_loss"]
                            )
                            PatternMoE_router_balance_losses.append(
                                conditional_stats["router_balance_loss"]
                            )
                            PatternMoE_diagnostics.append(conditional_stats)
                    else:
                        # Legacy PMME/CHTA path retained for controlled ablations.
                        x = torch.cat([x1, y1], dim=-1)
                        y = torch.cat([x2, y2], dim=-1)
                        if has_overlap:
                            OT_loss = (
                                torch.sum(criterion(alpha, x, beta, y))
                                * self.DAcoef_args
                                * OT_loss_scale
                            )
                        else:
                            OT_loss = 0.0
                        DA_losses.append(OT_loss)

                        if (
                            self.chta is not None
                            and has_overlap
                            and source_horizon_desc is not None
                            and target_horizon_desc is not None
                        ):
                            source_h_marginals, target_h_marginals, source_h_scores = (
                                self.chta.horizon_marginals(
                                    base_prob1, base_prob2,
                                    source_horizon_desc, target_horizon_desc
                                )
                            )
                            _, conditional_stats = self.chta.compute_alignment_loss(
                                x1,
                                x2,
                                source_horizon_desc,
                                target_horizon_desc,
                                source_h_marginals,
                                target_h_marginals,
                            )
                            CHTA_horizon_losses.append(
                                conditional_stats["horizon_losses"] * OT_loss_scale
                            )
                            CHTA_source_scores.append(source_h_scores)
                            CHTA_diagnostics.append(conditional_stats)

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
            
                    
        task_loss = sum(task_losses)
        DA_loss = sum(DA_losses) if DA_losses else task_loss.new_zeros(())
        if CHTA_horizon_losses:
            source_horizon_loss = torch.stack(CHTA_horizon_losses, dim=0)
            source_horizon_score = torch.stack(CHTA_source_scores, dim=0)
            # Normalize source-city compatibility independently at each
            # physical horizon, rather than summing the three source domains
            # with one fixed coefficient.
            source_city_weights = torch.softmax(
                source_horizon_score / max(self.chta_source_temperature, 1e-6), dim=0
            )
            CHTA_loss = (
                source_city_weights * source_horizon_loss
            ).sum(dim=0).mean() * self.chta_loss_weight * self.DAcoef_args
        else:
            source_city_weights = None
            CHTA_loss = task_loss.new_zeros(())

        target_loss_weight = 0.0
        if PatternMoE_horizon_losses:
            pattern_source_losses = torch.stack(PatternMoE_horizon_losses, dim=0)
            pattern_source_scores = torch.stack(PatternMoE_source_scores, dim=0)
            pattern_source_weights = torch.softmax(
                pattern_source_scores / max(self.pattern_moe_source_temperature, 1e-6),
                dim=0,
            )
            PatternMoE_loss = (
                pattern_source_weights * pattern_source_losses
            ).sum(dim=0).mean() * self.pattern_moe_loss_weight * self.DAcoef_args
        else:
            pattern_source_weights = None
            PatternMoE_loss = task_loss.new_zeros(())

        if self.pattern_moe is not None and self.DA == 'OT' and epoch >= self.DAepoch_args:
            # FiLM smoothness is retained. Router smoothness is a separate,
            # independently switchable term for the NoRouterSmooth ablation.
            PatternMoE_smoothness = (
                self.pattern_moe.smoothness_loss() * self.pattern_moe_smoothness_weight
            )
            if PatternMoE_router_smoothness_losses:
                PatternMoE_router_smoothness = (
                    torch.stack(PatternMoE_router_smoothness_losses).mean()
                    * self.pattern_moe_router_smoothness_weight
                )
            else:
                PatternMoE_router_smoothness = task_loss.new_zeros(())
            if PatternMoE_router_balance_losses:
                PatternMoE_router_balance = (
                    torch.stack(PatternMoE_router_balance_losses).mean()
                    * self.pattern_moe_router_balance_weight
                )
            else:
                PatternMoE_router_balance = task_loss.new_zeros(())
            target_loss_weight = self.pattern_moe_target_loss_weight
        else:
            PatternMoE_smoothness = task_loss.new_zeros(())
            PatternMoE_router_smoothness = task_loss.new_zeros(())
            PatternMoE_router_balance = task_loss.new_zeros(())

        if self.chta is not None and self.DA == 'OT' and epoch >= self.DAepoch_args:
            CHTA_smoothness = self.chta.smoothness_loss()
            CHTA_smoothness = self.chta_smoothness_weight * CHTA_smoothness
            target_loss_weight = self.chta_target_loss_weight
        else:
            CHTA_smoothness = task_loss.new_zeros(())
            if self.pattern_moe is None:
                target_loss_weight = 0.0

        # The few-shot target loss is now an explicit, low-weight Stage-I
        # objective.  In the projected-gradient path it remains the reference
        # direction and is also added back with target_loss_weight.
        transfer_loss = (
            task_loss + DA_loss + CHTA_loss + CHTA_smoothness
            + PatternMoE_loss + PatternMoE_smoothness
            + PatternMoE_router_smoothness + PatternMoE_router_balance
        )
        model_loss = transfer_loss + target_loss_weight * target_task_loss

        # Compare the conditional-OT gradient to the forecast gradient on
        # the SAME SOFTS parameters. This is a diagnostic only: no gradients
        # are written to .grad and the optimizer/projection path is untouched.
        # The OT gradient includes history/future costs in both experiment
        # arms; it must NOT be interpreted as a pattern-only gradient.
        if (
            self.pattern_moe is not None
            and epoch >= self.DAepoch_args
            and epoch % 100 == 0
            and PatternMoE_horizon_losses
            and PatternMoE_loss.requires_grad
        ):
            backbone_parameters = [
                parameter for parameter in self.model.parameters()
                if parameter.requires_grad
            ]
            forecast_gradients = torch.autograd.grad(
                task_loss, backbone_parameters, retain_graph=True, allow_unused=True
            )
            alignment_gradients = torch.autograd.grad(
                PatternMoE_loss, backbone_parameters, retain_graph=True, allow_unused=True
            )
            with torch.no_grad():
                forecast_sq = task_loss.detach().new_zeros(())
                alignment_sq = task_loss.detach().new_zeros(())
                dot_product = task_loss.detach().new_zeros(())
                for forecast_grad, alignment_grad in zip(
                    forecast_gradients, alignment_gradients
                ):
                    if forecast_grad is not None:
                        forecast_sq += forecast_grad.float().square().sum()
                    if alignment_grad is not None:
                        alignment_sq += alignment_grad.float().square().sum()
                    if forecast_grad is not None and alignment_grad is not None:
                        dot_product += (
                            forecast_grad.float() * alignment_grad.float()
                        ).sum()
                forecast_norm = forecast_sq.sqrt()
                alignment_norm = alignment_sq.sqrt()
                gradient_cosine = dot_product / (
                    forecast_norm * alignment_norm
                ).clamp_min(1e-12)
                self.diagnostics.log_alignment_gradient(
                    epoch,
                    alignment_norm,
                    forecast_norm,
                    gradient_cosine,
                    alignment_norm / forecast_norm.clamp_min(1e-12),
                    self.pattern_moe.pattern_weight,
                )
            del forecast_gradients, alignment_gradients

        if epoch % 25 == 0 and CHTA_diagnostics:
            mean_diversity = torch.stack([d["plan_diversity"] for d in CHTA_diagnostics]).mean()
            max_row_error = torch.stack([d["max_row_marginal_error"] for d in CHTA_diagnostics]).max()
            max_col_error = torch.stack([d["max_col_marginal_error"] for d in CHTA_diagnostics]).max()
            print(
                "CHTA_loss", float(CHTA_loss.detach()),
                "CHTA_plan_diversity", float(mean_diversity),
                "CHTA_row_marginal_error", float(max_row_error),
                "CHTA_col_marginal_error", float(max_col_error),
                "CHTA_source_city_weights",
                source_city_weights.detach().cpu().numpy().round(4).tolist()
                if source_city_weights is not None else None,
            )

        if epoch % 25 == 0 and PatternMoE_diagnostics:
            router = torch.stack([d["router_weights"] for d in PatternMoE_diagnostics], dim=0)
            entropy = torch.stack([d["router_entropy"] for d in PatternMoE_diagnostics]).mean()
            expert_costs = torch.stack([d["expert_cost_means"] for d in PatternMoE_diagnostics]).mean(dim=0)
            diversity = torch.stack([d["plan_diversity"] for d in PatternMoE_diagnostics]).mean()
            row_error = torch.stack([d["max_row_marginal_error"] for d in PatternMoE_diagnostics]).max()
            col_error = torch.stack([d["max_col_marginal_error"] for d in PatternMoE_diagnostics]).max()
            router_smoothness = torch.stack(
                [d["router_smoothness_loss"].detach() for d in PatternMoE_diagnostics]
            ).mean()
            print(
                "PatternMoE_loss", float(PatternMoE_loss.detach()),
                "PatternMoE_smoothness", float(PatternMoE_smoothness.detach()),
                "PatternMoE_router_smoothness", float(PatternMoE_router_smoothness.detach()),
                "PatternMoE_router_smoothness_weight", self.pattern_moe_router_smoothness_weight,
                "PatternMoE_router_balance", float(PatternMoE_router_balance.detach()),
                "PatternMoE_router_by_horizon",
                router.mean(dim=0).detach().cpu().numpy().round(4).tolist(),
                "PatternMoE_router_entropy", float(entropy),
                "PatternMoE_expert_costs", expert_costs.cpu().numpy().round(4).tolist(),
                "PatternMoE_router_smoothness", float(router_smoothness),
                "PatternMoE_plan_diversity", float(diversity),
                "PatternMoE_row_marginal_error", float(row_error),
                "PatternMoE_col_marginal_error", float(col_error),
                "PatternMoE_source_city_weights",
                pattern_source_weights.detach().cpu().numpy().round(4).tolist()
                if pattern_source_weights is not None else None,
            )

        for source_index, diagnostic in enumerate(PatternMoE_diagnostics):
            self.diagnostics.log_alignment(
                epoch,
                source_index,
                diagnostic,
                self.pattern_moe.anchor_steps,
                self.pattern_moe.minutes_per_step,
            )
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

        diagnostic_modules = {"backbone": self.model}
        if self.pattern_moe is not None:
            diagnostic_modules["pattern_moe"] = self.pattern_moe
        if self.chta is not None:
            diagnostic_modules["chta"] = self.chta

        def parameter_norm(module):
            squared = [p.detach().float().pow(2).sum() for p in module.parameters()]
            return torch.sqrt(torch.stack(squared).sum()) if squared else torch.zeros(())

        parameter_norm_before = {
            name: parameter_norm(module) for name, module in diagnostic_modules.items()
        }

        def parameter_sample(module, limit=4096):
            chunks = []
            remaining = limit
            for parameter in module.parameters():
                if remaining <= 0:
                    break
                flat = parameter.detach().reshape(-1)
                take = min(remaining, flat.numel())
                if take:
                    chunks.append(flat[:take].float().clone())
                    remaining -= take
            return torch.cat(chunks) if chunks else torch.zeros(1, device=self.device)

        parameter_samples_before = {
            name: parameter_sample(module) for name, module in diagnostic_modules.items()
        }

        def gradient_summary(module):
            params = list(module.parameters())
            total = max(len(params), 1)
            grad_sq, nonfinite_counts, none_count = [], [], 0
            for parameter in params:
                if parameter.grad is None:
                    none_count += 1
                    continue
                grad = parameter.grad.detach().float()
                nonfinite_counts.append((~torch.isfinite(grad)).sum())
                grad_sq.append(torch.nan_to_num(grad).pow(2).sum())
            grad_norm = torch.sqrt(torch.stack(grad_sq).sum()) if grad_sq else torch.zeros(())
            nonfinite_count = int(torch.stack(nonfinite_counts).sum().item()) if nonfinite_counts else 0
            return grad_norm, none_count / total, nonfinite_count

        if (self.projection_args == 0):
            self.main_optim.zero_grad()
            model_loss.backward()
            grad_summaries = {
                name: gradient_summary(module) for name, module in diagnostic_modules.items()
            }
            self.main_optim.step()
        else:
            self.main_optim.zero_grad()
            transfer_loss.backward(retain_graph=True)
            flat_A, mapping = flatten_and_pad_grads(self.optimized_parameters)

            self.main_optim.zero_grad()
            target_task_loss.backward(retain_graph=True)
            flat_B, _ = flatten_and_pad_grads(self.optimized_parameters)

            flat_proj_A = pcgrad_one_sided(flat_A, flat_B)
            flat_final = flat_proj_A + target_loss_weight * flat_B
            write_back_flat_grads(flat_final, mapping)
            grad_summaries = {
                name: gradient_summary(module) for name, module in diagnostic_modules.items()
            }

            self.main_optim.step()
            self.main_optim.zero_grad()

        for name, module in diagnostic_modules.items():
            grad_norm, none_fraction, nonfinite_count = grad_summaries[name]
            self.diagnostics.log_gradient(
                epoch, name, grad_norm, parameter_norm_before[name], none_fraction,
                nonfinite_count, parameter_norm(module),
                parameter_delta_norm=torch.norm(
                    parameter_sample(module) - parameter_samples_before[name]
                ),
            )

        def _mean_metric(values):
            return float(np.nanmean(np.asarray(values, dtype=np.float64))) if values else float("nan")

        epoch_mae = _mean_metric(total_mae)
        epoch_rmse = _mean_metric(total_rmse)
        epoch_mse = _mean_metric(total_mse)
        epoch_mape = _mean_metric(total_mape)
        self.diagnostics.log_epoch({
            "epoch": epoch,
            "total_loss": model_loss,
            "forecast_loss": task_loss,
            "target_loss": target_task_loss,
            "generic_da_loss": DA_loss,
            "chta_loss": CHTA_loss,
            "pattern_moe_loss": PatternMoE_loss,
            "pattern_moe_smoothness": PatternMoE_smoothness,
            "pattern_moe_router_smoothness": PatternMoE_router_smoothness,
            "pattern_moe_router_smoothness_weight": self.pattern_moe_router_smoothness_weight,
            "pattern_moe_router_balance": PatternMoE_router_balance,
            "mae_mean": epoch_mae,
            "rmse_mean": epoch_rmse,
            "mse_mean": epoch_mse,
            "mape_mean": epoch_mape,
            "source_city_weights": pattern_source_weights if pattern_source_weights is not None else [],
            "finite_total_loss": bool(torch.isfinite(model_loss.detach()).all()),
        })
        if train_horizon_metrics:
            horizon_mae = np.stack([item[0] for item in train_horizon_metrics], axis=0).mean(axis=0)
            horizon_rmse = np.stack([item[1] for item in train_horizon_metrics], axis=0).mean(axis=0)
            horizon_mse = np.stack([item[2] for item in train_horizon_metrics], axis=0).mean(axis=0)
            horizon_mape = np.stack([item[3] for item in train_horizon_metrics], axis=0).mean(axis=0)
            minutes_per_step = (
                self.pattern_moe.minutes_per_step if self.pattern_moe is not None
                else (self.chta.minutes_per_step if self.chta is not None else 5.0)
            )
            self.diagnostics.log_horizon(
                epoch, len(horizon_mae), minutes_per_step,
                horizon_mae, horizon_rmse, horizon_mse, horizon_mape,
            )
        optimizer_lr = self.main_optim.param_groups[0]["lr"] if self.main_optim.param_groups else self.lr
        self.diagnostics.log_runtime(
            epoch, time.perf_counter() - epoch_started, optimizer_lr, self.device
        )
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
    
        phase_name = "target_few_shot_train_eval"
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
            minutes_per_step = self.pattern_moe.minutes_per_step if self.pattern_moe is not None else (self.chta.minutes_per_step if self.chta is not None else 5.0)
            self.diagnostics.log_evaluation(
                phase_name, self.current_epoch, i, minutes_per_step,
                total_mae, total_rmse, total_mse, total_mape, total_len,
            )
            minutes_per_step = self.pattern_moe.minutes_per_step if self.pattern_moe is not None else (self.chta.minutes_per_step if self.chta is not None else 5.0)
            self.diagnostics.log_evaluation(
                phase_name, self.current_epoch, i, minutes_per_step,
                total_mae, total_rmse, total_mse, total_mape, total_len,
            )
            
        
        print("[INFO] Enter test phase")
        length = test_dataset.__len__()
       
        print('length', length) # 265
        total_mse_horizon,total_rmse_horizon, total_mae_horizon, total_mape_horizon, total_loss = self.test_batch(0,length+1, test_dataset,"test")   
        
        phase_name = "test"
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
    