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

from geomloss import SamplesLoss 
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.likelihoods import BernoulliLikelihood, LaplaceLikelihood

from SOFTS.SOFTS_new import Model as softs
from iTransformer.iTransformer_new import Model as iTransformer

from DA_tools import *
from stage1_transfer import SharedPrivateReliabilityTransfer
from rhsr_transfer import RegimeHorizonSourceRouter

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
    def __init__(self, classifier_args, DAepoch_args, DAcoef_args, xySquaredRatio_args, memory_size_args, projection_args, model_name, DA, data_args, task_args, model_args,PatchFSL_cfg, args):
        super(STRep, self).__init__()
        self.classifier_args = classifier_args
        self.DAepoch_args = DAepoch_args
        self.DAcoef_args = DAcoef_args
        self.xySquaredRatio_args = xySquaredRatio_args
        self.projection_args = projection_args 
        self.memory_size_args = memory_size_args 
        self.stage1_method = str(
            getattr(args, 'stage1_method', 'pmme')
        ).lower()

        # Stage-I transport safeguards. The domain classifier is useful for
        # ranking samples, but its probabilities must not become a hard gate:
        # an over-confident classifier can otherwise leave OT with only a few
        # samples from each city.
        self.stage1_mass_temperature = max(
            float(getattr(args, 'stage1_mass_temperature', 2.0)), 1.0
        )
        self.stage1_uniform_mass = min(
            max(float(getattr(args, 'stage1_uniform_mass', 0.75)), 0.0), 1.0
        )
        self.stage1_reliability_mix = min(
            max(float(getattr(args, 'stage1_reliability_mix', 0.25)), 0.0), 1.0
        )
        self.stage1_transfer_enabled = (
            self.stage1_method != 'rhsr'
            and bool(int(getattr(args, 'stage1_transfer', 1)))
        )
        self.DAwarmup_args = max(int(getattr(args, 'DAwarmup', 500)), 0)
        
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

        # Stage-I only: decompose latent representations into transferable and
        # city-specific components. The module starts as an identity partition.
        # SOFTS exposes a 128-dimensional latent representation per node.
        # The transfer module operates on that representation, not on the
        # 36-step forecast horizon.
        self.stage1_transfer = None
        self.rhsr_transfer = None
        if self.stage1_method == 'rhsr':
            self.rhsr_transfer = RegimeHorizonSourceRouter(
                input_dim=128,
                horizon=int(task_args['source']['pred_num']),
                num_regimes=int(getattr(args, 'rhsr_num_regimes', 4)),
                hidden_dim=int(getattr(args, 'rhsr_hidden_dim', 96)),
                route_temperature=float(getattr(args, 'rhsr_route_temperature', 1.5)),
                router_uniform_mix=float(
                    getattr(args, 'rhsr_router_uniform_mix', 0.0)
                ),
                route_logit_clip=float(
                    getattr(args, 'rhsr_route_logit_clip', 0.0)
                ),
            ).to(self.device)
        elif self.stage1_transfer_enabled:
            self.stage1_transfer = SharedPrivateReliabilityTransfer(
                input_dim=128,
                reliability_temperature=self.stage1_mass_temperature,
                reliability_uniform_mix=self.stage1_uniform_mass,
            ).to(self.device)
        stage1_parameters = list(self.model.parameters())
        if self.stage1_transfer is not None:
            stage1_parameters += list(self.stage1_transfer.parameters())
        if self.rhsr_transfer is not None:
            stage1_parameters += list(self.rhsr_transfer.parameters())
        self.main_optim = optim.AdamW(stage1_parameters, lr=self.lr, weight_decay=1e-2)
        
        self.loss_criterion = nn.SmoothL1Loss(reduction='mean', beta=5e-4) 
        
        input_dim = 36
        
        num_inducing = 256
        if self.stage1_method == 'rhsr':
            self.gp1 = None
            self.gp2 = None
            self.gp3 = None
            self.likelihood1 = None
            self.likelihood2 = None
            self.likelihood3 = None
            self.optimizer1 = None
            self.optimizer2 = None
            self.optimizer3 = None
        else:
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

    def _stage1_da_ramp(self, epoch):
        """Gradually enable Stage-I distribution alignment after DAepoch."""
        if epoch < self.DAepoch_args:
            return 0.0
        if self.DAwarmup_args <= 0:
            return 1.0
        return min(
            1.0,
            float(epoch - self.DAepoch_args + 1) / float(self.DAwarmup_args),
        )

    def meta_train_rhsr(self, data, matrix, epoch):
        """Train the non-PMME Regime-Horizon Source Routing method.

        RHSR deliberately replaces the PMME DVGP/conditional-OT path. It
        learns source weights directly from target-conditioned traffic-state
        summaries and applies them to horizon-wise supervised source losses.
        """
        if self.rhsr_transfer is None:
            raise RuntimeError("RHSR transfer module is not initialized")

        target_rep = None
        target_y = None
        target_task_loss = None
        source_records = []
        total_mse = []
        total_rmse = []
        total_mae = []
        total_mape = []

        for index, data_i in enumerate(data):
            out, y, rep, y_unscale, normed_memory = self.model(
                data_i, stage='train'
            )
            if index == 0:
                target_rep = rep.reshape(-1, rep.shape[-1])
                target_y = y.reshape(-1, y.shape[-1])
                target_task_loss = self.loss_criterion(out, y)
                continue

            source_records.append({
                'out': out,
                'y': y,
                'rep': rep.reshape(-1, rep.shape[-1]),
                'forecast': y.reshape(-1, y.shape[-1]),
            })
            mse, rmse, mae, mape = calc_metric(out, y)
            total_mse.append(mse.detach().cpu().numpy())
            total_rmse.append(rmse.detach().cpu().numpy())
            total_mae.append(mae.detach().cpu().numpy())
            total_mape.append(mape.detach().cpu().numpy())

        if not source_records:
            raise RuntimeError("RHSR requires at least one source task")

        target_encoded = self.rhsr_transfer.encode(target_rep, target_y)
        source_encoded = [
            self.rhsr_transfer.encode(record['rep'], record['forecast'])
            for record in source_records
        ]
        source_summaries = [item['summary'] for item in source_encoded]
        source_weights, route_logits = self.rhsr_transfer.route_weights(
            target_encoded['summary'], source_summaries
        )

        horizon_losses = []
        for record in source_records:
            pointwise = F.smooth_l1_loss(
                record['out'], record['y'], reduction='none'
            )
            horizon_losses.append(pointwise.mean(dim=(0, 1)))
        horizon_losses = torch.stack(horizon_losses, dim=0)
        source_task_loss = len(source_records) * (
            source_weights * horizon_losses
        ).mean()

        target_shared_mean = target_encoded['shared'].mean(dim=0)
        target_regime = target_encoded['regime_prob'].mean(dim=0)
        state_losses = []
        decomposition_losses = [target_encoded['decomposition_loss']]
        for encoded in source_encoded:
            source_shared_mean = encoded['shared'].mean(dim=0)
            shared_match = 1.0 - F.cosine_similarity(
                source_shared_mean.unsqueeze(0),
                target_shared_mean.unsqueeze(0),
                dim=-1,
            ).mean()
            source_regime = encoded['regime_prob'].mean(dim=0)
            regime_match = self.rhsr_transfer.symmetric_kl(
                source_regime, target_regime
            )
            state_losses.append(shared_match + regime_match)
            decomposition_losses.append(encoded['decomposition_loss'])

        state_consistency = torch.stack(state_losses).mean()
        decomposition_loss = torch.stack(decomposition_losses).mean()
        route_prob = source_weights.clamp_min(1e-8)
        route_entropy = -(route_prob * route_prob.log()).sum(dim=0).mean()
        route_balance_loss = -route_entropy

        target_anchor = float(getattr(self.args, 'rhsr_target_anchor', 0.1))
        state_weight = float(getattr(self.args, 'rhsr_state_weight', 0.1))
        decomposition_weight = float(
            getattr(self.args, 'rhsr_decomposition_weight', 0.1)
        )
        route_balance_weight = float(
            getattr(self.args, 'rhsr_route_balance_weight', 0.01)
        )
        model_loss = (
            source_task_loss
            + target_anchor * target_task_loss
            + state_weight * state_consistency
            + decomposition_weight * decomposition_loss
            + route_balance_weight * route_balance_loss
        )

        self.main_optim.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.rhsr_transfer.parameters()),
            max_norm=5.0,
        )
        self.main_optim.step()

        with torch.no_grad():
            mean_source_weights = source_weights.mean(dim=1)
            regime_hist = target_regime.detach().cpu().tolist()
            self.last_rhsr_diagnostics = {
                'source_weights': mean_source_weights.detach().cpu().tolist(),
                'source_horizon_weights': source_weights.detach().cpu().tolist(),
                'route_entropy': float(route_entropy.detach().cpu()),
                'state_consistency': float(state_consistency.detach().cpu()),
                'decomposition_loss': float(decomposition_loss.detach().cpu()),
                'target_regime_histogram': regime_hist,
                'route_logits_mean': route_logits.detach().mean(dim=1).cpu().tolist(),
            }

        return (
            model_loss.detach().cpu().numpy(),
            np.mean(total_mse),
            np.mean(total_rmse),
            np.mean(total_mae),
            np.mean(total_mape),
        )
        
    
    
    def meta_train_revise(self, data, matrix, epoch): 
        if self.stage1_method == 'rhsr':
            return self.meta_train_rhsr(data, matrix, epoch)
        
        total_mae = []
        total_mse = []
        total_rmse = []
        total_mape = []
        
        task_losses = []
        rep_source = []
        rep_target = []
        y_source = []
        y_target = []
        decomposition_losses = []

        target_task_loss = 0
        for i in range(len(data)):  
            task_loss = 0
            batch_size, node_num, seq_len, _ = data[i].x.shape
            
            out, y, rep, y_unscale, normed_memory = self.model(data[i], stage='train') 
            
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
                y_source.append(y_unscale)
                
            
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

                if i>0:
                    task_losses.append(task_loss)

            if i ==0 :
                loss = self.loss_criterion(out, y)
    
                target_task_loss = loss
            
        if not self.stage1_transfer_enabled:
            model_loss = sum(task_losses)
            self.main_optim.zero_grad()
            model_loss.backward()
            self.main_optim.step()
            return (
                model_loss.detach().cpu().numpy(),
                np.mean(total_mse),
                np.mean(total_rmse),
                np.mean(total_mae),
                np.mean(total_mape),
            )

            
        model_loss = 0 

        DA_loss = 0

        shape0_list = [207, 524, 627, 325]
        print(rep_target[0].shape[1]) # 325
        shape0_list.remove(rep_target[0].shape[1])
        print(shape0_list)

        DA_losses = []
        source_quality_scores = []
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
                    
            y_norm_sources.append(y1)
            y_norm_targets.append(y2)

            x1 = x1.view(-1, x1.size(-1))
            x2 = x2.view(-1, x2.size(-1))
            y1 = y1.view(-1, y1.size(-1))
            y2 = y2.view(-1, y2.size(-1))

            # Stage-I representation disentanglement: only the shared branch
            # participates in cross-city alignment; private information is
            # retained through reconstruction and orthogonality constraints.
            shared_x1, private_x1, reconstructed_x1, decomp1 = self.stage1_transfer(x1)
            shared_x2, private_x2, reconstructed_x2, decomp2 = self.stage1_transfer(x2)
            decomposition_losses.append(0.5 * (decomp1 + decomp2))

            # Reliability is estimated with cross-domain standardized centroid
            # distance (O(ND), not an N_source x N_target pairwise matrix).
            reliability_source, reliability_target = self.stage1_transfer.reliability_weights(
                shared_x1.detach(), shared_x2.detach())
            # Source-level transfer confidence complements sample-level weights.
            # It is detached so the model cannot game the gate by shrinking features.
            source_quality_scores.append(
                self.stage1_transfer.source_quality(shared_x1.detach(), shared_x2.detach())
            )
            alpha = torch.full((size1,), 1.0 / max(size1, 1), device=self.device)
            beta = torch.full((size2,), 1.0 / max(size2, 1), device=self.device)
            

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

                if epoch%25==0:
                    print('ratio1, ratio2', ratio1, ratio2)

                # Do not threshold probabilities into zero/non-zero masses.
                # Smooth the confidence and retain a large uniform component so
                # that a near-perfect domain classifier cannot collapse OT to a
                # few samples.
                alpha = self.stage1_transfer.stabilize_mass(
                    prob1,
                    temperature=self.stage1_mass_temperature,
                    uniform_mix=self.stage1_uniform_mass,
                )
                beta = self.stage1_transfer.stabilize_mass(
                    prob2,
                    temperature=self.stage1_mass_temperature,
                    uniform_mix=self.stage1_uniform_mass,
                )

                OT_loss_scale = torch.sqrt(
                    (prob1.mean() * prob2.mean()).clamp_min(1e-8)
                ).detach()

                if epoch%25==0:
                    print('OT_loss_scale', OT_loss_scale)
            
            else:
                alpha = torch.ones(size1, device=self.device)/size1  
                beta = torch.ones(size2, device=self.device)/size2  
                OT_loss_scale = 1

            # Reliability is a weak geometric prior. Blend it with the
            # classifier mass instead of multiplying two gates, which would
            # compound their concentration and discard most samples.
            reliability_mix = self.stage1_reliability_mix
            alpha = (1.0 - reliability_mix) * alpha + reliability_mix * reliability_source
            beta = (1.0 - reliability_mix) * beta + reliability_mix * reliability_target
            alpha = alpha / alpha.sum().clamp_min(1e-8)
            beta = beta / beta.sum().clamp_min(1e-8)

            if epoch % 25 == 0:
                source_eff = 1.0 / alpha.pow(2).sum().clamp_min(1e-8)
                target_eff = 1.0 / beta.pow(2).sum().clamp_min(1e-8)
                print(
                    "[Stage-I][SharedPrivate] decomp={:.6f}, "
                    "source_reliability_eff={:.1f}/{}, target_reliability_eff={:.1f}/{}".format(
                        float(decomposition_losses[-1].detach().cpu()),
                        float(source_eff.detach().cpu()), size1,
                        float(target_eff.detach().cpu()), size2
                    )
                )

            da_ramp = self._stage1_da_ramp(epoch)
            if self.DA == 'OT':
                if (da_ramp > 0.0):
                    
                    x_y_scale = math.sqrt(self.xySquaredRatio_args)
                    y1, y2 = y1*x_y_scale, y2*x_y_scale
                    
                    x = torch.cat((shared_x1, y1), dim=-1)
                    y = torch.cat((shared_x2, y2), dim=-1)  # target shared-pattern space
                    
                    OT_loss = (
                        torch.sum(criterion(alpha, x, beta, y))
                        * self.DAcoef_args
                        * OT_loss_scale
                        * da_ramp
                    )
                    DA_losses.append(OT_loss)

            elif self.DA == 'CMMD':
                if (da_ramp > 0.0):
                    sigma_x = compute_initial_bandwidth_full(shared_x1, shared_x2)
                    sigma_y = compute_initial_bandwidth_full(y1, y2)
            
                    cmmd = CMMDLoss(sigma_x=sigma_x, sigma_y=sigma_y)
                    DA_losses.append(cmmd(shared_x1, y1, shared_x2, y2)*self.DAcoef_args*da_ramp)
    
    
            elif self.DA == 'MMD':
                if (da_ramp > 0.0):
                    
                    sigma = compute_initial_bandwidth_full(shared_x1, shared_x2)
                    mmd = MMDLoss(sigma=sigma)
                    DA_losses.append(mmd(shared_x1, shared_x2)*self.DAcoef_args*da_ramp)
                
    
            elif self.DA == 'CORAL':
                if (da_ramp > 0.0):
                    coral = CORALLoss()
                    DA_losses.append(coral(shared_x1, shared_x2)*self.DAcoef_args*da_ramp)
            
                    
        task_loss = task_losses[0] + task_losses[1] + task_losses[2]
        if len(DA_losses) == 0:
            DA_loss = 0
        else:
            source_quality = torch.stack(source_quality_scores[:len(DA_losses)])
            source_quality = source_quality / source_quality.sum().clamp_min(1e-8)
            # Preserve the original sum-scale when all three source domains
            # receive equal confidence, while down-weighting mismatched sources.
            DA_loss = len(DA_losses) * sum(
                weight * loss for weight, loss in zip(source_quality, DA_losses)
            )
        
        decomposition_loss = torch.stack(decomposition_losses).mean() if decomposition_losses else 0.0
        model_loss += (task_loss + DA_loss + 0.1 * decomposition_loss) 

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
            
            flat_A, mapping = flatten_and_pad_grads(list(self.model.parameters()) + list(self.stage1_transfer.parameters()))
            
            self.main_optim.zero_grad()
            target_task_loss.backward(retain_graph=True)
            
            flat_B, _ = flatten_and_pad_grads(list(self.model.parameters()) + list(self.stage1_transfer.parameters()))
            
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
