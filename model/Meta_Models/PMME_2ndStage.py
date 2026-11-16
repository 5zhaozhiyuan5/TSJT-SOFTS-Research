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
from experiment_diagnostics import ExperimentDiagnostics

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
        
        num_model = 1
        #########
        
        if model_name == 'SOFTS':
            self.forecasting_model = softs()
        if model_name == 'iTransformer':
            self.forecasting_model = iTransformer()
        elif model_name in ['MLP']:  
            self.forecasting_model = MLP(288, 36)
        
        self.memory = None 
        
        if self.temperature_args>0:
            
            self.memory = nn.Parameter(torch.randn(memory_size_args, 128)) 
            print('self.memory.shape', self.memory.shape)
            D, D_out, H_w, H_b = 128, 36, 256, 256
            
            self.bias_mlp = nn.Sequential(
                
                nn.Linear(D, D_out),
                # nn.Tanh()
            )
            
            self.gating_mlp = nn.Sequential(
                nn.Linear(memory_size_args, H_b),
                #nn.ReLU(inplace=True),
                nn.GELU(),
                nn.Linear(H_b, H_b),
                nn.GELU(),
                nn.Linear(H_b, 1),
                nn.Sigmoid()
            )

            d_model = D
            self.Wq = nn.Linear(d_model, d_model, bias=False)
            self.Wk = nn.Linear(d_model, d_model, bias=False)
            self.Wv = nn.Linear(d_model, d_model, bias=False)
            
    
    def forward(self, data_i, stage = 'train'):  
        
        if(stage == 'train'):
            
            self.forecasting_model.train()
            if self.temperature_args>0:
                self.bias_mlp.train()
                self.gating_mlp.train()
                self.Wq.train()
                self.Wk.train()
                self.Wv.train()
        else:
            
            self.forecasting_model.eval()
            if self.temperature_args>0:
                self.bias_mlp.eval()
                self.gating_mlp.eval()
                self.Wq.eval()
                self.Wk.eval()
                self.Wv.eval()
            
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
        # y = (y - self.RevIn_mean) / self.RevIn_std 

        x = x.permute(0,2,1) # # [B, D, T] -> [B, T, D]

        if self.model_name in ['MLP']:
            x = x.permute(0,2,1)
        rep, out = self.forecasting_model(x) 

        out = out.detach()
        rep = rep.detach()

        B, N, D = rep.shape
        
        normed_memory = None
        if self.temperature_args>0:
            normed_memory = self.memory 
            r = rep.view(B*N, D)  # 
            q = self.Wq(r)                          # (B, d)
            K = self.Wk(self.memory)                # (N, d)
            V = self.Wv(self.memory)                # (N, d)

            scores = torch.matmul(q, K.t())     # (B, N)
            scores = scores / (D ** 0.5)       
            
            alpha = torch.softmax(scores, dim=1)  # (B, N)
            retrieved = torch.matmul(alpha, V)      # (B, d)
            delta_b  = self.bias_mlp(retrieved)       # [(B*N), D_out]

            delta_b = delta_b.view(B, N, -1) #/10    # [B, N, D_out]     
            out = out + delta_b # 
            
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
        self.diagnostics = getattr(args, "diagnostics", None)

        self.lr = model_args['STnet']['lr']
        self.model_name = args.STmodel
        self.device = PatchFSL_cfg['device']
        self.current_epoch = 0

        self.model = PatchFSL(data_args, model_args, task_args, memory_size_args, temperature_args, model_name, PatchFSL_cfg, args).to(self.device)
         
        # print(self.model)
        log_verbose(args,"model params: {}".format(count_parameters(self.model)))

        self.main_optim = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        self.loss_criterion = nn.SmoothL1Loss(reduction='mean', beta=5e-4) 
        if self.temperature_args>0:
            param_groups = [
                {'params': [p for name, p in self.model.named_parameters() 
                            if name != 'log_temp'], 'lr': self.lr, 'weight_decay': 1e-2,},  
            ]
            
            self.main_optim = optim.AdamW(param_groups)
    
    def meta_train_revise(self, data, matrix, epoch, target=False): # matrix没有用到
        epoch_started = time.perf_counter()
        params_for_diagnostics = list(self.model.parameters())
        param_norm_before = torch.sqrt(sum(
            p.detach().float().pow(2).sum() for p in params_for_diagnostics
        ))
        parameter_sample_before = []
        remaining_sample = 4096
        for parameter in params_for_diagnostics:
            if remaining_sample <= 0:
                break
            flat = parameter.detach().reshape(-1)
            take = min(remaining_sample, flat.numel())
            if take:
                parameter_sample_before.append(flat[:take].float().clone())
                remaining_sample -= take
        parameter_sample_before = (
            torch.cat(parameter_sample_before)
            if parameter_sample_before else torch.zeros(1, device=self.device)
        )

        def current_parameter_sample(limit=4096):
            chunks = []
            remaining = limit
            for parameter in params_for_diagnostics:
                if remaining <= 0:
                    break
                flat = parameter.detach().reshape(-1)
                take = min(remaining, flat.numel())
                if take:
                    chunks.append(flat[:take].float())
                    remaining -= take
            return torch.cat(chunks) if chunks else torch.zeros(1, device=self.device)
        
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

        target_task_loss = 0
        #origin_normed_memory = 0
        

        for i in range(len(data)):  # self.task_num):
            task_loss = 0
            #  for k in range(1):
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
                y_source.append(y_unscale) #y_unscale
                
            if True: 
                if i>0: #True:  
                    
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
    
                    #if i>0:
                    task_losses.append(task_loss)

                #elif target == True:
                if i ==0 :
                    loss = self.loss_criterion(out, y)
                    target_task_loss = loss
    
        
        OT_losses = []
        y_norm_sources = []
        y_norm_targets = []
        
        OT_loss_scale = 1
        model_loss = 0
        
        task_loss = sum(task_losses)

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
        write_back_flat_grads(flat_proj_A, mapping)
        grad_sq = []
        nonfinite_counts = []
        grad_none = 0
        for parameter in params_for_diagnostics:
            if parameter.grad is None:
                grad_none += 1
                continue
            grad = parameter.grad.detach().float()
            nonfinite_counts.append((~torch.isfinite(grad)).sum())
            grad_sq.append(torch.nan_to_num(grad).pow(2).sum())
        grad_norm = torch.sqrt(torch.stack(grad_sq).sum()) if grad_sq else torch.zeros(())
        nonfinite_grad_count = int(torch.stack(nonfinite_counts).sum().item()) if nonfinite_counts else 0
        self.main_optim.step()
        self.main_optim.zero_grad()

        model_loss = task_loss
        
        if epoch%25==0:
            print('len(task_losses), len(OT_losses)', len(task_losses), len(OT_losses))

        mse_value = float(np.mean(total_mse))
        rmse_value = float(np.mean(total_rmse))
        mae_value = float(np.mean(total_mae))
        mape_value = float(np.mean(total_mape))
        if self.diagnostics is not None:
            param_norm_after = torch.sqrt(sum(
                p.detach().float().pow(2).sum() for p in params_for_diagnostics
            ))
            self.diagnostics.write("stage2_epoch", {
                "epoch": epoch,
                "stage2_forecast_loss": model_loss,
                "source_mae_mean": mae_value,
                "source_rmse_mean": rmse_value,
                "source_mse_mean": mse_value,
                "source_mape_mean": mape_value,
                "target_reference_loss": target_task_loss,
                "finite_loss": bool(torch.isfinite(model_loss.detach()).all()),
            })
            if train_horizon_metrics:
                horizon_mae = np.stack([item[0] for item in train_horizon_metrics], axis=0).mean(axis=0)
                horizon_rmse = np.stack([item[1] for item in train_horizon_metrics], axis=0).mean(axis=0)
                horizon_mse = np.stack([item[2] for item in train_horizon_metrics], axis=0).mean(axis=0)
                horizon_mape = np.stack([item[3] for item in train_horizon_metrics], axis=0).mean(axis=0)
                minutes_per_step = float(self.data_args.get("processed_interval_minutes", 5)) if isinstance(self.data_args, dict) else 5.0
                self.diagnostics.log_horizon(
                    epoch, len(horizon_mae), minutes_per_step,
                    horizon_mae, horizon_rmse, horizon_mse, horizon_mape, stage="stage2",
                )
            self.diagnostics.log_gradient(
                epoch, "stage2_model", grad_norm, param_norm_before,
                grad_none / max(len(params_for_diagnostics), 1),
                nonfinite_grad_count, param_norm_after,
                parameter_delta_norm=torch.norm(current_parameter_sample() - parameter_sample_before),
                stage="stage2",
            )
            lr = self.main_optim.param_groups[0]["lr"] if self.main_optim.param_groups else self.lr
            self.diagnostics.log_runtime(
                epoch, time.perf_counter() - epoch_started, lr, self.device, stage="stage2"
            )
        
        return model_loss.detach().cpu().numpy(),mse_value,rmse_value,mae_value,mape_value
        
        
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
    
        validation_phase = "stage2_few_shot_validation"
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
            minutes_per_step = float(self.data_args.get("processed_interval_minutes", 5)) if isinstance(self.data_args, dict) else 5.0
            if self.diagnostics is not None:
                self.diagnostics.log_evaluation(
                    validation_phase, self.current_epoch, i, minutes_per_step,
                    total_mae, total_rmse, total_mse, total_mape, total_len,
                )
            
        
        print("[INFO] Enter test phase")
        length = test_dataset.__len__()
        # self.model = copy.deepcopy(best_model)
        print('length', length) # 265
        total_mse_horizon,total_rmse_horizon, total_mae_horizon, total_mape_horizon, total_loss = self.test_batch(0,length+1, test_dataset,"test")   

        test_phase = "stage2_test"
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
            minutes_per_step = float(self.data_args.get("processed_interval_minutes", 5)) if isinstance(self.data_args, dict) else 5.0
            if self.diagnostics is not None:
                self.diagnostics.log_evaluation(
                    test_phase, self.current_epoch, i, minutes_per_step,
                    total_mae, total_rmse, total_mse, total_mape, total_len,
                )
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
            minutes_per_step = float(self.data_args.get("processed_interval_minutes", 5)) if isinstance(self.data_args, dict) else 5.0
            if self.diagnostics is not None:
                self.diagnostics.log_evaluation(
                    "stage2_final_test", self.current_epoch, i, minutes_per_step,
                    total_mae, total_rmse, total_mse, total_mape, total_len,
                )

        return testing_score
    