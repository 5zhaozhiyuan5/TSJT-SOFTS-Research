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

try:
    from .residual_memory import (
        StationarityAwareResidualMemory,
        should_skip_stage2_optimizer_step,
    )
except ImportError:
    from residual_memory import (
        StationarityAwareResidualMemory,
        should_skip_stage2_optimizer_step,
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
        
        num_model = 1
        #########
        
        if model_name == 'SOFTS':
            self.forecasting_model = softs()
        if model_name == 'iTransformer':
            self.forecasting_model = iTransformer()
        elif model_name in ['MLP']:  
            self.forecasting_model = MLP(288, 36)
        
        self.residual_memory = None
        if self.temperature_args > 0 and self.memory_size_args > 0:
            source_city_names = [
                name for name in data_args.get('data_keys', [])
                if name != getattr(args, 'test_dataset', None)
            ]
            self.residual_memory = StationarityAwareResidualMemory(
                seq_len=288,
                pred_len=36,
                d_model=128,
                max_memory_size=memory_size_args,
                source_city_names=source_city_names,
                topk=8,
                candidate_pool=50,
                overlap_window=27,
            )
            
    
    def forward(self, data_i, stage='train', return_base=False):
        # The forecasting backbone is frozen in Stage-II. Keep it deterministic
        # when residual memory is enabled (SOFTS STAR has train/eval differences).
        if self.residual_memory is not None:
            self.forecasting_model.eval()
            if stage == 'train':
                self.residual_memory.train()
            else:
                self.residual_memory.eval()
        elif stage == 'train':
            self.forecasting_model.train()
        else:
            self.forecasting_model.eval()

        x, y, means, stds = data_i.x, data_i.y, data_i.means, data_i.stds
        x = torch.as_tensor(x, dtype=torch.float32, device=self.PatchFSL_cfg['device'])
        y = torch.as_tensor(y, dtype=torch.float32, device=self.PatchFSL_cfg['device'])
        means = torch.as_tensor(means, dtype=torch.float32, device=self.PatchFSL_cfg['device'])
        stds = torch.as_tensor(stds, dtype=torch.float32, device=self.PatchFSL_cfg['device'])

        # Input layout: [B, N, L, 2] -> speed channel [B, N, L].
        x = x.permute(0, 1, 3, 2)
        x = x[:, :, 0, -288:]
        self.eps = 1e-9
        self.RevIn_mean = x.mean(dim=2, keepdim=True).detach()
        var = x.var(dim=2, keepdim=True, unbiased=False).detach()
        self.RevIn_std = torch.sqrt(var + self.eps)
        x = (x - self.RevIn_mean) / self.RevIn_std
        history_for_memory = x.detach()

        x = x.permute(0, 2, 1)
        if self.model_name in ['MLP']:
            x = x.permute(0, 2, 1)
        rep, out = self.forecasting_model(x)

        out = out.detach()
        rep = rep.detach()
        base_out = out * self.RevIn_std + self.RevIn_mean
        base_out = unnorm(base_out, means, stds)

        normed_memory = None
        if self.residual_memory is not None and self.residual_memory.num_entries > 0:
            correction, normed_memory = self.residual_memory.retrieve(
                history_for_memory, rep, data_i
            )
            out = base_out + correction
        else:
            out = base_out

        y_unscale = norm(y, means, stds)
        y_unscale = (y_unscale - self.RevIn_mean) / self.RevIn_std

        if return_base:
            return out, y, rep, y_unscale, normed_memory, base_out
        return out, y, rep, y_unscale, normed_memory

    def update_residual_memory(self, data_i, base_prediction):
        if self.residual_memory is not None:
            self.residual_memory.update(data_i, base_prediction)

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
        self.loss_criterion = nn.SmoothL1Loss(reduction='mean', beta=5e-4) 
        if self.temperature_args>0:
            param_groups = [
                {'params': [p for name, p in self.model.named_parameters() 
                            if name != 'log_temp'], 'lr': self.lr, 'weight_decay': 1e-2,},  
            ]
            
            self.main_optim = optim.AdamW(param_groups)
    
    def meta_train_revise(self, data, matrix, epoch, target=False): # matrix没有用到

        # Stage-II has no trainable output path until the residual bank contains
        # source residuals. Keep the SOFTS backbone detached/frozen and warm up
        # the bank with the first source batch instead of backpropagating a
        # graph-free loss.
        residual_memory = self.model.residual_memory
        if residual_memory is None:
            raise RuntimeError(
                "Stage-II training requires residual memory; check MemSize > 0 "
                "and temperature_args > 0."
            )
        memory_was_empty = residual_memory.num_entries == 0

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
        base_outputs = []
        #origin_normed_memory = 0
        

        for i in range(len(data)):  # self.task_num):
            task_loss = 0
            #  for k in range(1):
            batch_size, node_num, seq_len, _ = data[i].x.shape
            out, y, rep, y_unscale, normed_memory, base_out = self.model(
                data[i], stage='train', return_base=True
            )
            base_outputs.append(base_out)
            
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
    
                    #if i>0:
                    task_losses.append(task_loss)

                #elif target == True:
                if i ==0 :
                    loss = self.loss_criterion(out, y)
                    target_task_loss = loss
    
        
        # Update the bank only after this batch has queried the previous bank.
        # Source cities are inserted; the target city is never used as memory.
        if self.model.residual_memory is not None:
            for memory_index in range(1, len(data)):
                self.model.update_residual_memory(data[memory_index], base_outputs[memory_index])

        # The first batch was forwarded while the bank was empty. Updating the
        # bank above does not retroactively attach gradients to those outputs.
        # Skip exactly this warm-up optimizer step; the next batch will retrieve
        # from the populated bank and train residual_gate normally.
        skip_optimizer_step = should_skip_stage2_optimizer_step(
            memory_was_empty, residual_memory.num_entries
        )
        if skip_optimizer_step:
            self.main_optim.zero_grad(set_to_none=True)
            model_loss = sum(task_losses)
            print(
                "[Stage-II] Residual-memory warm-up completed: "
                f"{residual_memory.num_entries} source entries; "
                "skipping backward/optimizer step for the first batch."
            )
            return (
                model_loss.detach().cpu().numpy(),
                np.mean(total_mse),
                np.mean(total_rmse),
                np.mean(total_mae),
                np.mean(total_mape),
            )

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
    