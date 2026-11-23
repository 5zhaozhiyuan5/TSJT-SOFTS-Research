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
    