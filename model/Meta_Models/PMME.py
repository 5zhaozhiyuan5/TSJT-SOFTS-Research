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
            shape0 = x1.shape[1]
                    
            y_norm_sources.append(y1)
            y_norm_targets.append(y2)

            x1 = x1.view(-1, x1.size(-1))
            x2 = x2.view(-1, x2.size(-1))
            y1 = y1.view(-1, y1.size(-1))
            y2 = y2.view(-1, y2.size(-1))
            

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
                    
                    prob1, prob2 = threshold(prob1).detach(), threshold(prob2).detach() 
                    alpha = (prob1/torch.sum(prob1)).detach()
                    beta = (prob2/torch.sum(prob2)).detach()
                    
                   
                    OT_loss_scale = torch.sqrt(((prob1.mean())*(prob2.mean())).detach())
                    
                    if epoch%25==0:
                        print('OT_loss_scale', OT_loss_scale)
            
            else:
                alpha = torch.ones(size1, device='cuda')/size1  
                beta = torch.ones(size2, device='cuda')/size2  
                OT_loss_scale = 1

            if self.DA == 'OT':
                if (epoch>=self.DAepoch_args): 
                    
                    x_y_scale = math.sqrt(self.xySquaredRatio_args)
                    y1, y2 = y1*x_y_scale, y2*x_y_scale
                    
                    x = torch.cat((x1,y1),dim=-1) 
                    y = torch.cat((x2,y2),dim=-1) # y: target  
                    
                    if (ratio1>0.00 and ratio2>0.00):
                        OT_loss = torch.sum(criterion(alpha, x, beta, y))*self.DAcoef_args *OT_loss_scale
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
        
        model_loss += (task_loss + DA_loss) 

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
    