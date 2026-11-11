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

from geomloss import SamplesLoss 
from torch.utils.data import TensorDataset, DataLoader
#from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls.variational_elbo import VariationalELBO
from gpytorch.likelihoods import BernoulliLikelihood, LaplaceLikelihood

from SOFTS.SOFTS_new import Model as softs
from iTransformer.iTransformer_new import Model as iTransformer

try:
    from .raft_cross_city import (
        MultiScaleRetrievalAdapter,
        configure_raft_stage2_parameters,
    )
except ImportError:
    from raft_cross_city import (
        MultiScaleRetrievalAdapter,
        configure_raft_stage2_parameters,
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
        
        self.raft_adapter = None
        if self.temperature_args > 0 and self.memory_size_args > 0:
            source_city_names = [
                name for name in data_args.get('data_keys', [])
                if name != getattr(args, 'test_dataset', None)
            ]
            self.raft_adapter = MultiScaleRetrievalAdapter(
                seq_len=288,
                pred_len=task_args['source']['pred_num'],
                max_memory_size=memory_size_args,
                source_city_names=source_city_names,
                periods=(4, 2, 1),
                topm=20,
                temperature=0.1,
            )
            
    
    def forward(self, data_i, stage='train', return_base=False):
        # Stage-II keeps the SOFTS forecasting backbone frozen and deterministic.
        if self.raft_adapter is not None:
            self.forecasting_model.eval()
            if stage == 'train':
                self.raft_adapter.train()
            else:
                self.raft_adapter.eval()
        elif stage == 'train':
            self.forecasting_model.train()
        else:
            self.forecasting_model.eval()

        x, y, means, stds = data_i.x, data_i.y, data_i.means, data_i.stds
        x = torch.as_tensor(x, dtype=torch.float32, device=self.PatchFSL_cfg['device'])
        y = torch.as_tensor(y, dtype=torch.float32, device=self.PatchFSL_cfg['device'])
        means = torch.as_tensor(means, dtype=torch.float32, device=self.PatchFSL_cfg['device'])
        stds = torch.as_tensor(stds, dtype=torch.float32, device=self.PatchFSL_cfg['device'])

        # Input layout: [B,N,L,2] -> globally standardized speed [B,N,L].
        x = x.permute(0, 1, 3, 2)
        x = x[:, :, 0, -288:]
        history_for_retrieval = x.detach()

        # Keep the original PMME/SOFTS RevIN path unchanged.
        self.eps = 1e-9
        self.RevIn_mean = x.mean(dim=2, keepdim=True).detach()
        var = x.var(dim=2, keepdim=True, unbiased=False).detach()
        self.RevIn_std = torch.sqrt(var + self.eps)
        x = (x - self.RevIn_mean) / self.RevIn_std

        x = x.permute(0, 2, 1)
        if self.model_name in ['MLP']:
            x = x.permute(0, 2, 1)
        rep, out = self.forecasting_model(x)

        # Stage-II must not backpropagate into the pretrained forecaster.
        out = out.detach()
        rep = rep.detach()
        base_out = out * self.RevIn_std + self.RevIn_mean
        base_out = unnorm(base_out, means, stds)

        retrieved_out = None
        retrieval_gate = None
        if self.raft_adapter is not None:
            out, retrieved_out, retrieval_gate = self.raft_adapter(
                history_for_retrieval, base_out, means, stds
            )
        else:
            out = base_out

        y_unscale = norm(y, means, stds)
        y_unscale = (y_unscale - self.RevIn_mean) / self.RevIn_std

        if return_base:
            return out, y, rep, y_unscale, retrieved_out, base_out, retrieval_gate
        return out, y, rep, y_unscale, retrieved_out

    def build_retrieval_bank(self, source_dataset):
        if self.raft_adapter is None:
            raise RuntimeError("RAFT adapter is disabled; set MemSize > 0 and temperature_args > 0.")
        target_city = getattr(self.args, 'test_dataset', None)
        count = self.raft_adapter.build_memory(
            source_dataset.x_list,
            source_dataset.y_list,
            source_dataset.means_list,
            source_dataset.stds_list,
            city_names=source_dataset.data_list,
            target_city=target_city,
        )
        return count, self.raft_adapter.city_counts()

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

        self.loss_criterion = nn.SmoothL1Loss(reduction='mean', beta=5e-4)
        if self.model.raft_adapter is not None:
            # Only the RAFT scale projections and reliability-aware horizon gate
            # are adapted. The pretrained SOFTS backbone stays frozen.
            self.trainable_params = configure_raft_stage2_parameters(self.model)
            self.main_optim = optim.AdamW(
                self.trainable_params, lr=self.lr, weight_decay=1e-4
            )
        else:
            self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.main_optim = optim.AdamW(
                self.trainable_params, lr=self.lr, weight_decay=1e-2
            )
    
    def meta_train_revise(self, data, matrix, epoch, target=False):
        """Few-shot target adaptation of the RAFT retrieval/fusion adapter.

        The source-only bank is built once before Stage-II training. This
        optimizer step uses all 46 target windows (the configured 3-day support
        set); it does not resample source batches or apply PCGrad.
        """
        if self.model.raft_adapter is None:
            raise RuntimeError(
                "Stage-II requires RAFT retrieval; check MemSize > 0 and temperature_args > 0."
            )
        if self.model.raft_adapter.num_entries == 0:
            raise RuntimeError(
                "RAFT source bank is empty. Call model.build_retrieval_bank(source_dataset) "
                "before Stage-II training."
            )

        data_i = data[0] if isinstance(data, (list, tuple)) else data
        self.model.train()
        self.main_optim.zero_grad(set_to_none=True)

        out, y, _, _, retrieved_out, base_out, retrieval_gate = self.model(
            data_i, stage='train', return_base=True
        )
        forecast_loss = self.loss_criterion(out, y)
        # Small shrinkage penalty lets the gate prefer the frozen PMME forecast
        # unless source retrieval contributes useful evidence on target support.
        loss = forecast_loss + 1e-3 * retrieval_gate.mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Stage-II RAFT adaptation loss.")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
        self.main_optim.step()

        MSE, RMSE, MAE, MAPE = calc_metric(out.detach(), y.detach())
        if epoch % 25 == 0:
            retrieval_mae = torch.mean(torch.abs(retrieved_out.detach() - y.detach()))
            base_mae = torch.mean(torch.abs(base_out.detach() - y.detach()))
            gate_parts = torch.tensor_split(retrieval_gate.detach(), 3, dim=-1)
            gate_short, gate_mid, gate_long = [
                float(part.mean().item()) for part in gate_parts
            ]
            print(
                f"[Stage-II][RAFT-ReliabilityGate] epoch={epoch} "
                f"loss={loss.item():.6f} forecast_loss={forecast_loss.item():.6f} "
                f"base_mae={base_mae.item():.6f} retrieval_mae={retrieval_mae.item():.6f} "
                f"gate_mean={float(retrieval_gate.detach().mean().item()):.4f} "
                f"gate_short/mid/long={gate_short:.4f}/{gate_mid:.4f}/{gate_long:.4f} "
                f"grad_norm={float(grad_norm):.6f} "
                f"bank_entries={self.model.raft_adapter.num_entries}"
            )

        return (
            loss.detach().cpu().numpy(),
            MSE.detach().cpu().numpy(),
            RMSE.detach().cpu().numpy(),
            MAE.detach().cpu().numpy(),
            MAPE.detach().cpu().numpy(),
        )

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
    