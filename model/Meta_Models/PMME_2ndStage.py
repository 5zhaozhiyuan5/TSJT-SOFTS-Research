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
    from .horizon_residual_memory import (
        HorizonConditionedResidualMemory,
        configure_horizon_stage2_parameters,
    )
except ImportError:
    from horizon_residual_memory import (
        HorizonConditionedResidualMemory,
        configure_horizon_stage2_parameters,
    )

def log_verbose(args, msg):
    print(msg)
    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
        f.write(str(msg) + "\n")

class PatchFSL(nn.Module):
    """PMME Stage-II wrapper with a frozen Stage-I forecasting backbone."""

    def __init__(self, data_args, model_args, task_args, memory_size_args,
                 temperature_args, model_name, PatchFSL_cfg, args, model='GWN'):
        super(PatchFSL, self).__init__()
        self.data_args = data_args
        self.model_args = model_args
        self.task_args = task_args
        self.memory_size_args = int(memory_size_args)
        self.temperature_args = temperature_args
        self.model_name = model_name
        self.PatchFSL_cfg = PatchFSL_cfg
        self.args = args

        if model_name == 'SOFTS':
            self.forecasting_model = softs()
        elif model_name == 'iTransformer':
            self.forecasting_model = iTransformer()
        elif model_name == 'MLP':
            self.forecasting_model = MLP(288, 36)
        else:
            raise ValueError(f"Unsupported PMME forecasting model: {model_name}")

        self.horizon_memory = None
        if self.temperature_args > 0 and self.memory_size_args > 0:
            self.horizon_memory = HorizonConditionedResidualMemory(
                d_model=128,
                pred_len=int(task_args['source']['pred_num']),
                memory_size=self.memory_size_args,
                num_bands=3,
                query_chunk=getattr(args, 'stage2_query_chunk', 256),
            )

    @torch.no_grad()
    def initialize_horizon_memory(self, target_support_dataset, seed=1):
        """Initialize memory from target-support SOFTS representations only.

        This pass reads support inputs to obtain representations. It does not
        use target labels to initialize slots and does not construct/read the
        held-out target test split.
        """
        if self.horizon_memory is None:
            raise RuntimeError("Horizon memory is disabled; set MemSize > 0 and temperature > 0.")

        feature_parts = []
        for index in range(len(target_support_dataset)):
            data_i, _ = target_support_dataset[index]
            _, _, representation, _, _, _, _ = self(
                data_i, stage='test', return_base=True
            )
            feature_parts.append(representation.reshape(-1, representation.shape[-1]))

        if not feature_parts:
            raise RuntimeError("Target support dataset produced no representations.")
        features = torch.cat(feature_parts, dim=0)
        count = self.horizon_memory.initialize_from_features(features, seed=seed)
        return count

    def forward(self, data_i, stage='train', return_base=False):
        # STAR in SOFTS has train/eval-dependent behavior. Keep Stage-I fixed
        # and deterministic while adapting only the Stage-II memory.
        if self.horizon_memory is not None:
            self.forecasting_model.eval()
            if stage == 'train':
                self.horizon_memory.train()
            else:
                self.horizon_memory.eval()
        elif stage == 'train':
            self.forecasting_model.train()
        else:
            self.forecasting_model.eval()

        x, y, means, stds = data_i.x, data_i.y, data_i.means, data_i.stds
        device = self.PatchFSL_cfg['device']
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)
        means = torch.as_tensor(means, dtype=torch.float32, device=device)
        stds = torch.as_tensor(stds, dtype=torch.float32, device=device)

        # Preserve the PMME/SOFTS RevIN preprocessing exactly.
        x = x.permute(0, 1, 3, 2)
        x = x[:, :, 0, -288:]
        eps = 1e-9
        revin_mean = x.mean(dim=2, keepdim=True).detach()
        revin_var = x.var(dim=2, keepdim=True, unbiased=False).detach()
        revin_std = torch.sqrt(revin_var + eps)
        x_normalized = (x - revin_mean) / revin_std
        x_normalized = x_normalized.permute(0, 2, 1)
        if self.model_name == 'MLP':
            x_normalized = x_normalized.permute(0, 2, 1)

        representation, base_normalized = self.forecasting_model(x_normalized)
        representation = representation.detach()
        base_normalized = base_normalized.detach()

        base_revin = base_normalized * revin_std + revin_mean
        base_out = unnorm(base_revin, means, stds)

        correction = None
        attention_entropy = None
        if self.horizon_memory is not None:
            correction, attention_entropy = self.horizon_memory(representation)
            corrected_normalized = base_normalized + correction
            corrected_revin = corrected_normalized * revin_std + revin_mean
            out = unnorm(corrected_revin, means, stds)
        else:
            out = base_out

        y_unscale = norm(y, means, stds)
        y_unscale = (y_unscale - revin_mean) / revin_std

        if return_base:
            return (
                out, y, representation, y_unscale,
                correction, base_out, attention_entropy,
            )
        return out, y, representation, y_unscale, correction

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
        if self.model.horizon_memory is not None:
            # Only the three horizon-conditioned memory queries/generators and
            # shared memory are trainable; all Stage-I SOFTS weights are frozen.
            self.trainable_params = configure_horizon_stage2_parameters(self.model)
            self.main_optim = optim.AdamW(
                self.trainable_params, lr=self.lr, weight_decay=1e-4
            )
        else:
            self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.main_optim = optim.AdamW(
                self.trainable_params, lr=self.lr, weight_decay=1e-2
            )
    
    def meta_train_revise(self, data, matrix, epoch, target=False):
        """Few-shot target adaptation of the horizon-conditioned residual memory."""
        memory = self.model.horizon_memory
        if memory is None:
            raise RuntimeError("Stage-II requires HorizonConditionedResidualMemory.")
        if not bool(memory.memory_initialized.item()):
            raise RuntimeError(
                "Horizon memory has not been initialized. Call "
                "model.initialize_horizon_memory(target_support_dataset) first."
            )

        data_i = data[0] if isinstance(data, (list, tuple)) else data
        self.model.train()
        self.main_optim.zero_grad(set_to_none=True)

        out, y, _, _, correction, base_out, attention_entropy = self.model(
            data_i, stage='train', return_base=True
        )
        forecast_loss = self.loss_criterion(out, y)
        if not torch.isfinite(forecast_loss):
            raise RuntimeError("Non-finite PMME Stage-II horizon-memory loss.")

        forecast_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
        self.main_optim.step()

        MSE, RMSE, MAE, MAPE = calc_metric(out.detach(), y.detach())
        if epoch % 25 == 0:
            pred_len = out.shape[-1]
            band_width = pred_len // 3
            band_stats = []
            for band in range(3):
                left = band * band_width
                right = (band + 1) * band_width
                base_mae = torch.mean(torch.abs(base_out[..., left:right] - y[..., left:right]))
                final_mae = torch.mean(torch.abs(out[..., left:right] - y[..., left:right]))
                correction_norm = correction[..., left:right].abs().mean()
                band_stats.append(
                    f"{base_mae.item():.4f}->{final_mae.item():.4f}"
                    f"(d={correction_norm.item():.4f})"
                )
            entropy_values = "/".join(
                f"{value:.3f}" for value in attention_entropy.detach().cpu().tolist()
            )
            print(
                f"[Stage-II][HorizonMemory] epoch={epoch} "
                f"loss={forecast_loss.item():.6f} "
                f"short/mid/long(base->final,|delta|)={' | '.join(band_stats)} "
                f"attention_entropy={entropy_values} "
                f"grad_norm={float(grad_norm):.6f} "
                f"memory_slots={memory.memory.shape[0]} "
                f"support_features={int(memory.initialization_feature_count.item())}"
            )

        return (
            forecast_loss.detach().cpu().numpy(),
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
    