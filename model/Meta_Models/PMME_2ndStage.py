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
    from .target_residual_calibrator import HorizonWiseResidualCalibrator
except ImportError:
    from target_residual_calibrator import HorizonWiseResidualCalibrator

def log_verbose(args, msg):
    print(msg)
    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
        f.write(str(msg) + "\n")

class PatchFSL(nn.Module):
    """PMME wrapper with a frozen Stage-I backbone and optional Stage-II adapter."""

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

        # Stage-II uses bounded horizon-wise robust residual calibration. Legacy
        # positive MemSize/temperature flags continue to enable Stage-II.
        self.residual_net = None
        if self.temperature_args > 0 and self.memory_size_args > 0:
            self.residual_net = HorizonWiseResidualCalibrator(
                pred_len=int(task_args['source']['pred_num']),
                horizon_bands=3,
                max_correction_ratio=getattr(args, 'stage2_max_correction_ratio', 0.25),
                purge_windows=3,
                min_calibration_windows=4,
            )

    def prepare_stage2(self, data_i):
        """Compute frozen Stage-I outputs and target-normalized supervision once."""
        x, y, means, stds = data_i.x, data_i.y, data_i.means, data_i.stds
        device = self.PatchFSL_cfg['device']
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)
        means = torch.as_tensor(means, dtype=torch.float32, device=device)
        stds = torch.as_tensor(stds, dtype=torch.float32, device=device)

        # Preserve the original PMME/SOFTS RevIN preprocessing.
        x = x.permute(0, 1, 3, 2)
        x = x[:, :, 0, -288:]
        eps = 1e-9
        revin_mean = x.mean(dim=2, keepdim=True).detach()
        revin_var = x.var(dim=2, keepdim=True, unbiased=False).detach()
        revin_std = torch.sqrt(revin_var + eps)
        history_normalized = (x - revin_mean) / revin_std
        x_normalized = history_normalized.permute(0, 2, 1)
        if self.model_name == 'MLP':
            x_normalized = x_normalized.permute(0, 2, 1)

        if self.residual_net is not None:
            # Stage-II never updates or backpropagates through Stage-I.
            self.forecasting_model.eval()
            with torch.no_grad():
                representation, base_normalized = self.forecasting_model(x_normalized)
        else:
            representation, base_normalized = self.forecasting_model(x_normalized)

        base_revin = base_normalized * revin_std + revin_mean
        base_out = unnorm(base_revin, means, stds)
        y_unscale = norm(y, means, stds)
        y_unscale = (y_unscale - revin_mean) / revin_std

        return {
            "history_normalized": history_normalized,
            "representation": representation,
            "base_normalized": base_normalized,
            "base_out": base_out,
            "y": y,
            "y_unscale": y_unscale,
            "revin_mean": revin_mean,
            "revin_std": revin_std,
            "means": means,
            "stds": stds,
        }

    def forward(self, data_i, stage='train', return_base=False):
        if self.residual_net is None:
            if stage == 'train':
                self.forecasting_model.train()
            else:
                self.forecasting_model.eval()
        else:
            self.forecasting_model.eval()
            self.residual_net.eval()

        prepared = self.prepare_stage2(data_i)
        history_normalized = prepared["history_normalized"]
        representation = prepared["representation"]
        base_normalized = prepared["base_normalized"]
        base_out = prepared["base_out"]
        y = prepared["y"]
        y_unscale = prepared["y_unscale"]

        correction = None
        if self.residual_net is not None:
            if not self.residual_net.is_fitted:
                raise RuntimeError(
                    "Stage-II residual calibrator has not been fitted on target support data."
                )
            # Correct directly in the same physical output space as base_out.
            # No RevIN-scale residual is added or inverted here.
            correction = self.residual_net(base_out)
            out = base_out + correction
        else:
            out = base_out

        if return_base:
            return (
                out, y, representation, y_unscale,
                correction, base_out, base_normalized,
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
        self.last_stage2_loss_components = None

        self.model = PatchFSL(data_args, model_args, task_args, memory_size_args, temperature_args, model_name, PatchFSL_cfg, args).to(self.device)

        self.loss_criterion = nn.SmoothL1Loss(reduction='mean', beta=5e-4)
        if self.model.residual_net is not None:
            # The Stage-II adapter has buffers only and is solved in closed form.
            # Freeze every Stage-I parameter; no optimizer or epoch loop is needed.
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.trainable_params = []
            self.main_optim = None
            log_verbose(
                args,
                "[Stage-II][HRRC] frozen Stage-I; "
                "trainable adapter parameters=0 (robust bias + support reliability calibration)."
            )
        else:
            self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.main_optim = optim.AdamW(
                self.trainable_params, lr=self.lr, weight_decay=1e-2
            )

        total_params = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        log_verbose(
            args,
            "model params total: {}, trainable: {}".format(
                total_params, trainable_params
            )
        )
    
    def _meta_train_stage1(self, data, epoch):
        """Original source/target PCGrad training path for Stage-I."""
        source_losses = []
        source_metrics = {"mse": [], "rmse": [], "mae": [], "mape": []}
        target_loss = None

        self.model.train()
        for task_index, data_i in enumerate(data):
            out, y, _, _, _ = self.model(data_i, stage='train')
            task_loss = self.loss_criterion(out, y)

            if task_index == 0:
                target_loss = task_loss
            else:
                source_losses.append(task_loss)
                mse, rmse, mae, mape = calc_metric(out, y)
                source_metrics["mse"].append(mse.detach().cpu().numpy())
                source_metrics["rmse"].append(rmse.detach().cpu().numpy())
                source_metrics["mae"].append(mae.detach().cpu().numpy())
                source_metrics["mape"].append(mape.detach().cpu().numpy())

        if target_loss is None or not source_losses:
            raise RuntimeError(
                "Stage-I PCGrad requires one target task followed by at least one source task."
            )

        source_loss = torch.stack(source_losses).sum()
        self.main_optim.zero_grad(set_to_none=True)
        source_loss.backward(retain_graph=True)

        def flatten_grads(params):
            flat, mapping, offset = [], [], 0
            for parameter in params:
                count = parameter.numel()
                grad = (
                    torch.zeros(count, device=parameter.device, dtype=parameter.dtype)
                    if parameter.grad is None
                    else parameter.grad.detach().reshape(-1).clone()
                )
                flat.append(grad)
                mapping.append((parameter, offset, count))
                offset += count
            return torch.cat(flat), mapping

        flat_source, mapping = flatten_grads(self.model.parameters())
        self.main_optim.zero_grad(set_to_none=True)
        target_loss.backward(retain_graph=True)
        flat_target, _ = flatten_grads(self.model.parameters())

        dot = torch.dot(flat_source, flat_target)
        if dot < 0:
            flat_source = flat_source - (
                dot / (flat_target.norm().square() + 1e-8)
            ) * flat_target

        for parameter, start, count in mapping:
            parameter.grad = flat_source[start:start + count].view_as(parameter).clone()

        self.main_optim.step()
        self.main_optim.zero_grad(set_to_none=True)

        def mean_metric(key):
            values = source_metrics[key]
            return float(np.mean(values)) if values else float("nan")

        if epoch % 25 == 0:
            print(
                f"[Stage-I][PCGrad] epoch={epoch} "
                f"source_loss={source_loss.item():.6f} "
                f"target_loss={target_loss.item():.6f} "
                f"gradient_dot={dot.item():.6f}"
            )

        return (
            source_loss.detach().cpu().numpy(),
            mean_metric("mse"),
            mean_metric("rmse"),
            mean_metric("mae"),
            mean_metric("mape"),
        )

    def fit_target_residual_adapter(self, target_dataset):
        """Fit horizon-wise robust residual calibration once on target support."""
        calibrator = self.model.residual_net
        if calibrator is None:
            raise RuntimeError("Stage-II residual calibrator is disabled.")

        if hasattr(target_dataset, "x"):
            data_items = [target_dataset]
        elif (
            isinstance(target_dataset, tuple)
            and len(target_dataset) > 0
            and hasattr(target_dataset[0], "x")
        ):
            data_items = [target_dataset[0]]
        elif (
            isinstance(target_dataset, list)
            and target_dataset
            and hasattr(target_dataset[0], "x")
        ):
            data_items = target_dataset
        else:
            data_items = []
            for index in range(len(target_dataset)):
                item = target_dataset[index]
                data_i = item[0] if isinstance(item, (tuple, list)) else item
                if hasattr(data_i, "x"):
                    data_items.append(data_i)

        if not data_items:
            raise RuntimeError("The target support dataset contains no usable batches.")

        self.model.eval()
        start_time = time.time()
        base_batches = []
        target_batches = []
        with torch.no_grad():
            for data_i in data_items:
                prepared = self.model.prepare_stage2(data_i)
                base_prediction = prepared["base_out"]
                target_values = prepared["y"]
                residual_target = target_values - base_prediction
                if (
                    not torch.isfinite(base_prediction).all()
                    or not torch.isfinite(target_values).all()
                    or not torch.isfinite(residual_target).all()
                ):
                    raise RuntimeError(
                        "Non-finite physical-space base predictions/target residuals."
                    )
                base_batches.append(base_prediction)
                target_batches.append(target_values)

            base_all = torch.cat(base_batches, dim=0)
            target_all = torch.cat(target_batches, dim=0)
            residual_all = target_all - base_all
            fit_summary = calibrator.fit_from_residuals(
                residual_all, target_all
            )

            correction = calibrator(base_all)
            output_all = base_all + correction
            target_residual_all = residual_all

        predicted_residual_all = correction
        residual_loss = F.smooth_l1_loss(
            predicted_residual_all, target_residual_all, beta=5e-4
        )
        forecast_loss = F.smooth_l1_loss(
            output_all, target_all, beta=5e-4
        )
        correction_penalty = predicted_residual_all.square().mean()
        total_loss = residual_loss + 0.1 * forecast_loss + 1e-4 * correction_penalty
        mse, rmse, mae, mape = calc_metric(output_all, target_all)

        # Also report the uncorrected support baseline for direct comparison.
        base_mse, base_rmse, base_mae, base_mape = calc_metric(base_all, target_all)
        stats = {
            **fit_summary,
            "fit_seconds": float(time.time() - start_time),
            "total_loss": float(total_loss.detach().cpu().item()),
            "residual_loss": float(residual_loss.detach().cpu().item()),
            "forecast_loss": float(forecast_loss.detach().cpu().item()),
            "correction_l2": float(correction_penalty.detach().cpu().item()),
            "base_train_MAE": float(base_mae.detach().cpu().item()),
            "base_train_RMSE": float(base_rmse.detach().cpu().item()),
            "train_MSE": float(mse.detach().cpu().item()),
            "train_RMSE": float(rmse.detach().cpu().item()),
            "train_MAE": float(mae.detach().cpu().item()),
            "train_MAPE": float(mape.detach().cpu().item()),
        }
        self.last_stage2_loss_components = {
            key: stats[key]
            for key in (
                "total_loss",
                "residual_loss",
                "forecast_loss",
                "correction_l2",
            )
        }
        return stats

    def meta_train_revise(self, data, epoch, target=False):
        """Keep the Stage-I PCGrad path; Stage-II uses one closed-form support fit."""
        if self.model.residual_net is None:
            return self._meta_train_stage1(data, epoch)

        stats = self.fit_target_residual_adapter(data)
        return (
            np.asarray(stats["total_loss"]),
            np.asarray(stats["train_MSE"]),
            np.asarray(stats["train_RMSE"]),
            np.asarray(stats["train_MAE"]),
            np.asarray(stats["train_MAPE"]),
        )

    def forward(self, data, matrix):
        data_i = data[0] if isinstance(data, (list, tuple)) else data
        out, y, representation, y_unscale, correction = self.model(
            data_i, stage='test'
        )
        return out, correction

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
                    
                out, y, rep, y_unscale, correction = self.model(data_i, stage='test')
            
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
    