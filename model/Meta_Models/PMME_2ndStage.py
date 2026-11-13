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


def log_verbose(args, msg):
    print(msg)
    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
        f.write(str(msg) + "\n")

class TCFAAdapter(nn.Module):
    """Target-statistics-conditioned, channel-wise feature adapter.

    The adapter is initialized as an exact identity mapping. Statistics are
    estimated once from the target support set and stored as buffers, so
    inference never reads target labels or updates the adapter.
    """

    def __init__(self, dim=128, strength=0.1, eps=1e-4):
        super().__init__()
        self.dim = int(dim)
        self.strength = float(strength)
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.zeros(self.dim))
        self.shift = nn.Parameter(torch.zeros(self.dim))
        self.register_buffer("target_mean", torch.zeros(1, 1, self.dim))
        self.register_buffer("target_std", torch.ones(1, 1, self.dim))
        self.register_buffer(
            "statistics_ready", torch.tensor(False, dtype=torch.bool)
        )

    @torch.no_grad()
    def set_target_statistics(self, representation):
        if representation.ndim != 3 or representation.shape[-1] != self.dim:
            raise ValueError(
                "TCFA expects representation [batch, nodes, {}], got {}".format(
                    self.dim, tuple(representation.shape)
                )
            )
        features = representation.detach().float()
        mean = features.mean(dim=(0, 1), keepdim=True)
        std = features.std(dim=(0, 1), unbiased=False, keepdim=True)
        self.target_mean.copy_(mean.to(self.target_mean.dtype))
        self.target_std.copy_(std.clamp_min(self.eps).to(self.target_std.dtype))
        self.statistics_ready.fill_(True)

    def forward(self, representation):
        if not bool(self.statistics_ready.item()):
            raise RuntimeError(
                "TCFA target statistics are not initialized. "
                "Call fit_target_statistics() with the target support set first."
            )
        if representation.shape[-1] != self.dim:
            raise ValueError(
                "TCFA feature dimension mismatch: expected {}, got {}".format(
                    self.dim, representation.shape[-1]
                )
            )
        normalized = (
            representation - self.target_mean.to(representation.dtype)
        ) / self.target_std.to(representation.dtype)
        scale = torch.tanh(self.scale).view(1, 1, self.dim)
        shift = torch.tanh(self.shift).view(1, 1, self.dim)
        delta = scale * normalized + shift
        return representation + self.strength * delta

    def regularization_loss(self):
        return self.scale.square().mean() + self.shift.square().mean()

    @torch.no_grad()
    def diagnostics(self, representation=None):
        result = {
            "scale_rms": float(self.scale.detach().square().mean().sqrt().cpu()),
            "shift_rms": float(self.shift.detach().square().mean().sqrt().cpu()),
            "statistics_ready": bool(self.statistics_ready.item()),
        }
        if representation is not None:
            adapted = self.forward(representation)
            result["feature_delta_rms"] = float(
                (adapted - representation).detach().square().mean().sqrt().cpu()
            )
        return result


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

        # Stage-II V1: lightweight target-statistics-conditioned feature adaptation.
        self.feature_adapter = None
        if self.temperature_args > 0 and self.memory_size_args > 0:
            if not hasattr(self.forecasting_model, "projection2"):
                raise NotImplementedError(
                    "TCFA currently requires a forecasting backbone exposing "
                    "projection2; supported backbones are SOFTS and iTransformer."
                )
            self.feature_adapter = TCFAAdapter(
                dim=128,
                strength=getattr(args, "stage2_strength", 0.1),
                eps=getattr(args, "stage2_stat_eps", 1e-4),
            )
            self.forecasting_model.requires_grad_(False)

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

        if self.feature_adapter is not None:
            # Stage-II never updates Stage-I. Gradients are needed only after
            # projection2, with respect to the adapted representation.
            self.forecasting_model.eval()
            with torch.no_grad():
                representation, base_normalized = self.forecasting_model(x_normalized)
        else:
            representation, base_normalized = self.forecasting_model(x_normalized)

        base_revin = base_normalized * revin_std + revin_mean
        base_out = unnorm(base_revin, means, stds)
        y_unscale = norm(y, means, stds)
        y_unscale = (y_unscale - revin_mean) / revin_std.clamp_min(1e-4)

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

    @torch.no_grad()
    def fit_target_statistics(self, data_i):
        """Estimate fixed TCFA statistics from the target support set only."""
        if self.feature_adapter is None:
            raise RuntimeError("TCFA is not enabled for this model.")
        self.forecasting_model.eval()
        prepared = self.prepare_stage2(data_i)
        self.feature_adapter.set_target_statistics(prepared["representation"])
        return self.feature_adapter.diagnostics()

    def forward(self, data_i, stage='train', return_base=False):
        if self.feature_adapter is None:
            if stage == 'train':
                self.forecasting_model.train()
            else:
                self.forecasting_model.eval()
        else:
            self.forecasting_model.eval()
            if stage == 'train':
                self.feature_adapter.train()
            else:
                self.feature_adapter.eval()

        prepared = self.prepare_stage2(data_i)
        representation = prepared["representation"]
        base_normalized = prepared["base_normalized"]
        base_out = prepared["base_out"]
        y = prepared["y"]
        y_unscale = prepared["y_unscale"]

        feature_delta = None
        if self.feature_adapter is not None:
            adapted_representation = self.feature_adapter(representation)
            # Both supported forecasting backbones use the same 128 -> horizon
            # projection. Reusing the frozen projection keeps Stage-I unchanged.
            adapted_normalized = self.forecasting_model.projection2(
                adapted_representation
            )
            adapted_revin = (
                adapted_normalized * prepared["revin_std"]
                + prepared["revin_mean"]
            )
            out = unnorm(adapted_revin, prepared["means"], prepared["stds"])
            feature_delta = adapted_representation - representation
        else:
            out = base_out

        if return_base:
            return (
                out, y, representation, y_unscale,
                feature_delta, base_out, base_normalized,
            )
        return out, y, representation, y_unscale, feature_delta

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
        base_lr = float(getattr(args, "stage2_lr", 1e-3))
        adapter = self.model.feature_adapter
        if adapter is not None:
            self.main_optim = optim.AdamW(
                adapter.parameters(),
                lr=base_lr,
                weight_decay=float(getattr(args, "stage2_weight_decay", 1e-3)),
            )
        else:
            self.main_optim = optim.AdamW(
                self.model.parameters(), lr=base_lr,
                weight_decay=float(getattr(args, "stage2_weight_decay", 1e-3)),
            )
        self.trainable_params = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if adapter is not None:
            log_verbose(
                args,
                "[Stage-II][TCFA-V1] frozen Stage-I; trainable adapter parameters={}".format(
                    sum(parameter.numel() for parameter in self.trainable_params)
                )
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


    def fit_target_statistics(self, finetune_dataset):
        """Fit fixed feature statistics from every target support window."""
        if self.model.feature_adapter is None:
            raise RuntimeError("TCFA is not enabled.")
        # The target few-shot dataset contains the complete 3-day support set.
        # Force one complete batch so no support windows are silently dropped.
        target_name = finetune_dataset.data_list[0]
        support_count = int(finetune_dataset.x_list[target_name].shape[0])
        finetune_dataset.task_args = dict(finetune_dataset.task_args)
        finetune_dataset.task_args["batch_size"] = support_count
        data_i, _ = finetune_dataset[0]
        stats = self.model.fit_target_statistics(data_i)
        stats["support_samples"] = support_count
        return stats

    def _stage2_task_loss(self, data_i):
        out, y, _, _, feature_delta, _, _ = self.model(
            data_i, stage="train", return_base=True
        )
        forecast_loss = self.loss_criterion(out, y)
        adapter_reg = self.model.feature_adapter.regularization_loss()
        reg_weight = float(getattr(self.args, "stage2_adapter_reg", 1e-3))
        total = forecast_loss + reg_weight * adapter_reg
        with torch.no_grad():
            delta_rms = (
                float(feature_delta.detach().square().mean().sqrt().cpu())
                if feature_delta is not None else 0.0
            )
        return total, forecast_loss, out, y, delta_rms, adapter_reg

    def _meta_train_stage2(self, data, epoch):
        """Target-only TCFA adaptation; Stage-I remains frozen."""
        if not data:
            raise RuntimeError("TCFA Stage-II received an empty target batch.")
        if len(data) != 1:
            raise ValueError(
                "TCFA-V1 is target-only. Pass exactly one target support batch; "
                "source-city batches are not used in this version."
            )
        if self.model.feature_adapter is None:
            raise RuntimeError("TCFA adapter is not initialized.")
        if not bool(self.model.feature_adapter.statistics_ready.item()):
            raise RuntimeError(
                "Call fit_target_statistics() before starting TCFA training."
            )

        self.model.eval()
        self.model.feature_adapter.train()
        self.main_optim.zero_grad(set_to_none=True)
        total_loss, forecast_loss, target_out, target_y, delta_rms, adapter_reg = (
            self._stage2_task_loss(data[0])
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
        self.main_optim.step()
        self.main_optim.zero_grad(set_to_none=True)

        with torch.no_grad():
            _, rmse, mae, mape = calc_metric(target_out, target_y)
            adapter_diag = self.model.feature_adapter.diagnostics()
            metrics = {
                "total_loss": float(total_loss.detach().cpu()),
                "target_loss": float(forecast_loss.detach().cpu()),
                "target_mae": float(mae.detach().cpu()),
                "target_rmse": float(rmse.detach().cpu()),
                "target_mape": float(mape.detach().cpu()),
                "adapter_reg": float(adapter_reg.detach().cpu()),
                "feature_delta_rms": delta_rms,
                "adapter_scale_rms": adapter_diag["scale_rms"],
                "adapter_shift_rms": adapter_diag["shift_rms"],
            }
        if epoch % 5 == 0:
            print(
                "[Stage-II][TCFA-V1] epoch={} target_MAE={:.6f} "
                "target_loss={:.6f} feature_delta_rms={:.6e}".format(
                    epoch + 1, metrics["target_mae"],
                    metrics["target_loss"], metrics["feature_delta_rms"]
                )
            )
        return metrics

    def meta_train_revise(
        self, data, matrix=None, epoch=0, target=False, phase="target_adapt"
    ):
        """Dispatch to original Stage-I training or target-only TCFA Stage-II."""
        if self.model.feature_adapter is None:
            return self._meta_train_stage1(data, epoch)
        return self._meta_train_stage2(data, epoch)

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
    