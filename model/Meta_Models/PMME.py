import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import Parameter

import time
from copy import deepcopy
from torch.nn.utils import weight_norm
import math
import os

import sys
from pathlib import Path
sys.path.append('../../')
from utils import *
import datetime
from block import *

from geomloss.samples_loss import SamplesLoss 
from torch.utils.data import TensorDataset, DataLoader

from SOFTS.SOFTS_new import Model as softs
from iTransformer.iTransformer_new import Model as iTransformer

from DA_tools import *
from transfer_utility import VARIANTS, compute_weights
from feedback_utility import (
    blend_source_weights, evaluate_feedback_candidates, select_feedback_policy
)
from h1_shadow_audit import audit_h1_shadow_update
from h2_candidate_audit import (
    detached_model_gradients, evaluate_h2_candidate_audit
)
from gradient_pair_ot import (
    VARIANTS as PAIR_VARIANTS, build_pairwise_cost,
    assign_city_target_groups
)

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
        
        # F1-F4 replace DVGP with forecasting-task gradient utilities.
        # The backbone, Sinkhorn operator, Stage-II and training runner are not changed.
        self.transfer_variant = os.environ.get(
            "PMME_TRANSFER_VARIANT", "uniform"
        ).strip().lower()
        if self.transfer_variant not in VARIANTS:
            raise ValueError("PMME_TRANSFER_VARIANT must be one of " + str(VARIANTS))
        self._capture_transfer = False
        self._transfer_records = []

        def _record_projection2(_module, inputs, output):
            if self._capture_transfer:
                self._transfer_records.append((inputs[0], output))
        self._transfer_hook = self.model.forecasting_model.projection2.register_forward_hook(
            _record_projection2
        )
        # Separate sample selection from OT optimization strength.
        # All F1-F4 comparisons must use the same fixed scale.
        self.transfer_ot_scale = float(os.environ.get(
            "PMME_TRANSFER_OT_SCALE", "0.01"
        ))
        if not math.isfinite(self.transfer_ot_scale) or self.transfer_ot_scale <= 0:
            raise ValueError("PMME_TRANSFER_OT_SCALE must be finite and positive")
        self.transfer_temperature = float(os.environ.get(
            "PMME_TRANSFER_TEMPERATURE", "8.0"
        ))
        if not math.isfinite(self.transfer_temperature) or self.transfer_temperature <= 0:
            raise ValueError("PMME_TRANSFER_TEMPERATURE must be finite and positive")
        self.transfer_min_ess = float(os.environ.get(
            "PMME_TRANSFER_MIN_ESS", "0.5"
        ))
        if not 0 < self.transfer_min_ess <= 1:
            raise ValueError("PMME_TRANSFER_MIN_ESS must be in (0, 1]")
        # G0: exact F2 route. G1: probe only. G2/G3: feedback-controlled
        # F2/Uniform blending with optional no-OT fallback, per source city.
        self.feedback_mode = os.environ.get(
            "PMME_FEEDBACK_MODE", "g0"
        ).strip().lower()
        if self.feedback_mode not in ("g0", "g1", "g2", "g3", "h1", "h2a"):
            raise ValueError("PMME_FEEDBACK_MODE must be g0/g1/g2/g3/h1/h2a")
        # H2: full-batch candidate diagnostics, without adjusting F2 OT.
        self.h2_citywise = os.environ.get(
            "PMME_H2_CITYWISE", "1"
        ).strip() not in ("0", "false", "False")
        self.h2_gain_tol = float(os.environ.get(
            "PMME_H2_GAIN_TOL", "1e-6"
        ))
        if not math.isfinite(self.h2_gain_tol) or self.h2_gain_tol < 0:
            raise ValueError("PMME_H2_GAIN_TOL must be finite and >=0")
        if self.transfer_variant != "gradient":
            raise ValueError("G0-G3 require PMME_TRANSFER_VARIANT=gradient")
        self.feedback_interval = int(os.environ.get(
            "PMME_FEEDBACK_INTERVAL", "200"
        ))
        self.feedback_source_n = int(os.environ.get(
            "PMME_FEEDBACK_SOURCE_N", "256"
        ))
        self.feedback_target_n = int(os.environ.get(
            "PMME_FEEDBACK_TARGET_N", "128"
        ))
        self.feedback_step = float(os.environ.get(
            "PMME_FEEDBACK_STEP", "0.05"
        ))
        self.feedback_max_norm = float(os.environ.get(
            "PMME_FEEDBACK_MAX_NORM", "0.02"
        ))
        self.feedback_risk = float(os.environ.get(
            "PMME_FEEDBACK_RISK", "1.0"
        ))
        if (self.feedback_interval < 1 or self.feedback_source_n < 1 or
                self.feedback_target_n < 1 or
                not all(math.isfinite(v) and v > 0 for v in
                        (self.feedback_step, self.feedback_max_norm)) or
                not math.isfinite(self.feedback_risk) or self.feedback_risk < 0):
            raise ValueError("Invalid PMME_FEEDBACK_* configuration")
        self.feedback_choices = {}
        self._feedback_inputs = []
        self._capture_feedback = False

        def _record_projection1(_module, inputs, _output):
            if self._capture_feedback:
                # Pre-projection1 inputs provide the EXACT linear Jacobian.
                self._feedback_inputs.append(inputs[0].detach())
        self._feedback_hook = (
            self.model.forecasting_model.projection1.register_forward_hook(
                _record_projection1
            )
        )
        # Four independent F2 Stage-I training variants; no feedback controller.
        self.pair_variant = os.environ.get("PMME_PAIR_VARIANT", "f2").strip().lower()
        if self.pair_variant not in PAIR_VARIANTS:
            raise ValueError("PMME_PAIR_VARIANT must be one of " + str(PAIR_VARIANTS))
        if self.pair_variant != "f2" and (
            self.transfer_variant != "gradient" or self.feedback_mode != "g0"
        ):
            raise ValueError("P1-P4 require F2 and PMME_FEEDBACK_MODE=g0")
        self.pair_strength = float(os.environ.get("PMME_PAIR_STRENGTH", "0.02"))
        self.pair_anchor_count = int(os.environ.get("PMME_PAIR_ANCHORS", "8"))
        self.pair_block_size = int(os.environ.get("PMME_PAIR_BLOCK_SIZE", "256"))
        self.p4_momentum = float(os.environ.get("PMME_P4_MOMENTUM", "0.8"))
        self.p4_repeat_penalty = float(os.environ.get("PMME_P4_REPEAT_PENALTY", "0.08"))
        if (not math.isfinite(self.pair_strength) or
                not 0 <= self.pair_strength <= 1 or
                self.pair_anchor_count < 1 or self.pair_block_size < 1 or
                not 0 <= self.p4_momentum < 1 or
                not math.isfinite(self.p4_repeat_penalty) or
                self.p4_repeat_penalty < 0):
            raise ValueError("Invalid PMME_PAIR_* or PMME_P4_* setting")
        self.p4_assignment_ema = {}
        self.p4_assignment_counts = {}
        print("PMME_PAIR_VARIANT", self.pair_variant,
              "strength", self.pair_strength,
              "anchors", self.pair_anchor_count,
              "block", self.pair_block_size)
        print("PMME_STAGE1_TRANSFER_VARIANT", self.transfer_variant)
        print("PMME_STAGE1_FEEDBACK", self.feedback_mode,
              "interval", self.feedback_interval,
              "subset", self.feedback_source_n, self.feedback_target_n)
        print("PMME_STAGE1_TRANSFER_CONFIG", "ot_scale", self.transfer_ot_scale,
              "temperature", self.transfer_temperature,
              "min_ess", self.transfer_min_ess)

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

        # Only capture the final forecast head when OT is actually active.
        need_transfer = (
            self.DA == 'OT' and epoch >= self.DAepoch_args
            and self.transfer_variant != 'uniform'
        )
        self._transfer_records = []
        self._capture_transfer = need_transfer
        # Do not capture another full set of encoder inputs at every step.
        need_feedback = (
            need_transfer and self.feedback_mode != "g0" and
            (epoch - self.DAepoch_args) % self.feedback_interval == 0
        )
        self._feedback_inputs = []
        # H2 needs none of the old G/H1 local-head hooks. Candidate OT
        # gradients are read from the actual full-sized Sinkhorn graph.
        h2_active = need_feedback and self.feedback_mode == "h2a"
        h2_city_gradients = []
        self._capture_feedback = need_feedback and not h2_active
        target_probe_prediction = None
        target_probe_groundtruth = None
        target_probe_scale = None
        target_fit_loss = None
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
                if need_feedback and not h2_active:
                    # Internal decision-only split: probe labels must not
                    # enter the local proxy's target alignment weights or
                    # one-sided projection direction.
                    fit_window_mask = (
                        torch.arange(out.shape[0], device=out.device) % 5 != 0
                    )
                    target_fit_loss = self.loss_criterion(
                        out[fit_window_mask], y[fit_window_mask]
                    )
                    # ReVIN + data unnormalization is an affine scale after
                    # projection2. Record it before the next city's forward.
                    global_std = torch.as_tensor(
                        data[i].stds, device=out.device, dtype=out.dtype
                    ).reshape(-1)[0]
                    target_probe_scale = (
                        self.model.RevIn_std.detach() * global_std
                    )
                    target_probe_prediction = out.detach()
                    target_probe_groundtruth = y.detach()
            
            
        self._capture_transfer = False
        self._capture_feedback = False
        if need_feedback and not h2_active and len(self._feedback_inputs) != len(data):
            raise RuntimeError(
                "Expected one pre-projection1 record per city, got %d" %
                len(self._feedback_inputs)
            )
        if need_feedback and not h2_active:
            # Fixed support split for all candidate policies.
            # With original 15-window per-city target grouping, every fifth
            # window is a candidate evaluation probe. The proxy OT never
            # contains those windows. Main PMME training remains unchanged.
            target_windows = self._feedback_inputs[0].shape[0]
            probe_idx = torch.arange(
                0, target_windows, 5, device=self.device
            )
            probe_pre = self._feedback_inputs[0].index_select(
                0, probe_idx
            )
            probe_pred = target_probe_prediction.index_select(
                0, probe_idx
            )
            probe_true = target_probe_groundtruth.index_select(
                0, probe_idx
            )
            probe_scale = target_probe_scale.index_select(
                0, probe_idx
            )
            if self.feedback_mode in ("g1", "g2", "g3"):
                shared_head = self.model.forecasting_model.projection1.weight
                task_head_gradient = torch.autograd.grad(
                    sum(task_losses), shared_head, retain_graph=True
                )[0].detach()
                target_head_gradient = torch.autograd.grad(
                    target_fit_loss, shared_head, retain_graph=True
                )[0].detach()
                head2_weights = (
                    self.model.forecasting_model.projection2.weight.detach()
                )

        if need_transfer:
            if len(self._transfer_records) != len(data):
                raise RuntimeError(
                    "Expected one shared-head forward per city, got %d" %
                    len(self._transfer_records)
                )
            target_features, target_head_output = self._transfer_records[0]
            target_deltas = torch.autograd.grad(
                target_task_loss, target_head_output, retain_graph=True
            )[0].detach()
            target_split = [15, 15, target_features.shape[0] - 30]
            target_feature_groups = torch.split(
                target_features.detach(), target_split, dim=0
            )
            target_delta_groups = torch.split(
                target_deltas, target_split, dim=0
            )

        # P4: balanced, task-aware source-city to target-window assignment.
        p4_target_assignment = list(range(len(rep_source)))
        p4_source_deltas = []
        if need_transfer and self.pair_variant == "p4" and self.pair_strength > 0:
            p4_source_features = [
                record[0] for record in self._transfer_records[1:]
            ]
            for k, (_, head_out) in enumerate(self._transfer_records[1:]):
                p4_source_deltas.append(torch.autograd.grad(
                    task_losses[k], head_out, retain_graph=True
                )[0].detach())
            p4_target_assignment, p4_stats = assign_city_target_groups(
                p4_source_features, p4_source_deltas,
                target_feature_groups, target_delta_groups,
                [str(data[k + 1].data_name) for k in range(len(rep_source))],
                ema=self.p4_assignment_ema,
                counts=self.p4_assignment_counts,
                momentum=self.p4_momentum,
                repeat_penalty=self.p4_repeat_penalty
            )
            if epoch % 25 == 0:
                print("pair_ot_assignment", "epoch", epoch, p4_stats)

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
            target_group = p4_target_assignment[index]

            size1, size2 = rep_source[index].shape[0]*rep_source[index].shape[1], rep_target[target_group].shape[0]*rep_target[target_group].shape[1]
            # print(size1, size2)

            x1,y1 = rep_source[index], y_source[index] 
            x2,y2 = rep_target[target_group], y_target[target_group]
            shape0 = x1.shape[1]
                    
            y_norm_sources.append(y1)
            y_norm_targets.append(y2)

            x1 = x1.view(-1, x1.size(-1))
            x2 = x2.view(-1, x2.size(-1))
            y1 = y1.view(-1, y1.size(-1))
            y2 = y2.view(-1, y2.size(-1))
            

            # F1: uniform source/target OT marginals.
            # F2-F4: task-conditioned source marginals; target stays uniform.
            # No domain classifier, BCE, pseudo-domain labels or DVGP.
            alpha = torch.full((size1,), 1.0 / size1, device=x1.device)
            beta = torch.full((size2,), 1.0 / size2, device=x2.device)
            OT_loss_scale = self.transfer_ot_scale
            ratio1, ratio2 = 1.0, 1.0
            if need_transfer:
                source_features, source_head_output = self._transfer_records[index + 1]
                source_deltas = (
                    p4_source_deltas[index]
                    if self.pair_variant == "p4" else
                    torch.autograd.grad(
                        task_losses[index], source_head_output,
                        retain_graph=True
                    )[0].detach()
                )
                alpha, beta, utility_stats = compute_weights(
                    source_features.detach(), source_deltas,
                    target_feature_groups[target_group],
                    target_delta_groups[target_group],
                    self.transfer_variant,
                    temperature=self.transfer_temperature,
                    min_ess_fraction=self.transfer_min_ess
                )
                if epoch % 25 == 0:
                    print("task_utility", self.transfer_variant, index, utility_stats)

                if need_feedback and self.feedback_mode in ("g1", "g2", "g3"):
                    # Proxy OT and held-out within-batch probe use disjoint
                    # target WINDOWS. Existing official full OT remains intact.
                    target_group_rep = rep_target[index].detach()
                    target_group_y = y_target[index].detach()
                    target_group_pre = torch.split(
                        self._feedback_inputs[0],
                        [15, 15, self._feedback_inputs[0].shape[0] - 30],
                        dim=0
                    )[index]
                    allowed = (
                        torch.arange(
                            target_group_rep.shape[0],
                            device=target_group_rep.device
                        ) % 5 != 0
                    )
                    # All target observations used here still come from
                    # target's first three days; no later test labels.
                    subset_r = target_group_rep[allowed]
                    subset_y = target_group_y[allowed]
                    subset_pre = target_group_pre[allowed]
                    source_pre = self._feedback_inputs[index + 1]
                    # The main F2 training weights intentionally use the
                    # official full support batch (G0 unchanged). The
                    # diagnostic/decision candidates are recomputed from
                    # only the non-probe target windows to avoid direct
                    # probe-label contamination of their similarity scores.
                    probe_f2_alpha, _, probe_weight_stats = compute_weights(
                        source_features.detach(), source_deltas,
                        target_feature_groups[index][allowed],
                        target_delta_groups[index][allowed],
                        "gradient",
                        temperature=self.transfer_temperature,
                        min_ess_fraction=self.transfer_min_ess
                    )
                    results = evaluate_feedback_candidates(
                        source_rep=x1.detach(),
                        target_rep=subset_r.reshape(-1, subset_r.shape[-1]),
                        source_y=y1.detach(),
                        target_y=subset_y.reshape(-1, subset_y.shape[-1]),
                        source_pre=source_pre.reshape(
                            -1, source_pre.shape[-1]
                        ),
                        target_pre=subset_pre.reshape(
                            -1, subset_pre.shape[-1]
                        ),
                        f2_alpha=probe_f2_alpha.detach(),
                        target_probe_pre=probe_pre,
                        target_probe_output=probe_pred,
                        target_probe_y=probe_true,
                        target_probe_scale=probe_scale,
                        projection2_weight=head2_weights,
                        task_head_grad=task_head_gradient,
                        target_head_grad=target_head_gradient,
                        criterion=criterion,
                        ot_coefficient=(
                            self.DAcoef_args * self.transfer_ot_scale
                        ),
                        xy_scale=math.sqrt(self.xySquaredRatio_args),
                        n_source=self.feedback_source_n,
                        n_target=self.feedback_target_n,
                        step_size=self.feedback_step,
                        max_step_norm=self.feedback_max_norm,
                        risk_penalty=self.feedback_risk,
                    )
                    if self.feedback_mode in ("g2", "g3"):
                        selected = select_feedback_policy(
                            results, self.feedback_mode
                        )
                        self.feedback_choices[int(shape0)] = selected
                    else:
                        selected = 1.0  # G1 diagnostics only, same training as G0.
                    print("transfer_feedback", self.feedback_mode,
                          "epoch", epoch, "source_nodes", int(shape0),
                          "selected", selected,
                          "probe_weight", probe_weight_stats,
                          "results", results)

                if self.feedback_mode in ("g2", "g3"):
                    selected = self.feedback_choices.get(int(shape0), 1.0)
                    if selected == "off":
                        OT_loss_scale = 0.0
                    else:
                        alpha = blend_source_weights(alpha, float(selected))
            elif epoch % 25 == 0:
                print("task_utility", self.transfer_variant, index,
                      "uniform or OT not active")

            if self.DA == 'OT':
                if (epoch>=self.DAepoch_args): 
                    
                    x_y_scale = math.sqrt(self.xySquaredRatio_args)
                    y1, y2 = y1*x_y_scale, y2*x_y_scale
                    
                    x = torch.cat((x1,y1),dim=-1) 
                    y = torch.cat((x2,y2),dim=-1) # y: target  
                    
                    if OT_loss_scale == 0.0:
                        # G2/G3 "off": skip the expensive full OT calculation.
                        OT_loss = x1.new_zeros(())
                    elif (ratio1>0.00 and ratio2>0.00):
                        ot_operator = criterion
                        if (self.pair_variant in ("p1", "p2", "p3") and
                                self.pair_strength > 0):
                            pair_args = {}
                            if self.pair_variant == "p2":
                                # Dataset history is city-standardized speed;
                                # convert physical future to the SAME units.
                                def standardized_future(city_data):
                                    truth = torch.as_tensor(
                                        city_data.y, device=x.device, dtype=x.dtype)
                                    mean = torch.as_tensor(
                                        city_data.means, device=x.device,
                                        dtype=x.dtype).reshape(1, 1, 1)
                                    std = torch.as_tensor(
                                        city_data.stds, device=x.device,
                                        dtype=x.dtype).reshape(1, 1, 1)
                                    return (truth - mean) / std.clamp_min(1e-6)

                                target_hist = torch.as_tensor(
                                    data[0].x, device=x.device,
                                    dtype=x.dtype)[..., 0]
                                target_hist_groups = torch.split(
                                    target_hist, target_split, dim=0)
                                target_future_groups = torch.split(
                                    standardized_future(data[0]),
                                    target_split, dim=0)
                                source_hist = torch.as_tensor(
                                    data[index + 1].x, device=x.device,
                                    dtype=x.dtype)[..., 0]
                                target_h = target_hist_groups[target_group]
                                pair_args = {
                                    "source_history": source_hist.reshape(
                                        -1, source_hist.shape[-1]),
                                    "source_future": standardized_future(
                                        data[index + 1]).reshape(-1, 36),
                                    "target_history": target_h.reshape(
                                        -1, target_h.shape[-1]),
                                    "target_future": target_future_groups[
                                        target_group].reshape(-1, 36),
                                }
                            pair_cost, pair_stats = build_pairwise_cost(
                                variant=self.pair_variant, x=x, y=y,
                                source_rep=x1, target_rep=x2,
                                source_delta=source_deltas.reshape(-1, 36),
                                target_delta=target_delta_groups[
                                    target_group].reshape(-1, 36),
                                source_alpha=alpha,
                                strength=self.pair_strength,
                                anchor_count=self.pair_anchor_count,
                                block_size=self.pair_block_size,
                                **pair_args
                            )
                            ot_operator = SamplesLoss(
                                loss="sinkhorn", p=1, backend="tensorized",
                                debias=False, blur=0.001, cost=pair_cost
                            )
                            if epoch % 25 == 0:
                                print("pair_ot_cost", "epoch", epoch,
                                      "city", str(data[index + 1].data_name),
                                      "target_group", target_group,
                                      pair_stats, flush=True)
                        OT_loss = (torch.sum(ot_operator(alpha, x, beta, y)) *
                                   self.DAcoef_args * OT_loss_scale)
                    else:
                        OT_loss = 0
                    DA_losses.append(OT_loss)

                    if h2_active:
                        # H2-A1: original full-sized, same y and same OT
                        # strength; ONLY sample marginals differ. Record
                        # detached all-parameter OT gradient tuples.
                        coefficient = self.DAcoef_args * self.transfer_ot_scale
                        uniform_alpha = torch.full_like(
                            alpha, 1.0 / alpha.numel()
                        )
                        half_alpha = blend_source_weights(alpha, 0.5)
                        f2_grad = detached_model_gradients(
                            OT_loss, self.model
                        )
                        ot_uniform = torch.sum(
                            criterion(uniform_alpha, x, beta, y)
                        ) * coefficient
                        uniform_grad = detached_model_gradients(
                            ot_uniform, self.model
                        )
                        del ot_uniform
                        ot_mixed = torch.sum(
                            criterion(half_alpha, x, beta, y)
                        ) * coefficient
                        mixed_grad = detached_model_gradients(
                            ot_mixed, self.model
                        )
                        del ot_mixed
                        h2_city_gradients.append({
                            "city": str(data[index + 1].data_name),
                            # OT target 15-window block follows randomized
                            # source mini-batch ordering each epoch.
                            "target_group": int(index),
                            "f2": f2_grad,
                            "uniform": uniform_grad,
                            "mixed": mixed_grad,
                        })

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

        # H2-A1 global, H2-A2 city-wise and H2-A3 temporal fold checks.
        # Shadow PCGrad/AdamW only; actual F2 optimization stays unchanged.
        if h2_active:
            report = evaluate_h2_candidate_audit(
                model=self.model,
                optimizer=self.main_optim,
                source_task_loss=task_loss,
                target_task_loss=target_task_loss,
                city_gradients=h2_city_gradients,
                target_data=data[0],
                epoch=epoch,
                beta=self.loss_criterion.beta,
                citywise=self.h2_citywise,
                confidence_tolerance=self.h2_gain_tol,
            )
            print("h2_candidate_audit", report, flush=True)

        # H1 audits the ORIGINAL F2 OT update, without changing optimizer
        # state, model weights, or which OT marginal is used for training.
        # Shadow A = source task only; shadow B = source task plus full F2 OT.
        # All-parameter PCGrad and the current AdamW state are replayed.
        if need_feedback and self.feedback_mode == "h1":
            audit_record = audit_h1_shadow_update(
                model=self.model,
                optimizer=self.main_optim,
                source_task_loss=task_loss,
                da_loss=DA_loss,
                target_task_loss=target_task_loss,
                target_data=data[0],
                probe_input=probe_pre,
                probe_prediction=probe_pred,
                probe_target=probe_true,
                probe_scale=probe_scale,
                probe_indices=probe_idx,
                epoch=epoch,
                beta=self.loss_criterion.beta
            )
            print("h1_shadow_audit", audit_record, flush=True)

        # Diagnostic ONLY: OT acts on the shared representation upstream of
        # projection2; its gradient on projection2.weight is necessarily zero.
        # Inspect projection1.weight where BOTH OT and prediction can act.
        # This does not modify the model gradients or its optimizer step.
        if self.DA == 'OT' and epoch >= self.DAepoch_args and epoch % 25 == 0:
            shared_parameter = self.model.forecasting_model.projection1.weight
            if isinstance(DA_loss, torch.Tensor) and DA_loss.requires_grad:
                ot_grad = torch.autograd.grad(
                    DA_loss, shared_parameter,
                    retain_graph=True, allow_unused=True
                )[0]
                target_grad = torch.autograd.grad(
                    target_task_loss, shared_parameter,
                    retain_graph=True, allow_unused=True
                )[0]
                if ot_grad is not None and target_grad is not None:
                    g_ot = ot_grad.detach().float().reshape(-1)
                    g_target = target_grad.detach().float().reshape(-1)
                    norm_ot, norm_target = g_ot.norm(), g_target.norm()
                    norm_ratio = (norm_ot / (norm_target + 1e-12)).item()
                    cosine = (
                        (g_ot @ g_target) /
                        (norm_ot * norm_target + 1e-12)
                    ).item()
                    print("task_utility_gradient_diagnostic",
                          "ot_to_target_norm_ratio", norm_ratio,
                          "ot_vs_target_cosine", cosine,
                          "ot_scale", self.transfer_ot_scale)
                else:
                    print("task_utility_gradient_diagnostic missing_gradient")

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
        
            
        self._transfer_records = []
        self._feedback_inputs = []
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
    