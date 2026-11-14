import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import traffic_dataset
from utils import *
import argparse
import yaml
import time
import sys
import json
sys.path.append('./model')
sys.path.append('./model/Meta_Models')

from PMME import *
from pathlib import Path
import random


def summarize_test_metrics(model, test_dataset, pred_num):
    """Aggregate held-out metrics by forecast horizon for experiment records."""
    raw_mse, raw_rmse, raw_mae, raw_mape, _ = model.test_batch(
        0, len(test_dataset) + 1, test_dataset, "test"
    )
    metrics = []
    for horizon in range(pred_num):
        total_len = 0
        mse_values = []
        mae_values = []
        mape_values = []
        for batch_mse, batch_rmse, batch_mae, batch_mape in zip(
            raw_mse, raw_rmse, raw_mae, raw_mape
        ):
            batch_size = int(batch_mse[1])
            total_len += batch_size
            mse_values.append(float(batch_mse[0][horizon]) * batch_size)
            mae_values.append(float(batch_mae[0][horizon]) * batch_size)
            mape_values.append(float(batch_mape[0][horizon]) * batch_size)
        mse = sum(mse_values) / max(total_len, 1)
        mae = sum(mae_values) / max(total_len, 1)
        mape = sum(mape_values) / max(total_len, 1)
        metrics.append({
            "horizon_index": horizon,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": mae,
            "mape": mape,
        })
    return metrics


def load_stage1_model_for_evaluation(model_path, eval_model):
    checkpoint = torch.load(model_path, map_location=eval_model.device)
    model_state = eval_model.model.state_dict()
    filtered_state = {
        key[len("model."):]: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("model.")
    }
    model_state.update({
        key: value for key, value in filtered_state.items()
        if key in model_state
    })
    eval_model.model.load_state_dict(model_state)
    return checkpoint


def save_stage1_test_result(
    model_path,
    finetune_dataset,
    test_dataset,
    args,
    config,
    data_list,
    model_args,
    task_args,
    data_args,
    PatchFSL_cfg,
    best_stage1_epoch,
    best_stage1_source_mae,
    best_stage1_selection_score=None,
    stage1_selection_metric='source_mae',
):
    """Evaluate the selected Stage-I checkpoint and persist a complete JSON record."""
    eval_model = STRep(
        args.classifier,
        args.DAepoch,
        args.DAcoef,
        args.xySquaredRatio,
        0,
        0,
        args.model_name,
        args.DA,
        data_args,
        task_args,
        model_args,
        PatchFSL_cfg,
        args,
    )
    checkpoint = load_stage1_model_for_evaluation(model_path, eval_model)
    eval_model.eval()
    metrics = summarize_test_metrics(
        eval_model,
        test_dataset,
        task_args["source"]["pred_num"],
    )
    selected_indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
    selected = [metrics[index] for index in selected_indices]
    result = {
        "experiment": args.description,
        "seed": int(args.seed),
        "target_dataset": args.test_dataset,
        "source_datasets": [str(item) for item in data_list],
        "model_name": args.model_name,
        "stage1_only": True,
        "stage1_method": args.stage1_method,
        "stage1_transfer": int(args.stage1_transfer),
        "stage1_selection_metric": stage1_selection_metric,
        "best_epoch": int(best_stage1_epoch),
        "best_source_mae": float(best_stage1_source_mae),
        "best_selection_score": float(
            best_stage1_source_mae
            if best_stage1_selection_score is None
            else best_stage1_selection_score
        ),
        "checkpoint": str(model_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", best_stage1_epoch)),
        "config": config,
        "metrics_all_horizons": metrics,
        "selected_horizon_indices": selected_indices,
        "metrics_selected_horizons": selected,
        "selected_mae_sum": float(sum(item["mae"] for item in selected)),
    }
    result_path = Path(args.save_dir) / f"stage1_test_metrics_seed{args.seed}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
    print(f"[RESULT] Stage-I metrics saved to {result_path}")
    for item in selected:
        print(
            "[RESULT] horizon={horizon_index} MSE={mse:.6f} RMSE={rmse:.6f} "
            "MAE={mae:.6f} MAPE={mape:.6f}".format(**item)
        )
    return result

curr_time = time.strftime('%Y-%m-%d-%H-%M-%S')

parser = argparse.ArgumentParser(description='PMME')
parser.add_argument('--config_filename', default='./configs/config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='pems-bay', type=str)
parser.add_argument('--train_epochs', default=0, type=int)
parser.add_argument('--finetune_epochs', default=1, type=int, help='Legacy Stage-II enable flag; the low-rank adapter is fitted once in closed form, not iteratively.')
parser.add_argument('--description', default='PMME', type=str)
parser.add_argument(
    '--evaluate_checkpoint',
    default='',
    type=str,
    help='Evaluate an existing Stage-I checkpoint without retraining.',
)

parser.add_argument('--seed',default=7,type=int)
parser.add_argument('--data_list', default='chengdu_shenzhen_metr',type=str)
parser.add_argument('--target_days', default=3,type=int)
parser.add_argument('--gpu', default=0, type = int)
parser.add_argument('--STmodel',default='SOFTS',type=str)
parser.add_argument('--his_num',default=288,type=int)
parser.add_argument('--save_dir', type=str)

parser.add_argument('--classifier',default=0.3,type=float)
parser.add_argument('--xySquaredRatio',default=10,type=float) 
parser.add_argument('--DAcoef',default=0.1,type=float)
parser.add_argument('--DAepoch',default=50000,type=int)
parser.add_argument('--DAwarmup',default=500,type=int,
                    help='Stage-I OT warmup epochs after DAepoch.')
parser.add_argument('--stage1_mass_temperature',default=2.0,type=float,
                    help='Temperature for smoothing Stage-I transport masses.')
parser.add_argument('--stage1_uniform_mass',default=0.75,type=float,
                    help='Uniform mass mixture preventing classifier collapse.')
parser.add_argument('--stage1_reliability_mix',default=0.25,type=float,
                    help='Small geometric reliability mixture; avoids double gating.')
parser.add_argument('--stage1_transfer', default=1, type=int,
                    help='Enable the current Stage-I transfer module; set 0 for a pure SOFTS baseline.')
parser.add_argument(
    '--stage1_selection_metric',
    default='source_mae',
    choices=['source_mae', 'target_support_mae'],
    help=(
        'Checkpoint metric for Stage-I. target_support_mae uses only the '
        'observed target support prefix, never the held-out target test split.'
    ),
)
parser.add_argument('--stage1_method', default='pmme', choices=['pmme', 'rhsr'],
                    help='Stage-I method: PMME path or non-PMME RHSR path.')
parser.add_argument('--rhsr_num_regimes', default=4, type=int)
parser.add_argument('--rhsr_hidden_dim', default=96, type=int)
parser.add_argument('--rhsr_route_temperature', default=1.5, type=float)
parser.add_argument('--rhsr_router_uniform_mix', default=0.0, type=float)
parser.add_argument('--rhsr_route_logit_clip', default=0.0, type=float)
parser.add_argument('--rhsr_target_anchor', default=0.1, type=float)
parser.add_argument('--rhsr_state_weight', default=0.1, type=float)
parser.add_argument('--rhsr_decomposition_weight', default=0.1, type=float)
parser.add_argument('--rhsr_route_balance_weight', default=0.01, type=float)
parser.add_argument('--MemSize', default=1, type=int, help='Legacy enable flag: any value > 0 enables the target-only low-rank Stage-II adapter; no memory slots are allocated.')
parser.add_argument('--stage2_feature_bins', default=12, type=int, help='Number of history segments used by the lightweight Stage-II adapter.')
parser.add_argument('--stage2_residual_rank', default=6, type=int, help='Total number of low-rank residual basis vectors.')
parser.add_argument('--stage2_residual_smooth_rank', default=3, type=int, help='Number of fixed low-frequency residual basis vectors.')
parser.add_argument('--stage2_residual_ridge', default=10.0, type=float, help='Ridge regularization for the closed-form residual coefficient map.')
parser.add_argument('--projection',default=0,type=float)
parser.add_argument('--model_name',default='SOFTS',type=str)
parser.add_argument('--DA',default='OT',type=str)
args = parser.parse_args()

args.curr_time = curr_time
print('args.test_dataset',args.test_dataset)
if args.save_dir:
    save_dir = Path(args.save_dir)
else:
    save_dir = Path(os.path.join(
        ".", "logged_files", args.test_dataset, args.data_list, args.curr_time
    ))
save_dir.mkdir(parents=True, exist_ok=True)
args.save_dir = save_dir

#a = torch.rand(8000, 1000, 1000, device='cuda:0')

args.new=1
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float32)


if __name__ == '__main__':
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        log_verbose(args,"INFO: GPU : {}".format(args.gpu))
    else:
        args.device = torch.device('cpu')
        log_verbose(args,"INFO: CPU")

    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['task']['source']['train_epochs'] = args.train_epochs
    config['task']['source']['finetune_epochs'] = args.finetune_epochs

    args.batch_size = config['task']['source']['batch_size']

    data_args, task_args, model_args = config['data'], config['task'], config['model']
    data_list = get_data_list(args.data_list)

    log_verbose(args,"INFO: train on {}. test on {}.".format(data_list, args.test_dataset))
    PatchFSL_cfg = {
        'data_list' : args.data_list,
        'base_dir': Path(sys.path[0]),
        'device':args.device
    }

    ## dataset
    # Stage-II source batches must never sample the target city. Pass a copy
    # and disable the dataset helper's default target append to avoid leakage.
    source_dataset = traffic_dataset(
        data_args,
        task_args['source'],
        data_list.copy(),
        "source",
        test_data=args.test_dataset,
        add_target=False,
    )
    if args.test_dataset in source_dataset.data_list:
        raise RuntimeError(
            f"Target city {args.test_dataset} leaked into the source dataset: "
            f"{source_dataset.data_list}"
        )
    log_verbose(args, f"Verified source-only training cities: {source_dataset.data_list}")
    ## check source dataset

    for data in source_dataset.data_list:
        log_verbose(args,"source dataset has {}. X : {}, y : {}".format(data,source_dataset.x_list[data].shape,source_dataset.y_list[data].shape))
    print('data_args', data_args)
    finetune_dataset = traffic_dataset(data_args, task_args['target_few_shot'], data_list, 'target_maml', test_data=args.test_dataset, target_days=args.target_days)
    # Construct the held-out target test split only after Stage-II adaptation.
    log_verbose(args,"data args : {}\ntask_args : {}\nmodel_args : {}\nPatchFSL_cfg : {}\nSTmodel : {}".format(data_args, task_args, model_args, PatchFSL_cfg, args.STmodel))

    rep_model = STRep(args.classifier, args.DAepoch, args.DAcoef, args.xySquaredRatio, 0, args.projection, args.model_name, args.DA, data_args, task_args, model_args, PatchFSL_cfg, args) 
    
    # rep_model = STRep(data_args, task_args, model_args, PatchFSL_cfg, args)
    best_loss = 9999999999999.0
    best_model = None
    ## train on big dataset
    rep_tasknum = task_args['source']['task_num']
    
    # Stage-I must not select a checkpoint using the held-out target test set.
    # The default is source forecasting MAE. An optional target-support metric
    # uses only the observed target prefix and is useful for checking whether
    # source loss is a poor proxy for target transfer quality.
    best_stage1_source_mae = float('inf')
    best_stage1_selection_score = float('inf')
    best_stage1_epoch = -1
    
    log_verbose(args,dict(vars(args)))
    log_verbose(args,config)

    yaml.dump([dict(vars(args)), config], open(os.path.join(args.save_dir, "config.yaml"), "w"))
    
    for i in range(task_args['source']['train_epochs']):
        length = source_dataset.__len__()
        # length=40
        if i%25==0:
            log_verbose(args,'----------------------')
        time_1 = time.time()
        
        data = []
        matrix = []
        idx = 0
        #a = time.time()
        current_shape = set()

        data_i, A = finetune_dataset[idx] 
        # print('fffffffffffffff')
        data.append(data_i)
        matrix.append(A)
        current_shape.add(A.shape[1])
        
        for idx in range(args.target_days):
            while(True):
                data_i, A = source_dataset[idx] 
                # print(A.shape[1], matrix[0].shape[1])
                if(A.shape[1]) not in current_shape:
                    current_shape.add(A.shape[1])
                    # print('true for read') 
                    break
            # print('use', A.shape)
            data.append(data_i)
            matrix.append(A)
            idx+=1
        
        if True:
            model_loss ,mse_loss, rmse_loss, mae_loss, mape_loss = rep_model.meta_train_revise(data, matrix, i)
    
            if i%25==0:
                log_verbose(args,'Epochs {}/{}'.format(i,task_args['source']['train_epochs']))
                log_verbose(args,'in meta-training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, reconstruction Loss : {:.5f}.'.format(mse_loss, rmse_loss, mae_loss,mape_loss,model_loss))
                if args.stage1_method == 'rhsr':
                    diagnostics = {
                        'epoch': int(i),
                        'model_loss': float(model_loss),
                        **getattr(rep_model, 'last_rhsr_diagnostics', {}),
                    }
                    with open(
                        Path(args.save_dir) / 'rhsr_diagnostics.jsonl',
                        'a',
                        encoding='utf-8',
                    ) as handle:
                        handle.write(json.dumps(diagnostics, ensure_ascii=False) + '\n')
            log_verbose(args,"This epoch cost {:.3}s.".format(time.time() - time_1))
            
            # if((i+1)%50==0):
            if((i+1)%100==0)or i==0:
                selection_score = float(mae_loss)
                target_support_mae = None
                if args.stage1_selection_metric == 'target_support_mae':
                    was_training = rep_model.training
                    rep_model.eval()
                    support_metrics = summarize_test_metrics(
                        rep_model,
                        finetune_dataset,
                        task_args['source']['pred_num'],
                    )
                    target_support_mae = float(
                        np.mean([item['mae'] for item in support_metrics])
                    )
                    if was_training:
                        rep_model.train()
                    selection_score = target_support_mae
                state = {
                    'epoch': i,  
                    'model_state_dict': rep_model.state_dict(), 
                    'optimizer_state_dict': rep_model.main_optim.state_dict(),  
                    'stage1_selection_metric': args.stage1_selection_metric,
                    'stage1_source_mae': float(mae_loss),
                    'stage1_target_support_mae': target_support_mae,
                    'stage1_selection_score': selection_score,
                }
                current_source_mae = float(mae_loss)
                best_stage1_source_mae = min(
                    best_stage1_source_mae, current_source_mae
                )
                if selection_score < best_stage1_selection_score:
                    torch.save(state, f"l1_{args.test_dataset}_{args.model_name}_seed{seed}.pth")
                    best_stage1_selection_score = selection_score
                    best_stage1_epoch = i
                    print(
                        f'best Stage-I checkpoint saved at epoch:{i}, '
                        f'source_mae:{current_source_mae:.6f}, '
                        f'selection={args.stage1_selection_metric}:'
                        f'{selection_score:.6f}'
                    )
                else:
                    print(
                        f'Stage-I checkpoint skipped at epoch:{i}; '
                        f'source_mae:{current_source_mae:.6f}, '
                        f'selection:{selection_score:.6f}, '
                        f'best:{best_stage1_selection_score:.6f} '
                        f'(epoch {best_stage1_epoch})'
                    )
       
                
    print('Source training Ending!!!')
    # del rep_model
    torch.cuda.empty_cache()
    
    
    args.new=1
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    
    import PMME_2ndStage
    import PMME

    if task_args['source']['finetune_epochs']>0:
    
        # Load the Stage-I checkpoint directly into the Stage-II model.
        # The held-out target test set is not evaluated before adaptation.
        model_path = f"l1_{args.test_dataset}_{args.model_name}_seed{seed}.pth"

        rep_model = PMME_2ndStage.STRep(args.classifier, args.DAepoch, args.DAcoef, args.xySquaredRatio, args.MemSize, 1, args.model_name, data_args, task_args, model_args, PatchFSL_cfg, args) # args.temperature
    
        model_path = f"l1_{args.test_dataset}_{args.model_name}_seed{seed}.pth" 
        
        ckpt = torch.load(model_path, map_location=args.device)
        old_state = ckpt['model_state_dict']    
    
        filtered_state_dict = {}
        for full_key, tensor in old_state.items():
            if full_key.startswith("model."):
                new_key = full_key[len("model."):]
                filtered_state_dict[new_key] = tensor
        old_state = filtered_state_dict
        
        new_state = rep_model.model.state_dict()
        
        filtered = {k: v for k, v in old_state.items() if k in new_state}
        
        print('------------------------------------')
        for i in filtered:
            print(i)
        
        new_state.update(filtered)
        
        rep_model.model.load_state_dict(new_state)

        # Stage-II is a one-shot, target-only low-rank residual fit.
        # The frozen SOFTS backbone is evaluated once per support batch, then
        # a mixed smooth/detail residual basis and ridge map are solved in closed form.
        log_verbose(
            args,
            "[Stage-II][LowRankResidualAdapter] enabled; "
            "feature compression + mixed low-rank residual basis + closed-form ridge."
        )

        log_verbose(args, dict(vars(args)))
        log_verbose(args, config)
        yaml.dump(
            [dict(vars(args)), config],
            open(os.path.join(args.save_dir, "config.yaml"), "w"),
        )

        stage2_start = time.time()
        adapter_stats = rep_model.fit_target_residual_adapter(finetune_dataset)
        log_verbose(
            args,
            "[Stage-II][LowRankResidualAdapter] fit_complete "
            "support_rows={support_rows} feature_dim={feature_dim} "
            "rank={rank} smooth_rank={smooth_rank} detail_rank={detail_rank} "
            "ridge_alpha={ridge_alpha:.4f} "
            "smooth_energy={smooth_energy_ratio:.4f} "
            "detail_energy={detail_energy_ratio:.4f} "
            "captured_energy={captured_energy_ratio:.4f} "
            "total_loss={total_loss:.6f} residual_loss={residual_loss:.6f} "
            "forecast_loss={forecast_loss:.6f} correction_l2={correction_l2:.6f} "
            "support_MSE={train_MSE:.5f} support_MAE={train_MAE:.5f} "
            "support_RMSE={train_RMSE:.5f} support_MAPE={train_MAPE:.5f} "
            "fit_time={fit_seconds:.3f}s wall_time={wall_time:.3f}s".format(
                **adapter_stats,
                wall_time=time.time() - stage2_start,
            )
        )

        # Save fitted adapter buffers. No optimizer state is needed because
        # the ridge solution is obtained directly rather than by epoch training.
        final_state = {
            'epoch': 0,
            'model_state_dict': rep_model.state_dict(),
            'optimizer_state_dict': None,
            'adapter_fit_summary': adapter_stats,
        }
        checkpoint_path = f"l2_{args.test_dataset}_{args.model_name}_seed{seed}.pth"
        torch.save(final_state, checkpoint_path)
        log_verbose(args, f"[Stage-II][LowRankResidualAdapter] checkpoint saved: {checkpoint_path}")

        # Only after adaptation is complete do we construct/read the held-out test split.
        test_dataset = traffic_dataset(
            data_args,
            task_args['source'],
            data_list,
            'test',
            test_data=args.test_dataset,
            target_days=args.target_days,
        )
        final_testing_score = rep_model.test_evaluation(
            finetune_dataset,
            test_dataset,
            task_args['source']['finetune_epochs'],
        )
        indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
        selected_test_scores = [final_testing_score[j] for j in indices]
        log_verbose(args, "[Stage-II][LowRankResidualAdapter] final held-out test MAE: {}".format(selected_test_scores))
        log_verbose(args, "[Stage-II][LowRankResidualAdapter] sum(final held-out test MAE): {:.6f}".format(sum(selected_test_scores)))

        # Evaluate Stage-I-only SOFTS on the same held-out split after adaptation,
        # solely for reporting and never for checkpoint/epoch selection.
        baseline_model = PMME_2ndStage.STRep(
            args.classifier, args.DAepoch, args.DAcoef, args.xySquaredRatio,
            0, 0, args.model_name, data_args, task_args, model_args, PatchFSL_cfg, args
        )
        baseline_ckpt = torch.load(model_path, map_location=args.device)
        baseline_state = {
            key[len("model."):]: value
            for key, value in baseline_ckpt['model_state_dict'].items()
            if key.startswith("model.")
        }
        baseline_model_state = baseline_model.model.state_dict()
        baseline_model_state.update({
            key: value for key, value in baseline_state.items()
            if key in baseline_model_state
        })
        baseline_model.model.load_state_dict(baseline_model_state)
        baseline_testing_score = baseline_model.test_evaluation(
            finetune_dataset,
            test_dataset,
            task_args['source']['finetune_epochs'],
        )
        baseline_selected_scores = [baseline_testing_score[j] for j in indices]
        log_verbose(args, "[Stage-I-only] final held-out test MAE: {}".format(baseline_selected_scores))
        log_verbose(args, "[Stage-I-only] sum(final held-out test MAE): {:.6f}".format(sum(baseline_selected_scores)))
        del baseline_model

        print('2ndStage Source training Ending!!!')
    else:
        # Stage-I-only runs still need a complete held-out evaluation record.
        # The test split is constructed only after the source checkpoint has
        # been selected, and it is never used to select the checkpoint.
        model_path = args.evaluate_checkpoint or f"l1_{args.test_dataset}_{args.model_name}_seed{seed}.pth"
        if args.evaluate_checkpoint:
            evaluation_checkpoint = torch.load(model_path, map_location=args.device)
            best_stage1_epoch = int(evaluation_checkpoint.get("epoch", -1))
            best_stage1_source_mae = float(
                evaluation_checkpoint.get("stage1_source_mae", float("inf"))
            )
        test_dataset = traffic_dataset(
            data_args,
            task_args['source'],
            data_list,
            'test',
            test_data=args.test_dataset,
            target_days=args.target_days,
        )
        save_stage1_test_result(
            model_path=model_path,
            finetune_dataset=finetune_dataset,
            test_dataset=test_dataset,
            args=args,
            config=config,
            data_list=source_dataset.data_list,
            model_args=model_args,
            task_args=task_args,
            data_args=data_args,
            PatchFSL_cfg=PatchFSL_cfg,
            best_stage1_epoch=best_stage1_epoch,
            best_stage1_source_mae=best_stage1_source_mae,
            best_stage1_selection_score=best_stage1_selection_score,
            stage1_selection_metric=args.stage1_selection_metric,
        )
        print('Stage-I-only evaluation and result saving complete.')
    # del rep_model
    # torch.cuda.empty_cache()
##################################################################################################################################
