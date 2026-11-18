import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import traffic_dataset
from utils import *
import argparse
import csv
import json
import yaml
import time
import sys
sys.path.append('./model')
sys.path.append('./model/Meta_Models')

from PMME import *
from pattern_moe import (
    build_multiscale_pattern_bank,
    physical_minutes_to_steps,
)
from pathlib import Path
import random
from experiment_diagnostics import ExperimentDiagnostics

curr_time = time.strftime('%Y-%m-%d-%H-%M-%S')

parser = argparse.ArgumentParser(description='PMME')
parser.add_argument('--config_filename', '--config_file', dest='config_filename',
                    default='./configs/config.yaml', type=str,
                    help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='pems-bay', type=str)
parser.add_argument('--train_epochs', default=0, type=int)
parser.add_argument('--finetune_epochs', default=300,type=int)
parser.add_argument('--description', default='PMME', type=str)

parser.add_argument('--seed',default=7,type=int)
parser.add_argument('--data_list', default='chengdu_shenzhen_metr',type=str)
parser.add_argument('--target_days', default=3,type=int)
parser.add_argument('--gpu', default=0, type = int)
parser.add_argument('--STmodel',default='SOFTS',type=str)
parser.add_argument('--his_num',default=288,type=int)
parser.add_argument('--save_dir', type=str)
parser.add_argument(
    '--pattern_moe_router_mode',
    choices=['learned', 'uniform', 'expert1', 'expert2', 'expert3'],
    default=None,
    help='Override pattern_moe.router_mode for controlled expert ablations.',
)

parser.add_argument(
    '--pattern_moe_pattern_weight', default=None, type=float,
    help='Override pattern_moe.pattern_weight (0 removes only the pattern matching cost).',
)
parser.add_argument(
    '--fixed_final_checkpoint', action='store_true',
    help='Strict protocol: use last-epoch Stage-I and Stage-II checkpoints, never select on target/test metrics.',
)
parser.add_argument('--classifier',default=0.3,type=float)
parser.add_argument('--xySquaredRatio',default=10,type=float) 
parser.add_argument('--DAcoef',default=0.1,type=float)
parser.add_argument('--DAepoch',default=50000,type=int)
parser.add_argument('--MemSize',default=10000,type=int)
parser.add_argument('--projection',default=0,type=float)
parser.add_argument('--model_name',default='SOFTS',type=str)
parser.add_argument('--DA',default='OT',type=str)
args = parser.parse_args()

args.curr_time = curr_time
print('args.test_dataset',args.test_dataset)
run_description = str(args.description).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
requested_router_mode = args.pattern_moe_router_mode or "config"
if args.save_dir:
    save_dir = Path(args.save_dir)
else:
    save_dir = (
        Path(".") / "logged_files" / args.test_dataset / args.data_list
        / f"{args.curr_time}_{run_description}_router-{requested_router_mode}_seed{args.seed}"
    )
save_dir.mkdir(parents=True, exist_ok=True)
args.save_dir = save_dir
args.diagnostics = ExperimentDiagnostics(
    save_dir,
    metadata={
        "cli_args": {key: str(value) for key, value in vars(args).items()},
        "run_description": args.description,
        "seed": args.seed,
        "target_dataset": args.test_dataset,
        "source_dataset_selector": args.data_list,
        "target_days": args.target_days,
        "requested_model": args.model_name,
        "alignment": args.DA,
        "requested_router_mode": requested_router_mode,
        "target_normalization_scope": "target_train",
    },
)
args.diagnostics.install_exception_hook()

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

    pattern_cfg_override = config.setdefault('model', {}).setdefault('pattern_moe', {})
    if args.pattern_moe_router_mode is not None:
        pattern_cfg_override['router_mode'] = args.pattern_moe_router_mode
    if args.pattern_moe_pattern_weight is not None:
        if not (0.0 <= args.pattern_moe_pattern_weight <= 1.0):
            raise ValueError("--pattern_moe_pattern_weight must lie in [0, 1]")
        pattern_cfg_override['pattern_weight'] = args.pattern_moe_pattern_weight
    if args.fixed_final_checkpoint and (
        args.train_epochs <= args.DAepoch or args.finetune_epochs <= 0
    ):
        raise ValueError("Strict protocol needs train_epochs > DAepoch and finetune_epochs > 0")
    resolved_router_mode = str(pattern_cfg_override.get('router_mode', 'learned')).lower()
    if resolved_router_mode not in ('learned', 'uniform', 'expert1', 'expert2', 'expert3'):
        raise ValueError(f"Unsupported pattern_moe.router_mode: {resolved_router_mode}")
    args.diagnostics.update_metadata({
        "resolved_router_mode": resolved_router_mode,
        "resolved_pattern_weight": float(pattern_cfg_override.get('pattern_weight', 0.5)),
        "stage1_checkpoint_policy": "final_epoch" if args.fixed_final_checkpoint else "legacy_best_target",
        "stage2_checkpoint_policy": "final_epoch" if args.fixed_final_checkpoint else "legacy_best_target",
        "target_normalization_scope": config.get('data', {}).get(
            'target_normalization_scope', 'target_train'
        ),
    })
    stage1_checkpoint_path = args.save_dir / (
        f"stage1_{args.test_dataset}_{args.model_name}_router-{resolved_router_mode}_seed{args.seed}.pth"
    )
    stage2_checkpoint_path = args.save_dir / (
        f"stage2_{args.test_dataset}_{args.model_name}_router-{resolved_router_mode}_seed{args.seed}.pth"
    )
    pre_alignment_checkpoint_path = args.save_dir / "stage1_pre_alignment_backbone.pth"

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
    source_dataset = traffic_dataset(
        data_args, task_args['source'], data_list, "source",
        test_data=args.test_dataset, target_days=args.target_days
    )#####
    ## check source dataset

    for data in source_dataset.data_list:
        log_verbose(args,"source dataset has {}. X : {}, y : {}".format(data,source_dataset.x_list[data].shape,source_dataset.y_list[data].shape))

    # Build the MTPB-inspired pattern bank from source-city training windows
    # only. Dataset preprocessing first maps METR-LA/PEMS-BAY and the
    # interpolated Chengdu/Shenzhen series onto one declared time grid.
    pattern_bank = None
    pattern_cfg = model_args.get("pattern_moe", {})
    if bool(pattern_cfg.get("enabled", False)):
        processed_minutes = int(source_dataset.processed_interval_minutes)
        declared_minutes = pattern_cfg.get("minutes_per_step")
        if declared_minutes is not None and int(declared_minutes) != processed_minutes:
            raise ValueError(
                "pattern_moe.minutes_per_step does not match the dataset's "
                f"processed interval ({declared_minutes} vs {processed_minutes} minutes)"
            )

        scale_minutes = tuple(
            int(value) for value in pattern_cfg.get(
                "scale_minutes", [60, 180, 360, 720, 1440]
            )
        )
        anchor_minutes = tuple(
            int(value) for value in pattern_cfg.get(
                "anchor_minutes", [10, 60, 120, 180]
            )
        )
        scale_steps = physical_minutes_to_steps(scale_minutes, processed_minutes)
        anchor_steps = physical_minutes_to_steps(anchor_minutes, processed_minutes)
        horizon_steps = int(task_args["source"]["pred_num"])
        if max(anchor_steps) > horizon_steps:
            raise ValueError(
                f"largest horizon anchor is {max(anchor_steps)} steps, but "
                f"the forecasting horizon is only {horizon_steps} steps"
            )
        if max(scale_steps) > int(source_dataset.his_num):
            raise ValueError(
                f"largest pattern scale requires {max(scale_steps)} history steps, "
                f"but task.source.his_num={source_dataset.his_num}. Increase the "
                "history window or reduce scale_minutes."
            )

        # Pass the resolved physical grid into PMME so the module and bank use
        # exactly the same step mapping.
        pattern_cfg["minutes_per_step"] = processed_minutes
        pattern_cfg["scale_steps"] = list(scale_steps)
        pattern_cfg["anchor_steps"] = list(anchor_steps)
        pattern_cfg["horizon_steps"] = horizon_steps

        source_pattern_arrays = {
            name: source_dataset.x_list[name]
            for name in source_dataset.data_list
            if name != args.test_dataset
        }
        if not source_pattern_arrays:
            raise RuntimeError(
                "pattern_moe is enabled but no source-city windows were found."
            )
        for city_name, windows in source_pattern_arrays.items():
            if windows.shape[-2] != int(source_dataset.his_num):
                raise ValueError(
                    f"{city_name}: expected {source_dataset.his_num} history steps "
                    f"on the {processed_minutes}-minute grid, got {windows.shape}"
                )

        log_verbose(
            args,
            "[PatternBank] physical scales={}min -> {} steps; "
            "horizons={}min -> {} steps; processed_grid={}min".format(
                list(scale_minutes),
                list(scale_steps),
                list(anchor_minutes),
                list(anchor_steps),
                processed_minutes,
            ),
        )
        log_verbose(
            args,
            "[PatternBank] source-only cities={} (target {} excluded)".format(
                sorted(source_pattern_arrays), args.test_dataset
            ),
        )
        pattern_bank = build_multiscale_pattern_bank(
            source_pattern_arrays,
            scale_steps=scale_steps,
            num_prototypes=int(pattern_cfg.get("num_prototypes", 10)),
            descriptor_bins=int(pattern_cfg.get("descriptor_bins", 16)),
            max_windows_per_city=int(pattern_cfg.get("max_windows_per_city", 24)),
            max_nodes_per_city=int(pattern_cfg.get("max_nodes_per_city", 24)),
            max_patches_per_scale=int(pattern_cfg.get("max_patches_per_scale", 8000)),
            seed=seed,
        ).to(args.device)
        log_verbose(
            args,
            "[PatternBank] built source-only multi-scale bank with shape {}".format(
                tuple(pattern_bank.shape)
            ),
        )

    print('data_args', data_args)
    finetune_dataset = traffic_dataset(
        data_args, task_args['target_few_shot'], data_list, 'target_maml',
        test_data=args.test_dataset, target_days=args.target_days
    )
    test_dataset = traffic_dataset(
        data_args, task_args['source'], data_list, 'test',
        test_data=args.test_dataset, target_days=args.target_days
    )
    args.diagnostics.update_metadata({
        "resolved_source_cities": list(source_dataset.data_list),
        "target_dataset": args.test_dataset,
        "processed_interval_minutes": int(source_dataset.processed_interval_minutes),
        "history_steps": int(source_dataset.his_num),
        "prediction_steps": int(task_args["source"]["pred_num"]),
        "target_few_shot_samples": len(finetune_dataset),
        "test_samples": len(test_dataset),
        "source_window_shapes": {
            name: list(source_dataset.x_list[name].shape)
            for name in source_dataset.data_list
        },
        "pattern_bank_enabled": bool(pattern_cfg.get("enabled", False)),
        "configured_anchor_minutes": pattern_cfg.get("anchor_minutes", []),
        "resolved_anchor_steps": pattern_cfg.get("anchor_steps", []),
        "configured_scale_minutes": pattern_cfg.get("scale_minutes", []),
        "resolved_scale_steps": pattern_cfg.get("scale_steps", []),
        "source_train_epochs": int(task_args["source"]["train_epochs"]),
        "stage2_finetune_epochs": int(task_args["source"]["finetune_epochs"]),
    })
    log_verbose(args,"data args : {}\ntask_args : {}\nmodel_args : {}\nPatchFSL_cfg : {}\nSTmodel : {}".format(data_args, task_args, model_args, PatchFSL_cfg, args.STmodel))

    rep_model = STRep(args.classifier, args.DAepoch, args.DAcoef, args.xySquaredRatio, 0, args.projection, args.model_name, args.DA, data_args, task_args, model_args, PatchFSL_cfg, args, pattern_bank=pattern_bank) 
    
    # rep_model = STRep(data_args, task_args, model_args, PatchFSL_cfg, args)
    best_loss = 9999999999999.0
    best_model = None
    ## train on big dataset
    rep_tasknum = task_args['source']['task_num']
    
    best_validation_score = [1e5,1e5,1e5]
    best_testing_score = [1e5,1e5,1e5]
    
    log_verbose(args,{key: value for key, value in vars(args).items() if key != "diagnostics"})
    log_verbose(args,config)

    yaml.dump([{key: value for key, value in vars(args).items() if key != "diagnostics"}, config], open(os.path.join(args.save_dir, "config.yaml"), "w"))
    
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
            if args.fixed_final_checkpoint and i == args.DAepoch - 1:
                # Independently verify that both controlled arms begin OT from
                # the same backbone (and do not use test feedback to select it).
                torch.save({
                    'epoch': i,
                    'backbone_state_dict': rep_model.model.state_dict(),
                }, pre_alignment_checkpoint_path)
    
            if i%25==0:
                log_verbose(args,'Epochs {}/{}'.format(i,task_args['source']['train_epochs']))
                log_verbose(args,'in meta-training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, reconstruction Loss : {:.5f}.'.format(mse_loss, rmse_loss, mae_loss,mape_loss,model_loss))
            log_verbose(args,"This epoch cost {:.3}s.".format(time.time() - time_1))
            
            # if((i+1)%50==0):
            if (((i + 1) % 100 == 0) or i == 0) and not args.fixed_final_checkpoint:
                state = {
                    'epoch': i,  
                    'model_state_dict': rep_model.state_dict(), 
                    'optimizer_state_dict': rep_model.main_optim.state_dict(),  
                }
                print('epoch:',i)
                validation_score, testing_score = rep_model.evaluation(finetune_dataset, test_dataset, task_args['source']['finetune_epochs'])
                print('finetune_dataset[0][1].shape[1]', finetune_dataset[0][1].shape[1])
                #print('validation_score', validation_score)
                if i >= 0: #1000:
                    indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35] 
                    current_validation_score = [validation_score[i] for i in indices]  

                    current_testing_score = [testing_score[i] for i in indices]                
                    print('sum(current_validation_score),sum(best_validation_score)',sum(current_validation_score),sum(best_validation_score))
                    #print('sum(current_testing_score),sum(best_testing_score)',sum(current_testing_score),sum(best_testing_score))
                    if sum(current_testing_score)<sum(best_testing_score):
                        best_testing_score = current_testing_score  # check testing score
                    
                    if sum(current_validation_score)<sum(best_validation_score):
                        torch.save(state, stage1_checkpoint_path)
                        best_validation_score = current_validation_score
                        best_epoch = i
                        print(f'epoch:{i} best!')
                    else:
                        print('not_best')
                        #if (i-best_epoch)>=1000:
                        #    print('No better epoch for over 1000 epochs!!!')
                        #    break
       
                
    if args.fixed_final_checkpoint:
        final_stage1_epoch = task_args['source']['train_epochs'] - 1
        torch.save({
            'epoch': final_stage1_epoch,
            'model_state_dict': rep_model.state_dict(),
            'optimizer_state_dict': rep_model.main_optim.state_dict(),
        }, stage1_checkpoint_path)
        args.diagnostics.update_metadata({
            'stage1_checkpoint_epoch': final_stage1_epoch,
            'stage1_checkpoint_path': str(stage1_checkpoint_path),
            'pre_alignment_checkpoint_path': str(pre_alignment_checkpoint_path),
        })
        print(f'[StrictProtocol] Stage-I FINAL checkpoint epoch={final_stage1_epoch}: {stage1_checkpoint_path}')

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
    
        rep_model = PMME_2ndStage.STRep(args.classifier, args.DAepoch, args.DAcoef, args.xySquaredRatio, 0, 0, args.model_name, data_args, task_args, model_args, PatchFSL_cfg, args) 
        
        print()
        print('Begin ckpt check!!!!!')
        print()
        
        model_path = str(stage1_checkpoint_path)   
        
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
        
        for i in old_state:
            print(i)
        print('------------------------------------')
        for i in new_state:
            print(i)
        print('------------------------------------')
        for i in filtered:
            print(i)
            
        new_state.update(filtered)
        
        rep_model.model.load_state_dict(new_state)
        
    
        if not args.fixed_final_checkpoint:
            testing_score = rep_model.test_evaluation(finetune_dataset, test_dataset, task_args['source']['finetune_epochs'])
            indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
            current_testing_score = [testing_score[k] for k in indices]
            print('current_testing_score', current_testing_score)
            print('sum(current_testing_score)', sum(current_testing_score))
        
        print('####################')
        print('####################')
        print('####################')
        print('####################')
        print('####################')
    
        rep_model = PMME_2ndStage.STRep(args.classifier, args.DAepoch, args.DAcoef, args.xySquaredRatio, args.MemSize, 1, args.model_name, data_args, task_args, model_args, PatchFSL_cfg, args) # args.temperature
    
        model_path = str(stage1_checkpoint_path) 
        
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
    
        if not args.fixed_final_checkpoint:
            testing_score = rep_model.test_evaluation(finetune_dataset, test_dataset, task_args['source']['finetune_epochs'])
            indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
            current_testing_score = [testing_score[k] for k in indices]
            print('current_testing_score', current_testing_score)
            print('sum(current_testing_score)', sum(current_testing_score))
    
        print()
        print('End ckpt check!!!!!')
        print()
        
        print('####################')
    
        best_loss = 9999999999999.0
        best_model = None
        ## train on big dataset
        rep_tasknum = task_args['source']['task_num']
        
        best_validation_score = [1e5,1e5,1e5]
        best_testing_score = [1e5,1e5,1e5]
        
        log_verbose(args,{key: value for key, value in vars(args).items() if key != "diagnostics"})
        log_verbose(args,config)
    
        yaml.dump([{key: value for key, value in vars(args).items() if key != "diagnostics"}, config], open(os.path.join(args.save_dir, "config.yaml"), "w"))
        
        for i in range(task_args['source']['finetune_epochs']):
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
            
            data.append(data_i)
            matrix.append(A)
            current_shape.add(A.shape[1])
            
            for idx in range(args.target_days):
                while(True):
                    data_i, A = source_dataset[idx] 
                    if(A.shape[1]) not in current_shape:
                        current_shape.add(A.shape[1])
                        # print('true for read') 
                        break
                # print('use', A.shape)
                data.append(data_i)
                matrix.append(A)
                idx+=1
            #b = time.time()
            if True:
                model_loss ,mse_loss, rmse_loss, mae_loss, mape_loss = rep_model.meta_train_revise(data, matrix, i)
    
                
                if i%25==0:
                    log_verbose(args,'Epochs {}/{}'.format(i,task_args['source']['train_epochs']))
                    log_verbose(args,'in meta-training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, reconstruction Loss : {:.5f}.'.format(mse_loss, rmse_loss, mae_loss,mape_loss,model_loss))
                log_verbose(args,"This epoch cost {:.3}s.".format(time.time() - time_1))
                
                # if((i+1)%50==0):
                if (((i + 1) % 100 == 0) or i == 0) and not args.fixed_final_checkpoint:
                    state = {
                        'epoch': i,  
                        'model_state_dict': rep_model.state_dict(),  
                        'optimizer_state_dict': rep_model.main_optim.state_dict(),  
                    }
                    print('epoch:',i)
                    validation_score, testing_score = rep_model.evaluation(finetune_dataset, test_dataset, task_args['source']['finetune_epochs'])
                    print('finetune_dataset[0][1].shape[1]', finetune_dataset[0][1].shape[1])
                    #print('validation_score', validation_score)
                    if i >= 0: #1000:
                        indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35] 
                        current_validation_score = [validation_score[i] for i in indices]  
    
                        current_testing_score = [testing_score[i] for i in indices]
                        
                        print('sum(current_validation_score),sum(best_validation_score)',sum(current_validation_score),sum(best_validation_score))
                        #print('sum(current_testing_score),sum(best_testing_score)',sum(current_testing_score),sum(best_testing_score))
                        if sum(current_testing_score)<sum(best_testing_score):
                            best_testing_score = current_testing_score  
                        
                        if sum(current_validation_score)<sum(best_validation_score):
                            torch.save(state, stage1_checkpoint_path) #####
                            best_validation_score = current_validation_score
                            best_epoch = i
                            print(f'epoch:{i} best!')
                        else:
                            print('not_best')
                            #if (i-best_epoch)>=1000:
                            #    print('No better epoch for over 1000 epochs!!!')
                            #    break
                    
        if args.fixed_final_checkpoint:
            final_stage2_epoch = task_args['source']['finetune_epochs'] - 1
            torch.save({
                'epoch': final_stage2_epoch,
                'model_state_dict': rep_model.state_dict(),
                'optimizer_state_dict': rep_model.main_optim.state_dict(),
                'stage1_checkpoint_path': str(stage1_checkpoint_path),
            }, stage2_checkpoint_path)
            args.diagnostics.update_metadata({
                'stage2_checkpoint_epoch': final_stage2_epoch,
                'stage2_checkpoint_path': str(stage2_checkpoint_path),
            })
            print(f'[StrictProtocol] Stage-II FINAL checkpoint epoch={final_stage2_epoch}: {stage2_checkpoint_path}')

            # Test once, after training is complete; no test-based selection.
            rep_model.current_epoch = final_stage2_epoch
            rep_model.test_evaluation(
                finetune_dataset, test_dataset, task_args['source']['finetune_epochs']
            )
            evaluation_csv = args.save_dir / "evaluation_metrics.csv"
            with evaluation_csv.open(newline="", encoding="utf-8") as f:
                final_rows = [
                    row for row in csv.DictReader(f)
                    if row['phase'] == 'stage2_final_test'
                    and int(row['epoch']) == final_stage2_epoch
                ]
            report_minutes = [10, 60, 120, 180]
            result = {
                str(minutes): {
                    'rmse': float(row['rmse']),
                    'mae': float(row['mae']),
                    'horizon_step': int(row['horizon_step']),
                }
                for minutes in report_minutes
                for row in final_rows
                if abs(float(row['horizon_minutes']) - minutes) < 1e-6
            }
            if len(result) != len(report_minutes):
                raise RuntimeError(
                    f'Incomplete final test metrics: found {list(result)}, expected {report_minutes}'
                )
            final_report = {
                'protocol': 'fixed_final_checkpoint_no_test_selection',
                'router_mode': resolved_router_mode,
                'pattern_weight': float(pattern_cfg_override.get('pattern_weight', 0.5)),
                'stage1_checkpoint_epoch': task_args['source']['train_epochs'] - 1,
                'stage2_checkpoint_epoch': final_stage2_epoch,
                'results': result,
            }
            with (args.save_dir / 'final_metrics.json').open('w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            print('[StrictProtocol] Final test metrics:', json.dumps(final_report, ensure_ascii=False))

        print('2ndStage Source training Ending!!!')
    # del rep_model
    # torch.cuda.empty_cache()
##################################################################################################################################
