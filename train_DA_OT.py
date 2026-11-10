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
sys.path.append('./model')
sys.path.append('./model/Meta_Models')

from PMME import *
from pathlib import Path
import random

curr_time = time.strftime('%Y-%m-%d-%H-%M-%S')

parser = argparse.ArgumentParser(description='PMME')
parser.add_argument('--config_filename', default='./configs/config.yaml', type=str,
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

parser.add_argument('--classifier',default=0.3,type=float)
parser.add_argument('--xySquaredRatio',default=10,type=float) 
parser.add_argument('--DAcoef',default=0.1,type=float)
parser.add_argument('--DAepoch',default=1500,type=int)
parser.add_argument('--MemSize',default=10000,type=int)
parser.add_argument('--DAwarmup',default=1500,type=int)
parser.add_argument('--projection',default=0,type=float)
parser.add_argument('--model_name',default='SOFTS',type=str)
parser.add_argument('--DA',default='OT',type=str)

# Dynamic Prototype Partial-OT (proposal 1)
parser.add_argument('--num_experts', default=4, type=int)
parser.add_argument('--top_k', default=2, type=int)
parser.add_argument('--moe_dense_fusion', action='store_true', default=False)
parser.add_argument('--moe_fusion_mode', default='add', choices=['add', 'gate', 'hybrid'])
parser.add_argument('--dvgp_soft_selector', action='store_true')  # soft+adaptive DVGP selector (CDTrans/FixMatch style)
parser.add_argument('--dvgp_temp', type=float, default=1.0)
parser.add_argument('--dvgp_chengdu_scale', type=float, default=1.0)  # per-city threshold relax for chengdu(524) whose DVGP is over-confident
parser.add_argument('--dvgp_chengdu_thresh_end', type=float, default=0.3)  # chengdu threshold end (lower keeps more of its long-horizon samples)  # temperature to de-saturate DVGP probs
parser.add_argument('--dvgp_thresh_start', type=float, default=0.05)  # adaptive threshold start
parser.add_argument('--dvgp_thresh_end', type=float, default=0.3)  # adaptive threshold end (=v31 fixed thresh)  # MoE fusion: add(v31)/gate/hybrid
parser.add_argument('--moe_gate_coef', type=float, default=0.0)  # hybrid gate modulation coef (0=v31)
parser.add_argument('--expert_bottleneck', default=16, type=int)
parser.add_argument('--router_hidden', default=64, type=int)
parser.add_argument('--partial_mass', default=0.8, type=float)
parser.add_argument('--adaptive_partial_mass', action='store_true', default=False)
parser.add_argument('--partial_mass_floor', default=0.6, type=float)
parser.add_argument('--partial_mass_ceiling', default=1.0, type=float)
parser.add_argument('--proto_epsilon', default=0.05, type=float)
parser.add_argument('--proto_iters', default=50, type=int)
parser.add_argument('--proto_label_weight', default=1.0, type=float)
parser.add_argument('--label_horizon_mod', action='store_true', default=False)
parser.add_argument('--label_horizon_gamma', default=0.5, type=float)
parser.add_argument('--dynamic_cost_weight', default=25.0, type=float)
parser.add_argument('--prototype_loss_coef', default=0.5, type=float)
parser.add_argument('--route_entropy_coef', default=0.0, type=float)
parser.add_argument('--route_cost_weight', default=0.0, type=float)
parser.add_argument('--expert_guide_scale', default=1.0, type=float)
parser.add_argument('--route_entropy_guide_scale', default=0.0, type=float)  # MoE route entropy as extra OT geometry dim
parser.add_argument('--ot_loss_scale_floor', default=0.0, type=float)  # floor on OT_loss_scale to prevent DVGP over-suppression
parser.add_argument('--source_use_raw_route', action='store_true', default=False)
parser.add_argument('--y_winsor_sigma', default=0.0, type=float)
parser.add_argument('--expert_guide_dense', action='store_true', default=False)
parser.add_argument('--expert_guide_temp', default=1.0, type=float)
parser.add_argument('--expert_guide_warmup', action='store_true', default=False)
parser.add_argument('--balance_coef', default=0.01, type=float)
parser.add_argument('--diversity_coef', default=0.01, type=float)
parser.add_argument('--adapter_l2_coef', default=0.0, type=float)
parser.add_argument('--memory_confidence_coef', default=0.001, type=float)
parser.add_argument('--memory_slots_per_expert', default=512, type=int)
parser.add_argument('--memory_top_k', default=32, type=int)
parser.add_argument('--residual_gate_init', default=0.0, type=float)
parser.add_argument('--memory_correction_clip', default=0.5, type=float)
parser.add_argument('--horizon_weight_gamma', default=0.0, type=float)
parser.add_argument('--stage1_horizon_gamma', default=0.0, type=float)
parser.add_argument('--stage2_horizon_gamma', default=0.0, type=float)
parser.add_argument('--target_memory_weight', default=0.25, type=float)
parser.add_argument('--smooth_l1_beta', default=5e-4, type=float)
parser.add_argument('--memory_correction_coef', default=0.01, type=float)
parser.add_argument('--memory_mode', default='learned', choices=['learned', 'prototype', 'pattern_proto', 'original', 'route_bias'])
parser.add_argument('--pattern_proto_horizon_decay', default=0.45, type=float)
parser.add_argument('--memory_source', default='source', choices=['source', 'target'])
parser.add_argument('--source_memory_batches', default=40, type=int)
parser.add_argument('--source_memory_cap', default=200000, type=int)
parser.add_argument('--prototype_memory_folds', default=3, type=int)
parser.add_argument('--prototype_memory_ridge', default=0.10, type=float)
parser.add_argument('--prototype_memory_temperature', default=0.10, type=float)
parser.add_argument('--prototype_residual_clip', default=3.0, type=float)
parser.add_argument('--prototype_gate_prior', default=0.25, type=float)
parser.add_argument('--prototype_gate_max', default=1.25, type=float)
parser.add_argument('--checkpoint_prefix', default='dp_pot_v2', type=str)
parser.add_argument('--stage1_checkpoint', default='', type=str)
args = parser.parse_args()

stage1_model_path = args.stage1_checkpoint or (
    f"{args.checkpoint_prefix}_stage1_{args.test_dataset}_{args.model_name}_seed{args.seed}.pth"
)
stage2_model_path = (
    f"{args.checkpoint_prefix}_stage2_{args.test_dataset}_{args.model_name}_seed{args.seed}.pth"
)

args.curr_time = curr_time
print('args.test_dataset',args.test_dataset)
save_dir = os.path.join(".", "logged_files", args.test_dataset, args.data_list , args.curr_time)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_dir = Path(save_dir)
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
    reuse_stage1_for_prototype = (
        task_args['source']['train_epochs'] == 0
        and args.memory_mode == 'prototype'
        and bool(args.stage1_checkpoint)
    )
    source_dataset = None
    if not reuse_stage1_for_prototype:
        source_dataset = traffic_dataset(
            data_args, task_args['source'], data_list, "source",
            test_data=args.test_dataset,
        )
        ## check source dataset
        for data in source_dataset.data_list:
            log_verbose(
                args,
                "source dataset has {}. X : {}, y : {}".format(
                    data,
                    source_dataset.x_list[data].shape,
                    source_dataset.y_list[data].shape,
                ),
            )
    else:
        log_verbose(args, 'INFO: reuse Stage-I checkpoint; skip source dataset loading.')
    print('data_args', data_args)
    finetune_dataset = traffic_dataset(data_args, task_args['target_few_shot'], data_list, 'target_maml', test_data=args.test_dataset)
    test_dataset = traffic_dataset(data_args, task_args['source'], data_list, 'test', test_data=args.test_dataset)
    log_verbose(args,"data args : {}\ntask_args : {}\nmodel_args : {}\nPatchFSL_cfg : {}\nSTmodel : {}".format(data_args, task_args, model_args, PatchFSL_cfg, args.STmodel))

    rep_model = STRep(args.classifier, args.DAepoch, args.DAcoef, args.xySquaredRatio, 0, args.projection, args.model_name, args.DA, data_args, task_args, model_args, PatchFSL_cfg, args) 
    
    # rep_model = STRep(data_args, task_args, model_args, PatchFSL_cfg, args)
    best_loss = 9999999999999.0
    best_model = None
    ## train on big dataset
    rep_tasknum = task_args['source']['task_num']
    
    best_validation_score = [1e5,1e5,1e5]
    best_testing_score = [1e5,1e5,1e5]
    
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
            log_verbose(args,"This epoch cost {:.3}s.".format(time.time() - time_1))
            
            # if((i+1)%50==0):
            if((i+1)%100==0)or i==0:
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
                        torch.save(state, stage1_model_path)
                        best_validation_score = current_validation_score
                        best_epoch = i
                        print(f'epoch:{i} best!')
                    else:
                        print('not_best')
                        #if (i-best_epoch)>=1000:
                        #    print('No better epoch for over 1000 epochs!!!')
                        #    break
       
                
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
        
        model_path = stage1_model_path
        
        ckpt = torch.load(model_path, map_location=args.device)
        old_state = ckpt['model_state_dict']    
    
        filtered_state_dict = {}
        for full_key, tensor in old_state.items():
            if full_key.startswith("model."):
                new_key = full_key[len("model."):]
                filtered_state_dict[new_key] = tensor
        old_state = filtered_state_dict
        
        new_state = rep_model.model.state_dict()
        
        filtered = {
            k: v for k, v in old_state.items()
            if k in new_state and v.shape == new_state[k].shape
        }
        
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
        
    
        if not reuse_stage1_for_prototype:
            testing_score = rep_model.test_evaluation(
                finetune_dataset,
                test_dataset,
                task_args['source']['finetune_epochs'],
            )
            indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
            current_testing_score = [testing_score[i] for i in indices]
            print('current_testing_score', current_testing_score)
            print('sum(current_testing_score)', sum(current_testing_score))
        else:
            print('INFO: skip duplicate Stage-I test; checkpoint result is already available.')
        
        print('####################')
        print('####################')
        print('####################')
        print('####################')
        print('####################')
    
        rep_model = PMME_2ndStage.STRep(args.classifier, args.DAepoch, args.DAcoef, args.xySquaredRatio, args.MemSize, 1, args.model_name, data_args, task_args, model_args, PatchFSL_cfg, args) # args.temperature
    
        model_path = stage1_model_path
        
        ckpt = torch.load(model_path, map_location=args.device)
        old_state = ckpt['model_state_dict']    
    
        filtered_state_dict = {}
        for full_key, tensor in old_state.items():
            if full_key.startswith("model."):
                new_key = full_key[len("model."):]
                filtered_state_dict[new_key] = tensor
        old_state = filtered_state_dict
        
        new_state = rep_model.model.state_dict()
        
        filtered = {
            k: v for k, v in old_state.items()
            if k in new_state and v.shape == new_state[k].shape
        }
        
        print('------------------------------------')
        for i in filtered:
            print(i)
        
        new_state.update(filtered)
        
        rep_model.model.load_state_dict(new_state)
        rep_model.freeze_stage1()

        # learned-mode residual enhancement: seed the learnable memory bank from
        # source-city Stage-I residuals partitioned by dynamic pattern (expert
        # route), replacing random init; target 3-days then only fine-tune gates.
        if args.memory_mode in ('learned', 'pattern_proto') and args.memory_source == 'source':
            source_batches = []
            for _ in range(args.source_memory_batches):
                data_i, _ = source_dataset[0]
                source_batches.append(data_i)
            seed_stats = rep_model.fit_source_memory(source_batches)
            rep_model.finalize_source_seed()
            print('learned_source_seed_stats', seed_stats)

        if args.memory_mode == 'prototype':
            if args.memory_source == 'target':
                target_data, _ = finetune_dataset[0]
                memory_stats = rep_model.fit_target_memory(target_data)
            else:
                # Build the residual memory from *source-city* Stage-I
                # residuals.  Each `source_dataset[i]` returns a random city and
                # a random batch, so we gather `source_memory_batches` batches.
                source_batches = []
                for _ in range(args.source_memory_batches):
                    # stage=='source' __getitem__ ignores the index and samples
                    # a random city + random batch each call.
                    data_i, _ = source_dataset[0]
                    source_batches.append(data_i)
                memory_stats = rep_model.fit_source_memory(source_batches)
            print('prototype_memory_stats', memory_stats)
            validation_score, testing_score = rep_model.evaluation(
                finetune_dataset,
                test_dataset,
                task_args['source']['finetune_epochs'],
            )
            state = {
                'epoch': -1,
                'model_state_dict': rep_model.state_dict(),
                'memory_stats': memory_stats,
            }
            torch.save(state, stage2_model_path)
            report_indices = [1, 11, 23, 35]
            print(
                'FINAL_PROTOTYPE_RM_MAE_10_60_120_180',
                [testing_score[index] for index in report_indices],
            )
            print('2ndStage Prototype Memory Ending!!!')
            raise SystemExit(0)
    
        testing_score = rep_model.test_evaluation(finetune_dataset, test_dataset, task_args['source']['finetune_epochs'])
        indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35] 
        current_testing_score = [testing_score[i] for i in indices] 
        print('current_testing_score', current_testing_score) 
        print('sum(current_testing_score)',sum(current_testing_score))
    
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
        
        log_verbose(args,dict(vars(args)))
        log_verbose(args,config)
    
        yaml.dump([dict(vars(args)), config], open(os.path.join(args.save_dir, "config.yaml"), "w"))
        
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
                    log_verbose(args,'Stage-II Epochs {}/{}'.format(i,task_args['source']['finetune_epochs']))
                    log_verbose(args,'in meta-training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, reconstruction Loss : {:.5f}.'.format(mse_loss, rmse_loss, mae_loss,mape_loss,model_loss))
                log_verbose(args,"This epoch cost {:.3}s.".format(time.time() - time_1))
                
                # if((i+1)%50==0):
                if((i+1)%100==0)or i==0:
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
                            torch.save(state, stage2_model_path)
                            best_validation_score = current_validation_score
                            best_epoch = i
                            print(f'epoch:{i} best!')
                        else:
                            print('not_best')
                            #if (i-best_epoch)>=1000:
                            #    print('No better epoch for over 1000 epochs!!!')
                            #    break
                    
        print('2ndStage Source training Ending!!!')
    # del rep_model
    # torch.cuda.empty_cache()
##################################################################################################################################
