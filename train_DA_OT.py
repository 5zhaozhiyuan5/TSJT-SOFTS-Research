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
parser.add_argument('--finetune_epochs', default=1, type=int, help='Legacy Stage-II enable flag; the low-rank adapter is fitted once in closed form, not iteratively.')
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
parser.add_argument('--DAepoch',default=50000,type=int)
parser.add_argument('--MemSize', default=1, type=int, help='Legacy enable flag: any value > 0 enables the target-only Stage-II residual network; no memory slots are allocated.')
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
            model_loss ,mse_loss, rmse_loss, mae_loss, mape_loss = rep_model.meta_train_revise(data, i)
    
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
                        torch.save(state, f"l1_{args.test_dataset}_{args.model_name}_seed{seed}.pth")
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
    # del rep_model
    # torch.cuda.empty_cache()
##################################################################################################################################
