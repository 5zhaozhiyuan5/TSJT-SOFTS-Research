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
parser.add_argument('--stage2_target_batch_size', default=4, type=int)
parser.add_argument('--stage2_source_batch_size', default=8, type=int)
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
parser.add_argument('--MemSize',default=10000,type=int)
parser.add_argument('--projection',default=0,type=float)
parser.add_argument(
    '--profile_coef',
    default=0.1,
    type=float,
    help='Stage-I multi-horizon traffic response profile supervision weight; set 0 to disable.',
)
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
    source_dataset = traffic_dataset(
        data_args, task_args['source'], data_list, "source",
        test_data=args.test_dataset
    )
    ## check source dataset

    for data in source_dataset.data_list:
        log_verbose(args,"source dataset has {}. X : {}, y : {}".format(data,source_dataset.x_list[data].shape,source_dataset.y_list[data].shape))
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
    # Preserve the original Stage-I data construction. Only after Stage-I has
    # finished, remove the target city from the Stage-II source sampler.
    source_dataset.data_list = np.asarray([
        name for name in source_dataset.data_list if name != args.test_dataset
    ])
    if len(source_dataset.data_list) == 0:
        raise RuntimeError("No source cities remain for Stage-II transfer.")
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
        
    
        testing_score = rep_model.test_evaluation(finetune_dataset, test_dataset, task_args['source']['finetune_epochs'])
        indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35] 
        current_testing_score = [testing_score[i] for i in indices] 
        print('current_testing_score', current_testing_score) 
        print('sum(current_testing_score)',sum(current_testing_score))
        
        print('####################')
        print('####################')
        print('####################')
        print('####################')
        print('####################')
    
        # Stage-II: budgeted HMRM-Mamba training.
        rep_model = PMME_2ndStage.STRep(
            args.classifier, args.DAepoch, args.DAcoef, args.xySquaredRatio,
            args.MemSize, 1, args.model_name, data_args, task_args,
            model_args, PatchFSL_cfg, args
        )

        model_path = f"l1_{args.test_dataset}_{args.model_name}_seed{seed}.pth"
        ckpt = torch.load(model_path, map_location=args.device)
        old_state = ckpt["model_state_dict"]
        stage1_state = {
            key[len("model."):]: value
            for key, value in old_state.items()
            if key.startswith("model.")
        }
        current_state = rep_model.model.state_dict()
        compatible_state = {
            key: value for key, value in stage1_state.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        missing_backbone = [
            key for key in rep_model.model.forecasting_model.state_dict()
            if "forecasting_model." + key not in compatible_state
        ]
        if missing_backbone:
            raise RuntimeError(
                "Stage-I checkpoint is incomplete for Stage-II backbone: "
                + ", ".join(missing_backbone[:10])
            )
        current_state.update(compatible_state)
        rep_model.model.load_state_dict(current_state, strict=True)

        # Stage-II uses micro-batches to keep the 10k-slot memory readers bounded.
        # The default target batch of 4 uses small updates; the final partial batch is retained.
        stage2_target_batch_size = max(1, int(args.stage2_target_batch_size))
        stage2_source_batch_size = max(1, int(args.stage2_source_batch_size))
        finetune_dataset.task_args = dict(finetune_dataset.task_args)
        finetune_dataset.task_args["batch_size"] = stage2_target_batch_size
        source_dataset.task_args = dict(source_dataset.task_args)
        source_dataset.task_args["batch_size"] = stage2_source_batch_size

        # Fixed Stage-II budget: retain the configured epoch count. During the
        # first 60% of updates, train on one target batch plus one source batch;
        # during the final 40%, adapt on target batches only. This avoids adding
        # a second full training schedule and cycles through every target batch.
        stage2_epochs = max(1, int(task_args["source"]["finetune_epochs"]))
        source_pretrain_epochs = (
            min(stage2_epochs - 1, max(1, int(round(stage2_epochs * 0.60))))
            if stage2_epochs > 1 else 0
        )
        stage2_start = time.time()
        source_length = len(source_dataset)
        target_sample_count = int(
            finetune_dataset.x_list[finetune_dataset.data_list[0]].shape[0]
        )
        target_length = (
            target_sample_count + stage2_target_batch_size - 1
        ) // stage2_target_batch_size
        if source_length <= 0:
            raise RuntimeError("Stage-II requires source training data.")
        if target_length <= 0:
            raise RuntimeError("Stage-II requires target support windows.")

        log_verbose(
            args,
            "[Stage-II][HMRM-Mamba] fixed budget: epochs={}, "
            "target-guided source phase={}, target-adaptation phase={}, "
            "target_support_batches={}, source_batch_size={}, max_source_batches_per_epoch=1".format(
                stage2_epochs, source_pretrain_epochs,
                stage2_epochs - source_pretrain_epochs, target_length,
                stage2_source_batch_size,
            )
        )

        for epoch in range(stage2_epochs):
            epoch_start = time.time()
            target_item = finetune_dataset[epoch % target_length]
            target_data = (
                target_item[0]
                if isinstance(target_item, (tuple, list)) else target_item
            )
            phase = (
                "cross_city_pretrain"
                if epoch < source_pretrain_epochs else "target_adapt"
            )
            stage2_tasks = [target_data]

            if phase == "cross_city_pretrain":
                # The target city was removed from source_dataset.data_list after Stage-I.
                source_data, _ = source_dataset[epoch % source_length]
                stage2_tasks.append(source_data)

            train_stats = rep_model.meta_train_revise(
                stage2_tasks, epoch=epoch, phase=phase
            )
            if epoch % 10 == 0 or epoch == stage2_epochs - 1:
                log_verbose(
                    args,
                    "[Stage-II][HMRM-Mamba] epoch {}/{} phase={} "
                    "target_MAE={:.6f} source_MAE={:.6f} "
                    "target_loss={:.6f} epoch_seconds={:.3f}".format(
                        epoch + 1, stage2_epochs, phase,
                        train_stats["target_mae"], train_stats["source_mae"],
                        train_stats["target_loss"], time.time() - epoch_start,
                    )
                )

            if (epoch + 1) % 25 == 0 or epoch == stage2_epochs - 1:
                stage2_path = (
                    f"stage2_hmamba_{args.test_dataset}_{args.model_name}_seed{seed}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "phase": phase,
                        "model_state_dict": rep_model.state_dict(),
                        "optimizer_state_dict": rep_model.main_optim.state_dict(),
                        "stage1_checkpoint": model_path,
                    },
                    stage2_path,
                )

        fit_seconds = time.time() - stage2_start
        log_verbose(
            args,
            f"[Stage-II][HMRM-Mamba] iterative training finished in {fit_seconds:.2f}s"
        )

        testing_score = rep_model.test_evaluation(
            finetune_dataset, test_dataset, stage2_epochs
        )
        indices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
        current_testing_score = [testing_score[index] for index in indices]
        print("[Stage-II][HMRM-Mamba] test MAE by selected horizons:", current_testing_score)
        print("[Stage-II][HMRM-Mamba] sum of selected horizon MAE:", sum(current_testing_score))
        print("[Stage-II][HMRM-Mamba] iterative training complete.")


    # del rep_model
    # torch.cuda.empty_cache()
##################################################################################################################################
