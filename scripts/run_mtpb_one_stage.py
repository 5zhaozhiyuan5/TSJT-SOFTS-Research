"""One-stage MTPB: source meta-training guided by target 3-day gradients.

NO target-city finetuning. Evaluation starts directly from the source-trained
weights. Requires a separate official zhyliu00/MTPB checkout and its data.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml

SOURCE_CITIES = ('chengdu_m', 'shenzhen', 'metr-la')
TARGET_CITY = 'pems-bay'
PATTERN_HOURS = (1, 3, 6, 12, 24)


def evaluate_without_adaptation(rep, test_dataset):
    """Direct held-out test, with no optimizer, no model selection, no weight edits.

    The upstream PatchFSL.forward(..., stage='test') toggles Longterm_model
    back into training mode. A pre-forward hook forces eval for that branch,
    avoiding batch norm/dropout adaptation on held-out target test inputs.
    """
    model = rep.model
    model.eval()
    longterm_hook = None
    if hasattr(model, 'model_list') and len(model.model_list) >= 6:
        def force_longterm_eval(module, inputs):
            module.eval()
        longterm_hook = model.model_list[5].register_forward_pre_hook(force_longterm_eval)
    parameter_snapshot = [p.detach().clone() for p in model.parameters()]
    buffer_snapshot = [b.detach().clone() for b in model.buffers()]
    sum_abs = None
    sum_sq = None
    count = 0
    num_batches = 0
    try:
        with torch.no_grad():
            for index in range(len(test_dataset)):
                data, adjacency = test_dataset[index]
                batch_size = int(data.x.shape[0])
                if batch_size == 0:
                    continue
                graph = adjacency.float().unsqueeze(0).expand(
                    batch_size, -1, -1).to(rep.device)
                prediction, truth, _ = model(data, graph, stage='test')
                if prediction.shape != truth.shape or prediction.shape[-1] != 36:
                    raise ValueError('MTPB evaluation requires [B,N,36] predictions')
                difference = prediction.float() - truth.float()
                if not bool(torch.isfinite(difference).all()):
                    raise FloatingPointError('non-finite direct target predictions')
                abs_per_step = difference.abs().sum(dim=(0, 1)).cpu().double()
                sq_per_step = difference.square().sum(dim=(0, 1)).cpu().double()
                if sum_abs is None:
                    sum_abs = abs_per_step
                    sum_sq = sq_per_step
                else:
                    sum_abs += abs_per_step
                    sum_sq += sq_per_step
                count += prediction.shape[0] * prediction.shape[1]
                num_batches += 1
    finally:
        if longterm_hook is not None:
            longterm_hook.remove()
        for before, after in zip(parameter_snapshot, model.parameters()):
            if not torch.equal(before, after.detach()):
                raise AssertionError('test evaluation modified learned model parameters')
        for before, after in zip(buffer_snapshot, model.buffers()):
            if not torch.equal(before, after.detach()):
                raise AssertionError('test evaluation modified model buffers')
    if count == 0:
        raise ValueError('empty target test dataset')
    mae_steps = (sum_abs/count).tolist()
    rmse_steps = torch.sqrt(sum_sq/count).tolist()
    return {
        'protocol': 'direct_target_test_no_finetune',
        'target_city': TARGET_CITY,
        'target_finetune_epochs': 0,
        'test_batches':num_batches,
        'test_node_window_pairs':count,
        'overall_mae': float(sum_abs.sum()/(count*36)),
        'overall_rmse':float(torch.sqrt(sum_sq.sum()/(count*36))),
        'mae_by_horizon':mae_steps,
        'rmse_by_horizon':rmse_steps,
        'selected_horizons':{
            str(h):{'minutes':h*5,'mae':mae_steps[h-1],
                    'rmse':rmse_steps[h-1]}
            for h in (2,12,24,36)
        }
    }


def main(argv=None):
    ap=argparse.ArgumentParser(
        description='MTPB source meta-training with target guidance, no target finetuning.')
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--mode',choices=['native','target_guided'],default='target_guided',
                    help='native source-only update or target-guided replacement')
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--order-count',type=int,default=3)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--target-days',type=int,default=3)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--skip-eval',action='store_true',
                    help='only for a source-train smoke test; never target finetune')
    args=ap.parse_args(argv)
    if args.epochs<1 or args.order_count<1 or args.target_days!=3:
        ap.error('epochs/order-count must be positive; target-days must be 3')
    upstream=args.upstream.resolve()
    if not (upstream/'model/Meta_Models/rep_model_final.py').is_file():
        ap.error('--upstream must point to the official MTPB checkout')
    sys.path[:0]=[str(upstream),str(upstream/'model'),
                  str(upstream/'model/Meta_Models'),str(upstream/'model/TSFormer')]
    from datasets import traffic_dataset
    from rep_model_final import STRep
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with (upstream/'configs/config.yaml').open(encoding='utf-8') as file:
        conf=yaml.safe_load(file)
    data_args, task_args, model_args=conf['data'],conf['task'],conf['model']
    if (task_args['maml']['his_num'],task_args['maml']['pred_num'])!=(288,36):
        raise RuntimeError('expected official MTPB 288 history, 36 horizon')
    task_args['maml']['train_epochs']=args.epochs
    for city in data_args['data_keys']:
        for key in ('dataset_path','adjacency_matrix_path'):
            path=Path(data_args[city][key])
            path=path if path.is_absolute() else upstream/path
            if not path.is_file():
                raise FileNotFoundError(f'missing official {city} {key}: {path}')
            data_args[city][key]=str(path)
    pattern_dir=upstream/'pattern/chengdu_shenzhen_metr'
    for hours in PATTERN_HOURS:
        artifact=pattern_dir/f'cosine_10_cl_{hours}.pt'
        if not artifact.is_file():
            raise FileNotFoundError(f'missing official pattern bank: {artifact}')

    # The author defaults to add_target=True; exclude target here so
    # target labels are consumed exclusively as gradient guidance.
    source=traffic_dataset(data_args,task_args['maml'],list(SOURCE_CITIES),
                           'source_train',test_data=TARGET_CITY,
                           add_target=False,target_days=3)
    target_guide=traffic_dataset(data_args,task_args['maml'],[],
                                 'target_maml',test_data=TARGET_CITY,
                                 add_target=False,target_days=3)
    # Create held-out dataset for evaluation only; never supply it to the updater.
    target_test=traffic_dataset(data_args,task_args['maml'],[],
                                'test',test_data=TARGET_CITY,
                                add_target=False,target_days=3)
    from tsjt_pgrad.mtpb.train_only_norm import apply_train_only_target_normalization
    train_stats=apply_train_only_target_normalization(
        target_guide,target_test,data_args[TARGET_CITY]['dataset_path'],train_days=3)

    args.output.mkdir(parents=True,exist_ok=True)
    run_args=argparse.Namespace(
        STmodel='GWN',device=device,seed=args.seed,save_dir=args.output,
        curr_time=datetime.now().isoformat(),
        batch_size=task_args['maml']['batch_size'])
    patch_cfg=dict(data_list='chengdu_shenzhen_metr',sim='cosine',K=10,
                   patch_encoder='pattern',base_dir=upstream,device=device)
    rep=STRep(data_args,task_args,model_args,patch_cfg,run_args)
    from tsjt_pgrad.mtpb.upstream_compat import repair_mtpb_gwnet_convs
    repairs=repair_mtpb_gwnet_convs(rep)
    updater=None
    if args.mode=='target_guided':
        from tsjt_pgrad.mtpb.integrated_meta import (
            IntegratedMetaConfig,TargetGuidedMTPBMetaUpdate)
        updater=TargetGuidedMTPBMetaUpdate(
            rep,IntegratedMetaConfig(
                inner_lr=float(model_args['STnet']['update_lr']),
                inner_steps=int(model_args['STnet']['update_step']),
                meta_lr=float(model_args['STnet']['update_lr']),
                order_count=args.order_count))

    protocol={
        'mode':args.mode,'model':'official_MTPB_PatchFSL_GWN',
        'history':288,'horizon':36,
        'source_cities':list(SOURCE_CITIES),'target_city':TARGET_CITY,
        'target_train_days':3,'target_train_role':'gradient_guidance_only',
        'target_inner_adaptation_steps':0,'target_finetune_epochs':0,
        'test_only_after_source_training':True,
        'source_stage_add_target':False,
        'target_train_only_normalization':train_stats,
        'seed':args.seed,'source_epochs':args.epochs,
        'gwn_compat_repairs':repairs,'skip_eval':args.skip_eval,
        'uses_prebuilt_source_pattern_bank':True,
    }
    (args.output/'protocol.json').write_text(
        json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    with (args.output/'diagnostics.jsonl').open('w',encoding='utf-8') as out:
        for epoch in range(args.epochs):
            if updater is not None:
                diag=updater.step(source,target_guide)
            else:
                support,matrix_s,query,matrix_q=[],[],[],[]
                for _ in range(task_args['maml']['task_num']):
                    idx=int(torch.randperm(source.pretrain_batchnum-1)[0])
                    d,a=source[idx]
                    support.append(d);matrix_s.append(a)
                    d,a=source[idx+1]
                    query.append(d);matrix_q.append(a)
                native_metrics=rep.meta_train_revise(
                    support,matrix_s,query,matrix_q)
                diag={'source_native_metrics':[float(v) for v in native_metrics]}
            record=dict(epoch=epoch+1,mode=args.mode,**diag)
            out.write(json.dumps(record,ensure_ascii=False)+'\n')
            out.flush()
            print(json.dumps(record,ensure_ascii=False))

    # Crucially, no rep.finetuning() / train_batch(target_test) call anywhere.
    torch.save(rep.model.state_dict(),args.output/'source_trained_final.pt')
    if not args.skip_eval:
        result=evaluate_without_adaptation(rep,target_test)
        (args.output/'direct_target_test.json').write_text(
            json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
