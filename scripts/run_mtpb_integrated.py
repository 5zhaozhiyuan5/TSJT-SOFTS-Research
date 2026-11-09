"""MTPB native baseline vs direct replacement of its source meta-optimizer.

Requires separately checked-out zhyliu00/MTPB at pinned commit; upstream
pretraining, pattern bank, datasets, and native 288->36 prediction stay intact.
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


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--upstream',type=Path,required=True)
    ap.add_argument('--mode',choices=['native','integrated'],default='integrated')
    ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--finetune-epochs',type=int,default=300)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--target-days',type=int,default=3)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--skip-finetune',action='store_true')
    ap.add_argument('--order-count',type=int,default=3)
    args=ap.parse_args(argv)
    upstream=args.upstream.resolve()
    if not (upstream/'model/Meta_Models/rep_model_final.py').exists():
        ap.error('--upstream must point to an official MTPB checkout')
    if args.epochs<1 or args.finetune_epochs<0 or args.target_days!=3:
        ap.error('epochs must be positive and target-days must equal 3')
    sys.path[:0]=[str(upstream),str(upstream/'model'),
                  str(upstream/'model/Meta_Models'),str(upstream/'model/TSFormer')]
    from datasets import traffic_dataset
    from rep_model_final import STRep
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(upstream/'configs/config.yaml',encoding='utf-8') as f:
        config=yaml.safe_load(f)
    task_args,model_args,data_args=config['task'],config['model'],config['data']
    if task_args['maml']['his_num']!=288 or task_args['maml']['pred_num']!=36:
        raise RuntimeError('native MTPB must use history=288, horizon=36')
    task_args['maml']['train_epochs']=args.epochs
    task_args['maml']['finetune_epochs']=args.finetune_epochs
    for city in data_args['data_keys']:
        for key in ('dataset_path', 'adjacency_matrix_path'):
            raw=Path(data_args[city][key])
            data_args[city][key]=str(raw if raw.is_absolute() else upstream/raw)
    if not (upstream/'pattern/chengdu_shenzhen_metr/cosine_10_cl_1.pt').exists():
        raise FileNotFoundError('Official MTPB pattern bank missing: generate/download official pattern artifacts first')
    target='pems-bay'
    source_cities=['chengdu_m','shenzhen','metr-la']
    # Prevent source_dataset from mutating the caller's list in-place.
    source=traffic_dataset(data_args,task_args['maml'],source_cities.copy(),
                           'source_train',test_data=target,add_target=False,
                           target_days=3)
    target_train=traffic_dataset(data_args,task_args['maml'],[],
                                 'target_maml',test_data=target,add_target=False,
                                 target_days=3)
    target_test=traffic_dataset(data_args,task_args['maml'],[],
                                'test',test_data=target,add_target=False,
                                target_days=3)
    from tsjt_pgrad.mtpb.train_only_norm import apply_train_only_target_normalization
    target_stats=apply_train_only_target_normalization(
        target_train,target_test,data_args[target]['dataset_path'],train_days=3)
    args.output.mkdir(parents=True,exist_ok=True)
    run_args=argparse.Namespace(STmodel='GWN',device=device,seed=args.seed,
                                save_dir=args.output,curr_time=datetime.now().isoformat(),
                                batch_size=task_args['maml']['batch_size'])
    patch_cfg=dict(data_list='chengdu_shenzhen_metr',sim='cosine',K=10,
                   patch_encoder='pattern',base_dir=upstream,device=device)
    rep=STRep(data_args,task_args,model_args,patch_cfg,run_args)
    from tsjt_pgrad.mtpb.upstream_compat import repair_mtpb_gwnet_convs
    repaired=repair_mtpb_gwnet_convs(rep)
    print(f'MTPB upstream GWN compatibility repair: {repaired} conv layers')
    updater=None
    if args.mode=='integrated':
        from tsjt_pgrad.mtpb.integrated_meta import (
            TargetGuidedMTPBMetaUpdate,IntegratedMetaConfig)
        updater=TargetGuidedMTPBMetaUpdate(rep,IntegratedMetaConfig(
            inner_lr=float(model_args['STnet']['update_lr']),
            inner_steps=int(model_args['STnet']['update_step']),
            meta_lr=float(model_args['STnet']['update_lr']),
            order_count=args.order_count))
    protocol={'mode':args.mode,'model':'official_MTPB_PatchFSL_GWN',
       'history':288,'horizon':36,'target_days':3,
       'source_cities':source_cities,'target_city':target,
       'target_train_normalization':target_stats,
       'source_meta_update_replaced':updater is not None,
       'seed':args.seed,'epochs':args.epochs,
       'finetune_epochs':args.finetune_epochs,'gwn_compat_repairs':repaired}
    (args.output/'run_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    with open(args.output/'diagnostics.jsonl','w',encoding='utf-8') as f:
        for epoch in range(args.epochs):
            if updater is None:
                support,matrices,query,qmat=[],[],[],[]
                for _ in range(task_args['maml']['task_num']):
                    idx=int(torch.randperm(source.pretrain_batchnum-1)[0])
                    d,a=source[idx]
                    support.append(d);matrices.append(a)
                    d,a=source[idx+1]
                    query.append(d);qmat.append(a)
                metrics=rep.meta_train_revise(support,matrices,query,qmat)
                diag={'native_metrics':[float(v) for v in metrics]}
            else:
                # Sole real source meta step: original MTPB meta_train_revise
                # is intentionally not invoked in integrated mode.
                diag=updater.step(source,target_train)
            record=dict(epoch=epoch+1,mode=args.mode,**diag)
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
            f.flush()
            print(json.dumps(record,ensure_ascii=False))
    if not args.skip_finetune:
        rep.finetuning(target_train,target_test,args.finetune_epochs)


if __name__=='__main__':
    main()
