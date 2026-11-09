"""TSJT backbone training with independently testable PGrad / rotation / fusion.

All real parameter updates use *explicit* TSJT-style SGD.  Virtual PGrad
and angle Query-R steps never modify the backbone or its optimizer state.
"""
from __future__ import annotations
import copy
import json
import math
import random
from collections.abc import Mapping,Sequence
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
import yaml
from torch.func import functional_call
from torch.utils.data import Dataset

from tsjt_c1.training import engine
from tsjt_c1.training.tsb import filter_source_gradient
from ..models.pgrad_model import TSJTPGrad
from ..pgrad.grad_utils import flatten_gradients,unflatten_gradients
from ..pgrad.pgrad import pgrad_principal_gradient
from ..pgrad.source_interaction import source_interaction_principal_gradient
from ..pgrad.rotation import rotate_source_gradient,decompose_parallel_orthogonal,angle_statistics
from ..pgrad.fusion import residual_matching_fusion
from .meta import virtual_parameter_update,angle_meta_loss,apply_meta_update

GRADIENT_METHODS={
    'tsjt_original','per_source_tsb','average_rotation','pgrad_fixed_rotation',
    'pgrad_rotation_only','residual_only','scale_only','full_method',
}
PGRAD_METHODS={'pgrad_fixed_rotation','pgrad_rotation_only','full_method'}
ANGLE_METHODS={'average_rotation','pgrad_rotation_only','full_method','scale_only'}
FUSION_METHODS={'average_rotation','pgrad_fixed_rotation','residual_only','scale_only','full_method'}


def _build_model(config):
    m=config.get('model',{})
    kwargs=m.get('kwargs',m)
    return TSJTPGrad(**{k:v for k,v in kwargs.items() if k not in {'factory','backbone'}})


def _flatten_model(config):
    runtime=copy.deepcopy(config)
    model=runtime['model']
    kwargs=model.pop('kwargs',{}) or {}
    model.update(kwargs)
    model.setdefault('backbone','graph_wavenet')
    return runtime


def _with_test_override(config):
    runtime=copy.deepcopy(config)
    runtime.setdefault('run',{})['test_override']=True
    runtime['run']['formal']=False
    return runtime


def _validate_pgrad_config(config):
    method=config['train'].get('gradient_method','tsjt_original')
    if method not in GRADIENT_METHODS:
        raise ValueError(f'unknown gradient_method {method}')
    if int(config['train'].get('max_epochs', config['train']['epochs'])) != int(config['train']['epochs']):
        raise ValueError('train.max_epochs and train.epochs differ; set both explicitly to the intended run length')
    if int(config.get('evaluation', {}).get('evaluate_once_after_epoch', config['train']['epochs'])) != int(config['train']['epochs']):
        raise ValueError('evaluation.evaluate_once_after_epoch must equal train.epochs even when skip_final_evaluation is true')
    # Mirror the old adapter's validation. Do not silently change input config.
    engine._validate_config(_with_test_override(_flatten_model(config)))


def _grads(loss,params):
    return tuple(g.detach() if g is not None else torch.zeros_like(p)
                 for g,p in zip(torch.autograd.grad(loss,params,allow_unused=True),params))


def _zeros_like(params):
    return tuple(torch.zeros_like(p) for p in params)


def _aggregate(groups,params,weights=None):
    if not groups:
        return _zeros_like(params)
    if weights is None:
        weights=[1.]*len(groups)
    flat=sum((float(w)*flatten_gradients(params,grads) for w,grads in zip(weights,groups)),
             torch.zeros_like(flatten_gradients(params,groups[0])))
    return unflatten_gradients(params,flat)


def _norm(grads,params):
    return float(flatten_gradients(params,grads).norm().detach())


def _tsb_filtered(target_grads, source_grads, params):
    # Original TSB filter is per-parameter. Retain this precise behavior for baseline.
    out=[]
    for tg,sg,p in zip(target_grads,source_grads,params):
        t=tg if tg is not None else torch.zeros_like(p)
        s=sg if sg is not None else torch.zeros_like(p)
        if sg is None:
            out.append(torch.zeros_like(p));continue
        filtered,_=filter_source_gradient(t,s)
        out.append(filtered.detach())
    return tuple(out)


def _projected(params,src,target):
    return decompose_parallel_orthogonal(src,target,params)[0]


def _mean_direction(params, per_city):
    return _aggregate(per_city,params,weights=[1./len(per_city)]*len(per_city))


def _real_backbone_gradients(model,params,x,y,adj):
    pred=model(x,adj,horizon=y.shape[1])
    if isinstance(pred,dict):pred=pred['prediction']
    loss=F.l1_loss(pred,y)
    return loss,_grads(loss,params)


def _pgrad_direction(model,source_chunks,params,device,virtual_lr,num_components,rounds=1,
                     *,source_grads=None,target_grads=None,interaction_config=None,previous_weights=None):
    """Re-evaluate source forward/backward under each successive virtual weight state."""
    prepared=[(x.to(device),y.to(device),adj.to(device) if torch.is_tensor(adj) else adj)
              for x,y,adj in source_chunks]
    def callback(city,virtual_params):
        sx,sy,adj=prepared[city]
        buffers={n:b.detach().clone() for n,b in model.backbone.named_buffers()}
        pred=functional_call(model.backbone,(virtual_params,buffers),(sx,adj),{'horizon':sy.shape[1]})
        if isinstance(pred,dict):pred=pred['prediction']
        loss=F.l1_loss(pred,sy)
        grads=torch.autograd.grad(loss,tuple(virtual_params.values()),allow_unused=True)
        return grads
    cfg = interaction_config or {}
    mode = cfg.get('source_interaction_mode', 'legacy_pgrad')
    if mode == 'legacy_pgrad':
        return pgrad_principal_gradient(model.backbone,callback,virtual_lr=virtual_lr,
            num_cities=len(prepared),num_components=num_components,rounds=rounds)
    if mode != 'source_interaction_v2':
        raise ValueError(f'unknown source_interaction_mode: {mode}')
    if source_grads is None or target_grads is None:
        raise ValueError('source_interaction_v2 needs real source and target gradients')
    # The caller still receives exactly one PGrad reference tuple + metadata.
    # Rotation, angle meta learning and residual fusion are NOT modified.
    return source_interaction_principal_gradient(
        model.backbone, callback, params, source_grads, target_grads,
        virtual_lr=virtual_lr, num_components=num_components, rounds=rounds,
        order_count=int(cfg.get('source_interaction_order_count', 3)),
        previous_weights=previous_weights,
        target_strength=float(cfg.get('source_interaction_target_strength', 1.0)),
        synergy_strength=float(cfg.get('source_interaction_synergy_strength', 0.5)),
        target_consistent_synergy=float(cfg.get('source_interaction_target_consistent_synergy', 0.0)),
        conflict_strength=float(cfg.get('source_interaction_conflict_strength', 0.5)),
        redundancy_strength=float(cfg.get('source_interaction_redundancy_strength', 0.6)),
        redundancy_threshold=float(cfg.get('source_interaction_redundancy_threshold', 0.8)),
        temperature=float(cfg.get('source_interaction_temperature', 0.35)),
        uniform_mix=float(cfg.get('source_interaction_uniform_mix', 0.15)),
        weight_ema=float(cfg.get('source_interaction_weight_ema', 0.65)),
        trajectory_blend=float(cfg.get('source_interaction_trajectory_blend', 0.5)),
    )


def _scale_only_directions(model, per_city, target, principal, params, *, keep_phi):
    # A direction-preserving trainable scale comparator (no rotation).
    dirs=[]; phis=[]; metas=[]
    for src in per_city:
        stats=angle_statistics(src,target,principal,params)
        phi=model.angle_controller(stats)
        factor=0.75+0.5*phi/model.angle_controller.phi_max  # ~0.77 initially
        base=_projected(params,src,target)
        dirs.append(tuple(factor.squeeze()*g for g in base))
        phis.append(phi)
        metas.append({'phi':float(phi.detach()),'scale':float(factor.detach()),'rotated':False})
    return dirs,phis,metas


def _build_source_directions(model,params,per_city,target,principal,method,config):
    """Generate *one* candidate per source city for all nonbaseline modes."""
    if method in {'residual_only','per_source_tsb'}:
        return [_tsb_filtered(target,src,params) for src in per_city],[],[]
    if method=='scale_only':
        return _scale_only_directions(model,per_city,target,principal,params,keep_phi=True)
    fixed= float(config['train'].get('fixed_angle',math.pi/12)) if method=='pgrad_fixed_rotation' else None
    dirs=[];phis=[];metas=[]
    for src in per_city:
        direction,metadata,phi=rotate_source_gradient(src,target,principal,params,
            model.angle_controller,fixed_angle=fixed,return_phi_tensor=True)
        dirs.append(direction);phis.append(phi);metas.append(metadata)
    return dirs,phis,metas


def _weighted_source(params,directions,weights,lam=1.):
    return _aggregate(directions,params,weights=[lam*float(w) for w in weights])


def _effective_source_cap(params,target,source,gamma_t,gamma_s,budget_ratio:float):
    """Cap (never amplify) source real-update norm relative to target real-update."""
    t_norm=abs(gamma_t)*_norm(target,params)
    s_norm=abs(gamma_s)*_norm(source,params)
    cap_factor=1.
    if budget_ratio>0 and s_norm>budget_ratio*t_norm and s_norm>1e-12:
        cap_factor=budget_ratio*t_norm/s_norm
    return cap_factor, s_norm/(t_norm+1e-12)


def _train_step(
    model:TSJTPGrad,parameters:Sequence[torch.nn.Parameter],target_x,target_y,target_adjacency,
    source_chunks,config,device,step_index,*,state:dict|None=None,
    query_r=None,query_w=None,
):
    cfg=config['train'];method=cfg['gradient_method']
    if state is None:state={}
    gamma_t=float(cfg['gamma_target']);gamma_s=float(cfg['gamma_source'])
    meta_interval=max(1,int(cfg.get('meta_interval',10)))
    fusion_interval=max(1,int(cfg.get('fusion_interval',10)))
    pgrad_interval=max(1,int(cfg.get('pgrad_interval',10)))
    model.train()
    target_x=target_x.to(device);target_y=target_y.to(device)
    if torch.is_tensor(target_adjacency):target_adjacency=target_adjacency.to(device)
    target_loss,target_grads=_real_backbone_gradients(model,parameters,target_x,target_y,target_adjacency)
    per_city=[];losses=[]
    for x,y,adj in source_chunks:
        loss,grads=_real_backbone_gradients(model,parameters,x.to(device),y.to(device),
            adj.to(device) if torch.is_tensor(adj) else adj)
        per_city.append(grads);losses.append(float(loss.detach()))
    diag={'method':method,'step':step_index,'target_loss':float(target_loss.detach()),
          'source_loss':sum(losses)/len(losses)}
    if method=='tsjt_original':
        # Preserve original baseline: first sum city grads then apply TSB independently per parameter.
        source_term=_tsb_filtered(target_grads,_aggregate(per_city,parameters),parameters)
        diag['path']=['tsb_original']
    else:
        if method in PGRAD_METHODS:
            if step_index%pgrad_interval==0 or 'pgrad' not in state:
                principal,pm=_pgrad_direction(model,source_chunks,parameters,device,
                    float(cfg.get('pgrad_virtual_lr',1e-3)),int(cfg.get('pgrad_num_components',2)),
                    int(cfg.get('pgrad_rounds',1)),
                    source_grads=per_city,target_grads=target_grads,
                    interaction_config=cfg,
                    previous_weights=state.get('source_interaction_weights'))
                state['pgrad']=tuple(g.detach() for g in principal)
                state['pgrad_meta']=pm
                if pm.get('mode') == 'source_interaction_v2':
                    state['source_interaction_weights']=pm['source_weights']
                diag['pgrad_updated']=True
            principal=state['pgrad']
            diag['pgrad']=state['pgrad_meta']
        else:
            principal=_mean_direction(parameters,per_city)
        # Never put angle-controller parameters inside backbone gradient vectors.
        directions,phis,rot_meta=_build_source_directions(model,parameters,per_city,target_grads,principal,method,config)
        diag['rotation']=rot_meta
        diag['num_source_directions']=len(directions)
        diag['path']=['per_city_gradient']
        if method in PGRAD_METHODS:diag['path'].append('pgrad')
        if method in ANGLE_METHODS or method=='pgrad_fixed_rotation':diag['path'].append('rotation')
        if method in ANGLE_METHODS and step_index%meta_interval==0:
            if query_r is None:
                raise ValueError('Query-R batch required for learned angle updates')
            valid_angle_count=sum(bool(m.get('rotated',True)) for m in rot_meta) if method!='scale_only' else len(rot_meta)
            if valid_angle_count>0:
                old_w=state.get('weights',[1./len(directions)]*len(directions))
                angle_source=_weighted_source(parameters,directions,old_w,
                    float(cfg.get('fusion_lambda',1.)))
                # Virtual update must reflect REAL update coefficients, including gamma_source.
                meta_cap,_=_effective_source_cap(parameters,target_grads,angle_source,
                    gamma_t,gamma_s,float(cfg.get('source_budget_ratio',0.)))
                meta_joint=tuple(gamma_t*t+gamma_s*meta_cap*s for t,s in zip(target_grads,angle_source))
                virtual=virtual_parameter_update(model,parameters,meta_joint,
                    step_scale=float(cfg.get('meta_step_scale',1.)))
                rx,ry=query_r;rx=rx.to(device);ry=ry.to(device)
                loss,qloss,penalty=angle_meta_loss(model,virtual,rx,ry,target_adjacency,
                    phi_tensors=(phis if method=='scale_only' else
                        [phi for phi,m in zip(phis,rot_meta) if m.get('rotated',False)]),
                    mu_phi=float(cfg.get('mu_phi',1e-4)))
                meta=apply_meta_update(model.angle_controller,loss,qloss,
                    learning_rate=float(cfg.get('angle_learning_rate',cfg.get('meta_lr',1e-3))))
                diag['meta']={**meta,'loss':float(loss.detach()),'query_loss':float(qloss.detach()),
                              'angle_reg':float(penalty.detach())}
                # Regenerate directions to use newly updated angle controller.
                directions,phis,rot_meta=_build_source_directions(model,parameters,per_city,target_grads,principal,method,config)
                diag['rotation']=rot_meta
            else:
                diag['meta']={'meta_updated':False,'reason':'no_valid_rotation'}
        if method in FUSION_METHODS:
            if step_index%fusion_interval==0 or 'weights' not in state:
                if query_w is None:
                    raise ValueError('Query-W batch required for residual matching')
                wx,wy=query_w;wx=wx.to(device);wy=wy.to(device)
                _,qgrads=_real_backbone_gradients(model,parameters,wx,wy,target_adjacency)
                source_term,fusion_meta=residual_matching_fusion(
                    target_grads,directions,parameters,target_query_gradients=qgrads,
                    lam=float(cfg.get('fusion_lambda',1.)),mu_w=float(cfg.get('fusion_mu_w',1e-3)),
                    return_source_only=True)
                state['weights']=fusion_meta['weights']
                diag['fusion']=fusion_meta
                diag['fusion_updated']=True
            else:
                source_term=_weighted_source(parameters,directions,state['weights'],
                    float(cfg.get('fusion_lambda',1.)))
                diag['fusion']={'weights':state['weights'],'cached':True}
            diag['path'].append('residual_matching')
        elif method=='pgrad_rotation_only':
            source_term=_weighted_source(parameters,directions,[1/len(directions)]*len(directions))
        elif method=='per_source_tsb':
            source_term=_aggregate(directions,parameters)
        else:
            raise RuntimeError(f'unhandled method {method}')
    budget=(0. if method=='tsjt_original' else float(cfg.get('source_budget_ratio',0.))) # baseline untouched
    cap,raw_ratio=_effective_source_cap(parameters,target_grads,source_term,
        gamma_t,gamma_s,budget)
    src_norm=_norm(source_term,parameters)
    # Single real step; compare the float-dtype update against a target-only
    # counterfactual. A nonzero requested source gradient may round to zero.
    realized_source_sq=0.; realized_total_sq=0.
    with torch.no_grad():
        for p,t,s in zip(parameters,target_grads,source_term):
            before=p.detach().clone()
            target_only=before-gamma_t*t
            p.add_(gamma_t*t+gamma_s*cap*s,alpha=-1.)
            realized_source_sq+=float((p.detach()-target_only).double().square().sum())
            realized_total_sq+=float((p.detach()-before).double().square().sum())
    target_update_norm=abs(gamma_t)*_norm(target_grads,parameters)
    realized_source_norm=realized_source_sq**.5
    diag.update({'source_norm':src_norm,'target_norm':_norm(target_grads,parameters),
                 'effective_source_ratio':raw_ratio*cap,'source_cap_factor':cap,
                 'source_contribution_norm':abs(gamma_s)*src_norm*cap,
                 'target_contribution_norm':target_update_norm,
                 'realized_source_effect_norm':realized_source_norm,
                 'realized_source_effect_ratio':realized_source_norm/(target_update_norm+1e-12),
                 'realized_total_update_norm':realized_total_sq**.5})
    return diag


class IndexedWindowDataset(Dataset):
    """Preserve chronological sample ID while retaining the original TSJT dataset."""
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,i):
        x,y=self.dataset[i]
        return x,y,int(i)


class TrainOnlyQuerySampler:
    """Per-step episodic Support / Query-R / Query-W with disjoint label ranges.

    All samples are from the original 3-day *training* dataset. Samples can
    change roles on other steps; within a step no two roles share label stamps.
    """
    def __init__(self,dataset,history:int,horizon:int):
        if not hasattr(dataset,'indices'):
            raise ValueError('target dataset must expose .indices for timestamp-safe Query sampling')
        self.dataset=dataset
        self.intervals=[(int(v)+history,int(v)+history+horizon) for v in dataset.indices]
        if len(self.intervals)!=len(dataset):
            raise ValueError('target .indices mismatch')
    def sample(self,support_ids,*,query_r_size:int,query_w_size:int,seed:int):
        chosen=[self.intervals[int(i)] for i in support_ids]
        rng=random.Random(seed)
        order=list(range(len(self.intervals)))
        rng.shuffle(order)
        out=[]
        for cand in order:
            start,end=self.intervals[cand]
            if any(start<e and s<end for s,e in chosen):
                continue
            out.append(cand)
            chosen.append((start,end))
            if len(out)==query_r_size+query_w_size:break
        if len(out)<query_r_size+query_w_size:
            raise ValueError('not enough non-overlapping train-only Query windows')
        def collate(ids):
            batch=[self.dataset[i] for i in ids]
            return torch.stack([torch.as_tensor(x) for x,_ in batch]),torch.stack([torch.as_tensor(y) for _,y in batch])
        r=collate(out[:query_r_size]) if query_r_size else None
        w=collate(out[query_r_size:]) if query_w_size else None
        return r,w,{'query_r_ids':out[:query_r_size],'query_w_ids':out[query_r_size:],
                    'support_ids':[int(i) for i in support_ids]}


def run(config_path:str)->str:
    config_path=Path(config_path)
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    _validate_pgrad_config(config)
    engine._set_seed(int(config['run']['seed']))
    device=engine._resolve_device(str(config['run'].get('device','auto')))
    output_dir,config_hash=engine._prepare_output(config_path,config,device,())
    cities=[*(str(d['city']).lower() for d in config['data']['sources']),
            str(config['data']['target']['city']).lower()]
    engine._fingerprint_data(Path(config['data']['root']),cities,output_dir)
    bundle=engine._make_data(_with_test_override(_flatten_model(config)),device,output_dir)
    return _run_all_pgrad(config,output_dir,config_hash,device,bundle)


def _run_all_pgrad(config,output_dir,config_hash,device,data_bundle):
    complete_path=output_dir/'complete.json';final_metrics=output_dir/'final_metrics.json'
    if complete_path.exists():
        # Do not silently reuse finished results when skip_complete=false.
        if not config['run'].get('skip_complete', False):
            raise FileExistsError('Experiment already marked complete; use a fresh output_dir or set run.skip_complete=true to reuse its result')
        ckpt=output_dir/'last.pt'
        if not ckpt.exists() or not final_metrics.exists() or torch.load(ckpt,map_location='cpu',weights_only=False).get('config_hash')!=config_hash:
            raise RuntimeError('complete.json exists but checkpoint/config hash or final metrics are missing/mismatched. Use fresh output_dir.')
        return str(final_metrics)
    if (output_dir/'last.pt').exists() and not config['run'].get('resume',False):
        raise FileExistsError('Existing incomplete last.pt found but resume=false; use a fresh output_dir, or enable resume=true with matching config')
    (target_loaders,target_adj,stats,target_protocol,source_loaders,source_adj,_) =data_bundle
    model=_build_model(config).to(device)
    parameters=[p for p in model.backbone.parameters() if p.requires_grad] # exclude frozen and angle params
    train=config['train'];seed=int(config['run']['seed']);epochs=int(train['epochs'])
    dataset=target_loaders['train'].dataset
    source_datasets={city:loader.dataset for city,loader in source_loaders.items()}
    method=train['gradient_method']
    sampler=TrainOnlyQuerySampler(dataset,int(config['data']['target']['history']),
        int(config['data']['target']['horizon'])) if method in ANGLE_METHODS|FUSION_METHODS else None
    # Isolate outputs across modes/seeds and re-runs.
    log_path=output_dir/'train.jsonl'
    diag_path=output_dir/'grad_diagnostics.jsonl'
    state={};step_index=0;start_epoch=1
    if (output_dir/'last.pt').exists() and config['run'].get('resume',False):
        ckpt=torch.load(output_dir/'last.pt',map_location=device,weights_only=False)
        if ckpt.get('config_hash')!=config_hash:
            raise RuntimeError('last.pt config mismatch; choose new output_dir')
        if 'train_state' not in ckpt:
            raise RuntimeError('old checkpoint lacks training state; start with fresh output_dir')
        model.load_state_dict(ckpt['model'])
        state=ckpt['train_state'];step_index=ckpt['step_index'];start_epoch=ckpt['epoch']+1
    for epoch in range(start_epoch,epochs+1):
        target_loader=engine._epoch_loader(IndexedWindowDataset(dataset),
            batch_size=int(train['target_batch_size']),seed=seed+epoch*10007,
            shuffle=True,num_workers=int(config['run'].get('num_workers',0)),pin_memory=device.type=='cuda')
        source_cursors={}
        allocations=engine._source_allocation(list(source_datasets),int(train['source_batch_size']))
        for i,city in enumerate(source_datasets):
            loader=engine._epoch_loader(source_datasets[city],batch_size=int(train['source_microbatch_size']),
                seed=seed+epoch*10007+(i+1)*1000003,shuffle=True,
                num_workers=int(config['run'].get('num_workers',0)),pin_memory=device.type=='cuda')
            source_cursors[city]=engine._CyclingCursor(loader)
        epoch_loss=0.;epoch_updates=0
        for x,y,ids in target_loader:
            chunks=[]
            for city in source_datasets:
                mini=source_cursors[city].take(allocations[city])
                chunks.append((torch.cat([b[0] for b in mini],dim=0),
                               torch.cat([b[1] for b in mini],dim=0),source_adj[city]))
            r,w=None,None
            if sampler is not None:
                need_r=method in ANGLE_METHODS and step_index%max(1,int(train.get('meta_interval',10)))==0
                need_w=method in FUSION_METHODS and (step_index%max(1,int(train.get('fusion_interval',10)))==0 or 'weights' not in state)
                if need_r or need_w:
                    r,w,_=sampler.sample(ids.tolist(),
                        query_r_size=int(train.get('query_r_batch_size',2)) if need_r else 0,
                        query_w_size=int(train.get('query_w_batch_size',2)) if need_w else 0,
                        seed=seed+epoch*999983+step_index)
            diag=_train_step(model,parameters,x,y,target_adj,chunks,config,device,step_index,
                             state=state,query_r=r,query_w=w)
            epoch_loss+=diag['target_loss'];epoch_updates+=1;step_index+=1
            if step_index%max(1,int(train.get('diagnostic_interval',25)))==0 or 'meta' in diag or 'fusion_updated' in diag:
                engine._append_jsonl(diag_path,{'epoch':epoch,**diag})
            if 0<int(train.get('max_steps_per_epoch',0))<=epoch_updates:
                break
        engine._append_jsonl(log_path,{'epoch':epoch,'target_loss':epoch_loss/max(epoch_updates,1),
            'logical_updates':epoch_updates,'gradient_method':method})
        engine._atomic_checkpoint(output_dir/'last.pt',{'model':model.state_dict(),
            'epoch':epoch,'step_index':step_index,'train_state':state,'config_hash':config_hash})
    if bool(config.get('evaluation',{}).get('skip_final_evaluation',False)):
        test={'skipped':True,'reason':'diagnostic smoke only'}
    else:
        test_loader=engine._evaluation_loader(target_loaders['test'].dataset,config,device)
        test=engine._evaluate_full_test(model,test_loader,target_adj,stats,int(target_protocol['horizon']),device)
    result={'seed':seed,'trained_epochs':epochs,'gradient_method':method,'test':test,'protocol':target_protocol}
    engine._atomic_json(final_metrics,result)
    engine._atomic_json(complete_path,{'epoch':epochs,'final_metrics':str(final_metrics)})
    return str(final_metrics)

__all__=['run','GRADIENT_METHODS','_train_step','TrainOnlyQuerySampler']
