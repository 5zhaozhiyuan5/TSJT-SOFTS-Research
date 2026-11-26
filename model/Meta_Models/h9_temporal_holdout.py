"""H9 leakage-guarded chronological development validation (not official test).

The original PMME checkpoint uses global-series normalization. Reuse the
checkpoint's mean/std without re-fitting on held-out data. This is therefore
CHRONOLOGICALLY disjoint from target support, but NOT fully train-only
normalization. Only the first fixed days following support are read for
development validation; all subsequent time steps are reserved and untouched.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch


def chronological_bounds(total_steps, *, support_days=3, validation_days=4,
                         steps_per_day=288, history=288, horizon=36, stride=12):
    """Split raw time into support [0,S), dev [S,V), reserve [V,total).

    All dev windows have *entire history and future* within [S,V), with no
    shared time step with original target SUPPORT. This deliberately consumes
    the first post-support period, previously part of the official test set.
    """
    for key,value in {
        "support_days":support_days,"validation_days":validation_days,
        "steps_per_day":steps_per_day,"history":history,
        "horizon":horizon,"stride":stride,
    }.items():
        if not isinstance(value,int) or value<1:
            raise ValueError(f"{key} must be a positive integer")
    if validation_days*steps_per_day<history+horizon:
        raise ValueError("Validation interval too short for one full window")
    s=support_days*steps_per_day
    v=s+validation_days*steps_per_day
    if total_steps<=v:
        raise ValueError("No untouched future remains after dev validation")
    starts=list(range(
        s,v-history-horizon+1,stride
    ))
    if not starts:
        raise ValueError("Empty chronological dev-window list")
    for start in starts:
        if start<s or start+history+horizon>v:
            raise AssertionError("Validation history/future crossed a boundary")
    return {
        "support_end_exclusive":s,
        "development_start_inclusive":s,
        "development_end_exclusive":v,
        "untouched_future_start_inclusive":v,
        "full_time_steps":int(total_steps),
        "history":history,"horizon":horizon,"stride":stride,
        "starts":starts,
    }


def create_development_batches(dataset_path, *, normalization_mean,
                               normalization_std, support_days=3,
                               validation_days=4, steps_per_day=288,
                               history=288,horizon=36,stride=12,
                               batch_size=8,max_windows=0):
    """Load ONLY early post-support dev tensors, using original F2 scaler.

    Mimics datasets.py's feature [normalized speed, weekly time slot] and raw
    physical-speed future. Does not construct official Stage='test' dataset.
    max_windows=0 evaluates ALL predeclared dev windows; otherwise selects
    evenly spaced windows by fixed indices determined WITHOUT labels.
    """
    if batch_size<1 or max_windows<0:
        raise ValueError("Bad validation batch_size/max_windows")
    raw=np.load(str(dataset_path),mmap_mode="r")
    if raw.ndim!=3 or raw.shape[-1]<2:
        raise ValueError("Expected original PMME [time,nodes,features] array")
    means=np.asarray(normalization_mean,dtype=np.float64).reshape(-1)
    stds=np.asarray(normalization_std,dtype=np.float64).reshape(-1)
    if (len(means)!=1 or len(stds)!=1 or not np.isfinite(means).all() or
        not np.isfinite(stds).all() or float(stds[0])<=0):
        raise ValueError("Expected original positive finite scalar F2 mean/std")
    bounds=chronological_bounds(
        raw.shape[0],support_days=support_days,
        validation_days=validation_days,steps_per_day=steps_per_day,
        history=history,horizon=horizon,stride=stride
    )
    starts=bounds["starts"]
    if max_windows and max_windows<len(starts):
        # Distributed over chronological dev period; selection uses NO labels.
        picks=np.linspace(0,len(starts)-1,num=max_windows,dtype=np.int64)
        starts=[starts[int(j)] for j in picks]
    dev_start=bounds["development_start_inclusive"]
    dev_end=bounds["development_end_exclusive"]
    # This is the ONLY raw time slice loaded into memory for H9 development.
    speed=np.asarray(raw[dev_start:dev_end,:,0],dtype=np.float32)
    week=np.asarray(raw[dev_start:dev_end,:,-1],dtype=np.float32)
    batches=[]
    cursor=0
    for i in range(0,len(starts),batch_size):
        picked=starts[i:i+batch_size]
        xs=[];ys=[];time_spans=[];day_ids=[]
        for index in picked:
            start=index-dev_start
            stop=start+history
            end=stop+horizon
            x_speed=(speed[start:stop].T-float(means[0]))/float(stds[0])
            x_week=week[start:stop].T
            x=np.stack((x_speed,x_week),axis=-1).astype(np.float32)
            y=speed[stop:end].T.astype(np.float32)
            if x.shape[-2:]!=(history,2) or y.shape[-1]!=horizon:
                raise AssertionError("Invalid development slice")
            xs.append(x)
            ys.append(y)
            forecast_start=index+history
            time_spans.append({
                "input_start":int(index),
                "input_end_exclusive":int(index+history),
                "forecast_start":int(forecast_start),
                "forecast_end_exclusive":int(index+history+horizon),
            })
            day_ids.append((forecast_start-dev_start)//steps_per_day)
        batches.append((
            SimpleNamespace(
                x=torch.from_numpy(np.stack(xs)),
                # Evaluation forward must NOT receive true future labels:
                y=torch.zeros_like(torch.from_numpy(np.stack(ys))),
                means=means.copy(),stds=stds.copy(),
                data_name="pems-bay-dev-blind",
            ),
            torch.from_numpy(np.stack(ys)),
            time_spans,
            [int(d) for d in day_ids],
        ))
        cursor+=len(picked)
    if cursor!=len(starts):
        raise AssertionError("Dev windows lost")
    bounds=dict(bounds)
    bounds.pop("starts")
    bounds.update({
        "window_count":len(starts),
        "window_start_indices":[int(z) for z in starts],
        "forecast_start_min":int(min(starts)+history),
        "forecast_end_max_exclusive":int(max(starts)+history+horizon),
        "normalization":"Original F2 full-series mean/std (transductive baseline; not train-only)",
        "source":"original preprocessed dataset_expand.npy, first post-support days only",
        "no_target_dev_labels_in_gradient":True,
        "reserved_future_not_read_into_batches":True,
        "official_test_comparability":"Post-support development overlaps original paper test split; future formal evaluation must exclude it.",
    })
    return batches,bounds


class ForecastAccumulator:
    """Streaming overall, horizon, and calendar-day metrics from blind forward."""
    def __init__(self,horizon=36):
        self.horizon=horizon
        self.abs_sum=0.;self.sq_sum=0.;self.count=0
        self.habs=np.zeros(horizon,dtype=np.float64)
        self.hsq=np.zeros(horizon,dtype=np.float64)
        self.hcount=0
        self.days={}

    @torch.no_grad()
    def add(self,pred,truth,day_ids):
        if pred.shape!=truth.shape or pred.ndim!=3:
            raise ValueError("Mismatched [windows,nodes,horizon] tensors")
        if pred.shape[-1]!=self.horizon or len(day_ids)!=len(pred):
            raise ValueError("Horizon or time/day mismatch")
        if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(truth).all()):
            raise FloatingPointError("Nonfinite heldout prediction/label")
        err=(pred.float().cpu()-truth.float().cpu())
        abs_err=err.abs().double()
        sq_err=err.double().square()
        self.abs_sum+=float(abs_err.sum())
        self.sq_sum+=float(sq_err.sum())
        self.count+=err.numel()
        self.habs+=abs_err.sum((0,1)).numpy()
        self.hsq+=sq_err.sum((0,1)).numpy()
        self.hcount+=int(pred.shape[0]*pred.shape[1])
        for i,day in enumerate(day_ids):
            key=int(day)
            if key not in self.days:
                self.days[key]=[0.,0.,0]
            self.days[key][0]+=float(abs_err[i].sum())
            self.days[key][1]+=float(sq_err[i].sum())
            self.days[key][2]+=int(err[i].numel())

    def result(self):
        if self.count<=0:
            raise ValueError("No evaluation labels scored")
        return {
            "mae":self.abs_sum/self.count,
            "rmse":math.sqrt(self.sq_sum/self.count),
            "mae_at_minutes":{
                str(m):float(self.habs[i]/self.hcount)
                for m,i in ((10,1),(60,11),(120,23),(180,35))
            },
            "horizon_band_mae":{
                k:float(self.habs[a:b].sum()/(self.hcount*(b-a)))
                for k,a,b in (("short",0,12),("mid",12,24),("long",24,36))
            },
            "forecast_days":{
                str(k):{
                    "mae":v[0]/v[2],
                    "rmse":math.sqrt(v[1]/v[2]),
                    "value_count":v[2],
                }
                for k,v in sorted(self.days.items())
            },
            "value_count":self.count,
            "window_count":None,
        }


@torch.no_grad()
def score_development(model,batches,*,seed=20271127):
    """No gradient, fixed RNG per batch, BLINDED labels in model forward."""
    device=next(model.parameters()).device
    cuda_devices=([device.index if device.index is not None
                   else torch.cuda.current_device()] if device.type=="cuda" else [])
    acc=ForecastAccumulator()
    states={n:m.training for n,m in model.named_modules()}
    dynamic={
        key:(hasattr(model,key),getattr(model,key,None))
        for key in ("RevIn_mean","RevIn_std","eps")
    }
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            for index,(blind,labels,spans,days) in enumerate(batches):
                torch.manual_seed(int(seed)+index)
                prediction,*_=model(blind,stage="test")
                acc.add(prediction,labels,days)
                del prediction
    finally:
        modules=dict(model.named_modules())
        for k,v in states.items():
            modules[k].training=v
        for k,(exists,value) in dynamic.items():
            if exists:
                setattr(model,k,value)
            elif hasattr(model,k):
                delattr(model,k)
    result=acc.result()
    result["window_count"]=sum(len(batch[3]) for batch in batches)
    result["evaluation_rng_seed"]=int(seed)
    return result
