"""H10 exact original-F2 gradient bank with ALL target support labels.

The H9 audit used a 36/10 target FIT/PROBE split; official Stage-I uses the
full 46 target windows to form the target matching marginal and delta.
This new bank restores that original behavior. IMPORTANT: exact parity
against the author training path must still be checked numerically.
"""
from __future__ import annotations
import hashlib
import torch
from audit_h3_city_checkpoint import sampled_flat_indices
try:
    from .h1_shadow_audit import _grads
    from .h6_ot_gradient_effect import gradient_norm
    from .transfer_utility import compute_weights
except ImportError:
    from h1_shadow_audit import _grads
    from h6_ot_gradient_effect import gradient_norm
    from transfer_utility import compute_weights

def _grad_with_presence(loss,parameters):
    values=torch.autograd.grad(
        loss,parameters,retain_graph=True,allow_unused=True
    )
    return ([torch.zeros_like(p) if x is None else x.detach()
             for p,x in zip(parameters,values)],
            [x is not None for x in values])


def original_f2_full_support_bank(
    model, samples, *, loss_criterion, sinkhorn,source_points,target_points,
    ot_scale=.01,step=1
):
    """Compute F2 source/OT gradient with ALL 46 support windows.

    This is a fidelity repair relative to H9's 36 FIT windows.
    The full-OT option (source_points=target_points=0) is required
    for official Stage-I numerical parity. DEV labels are never loaded.
    """
    if len(samples)!=4 or not (source_points==0 or source_points>=32) or not (
        target_points==0 or target_points>=16
    ):
        raise ValueError("H9 expected 4 batches, source 0/>=32, target 0/>=16")
    named=list(model.named_parameters())
    params=[x for _,x in named]
    transfers=[]
    def record(_module,input_tensor,output):
        transfers.append((input_tensor[0],output))
    handle=model.forecasting_model.projection2.register_forward_hook(record)
    forecasts=[];truths=[];representations=[];futures=[]
    try:
        torch.manual_seed(352000+int(step))
        for sample in samples:
            pred,true,rep,norm_future,_=model(sample,stage="train")
            forecasts.append(pred);truths.append(true)
            representations.append(rep);futures.append(norm_future)
    finally:
        handle.remove()
    if len(transfers)!=4:
        raise RuntimeError("Missing projection2 capture from one or more cities")
    n=forecasts[0].shape[0]
    if n!=46:
        raise ValueError("Expected fixed 46 target-support windows like H8")
    fit=torch.ones(n,dtype=torch.bool,device=forecasts[0].device)
    unused_probe=torch.zeros_like(fit)
    t_loss=loss_criterion(forecasts[0],truths[0])
    s_losses=[
        loss_criterion(forecasts[i],truths[i]) for i in (1,2,3)
    ]
    source_grad,source_mask=_grad_with_presence(sum(s_losses),params)
    target_grad,target_mask=_grad_with_presence(t_loss,params)
    ot_mask=[False]*len(params)
    target_features,target_out=transfers[0]
    target_delta=torch.autograd.grad(t_loss,target_out,retain_graph=True)[0].detach()
    split_sizes=(15,15,n-30)
    t_rep=torch.split(representations[0],split_sizes,dim=0)
    t_y=torch.split(futures[0],split_sizes,dim=0)
    t_f=torch.split(target_features,split_sizes,dim=0)
    t_d=torch.split(target_delta,split_sizes,dim=0)
    t_fit=torch.split(fit,split_sizes,dim=0)
    entries=[]
    ot_summaries=[]
    for j in range(3):
        source_city=samples[j+1].data_name
        f_s,out_s=transfers[j+1]
        delta_s=torch.autograd.grad(s_losses[j],out_s,retain_graph=True)[0].detach()
        sx=representations[j+1].reshape(-1,representations[j+1].shape[-1])
        sy=futures[j+1].reshape(-1,36)
        sf=f_s.reshape(-1,f_s.shape[-1])
        sd=delta_s.reshape(-1,36)
        tx=t_rep[j][t_fit[j]].reshape(-1,sx.shape[-1])
        ty=t_y[j][t_fit[j]].reshape(-1,36)
        tf=t_f[j][t_fit[j]].reshape(-1,sf.shape[-1])
        td=t_d[j][t_fit[j]].reshape(-1,36)
        sids=sampled_flat_indices(len(sx),source_points,sx.device)
        tids=sampled_flat_indices(len(tx),target_points,tx.device)
        alpha,beta,stats=compute_weights(
            sf.index_select(0,sids).unsqueeze(0).detach(),
            sd.index_select(0,sids).unsqueeze(0),
            tf.index_select(0,tids).unsqueeze(0).detach(),
            td.index_select(0,tids).unsqueeze(0),
            variant="gradient",temperature=8.0,
        )
        source_geometry=torch.cat((
            sx.index_select(0,sids),
            sy.detach().index_select(0,sids)*10.,
        ),dim=-1)
        target_geometry=torch.cat((
            tx.index_select(0,tids),
            ty.detach().index_select(0,tids)*10.,
        ),dim=-1)
        loss_ot=sinkhorn(alpha,source_geometry,beta,target_geometry).sum()*ot_scale
        g_ot,city_mask=_grad_with_presence(loss_ot,params)
        ot_mask=[x or y for x,y in zip(ot_mask,city_mask)]
        entries.append({"city":str(source_city),"f2":g_ot})
        with torch.no_grad():
            a=alpha.detach()
            b=beta.detach()
            top,loc=torch.topk(a,min(24,a.numel()))
            ot_summaries.append({
                "source_city":str(source_city),
                "target_group":j,
                "source_ot_points":int(len(sids)),
                "target_fit_ot_points":int(len(tids)),
                "source_point_ids_sha256":hashlib.sha256(
                    sids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "target_point_ids_sha256":hashlib.sha256(
                    tids.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                "source_alpha_sum":float(a.sum()),
                "source_alpha_ess":float(1/a.square().sum()),
                "source_alpha_min":float(a.min()),
                "source_alpha_max":float(a.max()),
                "source_alpha_top24_values":top.cpu().tolist(),
                "source_alpha_top24_flat_indices":
                    sids.index_select(0,loc).cpu().tolist(),
                "target_beta_sum":float(b.sum()),
                "target_beta_ess":float(1/b.square().sum()),
                "target_beta_uniform":bool(torch.allclose(
                    b,torch.full_like(b,1.0/len(b)),atol=1e-6
                )),
                "ot_loss":float(loss_ot.detach()),
                "raw_gradient_norm":gradient_norm(g_ot),
            })
        del loss_ot,source_geometry,target_geometry
    if len({e["city"] for e in entries})!=3:
        raise RuntimeError("Duplicate source-city OT gradient")
    return {
        "source_gradient":source_grad,
        "target_fit_gradient":target_grad,
        "source_present_mask":source_mask,
        "target_present_mask":target_mask,
        "ot_present_mask":ot_mask,
        "city_entries":entries,
        "ot_metadata":ot_summaries,
        "target_fit_count":int(fit.sum()),
        "unused_support_probe_count":0,
        "source_task_loss_value":float(sum(s_losses).detach()),
        "target_task_loss_value":float(t_loss.detach()),
    }

