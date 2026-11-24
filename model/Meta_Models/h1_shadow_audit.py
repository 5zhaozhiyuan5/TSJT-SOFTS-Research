"""H1 only: paired, full-parameter one-step counterfactual audit of F2 OT.

This is a diagnostic. Actual F2 optimization is NEVER replaced/updated here.

For identical current model+optimizer state, compare:
 A: AdamW(PCGrad(g_source,          g_target))
 B: AdamW(PCGrad(g_source + g_OT_F2, g_target))

The shadow AdamW step covers ALL model parameters and their current moment
states. Both shadow models predict on exactly the SAME support probe with
the SAME sampled SOFTS train-time random pooling; functional_call prevents
mutation of persistent parameters/optimizer state.

A second readout-only counterfactual reconstructs the exact change from
the shadow projection1.weight updates on cached, fixed encoder activations.
This diagnoses the previous G2's head-only approximation error.

Important caveats: this is a ONE-STEP, IN-SUPPORT target diagnostic, not a
held-out or multi-step generalization estimate. G2/G3's subsampled OT
candidate objectives are NOT validated by this audit.
"""

import math

import torch
import torch.nn.functional as F


def projected_directions(task_grads, target_grads, ot_grads):
    """Match original PMME's one-sided PCGrad over the full flattened vector."""
    if not (len(task_grads) == len(target_grads) == len(ot_grads)):
        raise ValueError("Gradient tuples must have equal lengths")
    with torch.no_grad():
        t = [g.detach() for g in target_grads]
        a = [g.detach() for g in task_grads]
        ot = [g.detach() for g in ot_grads]
        target_sq = sum((x.float().square().sum() for x in t))
        baseline = a
        candidate = [x + y for x, y in zip(a, ot)]

        def project(items):
            dot = sum(((x.float() * y.float()).sum() for x, y in zip(items, t)))
            if bool(dot < 0):
                coef = dot / (target_sq + 1e-8)
                return [x - coef.to(dtype=x.dtype) * y
                        for x, y in zip(items, t)]
            return items

        return project(baseline), project(candidate)


def simulate_adamw_step(named_parameters, optimizer, directions):
    """Simulate default PyTorch AdamW without changing model or optimizer.

    directions are already PCGrad-adjusted and include one tensor per model
    parameter (zero tensors for unused parameters, as in official PMME).
    Supports the repository's ordinary AdamW (no fused/differentiable mode).
    """
    entries = list(named_parameters)
    if len(entries) != len(directions):
        raise ValueError("Direction and parameter length mismatch")
    by_parameter = {id(param): (name, grad) for (name, param), grad
                    in zip(entries, directions)}
    updated = {}
    with torch.no_grad():
        for group in optimizer.param_groups:
            if group.get("amsgrad", False) or group.get("differentiable", False):
                raise NotImplementedError("This H1 audit requires plain AdamW")
            lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))
            eps = float(group.get("eps", 1e-8))
            beta1, beta2 = group["betas"]
            if not all(math.isfinite(z) for z in (lr, wd, eps, beta1, beta2)):
                raise ValueError("Non-finite AdamW configuration")
            for p in group["params"]:
                if id(p) not in by_parameter:
                    raise ValueError("Optimizer parameter absent from model")
                name, incoming = by_parameter[id(p)]
                grad = incoming.detach()
                if group.get("maximize", False):
                    grad = -grad
                state = optimizer.state.get(p, {})
                step_value = state.get("step", 0)
                step_number = (
                    float(step_value.detach().item())
                    if torch.is_tensor(step_value) else float(step_value)
                ) + 1.0
                m = state.get("exp_avg")
                v = state.get("exp_avg_sq")
                if m is None:
                    m = torch.zeros_like(p)
                if v is None:
                    v = torch.zeros_like(p)
                new_m = beta1 * m + (1 - beta1) * grad
                new_v = beta2 * v + (1 - beta2) * grad.square()
                m_correction = 1.0 - beta1 ** step_number
                v_correction = 1.0 - beta2 ** step_number
                denom = new_v.sqrt() / math.sqrt(v_correction) + eps
                new_p = (p.detach() * (1 - lr * wd)
                         - (lr / m_correction) * (new_m / denom))
                updated[name] = new_p.detach()
    if len(updated) != len(entries):
        raise ValueError("Some parameters are absent from optimizer")
    return updated


def _grads(loss, parameters):
    values = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True,
        create_graph=False
    )
    return [torch.zeros_like(p) if g is None else g.detach()
            for p, g in zip(parameters, values)]


def _smooth_l1_bands(pred, truth, beta):
    with torch.no_grad():
        total = F.smooth_l1_loss(pred, truth, beta=beta).item()
        bands = [
            F.smooth_l1_loss(pred[..., a:b], truth[..., a:b],
                             beta=beta).item()
            for a, b in ((0, 12), (12, 24), (24, 36))
        ]
    return float(total), [float(v) for v in bands]


def _readout_delta(probe_input, delta_weight, delta_bias,
                   projection2_weight, physical_scale):
    hidden = F.linear(probe_input.float(), delta_weight.float(),
                      delta_bias.float())
    delta = F.linear(hidden, projection2_weight.float())
    return delta * physical_scale.float()


def audit_h1_shadow_update(*, model, optimizer, source_task_loss,
                            da_loss, target_task_loss, target_data,
                            probe_input, probe_prediction, probe_target,
                            probe_scale, probe_indices, epoch, beta=5e-4):
    """Evaluate paired shadow AdamW steps; return simple numeric diagnostics.

    Does not call optimizer.step/backward, set .grad, or edit model weights.
    It uses source/target/OT gradients computed on the actual training graph.
    The target probe's past labels ARE part of the three-day support training.
    """
    if not isinstance(da_loss, torch.Tensor) or not da_loss.requires_grad:
        raise ValueError("H1 requires the real active F2 OT loss graph")
    named = list(model.named_parameters())
    parameters = [p for _, p in named]
    source_grads = _grads(source_task_loss, parameters)
    ot_grads = _grads(da_loss, parameters)
    target_grads = _grads(target_task_loss, parameters)
    grad_no_ot, grad_with_ot = projected_directions(
        source_grads, target_grads, ot_grads
    )
    no_ot = simulate_adamw_step(named, optimizer, grad_no_ot)
    f2_ot = simulate_adamw_step(named, optimizer, grad_with_ot)

    device = parameters[0].device
    cuda_devices = ([device.index if device.index is not None
                     else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    original_training_flags = {name: m.training
                               for name, m in model.named_modules()}
    # PatchFSL dynamically updates these values during train forward.
    dynamic_attrs = ("RevIn_mean", "RevIn_std", "eps")
    saved_dynamic = {a: (hasattr(model, a), getattr(model, a, None))
                     for a in dynamic_attrs}
    from torch.func import functional_call

    # Cloned buffers ensure shadow forwards do not mutate training buffers.
    buffer_template = {name: b.detach().clone()
                       for name, b in model.named_buffers()}
    true_y = torch.as_tensor(
        target_data.y, dtype=parameters[0].dtype, device=device
    ).index_select(0, probe_indices)
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            def predict(shadow_parameters):
                # Identical stochastic pooling draws in each shadow forward.
                torch.manual_seed(170019 + int(epoch))
                state_buffers = {name: b.clone()
                                 for name, b in buffer_template.items()}
                prediction, _, _, _, _ = functional_call(
                    model, (shadow_parameters, state_buffers),
                    (target_data,), {"stage": "train"}
                )
                return prediction.index_select(0, probe_indices).detach()

            pred_no_ot = predict(no_ot)
            pred_f2_ot = predict(f2_ot)
            full_base, full_base_bands = _smooth_l1_bands(
                pred_no_ot, true_y, beta
            )
            full_f2, full_f2_bands = _smooth_l1_bands(
                pred_f2_ot, true_y, beta
            )

            # Local proxy: replay both exact AdamW projection1 updates on
            # original frozen encoder outputs (rather than comparing one
            # gradient delta against the UNUPDATED prediction as G2 did).
            name = "forecasting_model.projection1.weight"
            p1 = dict(named)[name].detach()
            bias_name = "forecasting_model.projection1.bias"
            p1_bias = dict(named)[bias_name].detach()
            p2 = dict(named)["forecasting_model.projection2.weight"].detach()
            pre = probe_input.detach()
            original_prediction = probe_prediction.detach().float()
            scale = probe_scale.detach()
            proxy_no_ot = original_prediction + _readout_delta(
                pre, no_ot[name] - p1,
                no_ot[bias_name] - p1_bias, p2, scale
            )
            proxy_f2_ot = original_prediction + _readout_delta(
                pre, f2_ot[name] - p1,
                f2_ot[bias_name] - p1_bias, p2, scale
            )
            local_base, local_base_bands = _smooth_l1_bands(
                proxy_no_ot, probe_target.float(), beta
            )
            local_f2, local_f2_bands = _smooth_l1_bands(
                proxy_f2_ot, probe_target.float(), beta
            )
            full_gain = full_base - full_f2
            local_gain = local_base - local_f2
            ot_sq = sum(g.float().square().sum() for g in ot_grads)
            target_sq = sum(g.float().square().sum() for g in target_grads)
            source_sq = sum(g.float().square().sum() for g in source_grads)
            ot_target_dot = sum(
                (g.float() * t.float()).sum()
                for g, t in zip(ot_grads, target_grads)
            )
            return {
                "epoch": int(epoch),
                "full_no_ot": full_base,
                "full_f2": full_f2,
                "full_gain": full_gain,
                "full_band_gains": [
                    a - b for a, b in zip(full_base_bands, full_f2_bands)
                ],
                "head_no_ot": local_base,
                "head_f2": local_f2,
                "head_gain": local_gain,
                "head_band_gains": [
                    a - b for a, b in zip(local_base_bands, local_f2_bands)
                ],
                "gain_sign_match": bool(
                    full_gain * local_gain > 0
                    or (full_gain == 0.0 and local_gain == 0.0)
                ),
                "ot_to_source_norm": float(
                    (ot_sq.sqrt() / (source_sq.sqrt() + 1e-12)).item()
                ),
                "ot_to_target_norm": float(
                    (ot_sq.sqrt() / (target_sq.sqrt() + 1e-12)).item()
                ),
                "ot_target_cos": float(
                    (ot_target_dot /
                     (ot_sq.sqrt() * target_sq.sqrt() + 1e-12)).item()
                ),
            }
    finally:
        # torch.func swaps parameters back automatically, but PatchFSL's
        # non-buffer forward caches and module.training flags need restoring.
        modules = dict(model.named_modules())
        for name, was_training in original_training_flags.items():
            modules[name].training = was_training
        for attr, (had_attr, value) in saved_dynamic.items():
            if had_attr:
                setattr(model, attr, value)
            elif hasattr(model, attr):
                delattr(model, attr)
