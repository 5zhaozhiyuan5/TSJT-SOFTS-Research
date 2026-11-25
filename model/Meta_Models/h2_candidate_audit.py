"""H2-A1/A2/A3: full-batch OT candidate audit built directly on H1.

All three experiments are DIAGNOSTIC ONLY; actual Stage-I F2 weights and
optimizer updates are never changed.

A1: global full-batch alternatives: no OT, uniform OT, 50/50 OT, F2 OT.
A2: hold 2 cities' F2 OT fixed, replace 1 city's OT with uniform.
A3: score the SAME candidate shadow models on three contiguous target
support window blocks to study decision stability.

Every OT candidate uses the UNCHANGED original full-batch Sinkhorn operator
and the same OT scale. Shadow updates replay all-parameter one-step PCGrad
and stateful AdamW from H1. Target windows are already part of original
three-day training support; A3 folds are NOT unseen validation data.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:  # Imported as model.Meta_Models.h2_candidate_audit in CPU tests.
    from .h1_shadow_audit import (
        _grads, projected_directions, simulate_adamw_step, _smooth_l1_bands
    )
except ImportError:  # Imported as a top-level module by official train_DA_OT.py.
    from h1_shadow_audit import (
        _grads, projected_directions, simulate_adamw_step, _smooth_l1_bands
    )


GLOBAL_CHOICES = ("off", "uniform", "mixed", "f2")


def detached_model_gradients(loss, model):
    """Retrieve all parameter gradients without writing .grad or stepping."""
    return _grads(loss, list(model.parameters()))


def sum_gradient_vectors(vectors):
    """Sum city gradients without modifying their individual tensors."""
    if not vectors:
        raise ValueError("Expected at least one OT gradient")
    size = len(vectors[0])
    if any(len(v) != size for v in vectors):
        raise ValueError("Inconsistent gradient lengths")
    with torch.no_grad():
        result = [v.detach().clone() for v in vectors[0]]
        for vec in vectors[1:]:
            for i, grad in enumerate(vec):
                result[i].add_(grad)
        return result


def global_and_city_candidate_gradients(city_gradients, include_city=True):
    """Build A1 and A2 OT gradient tuples from matched full-batch Sinkhorn grads.

    The input contains gradient lists keyed by stable source city names.
    No candidate's source or target OT geometry is approximated by subsetting.
    """
    if not city_gradients:
        raise ValueError("No source cities supplied")
    keys = [entry["city"] for entry in city_gradients]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate city keys would confound city ablation")
    result = {
        "uniform": sum_gradient_vectors(
            [entry["uniform"] for entry in city_gradients]
        ),
        "mixed": sum_gradient_vectors(
            [entry["mixed"] for entry in city_gradients]
        ),
        "f2": sum_gradient_vectors(
            [entry["f2"] for entry in city_gradients]
        ),
    }
    if include_city:
        for idx, entry in enumerate(city_gradients):
            result["city_uniform:" + entry["city"]] = sum_gradient_vectors([
                city["uniform"] if j == idx else city["f2"]
                for j, city in enumerate(city_gradients)
            ])
    return result


def support_fold_slices(n_windows):
    """Three contiguous support blocks, 15/15/16 for the 46-window protocol."""
    if n_windows < 6:
        raise ValueError("At least six target windows required for 3 folds")
    first = n_windows // 3
    second = n_windows // 3
    return (
        slice(0, first),
        slice(first, first + second),
        slice(first + second, n_windows),
    )


def score_shadow_predictions(predictions, truth, beta=5e-4,
                             confidence_tolerance=1e-6):
    """A1 scores + A2 per-city effects + A3 fold stability, all from same preds.

    Gain is positive when candidate improves over the NO-OT shadow update.
    City-effect is positive when replacing THAT city's F2 with uniform
    improves over the full-F2 shadow update.
    """
    if not all(k in predictions for k in GLOBAL_CHOICES):
        raise ValueError("All A1 candidate predictions are mandatory")
    if any(p.shape != truth.shape for p in predictions.values()):
        raise ValueError("Shadow predictions must have target dimensions")
    if truth.shape[-1] != 36:
        raise ValueError("This audit expects 36 prediction steps")
    if not math.isfinite(confidence_tolerance) or confidence_tolerance < 0:
        raise ValueError("Invalid positive-gain tolerance")

    def measure(window):
        actual = truth[window]
        values = {}
        for name, pred in predictions.items():
            loss, bands = _smooth_l1_bands(pred[window], actual, beta)
            values[name] = {"loss": loss, "band_losses": bands}
        base = values["off"]
        for row in values.values():
            row["gain_vs_off"] = base["loss"] - row["loss"]
            row["band_gains_vs_off"] = [
                x - y for x, y in zip(base["band_losses"], row["band_losses"])
            ]
        f2 = values["f2"]
        for name, row in values.items():
            if name.startswith("city_uniform:"):
                row["gain_vs_f2"] = f2["loss"] - row["loss"]
                row["band_gains_vs_f2"] = [
                    x - y for x, y in zip(f2["band_losses"], row["band_losses"])
                ]
        best = "off"
        best_gain = confidence_tolerance
        for name in GLOBAL_CHOICES[1:]:
            if values[name]["gain_vs_off"] > best_gain:
                best_gain = values[name]["gain_vs_off"]
                best = name
        return {"scores": values, "best_global": best}

    total = measure(slice(None))
    folds = [measure(part) for part in support_fold_slices(truth.shape[0])]
    bests = [fold["best_global"] for fold in folds]
    all_positive = {
        name: all(f["scores"][name]["gain_vs_off"] > confidence_tolerance
                  for f in folds)
        for name in GLOBAL_CHOICES[1:]
    }
    fold_consensus = len(set(bests)) == 1
    winner = total["best_global"]
    return {
        "total": total,
        "folds": folds,
        "fold_best_global": bests,
        "fold_consensus": fold_consensus,
        "overall_winner_is_fold_consensus": fold_consensus and winner == bests[0],
        "global_positive_in_all_folds": all_positive,
    }


def evaluate_h2_candidate_audit(*, model, optimizer, source_task_loss,
                                target_task_loss, city_gradients, target_data,
                                epoch, beta=5e-4, citywise=True,
                                confidence_tolerance=1e-6):
    """Return all full-batch candidate gains and fold results, not model changes.

    All candidates start from the same exact CURRENT real model/AdamW state.
    The SOFTS training randomness is reset identically for each shadow
    forward, and fork_rng restores outer CPU/GPU random states.
    """
    if not city_gradients:
        raise ValueError("H2 audit requires source-city OT gradients")
    named = list(model.named_parameters())
    params = [p for _, p in named]
    source_grads = _grads(source_task_loss, params)
    target_grads = _grads(target_task_loss, params)
    candidates = global_and_city_candidate_gradients(
        city_gradients, include_city=citywise
    )
    zeros = [torch.zeros_like(p) for p in params]
    baseline, _ = projected_directions(
        source_grads, target_grads, zeros
    )
    dev = params[0].device
    cuda_devices = (
        [dev.index if dev.index is not None else torch.cuda.current_device()]
        if dev.type == "cuda" else []
    )
    original_training = {
        name: module.training for name, module in model.named_modules()
    }
    dynamic_attrs = ("RevIn_mean", "RevIn_std", "eps")
    saved_dynamic = {
        attr: (hasattr(model, attr), getattr(model, attr, None))
        for attr in dynamic_attrs
    }
    saved_buffers = {
        name: buf.detach().clone() for name, buf in model.named_buffers()
    }
    truth = torch.as_tensor(
        target_data.y, device=dev, dtype=params[0].dtype
    ).detach()
    from torch.func import functional_call

    predictions = {}
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            for name in ("off", *candidates.keys()):
                # Materialize ONE shadow parameter set at a time. Holding
                # all seven full model copies would waste GPU memory.
                if name == "off":
                    direction = baseline
                else:
                    _, direction = projected_directions(
                        source_grads, target_grads, candidates[name]
                    )
                candidate_parameters = simulate_adamw_step(
                    named, optimizer, direction
                )
                # Match all candidates' STAR random-pooling draws.
                torch.manual_seed(820123 + int(epoch))
                buffers = {
                    bname: value.clone()
                    for bname, value in saved_buffers.items()
                }
                pred, _, _, _, _ = functional_call(
                    model, (candidate_parameters, buffers),
                    (target_data,), {"stage": "train"}
                )
                predictions[name] = pred.detach()
                del candidate_parameters, buffers
            measured = score_shadow_predictions(
                predictions, truth, beta=beta,
                confidence_tolerance=confidence_tolerance
            )
    finally:
        module_dict = dict(model.named_modules())
        for name, was_training in original_training.items():
            module_dict[name].training = was_training
        for attr, (was_present, value) in saved_dynamic.items():
            if was_present:
                setattr(model, attr, value)
            elif hasattr(model, attr):
                delattr(model, attr)

    measured["epoch"] = int(epoch)
    measured["cities"] = [entry["city"] for entry in city_gradients]
    measured["n_windows"] = int(truth.shape[0])
    measured["n_shadow_candidates"] = len(predictions)
    measured["h2_mode"] = "all" if citywise else "global"
    return measured
