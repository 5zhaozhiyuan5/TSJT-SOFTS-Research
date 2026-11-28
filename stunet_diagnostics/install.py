import importlib
import math
import sys
from functools import wraps
from pathlib import Path

import torch
import torch.nn.functional as torch_f

from .recorder import RECORDER


_INSTALLED = False


def _install_backbone_import_compatibility():
    module = importlib.import_module("src.models.STUNet.core.Backbone")
    if not hasattr(module, "GFBackBone") and hasattr(module, "BackBone"):
        module.GFBackBone = module.BackBone
        sys.modules["src.models.STUNet.core.GFBackbone"] = module
        RECORDER.write(
            "upstream_compatibility_shim",
            issue="STUNet.py imports GFBackbone.GFBackBone; upstream tree contains Backbone.py:BackBone",
            action="registered an in-memory alias; upstream submodule files remain unchanged",
        )


def _patch_attention(layers_module):
    cls = layers_module.QAAttentionLayer
    if getattr(cls.forward, "_stunet_diag_wrapped", False):
        return
    original = cls.forward

    @wraps(original)
    def wrapped(self, node_tokens, graph_tokens, N, N_T, N_S):
        capture = RECORDER.due("attention")
        maps = []
        original_softmax = layers_module.F.softmax
        if capture:
            def traced_softmax(input_tensor, *args, **kwargs):
                result = original_softmax(input_tensor, *args, **kwargs)
                if torch.is_tensor(result) and result.ndim == 4:
                    maps.append(result[:min(8, result.shape[0])].detach().clone())
                return result
            layers_module.F.softmax = traced_softmax
        before = node_tokens.detach()
        try:
            output = original(self, node_tokens, graph_tokens, N, N_T, N_S)
        finally:
            if capture:
                layers_module.F.softmax = original_softmax
        if capture:
            layer_idx = int(getattr(self, "_diag_layer_idx", -1))
            for index, attention in enumerate(maps):
                RECORDER.log_attention(layer_idx, "query" if index == 0 else "aggregate", attention)
            before_sample = before[:min(8, before.shape[0])]
            after_sample = output[0][:min(8, output[0].shape[0])].detach()
            graph_sample = graph_tokens[:min(8, graph_tokens.shape[0])]
            delta = after_sample - before_sample
            RECORDER.csv("attention_layer_stats.csv", {
                **RECORDER.context, "layer": layer_idx,
                "input_rms": float(before_sample.float().square().mean().sqrt().cpu()),
                "output_rms": float(after_sample.float().square().mean().sqrt().cpu()),
                "delta_rms": float(delta.float().square().mean().sqrt().cpu()),
                "delta_to_input": float(delta.float().norm().cpu() / before_sample.float().norm().clamp_min(1e-12).cpu()),
                "node_token_count": int(before.shape[1]),
                "graph_token_count": int(graph_sample.shape[1]),
                "diagnostic_batch_sample": int(before_sample.shape[0]),
            })
        return output

    wrapped._stunet_diag_wrapped = True
    cls.forward = wrapped


def _patch_backbone_residual(backbone_module):
    cls = backbone_module.BackBone
    if getattr(cls.backbone_forward, "_stunet_diag_wrapped", False):
        return
    original = cls.backbone_forward

    @wraps(original)
    def wrapped(self, node_tokens, graph_tokens, N, N_T, N_S):
        for index, layer in enumerate(self.backbone):
            layer._diag_layer_idx = index
            before = node_tokens
            layer_output, graph_tokens = layer(node_tokens, graph_tokens, N, N_T, N_S)
            after_outer_residual = layer_output + before
            if RECORDER.due("attention"):
                before_s = before[:min(8, before.shape[0])].detach().float()
                layer_s = layer_output[:min(8, layer_output.shape[0])].detach().float()
                after_s = after_outer_residual[:min(8, after_outer_residual.shape[0])].detach().float()
                outer_delta, layer_delta = after_s - layer_s, layer_s - before_s
                RECORDER.csv("backbone_residual_stats.csv", {
                    **RECORDER.context, "layer": index,
                    "input_rms": float(before_s.square().mean().sqrt().cpu()),
                    "attention_block_output_rms": float(layer_s.square().mean().sqrt().cpu()),
                    "post_outer_residual_rms": float(after_s.square().mean().sqrt().cpu()),
                    "attention_block_delta_rms": float(layer_delta.square().mean().sqrt().cpu()),
                    "outer_residual_delta_rms": float(outer_delta.square().mean().sqrt().cpu()),
                    "attention_delta_to_input": float(layer_delta.norm().cpu() / before_s.norm().clamp_min(1e-12).cpu()),
                    "outer_residual_delta_to_input": float(outer_delta.norm().cpu() / before_s.norm().clamp_min(1e-12).cpu()),
                    "post_to_input_norm_ratio": float(after_s.norm().cpu() / before_s.norm().clamp_min(1e-12).cpu()),
                })
            node_tokens = after_outer_residual
        return node_tokens, graph_tokens

    wrapped._stunet_diag_wrapped = True
    cls.backbone_forward = wrapped


def _patch_graph_generator(graph_gen_module):
    cls = graph_gen_module.GraphGeneratorAttn
    if getattr(cls.forward, "_stunet_diag_wrapped", False):
        return
    original = cls.forward

    @wraps(original)
    def wrapped(self, x):
        output = original(self, x)
        if RECORDER.due("attention") and torch.is_tensor(output.adj_matrix):
            a = output.adj_matrix[:min(8, output.adj_matrix.shape[0])].detach().float()
            p = a.clamp_min(1e-12)
            row_sum = a.sum(dim=-1)
            entropy = -(p * p.log()).sum(dim=-1)
            RECORDER.csv("generated_graph_stats.csv", {
                **RECORDER.context, "shape": str(tuple(output.adj_matrix.shape)),
                "mean": float(a.mean().cpu()), "std": float(a.std(unbiased=False).cpu()),
                "min": float(a.min().cpu()), "max": float(a.max().cpu()),
                "symmetry_mae": float((a - a.transpose(-1, -2)).abs().mean().cpu()),
                "diagonal_mean": float(a.diagonal(dim1=-2, dim2=-1).mean().cpu()),
                "row_sum_mean": float(row_sum.mean().cpu()),
                "row_sum_std": float(row_sum.std(unbiased=False).cpu()),
                "row_entropy_mean": float(entropy.mean().cpu()),
                "row_entropy_normalized": float((entropy / max(math.log(max(2, a.shape[-1])), 1e-12)).mean().cpu()),
                "row_top1_mean": float(p.max(dim=-1).values.mean().cpu()),
            })
        return output

    wrapped._stunet_diag_wrapped = True
    cls.forward = wrapped


def _patch_tokenizer(tokenizer_module):
    cls = tokenizer_module.Tokenizer
    if getattr(cls.forward, "_stunet_diag_wrapped", False):
        return
    original = cls.forward

    @wraps(original)
    def wrapped(self, batch):
        output = original(self, batch)
        if RECORDER.due("token"):
            graph_count = int(output.graph_patch_num or 0)
            if graph_count > 0:
                RECORDER.log_tokens(output.tokens[:, :-graph_count, :],
                                    output.tokens[:, -graph_count:, :], output.adj_matrix)
            if torch.is_tensor(output.adj_matrix):
                RECORDER.write("tokenizer_graph_mode", use_adj=bool(self.use_adj),
                               graph_encoder_patch_dim=int(self.patch_dim),
                               node_patch_num=int(output.node_patch_num),
                               graph_patch_num=graph_count)
        return output

    wrapped._stunet_diag_wrapped = True
    cls.forward = wrapped


def _patch_stunet(stunet_module):
    cls = stunet_module.STUNet
    if getattr(cls.forward, "_stunet_diag_wrapped", False):
        return
    original = cls.forward

    @wraps(original)
    def wrapped(self, batch):
        layers = getattr(getattr(self, "backbone", None), "backbone", [])
        for index, layer in enumerate(layers):
            layer._diag_layer_idx = index
        output = original(self, batch)
        if output is not None and torch.is_tensor(output.y_pred) and torch.is_tensor(output.y_true):
            target_mask = batch.get("target_node_mask") if isinstance(batch, dict) else None
            RECORDER.write(
                "prediction_masking_mode",
                model_training=bool(self.training),
                target_mask_present=torch.is_tensor(target_mask),
                target_node_count=int(target_mask.sum().cpu()) if torch.is_tensor(target_mask) else None,
                subgraph_node_count=int(batch["x"].shape[-1]) if isinstance(batch, dict) and torch.is_tensor(batch.get("x")) else None,
                prediction_node_count=int(output.y_pred.shape[-1]),
                loss_scope="all_sampled_subgraph_nodes" if self.training else "target_nodes_only_when_masked",
            )
            if RECORDER.due("prediction"):
                RECORDER.log_predictions(output.y_pred, output.y_true, batch=batch)
        return output

    wrapped._stunet_diag_wrapped = True
    cls.forward = wrapped


def _patch_encoder_loaders(graph_encoder_module, temporal_module):
    for cls in (graph_encoder_module.GraphEncoderModel, temporal_module.TrafficNodeEncoder):
        original = cls.load_encoder
        if getattr(original, "_stunet_diag_wrapped", False):
            continue

        @wraps(original)
        def wrapped(self, *args, __original=original, **kwargs):
            path = getattr(self, "save_path", None)
            exists = bool(path and Path(str(path)).expanduser().exists())
            RECORDER.write("pretrained_encoder_load", encoder=type(self).__name__,
                           checkpoint=path, exists=exists)
            try:
                return __original(self, *args, **kwargs)
            except Exception as exc:
                RECORDER.write("pretrained_encoder_load_error", encoder=type(self).__name__,
                               checkpoint=path, error=repr(exc))
                raise

        wrapped._stunet_diag_wrapped = True
        cls.load_encoder = wrapped


def _patch_pretraining_models(spatial_module, temporal_ae_module):
    spatial_cls = spatial_module.SpatialAE
    if not getattr(spatial_cls.forward, "_stunet_diag_wrapped", False):
        original = spatial_cls.forward

        @wraps(original)
        def spatial_forward(self, batch):
            captured = {}
            handle = self.decoder.register_forward_hook(
                lambda _module, _inputs, output: captured.update(decoder=output.detach())
            )
            try:
                result = original(self, batch)
            finally:
                handle.remove()
            if RECORDER.due("batch") and "decoder" in captured:
                adj = batch["adj_matrix"][:8].detach().float()
                patches = captured["decoder"][:8].float()
                patch_dim = int(self.graph_patch_len)
                batch_size, node_count, _ = adj.shape
                side = int(math.ceil(node_count / patch_dim))
                padded = side * patch_dim
                recon = patches.reshape(batch_size, side, side, patch_dim, patch_dim)
                recon = recon.permute(0, 1, 3, 2, 4).reshape(batch_size, padded, padded)
                target = torch_f.pad(adj, (0, padded - node_count, 0, padded - node_count))
                error = recon - target
                positive, predicted_positive = target > 0, recon > 0.5
                tp = (positive & predicted_positive).sum().float()
                fp = ((~positive) & predicted_positive).sum().float()
                fn = (positive & (~predicted_positive)).sum().float()
                precision, recall = tp / (tp + fp).clamp_min(1), tp / (tp + fn).clamp_min(1)
                RECORDER.csv("spatial_pretrain_stats.csv", {
                    **RECORDER.context, "loss": float(result.loss.detach().cpu()),
                    "reconstruction_mse": float(error.square().mean().cpu()),
                    "reconstruction_mae": float(error.abs().mean().cpu()),
                    "target_edge_density": float(positive.float().mean().cpu()),
                    "predicted_positive_density_at_0_5": float(predicted_positive.float().mean().cpu()),
                    "edge_precision_at_0_5": float(precision.cpu()),
                    "edge_recall_at_0_5": float(recall.cpu()),
                    "decoder_output_rms": float(patches.square().mean().sqrt().cpu()),
                })
            return result

        spatial_forward._stunet_diag_wrapped = True
        spatial_cls.forward = spatial_forward

    temporal_cls = temporal_ae_module.TemporalAE
    if not getattr(temporal_cls.forward, "_stunet_diag_wrapped", False):
        original = temporal_cls.forward

        @wraps(original)
        def temporal_forward(self, batch):
            captured = {}
            handle = self.decoder.register_forward_hook(
                lambda _module, _inputs, output: captured.update(decoder=output.detach())
            )
            try:
                result = original(self, batch)
            finally:
                handle.remove()
            if RECORDER.due("prediction") and "decoder" in captured:
                pred = captured["decoder"].permute(0, 2, 1)
                pred = pred * batch["std"] + batch["mean"]
                RECORDER.log_predictions(pred, batch["y"], batch=batch, stage="temporal_pretrain")
                RECORDER.csv("temporal_pretrain_stats.csv", {
                    **RECORDER.context, "loss": float(result.loss.detach().cpu()),
                    "decoder_output_rms": float(captured["decoder"].float().square().mean().sqrt().cpu()),
                })
            return result

        temporal_forward._stunet_diag_wrapped = True
        temporal_cls.forward = temporal_forward


def _patch_datamodule(datamodule_module):
    cls = datamodule_module.DataModule
    if getattr(cls.__init__, "_stunet_diag_wrapped", False):
        return
    original = cls.__init__

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            dataset, sampler = self.train_dataset, self.train_dataset.sampler
            RECORDER.write(
                "data_module_summary",
                dataset_name=self.dataset_name, seq_len=self.hparams.seq_len,
                pred_len=self.hparams.pred_len, step=self.step, batch_size=self.batch_size,
                sample_batch_size=self.sample_batch_size, num_workers=self.num_workers,
                train_samples=len(self.train_dataset), validation_samples=len(self.val_dataset),
                test_samples=len(self.test_dataset), data_shape=tuple(dataset.x.shape),
                node_count=int(dataset.x.shape[1]), edge_count=int(sampler.edge_index.shape[1]),
                global_mean=float(self.mean), global_std=float(self.std),
                sample_sizes=list(self.hparams.sample_sizes),
            )
        except Exception as exc:
            RECORDER.write("data_module_summary_error", error=repr(exc))

    wrapped._stunet_diag_wrapped = True
    cls.__init__ = wrapped


def _patch_trainer_class(cls, trainer_kind):
    hooks = (("on_train_batch_start", "train"),
             ("on_validation_batch_start", "validation"),
             ("on_test_batch_start", "test"))
    for hook_name, stage in hooks:
        original = getattr(cls, hook_name, None)
        if getattr(original, "_stunet_diag_wrapped", False):
            continue

        def make_start(orig, current_stage):
            def start(self, batch, batch_idx, *args, **kwargs):
                RECORDER.set_context(stage=f"{trainer_kind}_{current_stage}",
                                     epoch=getattr(self, "current_epoch", -1),
                                     step=getattr(self, "global_step", 0), batch_idx=batch_idx)
                if RECORDER.due("batch"):
                    RECORDER.log_batch(batch)
                return orig(self, batch, batch_idx, *args, **kwargs) if orig is not None else None
            if orig is not None:
                start = wraps(orig)(start)
            start._stunet_diag_wrapped = True
            return start

        setattr(cls, hook_name, make_start(original, stage))

    for hook_name in ("on_train_batch_end", "on_validation_batch_end", "on_test_batch_end"):
        original = getattr(cls, hook_name, None)
        if getattr(original, "_stunet_diag_wrapped", False):
            continue

        def make_end(orig):
            def end(self, outputs, batch, batch_idx, *args, **kwargs):
                result = orig(self, outputs, batch, batch_idx, *args, **kwargs) if orig is not None else None
                RECORDER.log_step_end(outputs, batch=batch)
                return result
            if orig is not None:
                end = wraps(orig)(end)
            end._stunet_diag_wrapped = True
            return end

        setattr(cls, hook_name, make_end(original))

    original_backward = getattr(cls, "on_after_backward", None)
    if not getattr(original_backward, "_stunet_diag_wrapped", False):
        def after_backward(self, *args, **kwargs):
            result = original_backward(self, *args, **kwargs) if original_backward is not None else None
            RECORDER.log_gradients(getattr(self, "model", self))
            return result
        after_backward._stunet_diag_wrapped = True
        cls.on_after_backward = after_backward

    for hook_name, stage in (("on_train_epoch_end", f"{trainer_kind}_train"),
                             ("on_validation_epoch_end", f"{trainer_kind}_validation"),
                             ("on_test_epoch_end", f"{trainer_kind}_test")):
        original = getattr(cls, hook_name, None)
        if getattr(original, "_stunet_diag_wrapped", False):
            continue

        def make_epoch_end(orig, current_stage):
            def epoch_end(self, *args, **kwargs):
                result = orig(self, *args, **kwargs) if orig is not None else None
                RECORDER.log_epoch(self, current_stage)
                return result
            if orig is not None:
                epoch_end = wraps(orig)(epoch_end)
            epoch_end._stunet_diag_wrapped = True
            return epoch_end

        setattr(cls, hook_name, make_epoch_end(original, stage))

    original_fit_start = getattr(cls, "on_fit_start", None)
    if not getattr(original_fit_start, "_stunet_diag_wrapped", False):
        def fit_start(self, *args, **kwargs):
            result = original_fit_start(self, *args, **kwargs) if original_fit_start is not None else None
            RECORDER.log_model(getattr(self, "model", self))
            return result
        fit_start._stunet_diag_wrapped = True
        cls.on_fit_start = fit_start


def install():
    global _INSTALLED
    if _INSTALLED:
        return RECORDER.run_dir
    _install_backbone_import_compatibility()
    layers = importlib.import_module("src.models.STUNet.core.layers")
    backbone = importlib.import_module("src.models.STUNet.core.Backbone")
    tokenizer = importlib.import_module("src.models.STUNet.core.Tokenizer")
    graph_gen = importlib.import_module("src.models.STUNet.core.graph_gen")
    stunet = importlib.import_module("src.models.STUNet.core.STUNet")
    graph_encoder = importlib.import_module("src.models.STUNet.core.graph_encoder")
    temporal_model = importlib.import_module("src.models.STUNet.core.temporal_model")
    spatial_ae = importlib.import_module("src.models.STUNet.core.SpatialAE")
    temporal_ae = importlib.import_module("src.models.STUNet.core.TemporalAE")
    backbone_trainer = importlib.import_module("src.models.STUNet.backbonetrainer")
    pretrainer = importlib.import_module("src.models.STUNet.pretrainer")
    datamodule = importlib.import_module("src.data.largest.lightning_datamodule")

    _patch_attention(layers)
    _patch_backbone_residual(backbone)
    _patch_graph_generator(graph_gen)
    _patch_tokenizer(tokenizer)
    _patch_stunet(stunet)
    _patch_encoder_loaders(graph_encoder, temporal_model)
    _patch_pretraining_models(spatial_ae, temporal_ae)
    _patch_datamodule(datamodule)
    _patch_trainer_class(backbone_trainer.Trainer, "backbone")
    _patch_trainer_class(pretrainer.Trainer, "pretrain")

    RECORDER.write("diagnostics_installed", run_dir=str(RECORDER.run_dir),
                   batch_every=RECORDER.batch_every, attention_every=RECORDER.attention_every,
                   gradient_every=RECORDER.gradient_every, token_every=RECORDER.token_every,
                   prediction_every=RECORDER.prediction_every,
                   patched_modules=["QAAttentionLayer.forward", "Tokenizer.forward",
                                    "STUNet.forward", "SpatialAE.forward", "TemporalAE.forward",
                                    "BackBone.backbone_forward residual path", "GraphGeneratorAttn.forward",
                                    "Lightning training hooks", "LargeST DataModule.__init__"])
    _INSTALLED = True
    return RECORDER.run_dir
