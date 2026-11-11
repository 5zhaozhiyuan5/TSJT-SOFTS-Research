"""Horizon-conditioned PMME Stage-II residual memory.

The module restores the empirically stronger V1 retrieval layout:
three horizon-conditioned queries share one memory bank, while three
independent generators produce the short/mid/long residual segments.

The only new initialization step is target-support prototype seeding.  It
uses frozen Stage-I representations from the authorized few-shot support
windows (not their labels and never the held-out test split).
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def configure_horizon_stage2_parameters(model: nn.Module):
    """Freeze the Stage-I forecaster and train only Stage-II memory parameters."""
    memory = getattr(model, "horizon_memory", None)
    if memory is None:
        raise RuntimeError("Stage-II requires HorizonConditionedResidualMemory.")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in memory.parameters():
        parameter.requires_grad_(True)

    trainable = [parameter for parameter in memory.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable horizon-memory parameters were found.")
    return trainable


class HorizonConditionedResidualMemory(nn.Module):
    """Three duration-conditioned retrievals over one shared learnable memory.

    For horizon band k:
        q_k = Wq(h + e_k)
        z_k = Attention(q_k, Wk(M), Wv(M))
        delta_k = f_k(z_k)

    The three 12-step corrections are concatenated in chronological order.
    """

    def __init__(
        self,
        d_model: int = 128,
        pred_len: int = 36,
        memory_size: int = 10000,
        num_bands: int = 3,
        query_chunk: int = 256,
    ):
        super().__init__()
        if pred_len % num_bands != 0:
            raise ValueError(
                f"pred_len={pred_len} must be divisible by num_bands={num_bands}."
            )
        if memory_size < 1:
            raise ValueError("memory_size must be positive.")

        self.d_model = int(d_model)
        self.pred_len = int(pred_len)
        self.memory_size = int(memory_size)
        self.num_bands = int(num_bands)
        self.band_width = self.pred_len // self.num_bands
        self.query_chunk = max(1, int(query_chunk))

        self.memory = nn.Parameter(torch.empty(self.memory_size, self.d_model))
        nn.init.normal_(self.memory, mean=0.0, std=0.02)

        self.horizon_embeddings = nn.Parameter(
            torch.empty(self.num_bands, self.d_model)
        )
        nn.init.normal_(self.horizon_embeddings, mean=0.0, std=0.02)

        # Shared query/key/value projections preserve a common memory space.
        self.Wq = nn.Linear(self.d_model, self.d_model, bias=False)
        self.Wk = nn.Linear(self.d_model, self.d_model, bias=False)
        self.Wv = nn.Linear(self.d_model, self.d_model, bias=False)

        # No horizon embedding is added at the generation side: each duration
        # has its own compact 12-step residual generator.
        self.bias_heads = nn.ModuleList(
            [nn.Linear(self.d_model, self.band_width) for _ in range(self.num_bands)]
        )

        self.register_buffer("memory_initialized", torch.tensor(False))
        self.register_buffer("initialization_feature_count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def initialize_from_features(self, features: torch.Tensor, seed: int = 1) -> int:
        """Seed memory slots from frozen target-support representations.

        If there are more support representations than slots, select a
        deterministic random subset. If there are fewer, repeat the support
        prototypes and add tiny noise to avoid exact duplicate slots.
        """
        features = torch.as_tensor(
            features, device=self.memory.device, dtype=self.memory.dtype
        )
        if features.ndim == 1:
            features = features.unsqueeze(0)
        features = features.reshape(-1, self.d_model)
        valid = torch.isfinite(features).all(dim=1)
        features = features[valid]
        if features.shape[0] == 0:
            raise ValueError("Cannot initialize horizon memory from empty/non-finite features.")

        generator = torch.Generator(device=self.memory.device)
        generator.manual_seed(int(seed))
        feature_count = int(features.shape[0])

        if feature_count >= self.memory_size:
            indices = torch.randperm(
                feature_count, device=self.memory.device, generator=generator
            )[: self.memory_size]
            prototypes = features[indices]
        else:
            repeats = (self.memory_size + feature_count - 1) // feature_count
            prototypes = features.repeat(repeats, 1)[: self.memory_size].clone()
            if repeats > 1:
                noise_scale = features.std(unbiased=False).clamp_min(1e-4) * 0.01
                noise = torch.randn(
                    prototypes.shape,
                    device=self.memory.device,
                    dtype=self.memory.dtype,
                    generator=generator,
                )
                prototypes = prototypes + noise * noise_scale

        self.memory.copy_(prototypes)
        self.memory_initialized.fill_(True)
        self.initialization_feature_count.fill_(feature_count)
        return feature_count

    def forward(self, representation: torch.Tensor):
        if representation.ndim != 3 or representation.shape[-1] != self.d_model:
            raise ValueError(
                "Expected representation [B,N,d_model], received "
                f"{tuple(representation.shape)}."
            )

        original_shape = representation.shape[:-1]
        flat = representation.detach().reshape(-1, self.d_model)
        keys = self.Wk(self.memory)
        values = self.Wv(self.memory)
        scale = 1.0 / math.sqrt(self.d_model)

        band_outputs = []
        band_entropies = []
        for band in range(self.num_bands):
            query = self.Wq(flat + self.horizon_embeddings[band])
            head = self.bias_heads[band]

            def retrieve_chunk(query_chunk, key_matrix, value_matrix, active_head=head):
                logits = torch.matmul(query_chunk, key_matrix.transpose(0, 1)) * scale
                attention = torch.softmax(logits, dim=-1)
                retrieved = torch.matmul(attention, value_matrix)
                correction = active_head(retrieved)
                entropy = -(
                    attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()
                ).sum(dim=-1)
                entropy = entropy / math.log(max(self.memory_size, 2))
                return correction, entropy

            chunk_outputs = []
            chunk_entropies = []
            for start in range(0, query.shape[0], self.query_chunk):
                query_chunk = query[start : start + self.query_chunk]
                if self.training and torch.is_grad_enabled():
                    correction, entropy = checkpoint(
                        retrieve_chunk,
                        query_chunk,
                        keys,
                        values,
                        use_reentrant=False,
                    )
                else:
                    correction, entropy = retrieve_chunk(query_chunk, keys, values)
                chunk_outputs.append(correction)
                chunk_entropies.append(entropy.detach())

            band_outputs.append(torch.cat(chunk_outputs, dim=0))
            band_entropies.append(torch.cat(chunk_entropies, dim=0).mean())

        correction = torch.cat(band_outputs, dim=-1).reshape(
            *original_shape, self.pred_len
        )
        entropy = torch.stack(band_entropies)
        return correction, entropy
