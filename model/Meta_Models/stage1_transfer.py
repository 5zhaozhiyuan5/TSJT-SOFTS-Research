import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPrivateReliabilityTransfer(nn.Module):
    """Lightweight shared/private decomposition for Stage-I domain alignment.

    Input:  [samples, feature_dim]
    Output: shared, private, reconstruction, disentanglement_loss.

    The initial projection/decoder weights form an exact partition/reassembly
    of the input features, so enabling this module does not initially destroy
    the pretrained representation.
    """
    def __init__(self, input_dim=36, shared_ratio=2.0 / 3.0,
                 reliability_temperature=1.0, min_reliability=0.05):
        super().__init__()
        self.input_dim = int(input_dim)
        self.shared_dim = max(1, min(self.input_dim - 1,
                                     int(round(self.input_dim * shared_ratio))))
        self.private_dim = self.input_dim - self.shared_dim
        self.reliability_temperature = float(reliability_temperature)
        self.min_reliability = float(min_reliability)

        self.shared_proj = nn.Linear(self.input_dim, self.shared_dim, bias=False)
        self.private_proj = nn.Linear(self.input_dim, self.private_dim, bias=False)
        self.reconstruct = nn.Linear(self.input_dim, self.input_dim, bias=False)
        self.reset_identity_partition()

    def reset_identity_partition(self):
        with torch.no_grad():
            self.shared_proj.weight.zero_()
            self.private_proj.weight.zero_()
            self.reconstruct.weight.zero_()
            self.shared_proj.weight[:, :self.shared_dim] = torch.eye(
                self.shared_dim, device=self.shared_proj.weight.device,
                dtype=self.shared_proj.weight.dtype)
            self.private_proj.weight[:, self.shared_dim:] = torch.eye(
                self.private_dim, device=self.private_proj.weight.device,
                dtype=self.private_proj.weight.dtype)
            self.reconstruct.weight.copy_(torch.eye(
                self.input_dim, device=self.reconstruct.weight.device,
                dtype=self.reconstruct.weight.dtype))

    def forward(self, features):
        shared = self.shared_proj(features)
        private = self.private_proj(features)
        joined = torch.cat((shared, private), dim=-1)
        reconstructed = self.reconstruct(joined)

        if features.shape[0] > 1:
            s = shared - shared.mean(dim=0, keepdim=True)
            p = private - private.mean(dim=0, keepdim=True)
            cross = s.transpose(0, 1).matmul(p) / float(features.shape[0] - 1)
            s_scale = s.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
            p_scale = p.pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
            normalized_cross = cross / (s_scale[:, None] * p_scale[None, :])
            orthogonality = normalized_cross.pow(2).mean()
        else:
            orthogonality = features.new_zeros(())

        reconstruction = F.smooth_l1_loss(reconstructed, features.detach())
        disentanglement_loss = reconstruction + 0.05 * orthogonality
        return shared, private, reconstructed, disentanglement_loss


    @torch.no_grad()
    def source_quality(self, source_shared, target_shared):
        """Scalar target-conditioned source-domain relevance in [0.05, 1]."""
        target_center = target_shared.mean(dim=0, keepdim=True)
        source_center = source_shared.mean(dim=0, keepdim=True)
        target_var = target_shared.var(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        shift = ((source_center - target_center).pow(2) / target_var).mean()
        return torch.exp(-shift).clamp_min(0.05)

    @torch.no_grad()
    def reliability_weights(self, source_shared, target_shared):
        """Linear-memory reliability from cross-domain standardized centroid distance."""
        eps = 1e-6
        source_center = target_shared.mean(dim=0, keepdim=True)
        target_center = source_shared.mean(dim=0, keepdim=True)
        source_var = target_shared.var(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
        target_var = source_shared.var(dim=0, unbiased=False, keepdim=True).clamp_min(eps)

        source_distance = ((source_shared - source_center).pow(2) / source_var).mean(dim=-1)
        target_distance = ((target_shared - target_center).pow(2) / target_var).mean(dim=-1)
        source_rel = torch.exp(-source_distance / max(self.reliability_temperature, eps))
        target_rel = torch.exp(-target_distance / max(self.reliability_temperature, eps))
        source_rel = source_rel.clamp_min(self.min_reliability)
        target_rel = target_rel.clamp_min(self.min_reliability)
        source_rel = source_rel / source_rel.sum().clamp_min(eps)
        target_rel = target_rel / target_rel.sum().clamp_min(eps)
        return source_rel, target_rel
