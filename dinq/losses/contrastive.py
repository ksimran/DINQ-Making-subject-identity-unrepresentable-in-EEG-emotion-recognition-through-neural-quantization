"""Supervised contrastive (SupCon) loss with same-emotion positives on L2-normalised codes."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SupCon on z (B, d) with labels y (B,). Positives = same-y (excluding self)."""
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature  # (B, B)
    B = z.shape[0]
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    # log-softmax denominator over all a != i, computed without -inf (which would
    # give 0 * -inf = NaN once multiplied by the zero self-position of `pos`).
    m = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - m) * (~self_mask).float()  # zero the self term
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)) + m
    log_prob = sim - log_denom  # finite everywhere

    pos = (y[:, None] == y[None, :]) & ~self_mask  # (B, B) positive pairs
    pos_counts = pos.sum(1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return z.new_zeros(())
    # mean log-prob over positives, averaged over anchors that have positives
    mean_log_prob_pos = (pos.float() * log_prob).sum(1)[valid] / pos_counts[valid]
    return -mean_log_prob_pos.mean()
