"""Minimal DINQ: an MLP encoder feeding a quantized content code z_c and a continuous
residual z_r, and a decoder that reconstructs the tangent vector from both.
Loss: L_rec + L_vq + lambda_con * L_con."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..losses.contrastive import supervised_contrastive


@dataclass
class DINQConfig:
    in_dim: int  # tangent dim (SEED-V 2112 / FACED 1860)
    quantizer: str = "fsq"  # 'fsq' | 'vq' | 'none' (continuous ablation)
    fsq_levels: tuple[int, ...] = (8, 8, 8, 8)  # 4096 codes
    vq_codebook: int = 256
    d_c: int = 64  # content dim for 'vq' / 'none'; FSQ uses len(fsq_levels)
    d_r: int = 64  # continuous residual dim
    hidden: int = 512
    depth: int = 3
    temperature: float = 0.1
    lambda_con: float = 1.0
    commit_weight: float = 0.25
    contrast_on: str = "zc"  # 'zc' or 'u' (pre-quantization; avoids a flat SupCon signal through the STE)


def _mlp(dims: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.GELU()]
    return nn.Sequential(*layers)


class MinimalDINQ(nn.Module):
    def __init__(self, cfg: DINQConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.encoder = _mlp([cfg.in_dim] + [h] * cfg.depth)

        if cfg.quantizer == "fsq":
            from vector_quantize_pytorch import FSQ
            self.content_dim = len(cfg.fsq_levels)
            self.quant = FSQ(levels=list(cfg.fsq_levels))
            self.codebook_size = int(self.quant.codebook_size)
        elif cfg.quantizer == "vq":
            from vector_quantize_pytorch import VectorQuantize
            self.content_dim = cfg.d_c
            self.quant = VectorQuantize(dim=cfg.d_c, codebook_size=cfg.vq_codebook,
                                        decay=0.99, commitment_weight=cfg.commit_weight,
                                        kmeans_init=True, threshold_ema_dead_code=2)
            self.codebook_size = cfg.vq_codebook
        elif cfg.quantizer in ("none", "continuous"):
            # Mechanism-isolation ablation: identity bottleneck at matched capacity
            # (same d_c content dim, same d_r residual, same contrastive target,
            # everything else identical), separating the effect of quantization
            # from that of the autoencoder bottleneck.
            self.content_dim = cfg.d_c
            self.quant = None
            self.codebook_size = 1
        else:
            raise ValueError(f"quantizer must be 'fsq' | 'vq' | 'none', got {cfg.quantizer!r}")

        self.g_c = nn.Linear(h, self.content_dim)
        self.g_r = nn.Linear(h, cfg.d_r)
        self.decoder = _mlp([self.content_dim + cfg.d_r] + [h] * (cfg.depth - 1) + [cfg.in_dim])

    def quantize(self, u: torch.Tensor):
        """u (B, content_dim) -> z_c (B, content_dim), indices (B,), vq_loss."""
        if self.quant is None:  # continuous ablation: pass-through
            return u, u.new_zeros(u.shape[0], dtype=torch.long), u.new_zeros(())
        out = self.quant(u.unsqueeze(1))  # expects (B, 1, dim)
        if self.cfg.quantizer == "fsq":
            z_c, idx = out
            vq_loss = u.new_zeros(())
        else:
            z_c, idx, vq_loss = out
        return z_c.squeeze(1), idx.reshape(-1), vq_loss

    def forward(self, v: torch.Tensor):
        h = self.encoder(v)
        u = self.g_c(h)
        z_c, idx, vq_loss = self.quantize(u)
        z_r = self.g_r(h)
        v_hat = self.decoder(torch.cat([z_c, z_r], dim=1))
        return dict(v_hat=v_hat, z_c=z_c, z_r=z_r, u=u, idx=idx, vq_loss=vq_loss)

    def loss(self, v: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        out = self(v)
        l_rec = F.mse_loss(out["v_hat"], v)
        l_vq = out["vq_loss"]
        con_target = out["u"] if self.cfg.contrast_on == "u" else out["z_c"]
        l_con = supervised_contrastive(con_target, y, self.cfg.temperature)
        total = l_rec + l_vq + self.cfg.lambda_con * l_con
        return total, dict(rec=float(l_rec), vq=float(l_vq), con=float(l_con),
                           perplexity=perplexity(out["idx"], self.codebook_size))

    @torch.no_grad()
    def encode_zc(self, v: torch.Tensor) -> torch.Tensor:
        """Content code z_c for probing (the certificate input)."""
        return self(v)["z_c"]


def perplexity(idx: torch.Tensor, codebook_size: int) -> float:
    """exp(-sum p_k log p_k) over code usage (codebook-collapse monitor)."""
    p = torch.bincount(idx, minlength=codebook_size).float()
    p = p / p.sum().clamp_min(1)
    return float(torch.exp(-(p * (p + 1e-10).log()).sum()))
