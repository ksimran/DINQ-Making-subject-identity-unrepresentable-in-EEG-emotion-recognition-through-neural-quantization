"""SCMM baseline (Liu et al., ACM MM 2025; github.com/Kyler-RL/SCMM) on DINQ's 4-band
DE features. The flattened encoder size is computed by a dummy forward pass because
SCMM hard-codes it for 5 bands."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .scmm_vendor.loss import (
    AutoWeightLoss, SoftContrastiveLoss, AggregateReconstruct, ReconstructiveLoss,
)
from .scmm_vendor.augmentation import masking
from ..features.featureset import FeatureSet


@dataclass
class SCMMConfig:
    in_channels: int = 32
    fea_length: int = 4  # cross-corpus band count (DEAP has no delta)
    encoder_out_dim: int = 128
    out_dim: int = 128
    num_classes: int = 2
    target_num_classes: int = 2
    pretrain_epoch: int = 200
    finetune_epoch: int = 50
    source_batch_size: int = 256
    target_batch_size: int = 128
    beta1: float = 0.9
    beta2: float = 0.99
    learning_rate: float = 5e-4
    weight_decay: float = 3e-4
    mask_ratio: float = 0.5
    mask_num: int = 1
    threshold: float = 0.1
    mask_mode: str = "hybrid_mask"
    upper_bound: float = 0.5
    sharpness: float = 0.2
    dist_metric: str = "cosine"
    temperature: float = 0.05


class SCMM(nn.Module):
    """SCMM base model; flattened encoder dim inferred dynamically for any fea_length."""

    def __init__(self, cfg: SCMMConfig):
        super().__init__()
        d = cfg.encoder_out_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(cfg.in_channels, d // 4, kernel_size=8, bias=False, padding=4),
            nn.BatchNorm1d(d // 4), nn.ReLU(), nn.MaxPool1d(2, 2, padding=1),
            nn.Conv1d(d // 4, d // 2, kernel_size=8, bias=False, padding=4),
            nn.BatchNorm1d(d // 2), nn.ReLU(), nn.MaxPool1d(2, 2, padding=1),
            nn.Conv1d(d // 2, d, kernel_size=8, bias=False, padding=4),
            nn.BatchNorm1d(d), nn.ReLU(), nn.MaxPool1d(2, 2, padding=1),
        )
        # Infer flattened size with a dummy pass (SCMM hardcodes (fea_length+1)//2 for 5 bands).
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.in_channels, cfg.fea_length)
            self.flat_dim = int(self.encoder(dummy).reshape(1, -1).shape[1])

        self.projector = nn.Sequential(
            nn.Linear(self.flat_dim, d * 2), nn.BatchNorm1d(d * 2), nn.ReLU(),
            nn.Linear(d * 2, d),
        )
        self.awl = AutoWeightLoss(loss_num=2)
        self.contrastive = SoftContrastiveLoss(cfg)
        self.aggregation = AggregateReconstruct(cfg)
        self.decoder = nn.Linear(self.flat_dim, cfg.in_channels * cfg.fea_length)
        self.mse = ReconstructiveLoss()

    def forward(self, x_in, train_mode="pre_train"):
        x = self.encoder(x_in)
        h = x.reshape(x.shape[0], -1)
        if train_mode != "pre_train":
            return h
        z = self.projector(h)
        sim_matrix, loss_c = self.contrastive(x_in, z)
        _, recon_emb = self.aggregation(sim_matrix, x)
        recon_x = self.decoder(recon_emb.reshape(recon_emb.size(0), -1))
        loss_r = self.mse(x_in, recon_x)
        loss = self.awl(loss_c, loss_r)
        return loss, loss_c, loss_r


class SCMMClassifier(nn.Module):
    """2-layer MLP emotion classifier on the encoder features h."""

    def __init__(self, cfg: SCMMConfig, flat_dim: int):
        super().__init__()
        self.fc = nn.Linear(flat_dim, cfg.out_dim)
        self.classifier = nn.Linear(cfg.out_dim, cfg.target_num_classes)

    def forward(self, h):
        return self.classifier(torch.sigmoid(self.fc(h)))


def featureset_to_scmm(fs: FeatureSet, cfg: SCMMConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """DE FeatureSet X (N, C*B) -> SCMM tensors (N, C, B) + labels, channel-major."""
    x = fs.X.reshape(fs.n, cfg.in_channels, cfg.fea_length)
    return torch.from_numpy(x).float(), torch.from_numpy(fs.label).long()


def pretrain_masked_batch(model: SCMM, data: torch.Tensor, cfg: SCMMConfig, device) -> tuple:
    """One SSL step's forward (masking + concat original|masked), matching SCMM's trainer."""
    _, mask_data = masking(data.cpu().numpy() if isinstance(data, torch.Tensor) else data,
                           mask_ratio=cfg.mask_ratio, mask_num=cfg.mask_num,
                           threshold=cfg.threshold, mask_mode=cfg.mask_mode)
    batch = torch.cat([data, mask_data.to(data.device)], dim=0).float().to(device)
    return model(batch, train_mode="pre_train")
