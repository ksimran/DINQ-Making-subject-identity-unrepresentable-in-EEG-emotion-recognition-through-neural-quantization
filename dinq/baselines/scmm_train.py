"""SCMM pretraining, finetuning and embedding on DINQ's features, following SCMM's trainer.py."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .scmm import SCMM, SCMMClassifier, SCMMConfig, featureset_to_scmm, pretrain_masked_batch
from ..features.featureset import FeatureSet
from ..utils.seed import set_seed


def pretrain(model: SCMM, source_fs: FeatureSet, cfg: SCMMConfig, device: str,
             epochs: int | None = None, log_every: int = 25) -> SCMM:
    """SSL pretrain the encoder on source (unlabeled). Drops last partial batch."""
    epochs = cfg.pretrain_epoch if epochs is None else epochs
    X, _ = featureset_to_scmm(source_fs, cfg)
    loader = DataLoader(TensorDataset(X), batch_size=cfg.source_batch_size,
                        shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    model.train()
    for ep in range(1, epochs + 1):
        losses = []
        for (data,) in loader:
            loss, _, _ = pretrain_masked_batch(model, data, cfg, device)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if ep % log_every == 0 or ep == epochs:
            print(f"    [pretrain] ep {ep}/{epochs} loss={np.mean(losses):.4f}")
    return model


def finetune(model: SCMM, cfg: SCMMConfig, target_train: FeatureSet, device: str,
             epochs: int | None = None, tune_encoder: bool = True) -> SCMMClassifier:
    """Supervised finetune classifier (and optionally encoder) on target train labels."""
    epochs = cfg.finetune_epoch if epochs is None else epochs
    clf = SCMMClassifier(cfg, model.flat_dim).to(device)
    X, y = featureset_to_scmm(target_train, cfg)
    loader = DataLoader(TensorDataset(X, y), batch_size=cfg.target_batch_size, shuffle=True)
    params = list(clf.parameters()) + (list(model.parameters()) if tune_encoder else [])
    opt = torch.optim.Adam(params, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
                           weight_decay=cfg.weight_decay)
    ce = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train() if tune_encoder else model.eval()
        clf.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            h = model(xb, train_mode="fine_tune")
            loss = ce(clf(h), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return clf


@torch.no_grad()
def embed(model: SCMM, fs: FeatureSet, cfg: SCMMConfig, device: str) -> np.ndarray:
    """Encoder features h for every window in a FeatureSet (the representation to probe)."""
    model.eval()
    X, _ = featureset_to_scmm(fs, cfg)
    out = []
    for i in range(0, X.shape[0], 512):
        out.append(model(X[i:i + 512].to(device), train_mode="fine_tune").cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def scmm_representation(source_fs: FeatureSet, deap_fs: FeatureSet, seedv_fs: FeatureSet,
                        cfg: SCMMConfig, device: str, seed: int = 0,
                        pretrain_epochs: int | None = None) -> tuple[FeatureSet, FeatureSet]:
    """Pretrain SCMM on `source_fs` and return DEAP/SEED-V encoder-feature FeatureSets.

    The returned X is SCMM's encoder representation (the certificate input). The
    task metric comes from the harness's own task probe on this representation,
    not from SCMM finetuning.
    """
    set_seed(seed)
    model = SCMM(cfg).to(device)
    pretrain(model, source_fs, cfg, device, epochs=pretrain_epochs)

    def as_repr(fs: FeatureSet) -> FeatureSet:
        z = embed(model, fs, cfg, device)
        return FeatureSet(X=z, label=fs.label, subject=fs.subject, domain=fs.domain,
                          trial_id=fs.trial_id, session=fs.session, feature_kind="scmm_h",
                          info={**fs.info, "repr": "scmm_encoder_h"})

    return as_repr(deap_fs), as_repr(seedv_fs)
