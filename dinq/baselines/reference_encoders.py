"""Capacity-matched reference encoders (random, task-only, MMD, GRL) on DE features,
used to recalibrate the certificate thresholds before DINQ is evaluated."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset

from ..features.featureset import FeatureSet
from ..utils.seed import set_seed


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hid: int = 256, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Linear(hid, out), nn.BatchNorm1d(out), nn.ReLU(),
        )
        self.out_dim = out

    def forward(self, x):
        return self.net(x)


def _mmd_rbf(xs, xt, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)):
    def k(a, b):
        d = torch.cdist(a, b) ** 2
        return sum(torch.exp(-d / (2 * s * s)) for s in sigmas) / len(sigmas)
    return k(xs, xs).mean() + k(xt, xt).mean() - 2 * k(xs, xt).mean()


def train_reference(
    mode: str,
    source_train: FeatureSet,
    target_fs: FeatureSet,
    n_classes: int = 2,
    device: str = "cpu",
    epochs: int = 60,
    lr: float = 1e-3,
    batch: int = 256,
    grl_lambda: float = 1.0,
    mmd_weight: float = 1.0,
    seed: int = 0,
) -> tuple[MLPEncoder, nn.Module | None]:
    """Train (or, for 'random', just init) an encoder. Returns (encoder, head).

    `head` is the source-task classifier (None for 'random'); used to report the
    model's actual DG accuracy (train on source, test target, no target labels).
    """
    set_seed(seed)
    in_dim = source_train.dim
    enc = MLPEncoder(in_dim).to(device)
    if mode == "random":
        return enc, None

    head = nn.Linear(enc.out_dim, n_classes).to(device)
    dom = nn.Linear(enc.out_dim, 2).to(device) if mode == "grl" else None

    Xs = torch.from_numpy(source_train.X).float()
    ys = torch.from_numpy(source_train.label).long()
    sloader = DataLoader(TensorDataset(Xs, ys), batch_size=batch, shuffle=True, drop_last=True)
    Xt = torch.from_numpy(target_fs.X).float().to(device)

    params = list(enc.parameters()) + list(head.parameters()) + (list(dom.parameters()) if dom else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    enc.train()
    for _ in range(epochs):
        for xb, yb in sloader:
            xb, yb = xb.to(device), yb.to(device)
            hs = enc(xb)
            loss = ce(head(hs), yb)
            if mode in ("mmd", "grl"):
                ti = rng.integers(0, Xt.shape[0], size=xb.shape[0])
                ht = enc(Xt[ti])
                if mode == "mmd":
                    loss = loss + mmd_weight * _mmd_rbf(hs, ht)
                else:  # grl: domain adversary via gradient reversal
                    hcat = torch.cat([grad_reverse(hs, grl_lambda), grad_reverse(ht, grl_lambda)], 0)
                    dlab = torch.cat([torch.zeros(hs.shape[0]), torch.ones(ht.shape[0])]).long().to(device)
                    loss = loss + ce(dom(hcat), dlab)
            opt.zero_grad(); loss.backward(); opt.step()
    return enc, head


@torch.no_grad()
def dg_accuracy(enc: MLPEncoder, head, target_fs, device: str) -> float:
    """Model's DG accuracy: source-trained head on target (no target labels used)."""
    from sklearn.metrics import balanced_accuracy_score
    if head is None:
        return float("nan")
    enc.eval(); head.eval()
    X = torch.from_numpy(target_fs.X).float()
    preds = []
    for i in range(0, X.shape[0], 1024):
        preds.append(head(enc(X[i:i + 1024].to(device))).argmax(1).cpu().numpy())
    return float(balanced_accuracy_score(target_fs.label, np.concatenate(preds)))


@torch.no_grad()
def embed_encoder(enc: MLPEncoder, fs: FeatureSet, device: str) -> FeatureSet:
    """Return a FeatureSet whose X is the encoder representation of `fs`."""
    enc.eval()
    X = torch.from_numpy(fs.X).float()
    out = [enc(X[i:i + 1024].to(device)).cpu().numpy() for i in range(0, X.shape[0], 1024)]
    return FeatureSet(X=np.concatenate(out).astype(np.float32), label=fs.label,
                      subject=fs.subject, domain=fs.domain, trial_id=fs.trial_id,
                      session=fs.session, feature_kind="ref_h", info=dict(fs.info))
