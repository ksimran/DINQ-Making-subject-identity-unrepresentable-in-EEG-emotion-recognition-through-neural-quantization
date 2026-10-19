"""Per-band tangent-space projection of covariances at a reference mean (pyRiemann TangentSpace)."""

from __future__ import annotations

import numpy as np


def tangent_dim(n_channels: int) -> int:
    return n_channels * (n_channels + 1) // 2


class BandTangent:
    """One pyRiemann TangentSpace per band; concatenates band vectors."""

    def __init__(self, metric: str = "riemann"):
        self.metric = metric
        self._ts: list = []
        self.n_bands: int | None = None
        self.n_channels: int | None = None

    def fit(self, covs: np.ndarray) -> "BandTangent":
        """covs: (N, B, C, C) source-train covariances."""
        from pyriemann.tangentspace import TangentSpace

        if covs.ndim != 4:
            raise ValueError(f"covs must be (N,B,C,C), got {covs.shape}")
        _, B, C, _ = covs.shape
        self.n_bands, self.n_channels = B, C
        self._ts = []
        for b in range(B):
            ts = TangentSpace(metric=self.metric)
            ts.fit(covs[:, b])
            self._ts.append(ts)
        return self

    def transform(self, covs: np.ndarray) -> np.ndarray:
        """covs: (N, B, C, C) -> (N, B * C(C+1)/2) tangent features."""
        if self.n_bands is None:
            raise RuntimeError("call fit() first")
        if covs.shape[1] != self.n_bands:
            raise ValueError("band count mismatch vs fit")
        vecs = [self._ts[b].transform(covs[:, b]) for b in range(self.n_bands)]
        return np.concatenate(vecs, axis=1).astype(np.float32)

    def fit_transform(self, covs: np.ndarray) -> np.ndarray:
        return self.fit(covs).transform(covs)

    def inverse_transform(self, vecs: np.ndarray) -> np.ndarray:
        """(N, B*d) tangent vectors -> (N, B, C, C) covariances (round-trip test)."""
        if self.n_bands is None:
            raise RuntimeError("call fit() first")
        d = tangent_dim(self.n_channels)
        out = []
        for b in range(self.n_bands):
            vb = vecs[:, b * d : (b + 1) * d]
            out.append(self._ts[b].inverse_transform(vb))
        return np.stack(out, axis=1)
