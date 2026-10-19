"""Per-band spatial covariances with Ledoit-Wolf shrinkage (pyRiemann), trace-normalised and cached."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .wavelet import decompose_bands
from ..data.trials import Trial
from ..data.trials import window_signal


def window_covariances(
    windows: np.ndarray,
    estimator: str = "lwf",
    normalize: str = "trace",
) -> np.ndarray:
    """(n_win, C, w) -> (n_win, C, C) shrunk SPD covariances.

    `normalize='trace'` rescales each covariance to mean-diagonal 1
    (Sigma * C / tr(Sigma)). This removes absolute amplitude (a cross-corpus
    confound: DEAP in uV and SEED-V in V from .cnt differ by ~1e10 in scale)
    while preserving the spatial covariance structure the tangent map reads.
    It also keeps covariances numerically O(1) regardless of the recording's
    units. Set `normalize='none'` to keep raw scale (within-corpus / ablation).
    """
    from pyriemann.estimation import Covariances

    if windows.shape[0] == 0:
        C = windows.shape[1]
        return np.empty((0, C, C), dtype=np.float64)
    covs = Covariances(estimator=estimator).transform(windows)
    if normalize == "trace":
        C = covs.shape[-1]
        tr = np.trace(covs, axis1=-2, axis2=-1)[:, None, None]
        covs = covs * (C / np.clip(tr, 1e-30, None))
    elif normalize != "none":
        raise ValueError(f"normalize must be 'trace' or 'none', got {normalize!r}")
    return covs


def trial_band_covariances(
    trial: Trial,
    bands: tuple[tuple[str, float, float], ...],
    win_sec: float = 4.0,
    hop_sec: float = 2.0,
    estimator: str = "lwf",
    band_method: str = "filterbank",
    normalize: str = "trace",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Trial -> (n_win, B, C, C) covariances + per-window metadata.

    Bands are extracted from the continuous trial first (no window-edge
    transient), then each band is windowed and covariance-estimated. Covariances
    are trace-normalized by default (cross-corpus scale invariance; see
    `window_covariances`).
    """
    band_sigs = decompose_bands(trial.eeg, trial.fs, bands, method=band_method)  # (B,C,T)
    per_band = []
    for bi in range(band_sigs.shape[0]):
        wins = window_signal(band_sigs[bi], trial.fs, win_sec, hop_sec)  # (n_win,C,w)
        covs = window_covariances(wins, estimator=estimator, normalize=normalize)
        per_band.append(covs)
    covs = np.stack(per_band, axis=1)  # (n_win, B, C, C)

    n = covs.shape[0]
    meta = {
        "label": np.full(n, trial.label, np.int64),
        "subject": np.full(n, trial.subject, np.int64),
        "domain": np.full(n, trial.domain, np.int64),
        "trial_id": np.full(n, trial.trial_id, np.int64),
        "session": np.full(n, trial.session, np.int64),
    }
    return covs, meta


def is_spd(mats: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Per-matrix SPD check: symmetric and positive-definite, scale-relative.

    Tolerances are relative to each matrix's own spectrum so the verdict is
    invariant to units: a covariance in volts (values ~1e-10) and one in
    microvolts (values ~1e2) are judged identically. PD requires
    min_eig > rtol * max_eig (also caps the condition number at ~1/rtol).
    """
    mats = np.asarray(mats)
    flat = mats.reshape(-1, mats.shape[-2], mats.shape[-1])
    ok = np.ones(flat.shape[0], dtype=bool)
    for i, m in enumerate(flat):
        scale = max(float(np.abs(m).max()), 1e-300)
        sym = np.allclose(m, m.T, atol=1e-8 * scale, rtol=1e-5)
        w = np.linalg.eigvalsh((m + m.T) / 2.0)
        ok[i] = sym and bool(w.min() > rtol * max(abs(w.max()), 1e-300))
    return ok.reshape(mats.shape[:-2])


_META =("label", "subject", "domain", "trial_id", "session")


@dataclass
class CovarianceSet:
    """Per-window per-band SPD covariances and aligned metadata, for one corpus.

    Kept separate from FeatureSet because tangent projection is deferred: the
    reference mean is fit on source-train only (leakage-safe) and recentering
    can be toggled without recomputing these covariances.
    """

    covs: np.ndarray  # (N, B, C, C) float32
    label: np.ndarray
    subject: np.ndarray
    domain: np.ndarray
    trial_id: np.ndarray
    session: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.covs.shape[0])

    def select(self, mask: np.ndarray) -> "CovarianceSet":
        return CovarianceSet(
            covs=self.covs[mask], label=self.label[mask], subject=self.subject[mask],
            domain=self.domain[mask], trial_id=self.trial_id[mask],
            session=self.session[mask], info=dict(self.info),
        )

    def subset_montage(self, target: tuple[str, ...]) -> "CovarianceSet":
        """Subset channels to `target`, then re-trace-normalize.

        A principal submatrix of an SPD matrix stays SPD, so a 32-ch cache can be
        aligned to a 30-ch cross-corpus montage without a raw re-read.
        `info['channels']` must list the current channel order.
        """
        from ..data.channels import channel_indices
        src = tuple(self.info["channels"])
        idx = channel_indices(src, target)  # indices of target within src
        covs = self.covs[:, :, idx][:, :, :, idx].astype(np.float64)
        C = covs.shape[-1]
        tr = np.trace(covs, axis1=-2, axis2=-1)[..., None, None]
        covs = (covs * (C / np.clip(tr, 1e-30, None))).astype(np.float32)  # re-trace-norm to mean-diag 1
        return CovarianceSet(
            covs=covs, label=self.label, subject=self.subject, domain=self.domain,
            trial_id=self.trial_id, session=self.session,
            info={**self.info, "channels": list(target)},
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            covs=self.covs.astype(np.float32),
            label=self.label, subject=self.subject, domain=self.domain,
            trial_id=self.trial_id, session=self.session,
            info=np.array(json.dumps(self.info)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CovarianceSet":
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["info"])) if "info" in d.files else {}
        return cls(
            covs=d["covs"], label=d["label"], subject=d["subject"],
            domain=d["domain"], trial_id=d["trial_id"], session=d["session"],
            info=info,
        )


def covset_from_trials(
    trials: list[Trial],
    bands: tuple[tuple[str, float, float], ...],
    win_sec: float = 4.0,
    hop_sec: float = 2.0,
    estimator: str = "lwf",
    band_method: str = "filterbank",
    normalize: str = "trace",
    corpus: str = "",
) -> CovarianceSet:
    """Build a CovarianceSet from a list of Trials (one corpus)."""
    covs_l, meta_l = [], {k: [] for k in _META}
    for tr in trials:
        c, m = trial_band_covariances(tr, bands, win_sec, hop_sec, estimator,
                                      band_method, normalize)
        if c.shape[0] == 0:
            continue
        covs_l.append(c.astype(np.float32))
        for k in _META:
            meta_l[k].append(m[k])
    covs = np.concatenate(covs_l, axis=0)
    return CovarianceSet(
        covs=covs,
        **{k: np.concatenate(meta_l[k]) for k in _META},
        info=dict(
            corpus=corpus,
            channels=list(trials[0].channels),
            bands=[b[0] for b in bands],
            win_sec=win_sec, hop_sec=hop_sec, estimator=estimator,
            band_method=band_method, normalize=normalize,
            shape="(N,B,C,C)",
        ),
    )
