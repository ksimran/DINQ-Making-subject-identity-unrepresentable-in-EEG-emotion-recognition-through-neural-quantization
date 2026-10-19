"""Differential-entropy features for DEAP on the same bands and channels as SEED-V's precomputed DE."""

from __future__ import annotations

import numpy as np

from .bands import band_set
from .wavelet import decompose_bands
from .featureset import FeatureSet, concat
from ..data.trials import Trial, window_signal

_LOG_2PIE = float(np.log(2.0 * np.pi * np.e))


def differential_entropy(var: np.ndarray) -> np.ndarray:
    """DE from variance: 0.5 * log(2*pi*e*var). Guards against var<=0."""
    var = np.maximum(np.asarray(var, dtype=np.float64), 1e-12)
    return 0.5 * (_LOG_2PIE + np.log(var))


def trial_de(
    trial: Trial,
    bands: tuple[tuple[str, float, float], ...],
    win_sec: float = 4.0,
    hop_sec: float = 4.0,
    band_method: str = "filterbank",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Trial -> (n_win, C, B) DE + per-window metadata (4 s non-overlap default)."""
    band_sigs = decompose_bands(trial.eeg, trial.fs, bands, method=band_method)  # (B,C,T)
    per_band = []
    for bi in range(band_sigs.shape[0]):
        wins = window_signal(band_sigs[bi], trial.fs, win_sec, hop_sec)  # (n_win,C,w)
        var = wins.var(axis=2)  # (n_win,C)
        per_band.append(differential_entropy(var))
    de = np.stack(per_band, axis=2)  # (n_win, C, B)

    n = de.shape[0]
    meta = {
        "label": np.full(n, trial.label, np.int64),
        "subject": np.full(n, trial.subject, np.int64),
        "domain": np.full(n, trial.domain, np.int64),
        "trial_id": np.full(n, trial.trial_id, np.int64),
        "session": np.full(n, trial.session, np.int64),
    }
    return de, meta


def deap_de_featureset(
    trials: list[Trial],
    bands: str = "four",
    win_sec: float = 4.0,
    hop_sec: float = 4.0,
    band_method: str = "filterbank",
) -> FeatureSet:
    """Build a DEAP DE FeatureSet (X = (N, C*B) channel-major) from Trials."""
    band_tuple = band_set(bands)
    parts: list[FeatureSet] = []
    for tr in trials:
        de, meta = trial_de(tr, band_tuple, win_sec, hop_sec, band_method)
        if de.shape[0] == 0:
            continue
        k = de.shape[0]
        parts.append(
            FeatureSet(
                X=de.reshape(k, -1).astype(np.float32),  # channel-major flatten
                label=meta["label"],
                subject=meta["subject"],
                domain=meta["domain"],
                trial_id=meta["trial_id"],
                session=meta["session"],
                feature_kind="de",
                info=dict(
                    corpus="deap",
                    channels=list(trials[0].channels),
                    bands=[b[0] for b in band_tuple],
                    window_sec=win_sec,
                    hop_sec=hop_sec,
                    source="computed_de",
                ),
            )
        )
    return concat(parts)
