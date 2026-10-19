"""Corpus-agnostic Trial container and windowing in seconds, so corpora with different
sampling rates give the same windows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trial:
    """One emotion trial, channel-aligned to the common montage.

    eeg      : (C, T) float64 continuous EEG for this trial.
    fs       : sampling rate in Hz.
    label    : binary emotion label (0 negative / 1 positive); -1 if dropped.
    subject  : global subject id (SEED-V and FACED ids offset to be disjoint).
    domain   : 0 = DEAP, 1 = SEED-V, 2 = FACED.
    trial_id : within-corpus trial index (for provenance / leakage checks).
    session  : recording session (0 for DEAP and FACED; 1..3 for SEED-V).
    channels : channel names in row order of `eeg` (common montage order).
    """

    eeg: np.ndarray
    fs: float
    label: int
    subject: int
    domain: int
    trial_id: int
    session: int
    channels: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.eeg.ndim != 2:
            raise ValueError(f"eeg must be (C, T), got {self.eeg.shape}")
        if self.channels and len(self.channels) != self.eeg.shape[0]:
            raise ValueError(
                f"channels ({len(self.channels)}) != eeg rows ({self.eeg.shape[0]})"
            )


def window_signal(
    eeg: np.ndarray,
    fs: float,
    win_sec: float,
    hop_sec: float,
    drop_last: bool = True,
) -> np.ndarray:
    """Slice (C, T) into (n_windows, C, win_samples) along time.

    Every window has exactly `win_samples` columns (required for equal-length
    covariances). If the hop does not tile the trial exactly, `drop_last=True`
    discards the uncovered tail; `drop_last=False` adds one extra window aligned
    to the end of the trial.
    """
    if eeg.ndim != 2:
        raise ValueError(f"eeg must be (C, T), got {eeg.shape}")
    C, T = eeg.shape
    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    if win <= 0 or hop <= 0:
        raise ValueError("win_sec and hop_sec must be positive")
    if win > T:
        return np.empty((0, C, win), dtype=eeg.dtype)

    starts = list(range(0, T - win + 1, hop))
    if not drop_last and (T - win) % hop != 0:
        starts.append(T - win)
    out = np.stack([eeg[:, s : s + win] for s in starts], axis=0)
    return out


def window_trial(
    trial: Trial,
    win_sec: float,
    hop_sec: float,
    drop_last: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Window a trial, returning (windows, per-window metadata arrays).

    Metadata arrays (label/subject/domain/trial_id/session) are broadcast to one
    entry per window so downstream code can concatenate across trials and keep
    provenance for leakage-safe splitting and the probe export.
    """
    windows = window_signal(trial.eeg, trial.fs, win_sec, hop_sec, drop_last)
    n = windows.shape[0]
    meta = {
        "label": np.full(n, trial.label, dtype=np.int64),
        "subject": np.full(n, trial.subject, dtype=np.int64),
        "domain": np.full(n, trial.domain, dtype=np.int64),
        "trial_id": np.full(n, trial.trial_id, dtype=np.int64),
        "session": np.full(n, trial.session, dtype=np.int64),
    }
    return windows, meta
