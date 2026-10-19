"""DEAP loader: data_preprocessed_python .dat pickles to channel-aligned Trials
(128 Hz, first 32 of 40 channels, 3 s pre-trial baseline dropped)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import channels as ch
from .labels import DOMAIN_DEAP, deap_binary_valence
from .trials import Trial

DEAP_FS = 128.0
DEAP_N_TRIALS = 40
DEAP_BASELINE_SEC = 3.0
DEAP_EEG_CHANNELS = 32  # first 32 of 40 are EEG


def _subject_id_from_path(path: Path) -> int:
    # s01.dat -> 1
    return int(path.stem[1:])


def load_deap_trials(
    root: str | Path | None = None,
    subjects: list[int] | None = None,
    remove_baseline_sec: float = DEAP_BASELINE_SEC,
    valence_threshold: float | None = None,
    montage_order: str = "deap",
) -> list[Trial]:
    """Load DEAP into channel-aligned, binary-labelled Trials.

    root      : directory of sXX.dat (defaults to dinq.paths.DEAP_ROOT).
    subjects  : optional subset of DEAP subject ids (1..32) for fast iteration.
    Returns one Trial per (subject, trial). Valence is binarized globally across
    the returned set (median split by default), matching the median-split
    convention of the companion framework's probe loader.
    """
    from ..paths import DEAP_ROOT

    root = Path(root) if root is not None else DEAP_ROOT
    files = sorted(root.glob("s*.dat"))
    if not files:
        raise FileNotFoundError(f"no DEAP s*.dat under {root}")

    common = ch.common_montage(montage_order)
    eeg_names = ch.DEAP_CHANNELS
    sel = ch.channel_indices(eeg_names, common)  # into the 32 EEG channels

    base = int(round(remove_baseline_sec * DEAP_FS))

    raw_trials: list[dict] = []
    valences: list[float] = []
    for f in files:
        sid = _subject_id_from_path(f)
        if subjects is not None and sid not in subjects:
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        data = np.asarray(d["data"], dtype=np.float64)  # (40, 40, 8064)
        labels = np.asarray(d["labels"], dtype=np.float64)  # (40, 4)
        for ti in range(data.shape[0]):
            eeg = data[ti, :DEAP_EEG_CHANNELS, :]  # (32, 8064)
            eeg = eeg[sel, :]  # (C_common, 8064), common order
            if base > 0:
                eeg = eeg[:, base:]
            raw_trials.append(
                dict(eeg=np.ascontiguousarray(eeg), subject=sid, trial_id=ti)
            )
            valences.append(float(labels[ti, 0]))  # column 0 = valence

    y = deap_binary_valence(np.asarray(valences), threshold=valence_threshold)

    trials: list[Trial] = []
    for rt, yi in zip(raw_trials, y):
        trials.append(
            Trial(
                eeg=rt["eeg"],
                fs=DEAP_FS,
                label=int(yi),
                subject=int(rt["subject"]),  # DEAP ids 1..32, no offset needed
                domain=DOMAIN_DEAP,
                trial_id=int(rt["trial_id"]),
                session=0,
                channels=common,
            )
        )
    return trials
