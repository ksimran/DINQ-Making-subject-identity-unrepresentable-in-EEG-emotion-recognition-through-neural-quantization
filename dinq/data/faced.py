"""FACED loader: preprocessed CLISA .pkl files, read from the zip, to channel-aligned
Trials (250 Hz, 30 EEG channels after dropping the A1/A2 mastoids)."""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np

from . import channels as ch
from .labels import DOMAIN_FACED, FACED_SUBJECT_OFFSET
from .trials import Trial

FACED_FS = 250.0
FACED_N_VIDEOS = 28
FACED_N_SUBJECTS = 123

# 0-based .pkl video -> binary (1 pos / 0 neg / -1 neutral-drop).
FACED_VIDEO_BINARY: tuple[int, ...] = tuple(
    (0 if v <= 11 else (-1 if v <= 15 else 1)) for v in range(FACED_N_VIDEOS)
)

# Row indices of the 30 EEG channels within the 32-channel .pkl (drop A1/A2).
_EEG_ROWS = [i for i, c in enumerate(ch.FACED_BDF_CHANNELS) if c not in ch.FACED_MASTOIDS]


def load_faced_trials(
    subjects: list[int] | None = None,
    clisa_zip: str | Path | None = None,
    montage: tuple[str, ...] | None = None,
) -> list[Trial]:
    """Load FACED into channel-aligned, binary-labelled Trials.

    montage : common-montage names to keep (e.g. ch.intersect(("deap","faced")));
              defaults to all 30 FACED EEG channels (within-FACED cross-subject).
    Neutral videos are dropped. One Trial per (subject, non-neutral video).
    """
    from ..paths import FACED_CLISA_ZIP

    zip_path = Path(clisa_zip) if clisa_zip is not None else FACED_CLISA_ZIP
    subjects = subjects if subjects is not None else list(range(FACED_N_SUBJECTS))
    common = montage if montage is not None else ch.FACED_CHANNELS
    sel = ch.channel_indices(ch.FACED_CHANNELS, common)  # into the 30 EEG channels

    trials: list[Trial] = []
    with zipfile.ZipFile(zip_path) as z:
        for sid in subjects:
            name = f"Clisa_data/sub{sid:03d}.pkl"
            with z.open(name) as fh:
                arr = np.asarray(pickle.load(fh), dtype=np.float64)  # (28, 32, 7500)
            eeg30 = arr[:, _EEG_ROWS, :]  # (28, 30, 7500)
            for vid in range(arr.shape[0]):
                y = FACED_VIDEO_BINARY[vid]
                if y < 0:
                    continue  # neutral dropped
                seg = eeg30[vid][sel, :]  # (C_common, 7500)
                trials.append(
                    Trial(
                        eeg=np.ascontiguousarray(seg),
                        fs=FACED_FS,
                        label=int(y),
                        subject=sid + FACED_SUBJECT_OFFSET,
                        domain=DOMAIN_FACED,
                        trial_id=vid,
                        session=0,
                        channels=common,
                    )
                )
    return trials
