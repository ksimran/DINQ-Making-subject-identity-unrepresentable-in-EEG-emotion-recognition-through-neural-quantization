"""SEED-V loaders: precomputed DE features and raw Neuroscan .cnt recordings."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from . import channels as ch
from .labels import (
    DOMAIN_SEEDV,
    SEEDV_SUBJECT_OFFSET,
    seedv_binary_labels,
)
from .trials import Trial
from ..features.bands import band_set, band_indices_in_5
from ..features.featureset import FeatureSet, concat

SEEDV_N_SUBJECTS = 16
SEEDV_TRIALS_PER_SESSION = 15
SEEDV_N_DE_CHANNELS = 62
SEEDV_N_DE_BANDS = 5


# Labels (shared by both paths)
def _load_de_npz(subject: int, de_root: Path) -> tuple[dict, dict]:
    f = de_root / f"{subject}_123.npz"
    if not f.exists():
        raise FileNotFoundError(f"SEED-V DE file missing: {f}")
    d = np.load(f, allow_pickle=True)
    data = pickle.loads(d["data"].tobytes())
    label = pickle.loads(d["label"].tobytes())
    return data, label


def seedv_trial_codes(subject: int, de_root: str | Path | None = None) -> dict[int, int]:
    """Per-trial emotion code (0..4) for a subject, from the DE label dict."""
    from ..paths import SEEDV_DE_ROOT

    de_root = Path(de_root) if de_root is not None else SEEDV_DE_ROOT
    _, label = _load_de_npz(subject, de_root)
    return {t: int(np.asarray(label[t]).ravel()[0]) for t in sorted(label.keys())}


# DE feature path (precomputed)
def load_seedv_de(
    de_root: str | Path | None = None,
    subjects: list[int] | None = None,
    bands: str = "four",
    neutral: str = "drop",
    montage_order: str = "deap",
) -> FeatureSet:
    """Load SEED-V DE into a FeatureSet on the common montage + chosen band set.

    Output X is (N, C_common * B) channel-major flattened, feature_kind='de'.
    Neutral segments are dropped (default). Subject ids are offset to be disjoint
    from DEAP.
    """
    from ..paths import SEEDV_DE_ROOT

    de_root = Path(de_root) if de_root is not None else SEEDV_DE_ROOT
    subjects = subjects or list(range(1, SEEDV_N_SUBJECTS + 1))

    band_tuple = band_set(bands)
    b_idx = band_indices_in_5(band_tuple)
    common = ch.common_montage(montage_order)
    c_idx = ch.seedv_common_indices(montage_order)

    per_subject: list[FeatureSet] = []
    for sid in subjects:
        data, label = _load_de_npz(sid, de_root)
        Xs, lab, subj, dom, tri, ses = [], [], [], [], [], []
        for t in sorted(data.keys()):
            seg = np.asarray(data[t], dtype=np.float64)  # (n_seg, 310)
            n_seg = seg.shape[0]
            seg = seg.reshape(n_seg, SEEDV_N_DE_CHANNELS, SEEDV_N_DE_BANDS)
            seg = seg[:, c_idx, :][:, :, b_idx]  # (n_seg, C_common, B)
            codes = np.asarray(label[t]).ravel().astype(int)
            if codes.shape[0] != n_seg:  # some label arrays are scalar-broadcast
                codes = np.full(n_seg, codes[0], dtype=int)
            ybin, keep = seedv_binary_labels(codes, neutral=neutral)
            if not keep.any():
                continue
            seg, ybin = seg[keep], ybin[keep]
            k = seg.shape[0]
            Xs.append(seg.reshape(k, -1).astype(np.float32))
            lab.append(ybin)
            subj.append(np.full(k, sid + SEEDV_SUBJECT_OFFSET, np.int64))
            dom.append(np.full(k, DOMAIN_SEEDV, np.int64))
            tri.append(np.full(k, t, np.int64))
            ses.append(np.full(k, t // SEEDV_TRIALS_PER_SESSION + 1, np.int64))
        if not Xs:
            continue
        per_subject.append(
            FeatureSet(
                X=np.concatenate(Xs),
                label=np.concatenate(lab),
                subject=np.concatenate(subj),
                domain=np.concatenate(dom),
                trial_id=np.concatenate(tri),
                session=np.concatenate(ses),
                feature_kind="de",
                info=dict(
                    corpus="seedv",
                    channels=list(common),
                    bands=[b[0] for b in band_tuple],
                    window_sec=4.0,
                    source="precomputed_de",
                ),
            )
        )
    return concat(per_subject)


# Raw .cnt path (for the covariance-tangent frontend)
def parse_trial_timestamps(meta_root: str | Path | None = None) -> dict[int, dict]:
    """Parse trial_start_end_timestamp.txt -> {session: {start:[...], end:[...]}}."""
    from ..paths import SEEDV_META_ROOT

    meta_root = Path(meta_root) if meta_root is not None else SEEDV_META_ROOT
    text = (meta_root / "trial_start_end_timestamp.txt").read_text()
    out: dict[int, dict] = {}
    cur = None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"Session\s+(\d+):", line)
        if m:
            cur = int(m.group(1))
            out[cur] = {}
            continue
        if cur is None:
            continue
        m = re.match(r"(start|end)_second:\s*\[(.*)\]", line)
        if m:
            key = m.group(1) + "_second"
            vals = [int(x) for x in m.group(2).split(",") if x.strip()]
            out[cur][key.replace("_second", "")] = vals
    return out


def _cnt_files_for_subject(subject: int, raw_root: Path) -> dict[int, Path]:
    """{session: path} for a subject's 3 .cnt files (session per filename).

    Prefers a `*_repaired.cnt` over the original when both exist (subject 7's
    session 1 ships a repaired copy of a corrupted recording).
    """
    chosen: dict[int, tuple[Path, bool]] = {}
    for f in sorted(raw_root.glob(f"{subject}_*.cnt")):
        m = re.match(rf"{subject}_(\d+)_\d+(_repaired)?\.cnt$", f.name)
        if not m:
            continue
        sess, repaired = int(m.group(1)), m.group(2) is not None
        if sess not in chosen or repaired:
            chosen[sess] = (f, repaired)
    return {s: fp for s, (fp, _) in chosen.items()}


def load_seedv_raw_trials(
    subject: int,
    raw_root: str | Path | None = None,
    de_root: str | Path | None = None,
    meta_root: str | Path | None = None,
    resample_hz: float = 200.0,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    notch_hz: float | None = 50.0,
    neutral: str = "drop",
    montage_order: str = "deap",
    verify_channels: bool = True,
) -> list[Trial]:
    """Read one subject's raw .cnt (3 sessions) into channel-aligned Trials.

    Heavy (~2.5 GB per subject), so features are extracted once and cached.
    Channels are selected by name from the .cnt header; with `verify_channels`,
    a missing common-montage channel raises rather than silently misaligning
    corpora.
    """
    import mne

    from ..paths import SEEDV_RAW_ROOT

    raw_root = Path(raw_root) if raw_root is not None else SEEDV_RAW_ROOT
    mne.set_log_level("ERROR")

    codes = seedv_trial_codes(subject, de_root)
    ts = parse_trial_timestamps(meta_root)
    files = _cnt_files_for_subject(subject, raw_root)
    if not files:
        raise FileNotFoundError(f"no .cnt for subject {subject} under {raw_root}")

    common = ch.common_montage(montage_order)
    trials: list[Trial] = []

    for session, path in sorted(files.items()):
        raw = mne.io.read_raw_cnt(str(path), preload=True)
        # Select the common montage by name: robust to channel order and to a
        # missing non-common channel (subject 7's repaired session 1 lacks FC4,
        # which is outside DEAP∩SEED-V). Only the common channels are required.
        present = {c.lower(): c for c in raw.ch_names}
        missing_common = [c for c in common if c.lower() not in present]
        if verify_channels and missing_common:
            raise AssertionError(
                f"subject {subject} sess {session}: common-montage channels "
                f"missing {missing_common}"
            )
        pick = [present[c.lower()] for c in common if c.lower() in present]
        raw.pick(pick)

        if notch_hz:
            raw.notch_filter(freqs=[notch_hz])
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        if resample_hz:
            raw.resample(resample_hz)
        fs = raw.info["sfreq"]

        # Reorder rows to the common montage.
        sel = ch.channel_indices(tuple(raw.ch_names), common)
        arr = raw.get_data()[sel, :]  # (C_common, T)

        starts = ts[session]["start"]
        ends = ts[session]["end"]
        for i, (s0, s1) in enumerate(zip(starts, ends)):
            global_trial = (session - 1) * SEEDV_TRIALS_PER_SESSION + i
            code = codes[global_trial]
            ybin, keep = seedv_binary_labels(np.array([code]), neutral=neutral)
            if not keep[0]:
                continue  # neutral trial dropped
            a, b = int(s0 * fs), int(s1 * fs)
            seg = arr[:, a:b]
            trials.append(
                Trial(
                    eeg=np.ascontiguousarray(seg, dtype=np.float64),
                    fs=fs,
                    label=int(ybin[0]),
                    subject=subject + SEEDV_SUBJECT_OFFSET,
                    domain=DOMAIN_SEEDV,
                    trial_id=global_trial,
                    session=session,
                    channels=common,
                )
            )
    return trials
