"""Binary emotion labels and the domain and subject id conventions shared by all corpora."""

from __future__ import annotations

import numpy as np

# Domain (dataset) ids.
DOMAIN_DEAP = 0
DOMAIN_SEEDV = 1
DOMAIN_FACED = 2

# Subject-id offsets so ids stay disjoint across corpora (DEAP 1..32, SEED-V
# 33..48, FACED 100..222).
SEEDV_SUBJECT_OFFSET = 32
FACED_SUBJECT_OFFSET = 100

# SEED-V integer emotion codes -> human label.
SEEDV_CODE_TO_NAME = {0: "disgust", 1: "fear", 2: "sad", 3: "neutral", 4: "happy"}

# SEED-V 5-class -> binary. neutral (3) is intentionally absent -> dropped.
SEEDV_CODE_TO_BINARY = {
    4: 1,  # happy -> positive
    2: 0,  # sad -> negative
    1: 0,  # fear -> negative
    0: 0,  # disgust -> negative
    # 3 (neutral) -> drop / hold out (handled by seedv_binary_labels)
}
SEEDV_NEUTRAL_CODE = 3

# Binary emotion axis convention (single axis, unlike DEAP's 2-axis valence/arousal).
POSITIVE = 1
NEGATIVE = 0


def seedv_binary_labels(
    codes: np.ndarray,
    neutral: str = "drop",
) -> tuple[np.ndarray, np.ndarray]:
    """Map SEED-V 5-class codes to binary, returning (labels, keep_mask).

    `neutral='drop'` marks neutral segments as not-kept (keep_mask False) so the
    caller removes them; `neutral='hold_out'` behaves identically for the binary
    training label but the caller may route neutrals to a separate 3-way check.
    Returned `labels` is int64 with a placeholder (-1) at dropped positions.
    """
    codes = np.asarray(codes).astype(int)
    labels = np.full(codes.shape, -1, dtype=np.int64)
    keep = np.ones(codes.shape, dtype=bool)
    for c in np.unique(codes):
        if c == SEEDV_NEUTRAL_CODE:
            keep[codes == c] = False
            continue
        if c not in SEEDV_CODE_TO_BINARY:
            raise ValueError(f"unexpected SEED-V code {c}")
        labels[codes == c] = SEEDV_CODE_TO_BINARY[c]
    if neutral not in ("drop", "hold_out"):
        raise ValueError("neutral must be 'drop' or 'hold_out'")
    return labels, keep


def deap_binary_valence(
    valence: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    """Binarize DEAP valence (1-9): above the cut -> positive (1), else negative (0).

    `threshold=None` cuts at the median of `valence`; `threshold=5.0` gives a
    fixed-threshold protocol (e.g. SCMM's).
    """
    valence = np.asarray(valence, dtype=np.float64)
    cut = float(np.median(valence)) if threshold is None else float(threshold)
    return (valence > cut).astype(np.int64)


def offset_seedv_subjects(subject_ids: np.ndarray) -> np.ndarray:
    """Shift SEED-V subject ids (1..16) into 33..48 so they stay disjoint from DEAP."""
    return np.asarray(subject_ids, dtype=np.int64) + SEEDV_SUBJECT_OFFSET
