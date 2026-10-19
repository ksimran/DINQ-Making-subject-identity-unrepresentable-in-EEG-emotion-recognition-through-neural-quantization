"""Channel montages of DEAP, SEED-V and FACED and their computed 10-20 intersections."""

from __future__ import annotations

# SEED-V: 62 channels, order per channel_62_pos.locs (== DE-feature channel order
# and the .cnt recording order). The DE bands (delta/theta/alpha/beta/gamma) are
# handled in dinq.features.
SEEDV_CHANNELS: tuple[str, ...] = (
    "Fp1", "Fpz", "Fp2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8",
    "CB1", "O1", "Oz", "O2", "CB2",
)

# FACED: raw .bdf channel order (32), validated against sub000/data.bdf header.
# Indices 16-17 are A1/A2 (mastoid references, dropped for EEG); old 10-20 names
# T3/T4/T5/T6 == modern T7/T8/P7/P8. Preprocessed .pkl keeps this 32-ch order at
# 250 Hz. FACED has no AF3/AF4, so DEAP-cap-FACED = 30 (DEAP minus AF3/AF4).
FACED_BDF_CHANNELS: tuple[str, ...] = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2",
)
FACED_MASTOIDS: tuple[str, ...] = ("A1", "A2")
_OLD_TO_NEW_1020: dict[str, str] = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# FACED 30 EEG channels in .pkl order (mastoids removed, names modernized).
FACED_CHANNELS: tuple[str, ...] = tuple(
    _OLD_TO_NEW_1020.get(c, c) for c in FACED_BDF_CHANNELS if c not in FACED_MASTOIDS
)

# DEAP: first 32 channels of data_preprocessed_python are EEG, Geneva order.
DEAP_CHANNELS: tuple[str, ...] = (
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
)


def _normalize(name: str) -> str:
    """Case/space-insensitive channel key for robust matching across corpora."""
    return name.strip().lower()


def common_montage(
    order: str = "deap",
) -> tuple[str, ...]:
    """Return the DEAP-cap-SEED-V channel names in a stable canonical order.

    `order` ("deap" or "seedv") selects which corpus's ordering is used; the set
    is identical either way. DEAP order is the default because it is the smaller
    montage and keeps the common index contiguous with DEAP's layout.
    """
    seedv_keys = {_normalize(c) for c in SEEDV_CHANNELS}
    deap_keys = {_normalize(c) for c in DEAP_CHANNELS}
    common_keys = deap_keys & seedv_keys

    ref = DEAP_CHANNELS if order == "deap" else SEEDV_CHANNELS
    if order not in ("deap", "seedv"):
        raise ValueError(f"order must be 'deap' or 'seedv', got {order!r}")
    return tuple(c for c in ref if _normalize(c) in common_keys)


MONTAGES: dict[str, tuple[str, ...]] = {
    "deap": DEAP_CHANNELS,
    "seedv": SEEDV_CHANNELS,
    "faced": FACED_CHANNELS,
}


def intersect(corpora: tuple[str, ...], order: str | None = None) -> tuple[str, ...]:
    """Common 10-20 channels across `corpora`, in `order`'s corpus ordering.

    E.g. intersect(("deap","faced")) -> DEAP-cap-FACED (30, DEAP minus AF3/AF4);
    intersect(("deap","seedv","faced")) -> the 3-way common montage (30).
    """
    order = order or corpora[0]
    keysets = [{_normalize(c) for c in MONTAGES[k]} for k in corpora]
    common = set.intersection(*keysets)
    return tuple(c for c in MONTAGES[order] if _normalize(c) in common)


def channel_indices(
    corpus_channels: tuple[str, ...],
    common: tuple[str, ...],
) -> list[int]:
    """Indices into `corpus_channels` that select `common`, in `common` order.

    Used to slice a corpus's native (C_native x ...) feature array down to the
    common montage in the exact canonical order, so DEAP and SEED-V features are
    channel-aligned before any covariance / DE comparison.
    """
    lookup = {_normalize(c): i for i, c in enumerate(corpus_channels)}
    missing = [c for c in common if _normalize(c) not in lookup]
    if missing:
        raise KeyError(f"common channels absent from corpus montage: {missing}")
    return [lookup[_normalize(c)] for c in common]


def deap_common_indices(order: str = "deap") -> list[int]:
    """Indices into DEAP's 32 EEG channels selecting the common montage."""
    return channel_indices(DEAP_CHANNELS, common_montage(order))


def seedv_common_indices(order: str = "deap") -> list[int]:
    """Indices into SEED-V's 62 channels selecting the common montage."""
    return channel_indices(SEEDV_CHANNELS, common_montage(order))


def assert_matches_names(
    constant: tuple[str, ...],
    actual_names: list[str],
    corpus: str,
) -> None:
    """Raise if a hard-coded montage disagrees with channel names read from data."""
    if len(constant) != len(actual_names):
        raise AssertionError(
            f"{corpus}: constant has {len(constant)} channels, data has "
            f"{len(actual_names)}"
        )
    mismatch = [
        (i, c, a)
        for i, (c, a) in enumerate(zip(constant, actual_names))
        if _normalize(c) != _normalize(a)
    ]
    if mismatch:
        raise AssertionError(f"{corpus}: channel order mismatch at {mismatch[:8]}")
