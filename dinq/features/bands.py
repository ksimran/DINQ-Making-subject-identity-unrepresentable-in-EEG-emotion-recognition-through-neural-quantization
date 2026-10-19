"""Frequency bands. Cross-corpus runs use the four bands all corpora share (theta, alpha,
beta, gamma), because DEAP's preprocessed EEG is band-passed at 4-45 Hz."""

from __future__ import annotations

# (name, lo_hz, hi_hz) in the canonical SEED-V DE order.
BANDS_5: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 14.0),
    ("beta", 14.0, 31.0),
    ("gamma", 31.0, 50.0),
)

# Cross-corpus default: drop delta (DEAP preprocessed has none).
BANDS_4: tuple[tuple[str, float, float], ...] = BANDS_5[1:]

# Index of each band within the 5-band DE layout (for slicing SEED-V DE).
BAND_NAME_TO_INDEX5: dict[str, int] = {b[0]: i for i, b in enumerate(BANDS_5)}


def band_set(name: str) -> tuple[tuple[str, float, float], ...]:
    """Resolve a named band set: 'five' (all) or 'four' (cross-corpus default)."""
    if name in ("four", "4", "cross_corpus"):
        return BANDS_4
    if name in ("five", "5", "full"):
        return BANDS_5
    raise ValueError(f"unknown band set {name!r} (use 'four' or 'five')")


def band_indices_in_5(bands: tuple[tuple[str, float, float], ...]) -> list[int]:
    """Column indices (in the 5-band DE layout) for the given band subset."""
    return [BAND_NAME_TO_INDEX5[b[0]] for b in bands]
