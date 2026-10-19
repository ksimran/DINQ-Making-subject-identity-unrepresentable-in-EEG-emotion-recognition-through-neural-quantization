"""Data and cache paths; each can be overridden with a DINQ_* environment variable."""

from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# All corpora live under one root (default: ./data, gitignored). Override the base
# via DINQ_DATA_ROOT, or any individual path via its own DINQ_* var.
DATA_ROOT = Path(os.environ.get("DINQ_DATA_ROOT", str(_REPO / "data")))


def _p(env: str, default) -> Path:
    return Path(os.environ[env]) if env in os.environ else Path(default)


# DEAP preprocessed .dat pickles (32 subjects, data_preprocessed_python, 128 Hz).
DEAP_ROOT = _p("DINQ_DEAP_ROOT", DATA_ROOT / "DEAP_dataset")

# SEED-V precomputed DE features (16 subjects, 62ch x 5 bands, 4 s windows).
SEEDV_DE_ROOT = _p("DINQ_SEEDV_DE_ROOT", DATA_ROOT / "SEED_V" / "EEG_DE_features")

# SEED-V raw EEG (.cnt, Neuroscan); read once, then cached as features.
SEEDV_RAW_ROOT = _p("DINQ_SEEDV_RAW_ROOT", DATA_ROOT / "SEED_V" / "EEG_raw")

# SEED-V metadata (channel order, trial timestamps, labels/stimuli order).
SEEDV_META_ROOT = _p("DINQ_SEEDV_META_ROOT", DATA_ROOT / "SEED_V")

# FACED (123 subjects, 32 ch), the large-N cross-subject corpus.
FACED_ROOT = _p("DINQ_FACED_ROOT", DATA_ROOT / "Faced")
# Preprocessed CLISA .pkl (28 videos x 32 ch x 7500 samples at 250 Hz), read
# per subject directly from the zip (no full extraction).
FACED_CLISA_ZIP = _p("DINQ_FACED_CLISA_ZIP", FACED_ROOT / "Clisa_data.zip")

# The companion framework (probe suite from the authors' companion study),
# MIT-licensed:
# https://github.com/ksimran/Diagnosing-Domain-Adaptation-Failures-in-EEG-Representations
# Default: cloned next to this repository; override via DINQ_PAPER1_REPO.
PAPER1_REPO = _p(
    "DINQ_PAPER1_REPO",
    _REPO.parent / "Diagnosing-Domain-Adaptation-Failures-in-EEG-Representations",
)

# Local cache for extracted features / splits (gitignored).
CACHE_ROOT = _p("DINQ_CACHE_ROOT", str(_REPO / "cache"))


def deap_available() -> bool:
    return DEAP_ROOT.exists() and any(DEAP_ROOT.glob("s*.dat"))


def seedv_de_available() -> bool:
    return SEEDV_DE_ROOT.exists() and any(SEEDV_DE_ROOT.glob("*_123.npz"))


def seedv_raw_available() -> bool:
    return SEEDV_RAW_ROOT.exists() and any(SEEDV_RAW_ROOT.glob("*.cnt"))
