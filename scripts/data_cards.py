"""Write a Markdown data card per corpus to docs/data_cards/.

Usage:
  python scripts/data_cards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from dinq.data import channels as ch
from dinq.data.deap import load_deap_trials, DEAP_FS
from dinq.data.seedv import load_seedv_de, SEEDV_N_SUBJECTS
from dinq.data.labels import SEEDV_SUBJECT_OFFSET
from dinq.features.bands import band_set
from dinq.features.de import deap_de_featureset
from dinq.features.tangent import tangent_dim
from dinq import paths as P

OUT = _REPO / "docs" / "data_cards"


def _balance(label: np.ndarray) -> str:
    """Class balance of the kept labels (label >= 0) as a short string."""
    v = label[label >= 0]
    pos = int((v == 1).sum())
    neg = int((v == 0).sum())
    tot = pos + neg
    return f"{pos} pos / {neg} neg ({pos / tot:.1%} pos)" if tot else "n/a"


def deap_card(bands: str) -> str:
    trials = load_deap_trials()
    subs = sorted({t.subject for t in trials})
    labels = np.array([t.label for t in trials])
    de = deap_de_featureset(trials, bands=bands)
    C = len(ch.common_montage())
    B = len(band_set(bands))
    return f"""# DEAP — data card

- **Source:** `{P.DEAP_ROOT}` (data_preprocessed_python, .dat)
- **Subjects:** {len(subs)} (ids {subs[0]}..{subs[-1]})
- **Trials:** {len(trials)} total, {len(trials)//len(subs)}/subject
- **Sampling:** {DEAP_FS:.0f} Hz; 60 s stimulus (3 s baseline dropped)
- **Label mapping:** valence median-split -> binary; trial balance: {_balance(labels)}
- **Channels:** 32 EEG (Geneva order) -> common montage C={C}
- **Bands ({bands}):** {[b[0] for b in band_set(bands)]}  (delta absent: DEAP is 4-45 Hz)
- **DE feature dim:** C*B = {C*B}
- **Covariance/tangent dim:** B * C(C+1)/2 = {B * tangent_dim(C)}
- **Domain id:** 0
"""


def seedv_card(bands: str) -> str:
    de = load_seedv_de(bands=bands)
    subs = sorted(set(np.unique(de.subject).tolist()))
    C = len(ch.common_montage())
    B = len(band_set(bands))
    raw = "reachable" if P.seedv_raw_available() else "NOT mounted (SFTP)"
    return f"""# SEED-V — data card

- **Source (DE):** `{P.SEEDV_DE_ROOT}` (precomputed DE, 4 s windows)
- **Source (raw):** `{P.SEEDV_RAW_ROOT}` — {raw}
- **Subjects:** {SEEDV_N_SUBJECTS} (offset ids {subs[0]}..{subs[-1]}, +{SEEDV_SUBJECT_OFFSET})
- **Trials:** 45/subject (3 sessions x 15)
- **Native channels:** 62 (10-20) -> common montage C={C}
- **Label mapping:** happy->pos; {{sad,fear,disgust}}->neg; neutral dropped.
  Kept-segment balance: {_balance(de.label)}
- **Bands ({bands}):** {[b[0] for b in band_set(bands)]}
- **DE feature dim:** C*B = {C*B} (channel-major reshape(n,62,5))
- **Covariance/tangent dim:** B * C(C+1)/2 = {B * tangent_dim(C)} (needs raw .cnt)
- **Domain id:** 1
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for bands in ("four",):
        (OUT / "deap.md").write_text(deap_card(bands))
        (OUT / "seedv.md").write_text(seedv_card(bands))
    common = ch.common_montage()
    (OUT / "common_montage.md").write_text(
        "# DEAP ∩ SEED-V common montage\n\n"
        f"- **|C|** = {len(common)}\n- **channels:** {', '.join(common)}\n"
    )
    print(f"[data_cards] wrote {OUT}/deap.md, seedv.md, common_montage.md")


if __name__ == "__main__":
    main()
