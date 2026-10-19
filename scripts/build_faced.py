"""Build the FACED caches: per-band covariances, recentred tangent features and DE.

Usage:
  python scripts/build_faced.py                 # all 123 subjects
  python scripts/build_faced.py --subjects 5    # first 5 (dev)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from dinq.data.faced import load_faced_trials, FACED_N_SUBJECTS
from dinq.features.bands import band_set
from dinq.features.covariance import covset_from_trials, CovarianceSet, is_spd
from dinq.features.de import deap_de_featureset
from dinq.features.featureset import concat
from dinq.features.recenter import recentered_tangent_featureset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=0, help="0=all 123; else first-N (dev)")
    ap.add_argument("--bands", default="four")
    ap.add_argument("--win-sec", type=float, default=4.0)
    ap.add_argument("--hop-sec", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cache = _REPO / "cache"
    (cache / "cov").mkdir(parents=True, exist_ok=True)
    (cache / "de").mkdir(parents=True, exist_ok=True)
    (cache / "tangent").mkdir(parents=True, exist_ok=True)
    subs = list(range(args.subjects if args.subjects else FACED_N_SUBJECTS))
    bands = band_set(args.bands)

    cov_out = cache / "cov" / f"faced_{args.bands}.npz"
    de_out = cache / "de" / f"faced_{args.bands}.npz"
    tan_out = cache / "tangent" / f"faced_recentered_{args.bands}.npz"

    # Covariances and DE, one subject (one .pkl from the zip) at a time.
    if cov_out.exists() and not args.force:
        print(f"[skip] {cov_out}")
        cov = CovarianceSet.load(cov_out)
    else:
        t0 = time.time(); parts_c, parts_d = [], []
        for i, sid in enumerate(subs):
            tr = load_faced_trials(subjects=[sid])  # 24 trials, 30 ch, 250 Hz
            cs = covset_from_trials(tr, bands, win_sec=args.win_sec, hop_sec=args.hop_sec,
                                    corpus="faced")
            parts_c.append(cs)
            parts_d.append(deap_de_featureset(tr, bands=args.bands, win_sec=args.win_sec,
                                              hop_sec=args.hop_sec))
            if (i + 1) % 10 == 0 or i == len(subs) - 1:
                print(f"  [faced] {i+1}/{len(subs)} subjects  ({time.time()-t0:.0f}s)")
        cov = CovarianceSet(
            covs=np.concatenate([p.covs for p in parts_c]),
            **{k: np.concatenate([getattr(p, k) for p in parts_c])
               for k in ("label", "subject", "domain", "trial_id", "session")},
            info=parts_c[0].info,
        )
        cov.save(cov_out)
        print(f"[write] {cov_out}  n_windows={cov.n} shape={cov.covs.shape}")
        de = concat(parts_d)
        if not de_out.exists() or args.force:
            de.save(de_out)
            neg, pos = np.bincount(de.label[de.label >= 0], minlength=2)[:2]
            print(f"[write] {de_out}  n={de.n} dim={de.dim} neg={neg} pos={pos}")

    # SPD check on a random sample of windows.
    idx = np.random.RandomState(0).choice(cov.n, min(4000, cov.n), replace=False)
    spd = is_spd(cov.covs[idx].astype(np.float64))
    print(f"[SPD] sample {len(idx)}: all SPD? {bool(spd.all())} frac={spd.mean():.4f}  "
          f"subjects={len(np.unique(cov.subject))} labels(neg,pos)={np.bincount(cov.label)}")

    # Per-subject recentered tangent features.
    if tan_out.exists() and not args.force:
        print(f"[skip] {tan_out}")
    else:
        print("[recenter] building per-subject recentered tangent (G1) ...")
        t0 = time.time()
        ft = recentered_tangent_featureset(cov)
        ft.save(tan_out)
        print(f"[write] {tan_out}  X={ft.X.shape}  ({time.time()-t0:.0f}s)")
    print("[build_faced] done.")


if __name__ == "__main__":
    main()
