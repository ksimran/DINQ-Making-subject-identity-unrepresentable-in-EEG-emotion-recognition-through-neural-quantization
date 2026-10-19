"""Build and cache DEAP/SEED-V features: DE or per-band covariances.

Usage:
  python scripts/build_features.py                      # DE, all subjects
  python scripts/build_features.py features=covariance  # covariances
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from dinq import paths as P
from dinq.features.bands import band_set
from dinq.utils.seed import set_seed


def _summary(name, X_or_cov, label):
    import numpy as np

    neg, pos = np.bincount(label[label >= 0], minlength=2).tolist()[:2]
    print(f"  [{name}] n={len(label)} shape={tuple(X_or_cov.shape)} "
          f"neg={neg} pos={pos} ({pos / max(pos + neg, 1):.1%} pos)")


def build_de(cfg: DictConfig, cache: Path) -> None:
    """Write DE features: computed from DEAP trials, loaded precomputed for SEED-V."""
    from dinq.data.deap import load_deap_trials
    from dinq.data.seedv import load_seedv_de
    from dinq.features.de import deap_de_featureset

    out_dir = cache / "de"
    fc = cfg.features

    if "deap" in cfg.corpora:
        out = out_dir / f"deap_{cfg.bands}.npz"
        if out.exists():
            print(f"[skip] {out}")
        else:
            trials = load_deap_trials(subjects=cfg.deap_subjects, montage_order=cfg.montage_order)
            fs = deap_de_featureset(trials, bands=cfg.bands, win_sec=fc.win_sec,
                                    hop_sec=fc.hop_sec, band_method=fc.band_method)
            fs.save(out)
            _summary("deap-de", fs.X, fs.label)
            print(f"[write] {out}")

    if "seedv" in cfg.corpora:
        out = out_dir / f"seedv_{cfg.bands}.npz"
        if out.exists():
            print(f"[skip] {out}")
        else:
            fs = load_seedv_de(subjects=cfg.seedv_subjects, bands=cfg.bands,
                               neutral=cfg.neutral, montage_order=cfg.montage_order)
            fs.save(out)
            _summary("seedv-de", fs.X, fs.label)
            print(f"[write] {out}")


def build_covariance(cfg: DictConfig, cache: Path) -> None:
    """Write per-band covariances; SEED-V raw .cnt data are read one subject at a time."""
    from dinq.data.deap import load_deap_trials
    from dinq.features.covariance import covset_from_trials

    out_dir = cache / "cov"
    fc = cfg.features
    bands = band_set(cfg.bands)

    if "deap" in cfg.corpora:
        out = out_dir / f"deap_{cfg.bands}.npz"
        if out.exists():
            print(f"[skip] {out}")
        else:
            trials = load_deap_trials(subjects=cfg.deap_subjects, montage_order=cfg.montage_order)
            cs = covset_from_trials(trials, bands, win_sec=fc.win_sec, hop_sec=fc.hop_sec,
                                    estimator=fc.estimator, band_method=fc.band_method,
                                    normalize=fc.normalize, corpus="deap")
            cs.save(out)
            _summary("deap-cov", cs.covs, cs.label)
            print(f"[write] {out}")

    if "seedv" in cfg.corpora:
        out = out_dir / f"seedv_{cfg.bands}.npz"
        if out.exists():
            print(f"[skip] {out}")
        elif not P.seedv_raw_available():
            print(f"[skip] seedv covariance: raw .cnt mount not reachable at "
                  f"{P.SEEDV_RAW_ROOT} (remount SFTP, then re-run).")
        else:
            from dinq.data.seedv import load_seedv_raw_trials
            from dinq.features.covariance import CovarianceSet
            import numpy as np

            subs = cfg.seedv_subjects or list(range(1, 17))
            parts = []
            for sid in subs:
                trials = load_seedv_raw_trials(sid, neutral=cfg.neutral,
                                               montage_order=cfg.montage_order)
                cs = covset_from_trials(trials, bands, win_sec=fc.win_sec, hop_sec=fc.hop_sec,
                                        estimator=fc.estimator, band_method=fc.band_method,
                                        normalize=fc.normalize, corpus="seedv")
                parts.append(cs)
                print(f"  [seedv-cov] subject {sid}: n={cs.n}")
            merged = CovarianceSet(
                covs=np.concatenate([p.covs for p in parts]),
                label=np.concatenate([p.label for p in parts]),
                subject=np.concatenate([p.subject for p in parts]),
                domain=np.concatenate([p.domain for p in parts]),
                trial_id=np.concatenate([p.trial_id for p in parts]),
                session=np.concatenate([p.session for p in parts]),
                info=parts[0].info,
            )
            merged.save(out)
            _summary("seedv-cov", merged.covs, merged.label)
            print(f"[write] {out}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    cache = Path(cfg.cache_dir)
    print(OmegaConf.to_yaml(cfg))
    print(f"[build_features] kind={cfg.features.kind} bands={cfg.bands} cache={cache}")
    if cfg.features.kind == "de":
        build_de(cfg, cache)
    elif cfg.features.kind == "covariance":
        build_covariance(cfg, cache)
    else:
        raise ValueError(f"unknown features.kind {cfg.features.kind!r}")
    print("[build_features] done.")


if __name__ == "__main__":
    main()
