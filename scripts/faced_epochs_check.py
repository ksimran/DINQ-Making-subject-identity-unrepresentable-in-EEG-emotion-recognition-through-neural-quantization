"""Matched DINQ row on FACED (60 epochs, cap 8000): emotion probed on the saved
faced_permtest z_c, joined with its per-probe subject recoverability.

Usage:  python scripts/faced_epochs_check.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def ci95(vals):
    """Mean and t-based 95% CI; a single value is returned as (m, m, m)."""
    v = np.asarray(vals, float)
    m = float(v.mean())
    if v.size < 2:
        return m, m, m
    from scipy import stats
    h = float(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--permtest", default=str(_REPO / "results" / "faced_permtest"))
    ap.add_argument("--out", default=str(_REPO / "results" / "faced_epochs_check"))
    args = ap.parse_args()

    from dinq.features.featureset import FeatureSet
    from dinq.eval.dinq_probe import emotion_unseen, subject_split
    from dinq.eval.blockbootstrap import trial_group_ids

    pt, out = Path(args.permtest), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = FeatureSet.load(_REPO / "cache" / "tangent" / "faced_recentered_four.npz")
    uid = trial_group_ids(rec.subject, rec.session, rec.trial_id)

    rows = []
    for sj in sorted(pt.glob("split*.json"), key=lambda p: int(p.stem[5:])):
        s = json.loads(sj.read_text())
        split = int(s["split"])
        zc = np.load(pt / f"zc_split{split}.npz")["zc"]
        train_mask, test_mask, n_train, n_test = subject_split(rec, split, 0.25)
        obs = s["observed"]
        row = {"split": split, "emotion_unseen": emotion_unseen(zc, rec.label, uid, train_mask, test_mask),
               "recov_heldout": {k: obs[k] for k in ("linear", "mlp", "hgb")},
               "recov_heldout_max": float(np.nanmax([obs["linear"], obs["mlp"], obs["hgb"]])),
               "K": obs["K"], "n_train_subjects": n_train, "n_test_subjects": n_test}
        rows.append(row)
        print(f"[60ep/cap8000] split{split}: emo={row['emotion_unseen']:.3f} "
              f"lin={obs['linear']:.3f} max={row['recov_heldout_max']:.3f}")

    sweep = [json.load(open(f)) for f in sorted(glob.glob(str(_REPO / "results" / "dinq_sweep_faced" / "cells" /
                                                            "vq512_conU__split*.json")))]
    summary = {}
    for name, cells, emo, lin, mx in [
        ("like_for_like_60ep_cap8000", rows, lambda c: c["emotion_unseen"],
         lambda c: c["recov_heldout"]["linear"], lambda c: c["recov_heldout_max"]),
        ("table_main_sweep_epochs_unconfirmed", sweep, lambda c: c["emotion_unseen"],
         lambda c: c["recov_heldout"]["linear"],
         lambda c: max(c["recov_heldout"][p] for p in ("linear", "mlp", "hgb")))]:
        summary[name] = {"n_splits": len(cells),
                         "emotion_unseen": ci95([emo(c) for c in cells]),
                         "recov_linear": ci95([lin(c) for c in cells]),
                         "recov_max": ci95([mx(c) for c in cells]),
                         "per_split_emotion": [round(emo(c), 4) for c in cells],
                         "per_split_max": [round(mx(c), 4) for c in cells]}
    (out / "like_for_like_60ep.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))

    print(f"\n{'row':40}{'emotion':24}{'linear':24}{'max over probes':24}")
    for name, sm in summary.items():
        f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
        print(f"{name:40}{f(sm['emotion_unseen']):24}{f(sm['recov_linear']):24}{f(sm['recov_max']):24}")
    print(f"\n[written] {out / 'like_for_like_60ep.json'}")


if __name__ == "__main__":
    main()
