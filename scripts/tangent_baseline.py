"""Print the cross-subject certificate of the recentred tangent baseline (no DINQ).

Usage:
  python scripts/tangent_baseline.py --corpus faced --splits 4
  python scripts/tangent_baseline.py --corpus seedv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from dinq.features.featureset import FeatureSet
from dinq.eval.blockbootstrap import trial_group_ids
from dinq.eval.dinq_probe import emotion_unseen, subject_recoverability, perm_null_ceiling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="faced")
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--cap", type=int, default=12000)
    ap.add_argument("--n-perm", type=int, default=10)
    args = ap.parse_args()

    rec = FeatureSet.load(_REPO / "cache" / "tangent" / f"{args.corpus}_recentered_four.npz")
    uid = trial_group_ids(rec.subject, rec.session, rec.trial_id)
    print(f"=== {args.corpus} recentered-tangent BASELINE (no DINQ), {args.splits} splits ===", flush=True)

    emos, held_hgb, held_lin, ceils = [], [], [], []
    for split in range(args.splits):
        subs = np.unique(rec.subject); rng = np.random.RandomState(split); rng.shuffle(subs)
        n_test = max(2, int(round(len(subs) * args.test_frac)))
        test = set(subs[:n_test].tolist())
        test_mask = np.isin(rec.subject, list(test)); train_mask = ~test_mask
        emo = emotion_unseen(rec.X, rec.label, uid, train_mask, test_mask)
        rh = subject_recoverability(rec.X, rec.subject, uid, test_mask, cap=args.cap, rs=split)
        ch = perm_null_ceiling(rec.X, rec.subject, uid, test_mask, cap=args.cap, n_perm=args.n_perm, rs=split)
        emos.append(emo); held_hgb.append(rh["hgb"]); held_lin.append(rh["linear"]); ceils.append(ch)
        print(f"  split{split} (held-out K={rh['K']}): emotion(unseen)={emo:.3f}  "
              f"held-out subj-recov HGB={rh['hgb']:.3f} lin={rh['linear']:.3f} (ceiling {ch:.3f})", flush=True)

    print(f"\nBASELINE emotion(unseen)      = {np.mean(emos):.3f} +/- {np.std(emos):.3f}")
    print(f"BASELINE held-out subj-recov  = HGB {np.mean(held_hgb):.3f} / lin {np.mean(held_lin):.3f} "
          f"(ceiling {np.mean(ceils):.3f})")


if __name__ == "__main__":
    main()
