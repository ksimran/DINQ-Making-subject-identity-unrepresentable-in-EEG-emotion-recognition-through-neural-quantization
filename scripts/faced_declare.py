"""Freeze the FACED task floor from an emotion-label permutation null, then label DINQ
against the floor and the subject-recoverability ceiling.

Usage:
  python scripts/faced_declare.py                       # freeze floor + declare
  python scripts/faced_declare.py --config vq512_conU --floor-splits 4 --n-perm 25
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from dinq.features.featureset import FeatureSet
from dinq.eval.blockbootstrap import trial_group_ids
from dinq.eval.dinq_probe import emotion_unseen, task_perm_null_ceiling


def ci95(x):
    """Mean and normal-approximation 95% CI (mean +/- 1.96 SE) over the finite values."""
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    m, s = x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0
    half = 1.96 * s / np.sqrt(len(x))
    return m, m - half, m + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="faced")
    ap.add_argument("--config", default="vq512_conU")
    ap.add_argument("--floor-splits", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=25)
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    rec = FeatureSet.load(_REPO / "cache" / "tangent" / f"{args.corpus}_recentered_four.npz")
    uid = trial_group_ids(rec.subject, rec.session, rec.trial_id)

    # Stage 1: task permutation-null floor on the baseline, then frozen.
    print(f"=== STAGE 1: task permutation-null floor ({args.corpus} recentered tangent baseline) ===")
    floors, base_emos = [], []
    for split in range(args.floor_splits):
        # Same seeded subject shuffle as dinq_probe.subject_split, so the splits match the DINQ cells.
        subs = np.unique(rec.subject); rng = np.random.RandomState(split); rng.shuffle(subs)
        n_test = max(2, int(round(len(subs) * args.test_frac)))
        test = set(subs[:n_test].tolist())
        test_mask = np.isin(rec.subject, list(test)); train_mask = ~test_mask
        f = task_perm_null_ceiling(rec.X, rec.label, uid, train_mask, test_mask, n_perm=args.n_perm, rs=split)
        e = emotion_unseen(rec.X, rec.label, uid, train_mask, test_mask)
        floors.append(f); base_emos.append(e)
        print(f"  split{split}: task-null 95th-pct={f:.3f}  baseline emotion(unseen)={e:.3f}")
    task_floor = float(np.nanmax(floors))  # conservative: max across splits
    base_emo_m, base_emo_lo, _ = ci95(base_emos)
    print(f"  FROZEN task floor = {task_floor:.3f} (max across splits); baseline emotion {base_emo_m:.3f}")
    frozen = {"faced_recentered": {"task_floor_permnull": round(float(task_floor), 4),
                                   "baseline_emotion_unseen": round(float(base_emo_m), 4),
                                   "n_floor_splits": int(args.floor_splits), "n_perm": int(args.n_perm)}}
    (_REPO / "configs" / "thresholds_faced.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False))
    print("  [frozen] configs/thresholds_faced.yaml")

    # Stage 2: DINQ cells across splits.
    cells = [json.loads(Path(p).read_text()) for p in
             glob.glob(str(_REPO / "results/dinq_sweep_faced/cells" / f"{args.config}__split*.json"))]
    if not cells:
        print(f"\n[!] no DINQ cells for {args.config}; run the sweep first."); return
    emo = [c["emotion_unseen"] for c in cells]
    recov = [c["recov_heldout"]["hgb"] for c in cells]
    ceil = [c["ceiling_heldout"] for c in cells]
    em, elo, ehi = ci95(emo); rm, rlo, rhi = ci95(recov); cm, _, _ = ci95(ceil)
    print(f"\n=== STAGE 2: DINQ {args.config} on {args.corpus}, {len(cells)} splits ===")
    print(f"  emotion(unseen)          = {em:.3f}  95%CI [{elo:.3f},{ehi:.3f}]")
    print(f"  held-out subj-recov HGB  = {rm:.3f}  95%CI [{rlo:.3f},{rhi:.3f}]")
    print(f"  subject perm-null ceiling= {cm:.3f}   (task floor {task_floor:.3f})")

    # Stage 3: declaration.
    print("\n=== STAGE 3: declaration (recentering held fixed for both) ===")
    print(f"  BASELINE (recentering only): emotion {base_emo_m:.3f}")
    print(f"  DINQ:                        emotion {em:.3f}, held-out HGB {rm:.3f}")
    emotion_above = elo > task_floor
    recov_at_ceiling = rlo <= cm + 0.005  # CI lower end at or within 0.005 of the ceiling
    recov_overlaps = rlo <= cm <= rhi
    if emotion_above and (recov_at_ceiling or recov_overlaps):
        verdict = "CLEAN DOMAIN-GENERALIZING C1-ANALOG"
    elif emotion_above:
        verdict = "NEAR-C1 with a small characterized residual (emotion clears floor; subject-recov CI above ceiling)"
    else:
        verdict = "emotion CI does not clear the task-null floor — investigate"
    print(f"  emotion CI above task-null floor? {emotion_above}")
    print(f"  held-out subject-recov CI at/overlapping ceiling? {recov_at_ceiling or recov_overlaps}")
    print(f"\n  >>> {verdict}")
    if len(cells) < 8:
        print(f"\n  (CIs from {len(cells)} splits — for the headline, run more splits:")
        print(f"   python scripts/dinq_sweep.py --corpus faced --configs {args.config} --splits 8 "
              f"--epochs 60 --cap 8000 --gpus 0,1 --out results/dinq_sweep_faced )")


if __name__ == "__main__":
    main()
