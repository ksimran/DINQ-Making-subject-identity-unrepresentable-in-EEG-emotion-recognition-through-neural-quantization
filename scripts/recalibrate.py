"""Recalibrate and freeze the certificate thresholds on the recentred reference
baselines, before DINQ is probed.

Usage:
  python scripts/recalibrate.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def _load_cov_reference():
    """Load the recentered covariance baseline cells and their point estimates per quantity."""
    cells = [json.load(open(f)) for f in
             glob.glob(str(_REPO / "results/baselines_recentered/cells/*__covariance__*.json"))]
    q = lambda k: np.array([c["quantities"][k]["point"] for c in cells])
    return cells, {
        "domain_linear": q("domain_linear_acc"),
        "domain_deep": q("domain_deep_acc"),
        "residual": q("residual_domain_acc"),
        "conditional": q("conditional_domain_acc"),
        "task": q("task_target_acc"),
    }


def derive():
    """Derive both threshold sets; returns (cells, reference distribution, thresholds)."""
    cells, d = _load_cov_reference()
    print("=== recentered covariance reference distribution (8 cells) ===")
    for k, v in d.items():
        print(f"  {k:14s}: min={v.min():.2f} med={np.median(v):.2f} max={v.max():.2f}  {np.round(v,2)}")

    # Cross-corpus (binary domain, chance 0.5).
    dom_low = 0.60  # chance + 0.10 (near-chance)
    residual_low = 0.60  # chance + 0.10
    # dom_high ("clearly above") marks the high cluster of the reference deep-domain
    # accuracies: their 60th percentile, floored at 0.70.
    dom_high = float(max(0.70, round(np.percentile(d["domain_deep"], 60), 2)))
    # Task floors keyed to the reference cross-corpus transfer distribution.
    task_floor = float(round(max(0.52, np.percentile(d["task"], 25)), 2))
    task_good = float(round(max(task_floor + 0.03, np.percentile(d["task"], 90)), 2))

    cross_corpus = dict(task_floor=task_floor, task_good=task_good, dom_low=dom_low,
                        dom_high=dom_high, residual_low=residual_low,
                        angle_align=0.5, pareto_steep=1.5, deep_linear_gap=0.10,
                        residual_slope_guard=0.65)

    # Cross-subject (multi-class subject recoverability, chance 1/K).
    # Near-chance = 1/K + 0.10 (absolute margin, as in the binary case). A
    # permutation null gives the exact per-K cutoff; the large gap between the
    # reference and chance makes the verdict robust to the exact margin.
    cross_subject = dict(
        subjrecov_margin=0.10,  # near-chance cutoff = 1/K + margin
        task_floor=0.55,  # chance 0.5 + margin
        task_good=0.62,
        recalib_note="subject-recov cutoff = 1/K + subjrecov_margin; validate with permutation null at DINQ time",
    )

    return cells, d, dict(setting="recentered_tangent", cross_corpus=cross_corpus,
                          cross_subject=cross_subject)


def classify_cov(cell, t):
    """Coarse certificate category of a reference cell under thresholds `t`."""
    q = cell["quantities"]
    task = q["task_target_acc"]["point"]; dl = q["domain_linear_acc"]["point"]
    dd = q["domain_deep_acc"]["point"]; res = q["residual_domain_acc"]["point"]
    cond = q["conditional_domain_acc"]["point"]
    if task < t["task_floor"]:
        return "C5_Null"
    if dl < t["dom_low"] and dd < t["dom_low"] and res < t["residual_low"] and cond < t["dom_low"]:
        return "C1_Invariant"
    if res > t["dom_high"]:
        return "C4_Entangled"
    if dd > t["dom_high"]:
        return "C3/C2_domain"
    return "Unclassified"


def main():
    cells, d, thr = derive()
    cc = thr["cross_corpus"]
    print("\n=== FROZEN thresholds (recentered) ===")
    print("  cross-corpus:", {k: cc[k] for k in ("task_floor","task_good","dom_low","dom_high","residual_low")})
    print("  cross-subject:", {k: thr["cross_subject"][k] for k in ("subjrecov_margin","task_floor","task_good")})

    print("\n=== reference cov baselines under frozen thresholds (expect NONE C1) ===")
    cats = [classify_cov(c, cc) for c in cells]
    for c, cat in sorted(zip(cells, cats), key=lambda x: x[0]["model"]):
        print(f"  {c['direction']:12} {c['model']:10} -> {cat}")
    n_c1 = sum(c == "C1_Invariant" for c in cats)
    print(f"  C1 count among reference = {n_c1} (must be 0: the bar)")

    # +/-10% sensitivity: perturb each cross-corpus threshold, count category flips.
    print("\n=== +/-10% sensitivity (cross-corpus) ===")
    base = cats
    flips = 0; total = 0
    for key in ("task_floor", "task_good", "dom_low", "dom_high", "residual_low"):
        for mult in (0.9, 1.1):
            t2 = dict(cc); t2[key] = round(cc[key] * mult, 3)
            c2 = [classify_cov(c, t2) for c in cells]
            f = sum(a != b for a, b in zip(base, c2)); flips += f; total += len(cells)
    print(f"  category flips under +/-10% on 5 thresholds: {flips}/{total} "
          f"({flips/total:.1%}) -> {'STABLE' if flips/total < 0.15 else 'SENSITIVE'}")

    out = _REPO / "configs" / "thresholds_recentered.yaml"
    out.write_text(yaml.safe_dump({"thresholds_recentered": thr}, sort_keys=False))
    print(f"\n[frozen] {out}")


if __name__ == "__main__":
    main()
