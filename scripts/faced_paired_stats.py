"""Paired per-split tests on FACED (t-test, Wilcoxon, t-based CIs) for the comparisons
reported in the paper; writes results/faced_epochs_check/paired_stats.json.

Usage:  python scripts/faced_paired_stats.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
RES = _REPO / "results"


def _cells(pattern):
    """Load the cell JSONs matching `pattern` under results/, keyed by split."""
    return {c["split"]: c for c in (json.load(open(f)) for f in glob.glob(str(RES / pattern)))}


def max3(c):
    r = c["recov_heldout"]; return max(r["linear"], r["mlp"], r["hgb"])


def max_deep(c):  # earlier max-over-probes definition: MLP and HGB only
    r = c["recov_heldout"]; return max(r["mlp"], r["hgb"])


def ci95(v):
    v = np.asarray(v, float)
    h = float(stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size))
    return [float(v.mean()), float(v.mean() - h), float(v.mean() + h)]


def paired(a, b):
    """Paired comparison of a minus b over splits: difference CI, paired t-test, Wilcoxon."""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    t = stats.ttest_rel(a, b); w = stats.wilcoxon(a, b)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "diff_per_split": d.round(5).tolist(),
            "n_a_greater": int((d > 0).sum()), "n": int(d.size), "mean_diff_ci95": ci95(d),
            "t": float(t.statistic), "t_p_two_sided": float(t.pvalue),
            "wilcoxon_p_two_sided": float(w.pvalue)}


def main():
    dinq = {r["split"]: r for r in json.loads((RES / "faced_epochs_check" / "like_for_like_60ep.json")
                                              .read_text())["rows"]}
    cont = _cells("dinq_sweep_faced/cells/continuous__split*.json")
    rec = _cells("baselines_faced_subject/cells/recentering__split*.json")
    old = _cells("dinq_sweep_faced/cells/vq512_conU__split*.json")  # archived 80 ep / cap 12000
    S = sorted(dinq)
    assert S == sorted(cont) == sorted(rec) == sorted(old), "split sets differ"
    emo = lambda d: [d[s]["emotion_unseen"] for s in S]
    out = {
        "a_continuous_minus_dinq_max3": paired([max3(cont[s]) for s in S], [max3(dinq[s]) for s in S]),
        "b_emotion_dinq_minus_continuous": paired(emo(dinq), emo(cont)),
        "b_emotion_dinq_minus_recentring": paired(emo(dinq), emo(rec)),
        "c_recentring_emotion_ci95": ci95(emo(rec)),
        "d_archival_80ep_continuous_minus_dinq_max3": paired([max3(cont[s]) for s in S], [max3(old[s]) for s in S]),
        "d_archival_80ep_continuous_minus_dinq_max_mlp_hgb": paired([max_deep(cont[s]) for s in S],
                                                                   [max_deep(old[s]) for s in S]),
    }
    (RES / "faced_epochs_check" / "paired_stats.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if isinstance(v, dict):
            m = v["mean_diff_ci95"]
            print(f"{k:52} {v['n_a_greater']}/{v['n']}  diff {m[0]:+.4f} [{m[1]:+.4f}, {m[2]:+.4f}]  "
                  f"t p={v['t_p_two_sided']:.4f}  Wilcoxon p={v['wilcoxon_p_two_sided']:.4f}")
        else:
            print(f"{k:52} {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]")
    print(f"[written] {RES / 'faced_epochs_check' / 'paired_stats.json'}")


if __name__ == "__main__":
    main()
