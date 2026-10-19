"""FACED permutation test: is DINQ's held-out subject recoverability distinguishable
from a pooled subject-label permutation null?

Usage:
  python scripts/faced_permtest.py --n-perm 125 --gpus 0,1
  python scripts/faced_permtest.py --analyze-only   # re-do tests+figure from cache
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

CFG = {"quantizer": "vq", "vq_codebook": 512, "d_r": 16, "contrast_on": "u"}
CORPUS = "faced"
TEST_FRAC = 0.25
EPOCHS = 60


def _split_masks(rec, split):
    """(test_mask, train_mask) for a split; same shuffle as dinq_probe.subject_split."""
    subs = np.unique(rec.subject)
    rng = np.random.RandomState(split); rng.shuffle(subs)
    n_test = max(2, int(round(len(subs) * TEST_FRAC)))
    test = set(subs[:n_test].tolist())
    test_mask = np.isin(rec.subject, list(test))
    return test_mask, ~test_mask


def process_split(split, args, device, out):
    """Train (or load cached) z_c for one split, then save observed recoverability and its null."""
    import torch
    from dinq.features.featureset import FeatureSet
    from dinq.models.minimal_dinq import DINQConfig
    from dinq.eval.dinq_probe import (train_minimal_dinq, extract_zc, subject_recoverability,
                                      perm_null_distribution)
    from dinq.eval.blockbootstrap import trial_group_ids

    cell = out / f"split{split}.json"
    if cell.exists() and not args.force:
        print(f"[skip] split{split}"); return
    rec = FeatureSet.load(_REPO / "cache" / "tangent" / f"{CORPUS}_recentered_four.npz")
    test_mask, train_mask = _split_masks(rec, split)
    uid = trial_group_ids(rec.subject, rec.session, rec.trial_id)

    zc_path = out / f"zc_split{split}.npz"
    if zc_path.exists():
        zc = np.load(zc_path)["zc"]
    else:
        torch.manual_seed(split); np.random.seed(split)
        cfg = DINQConfig(in_dim=rec.dim, **CFG)
        m = train_minimal_dinq(rec.X[train_mask], rec.label[train_mask], cfg,
                               epochs=EPOCHS, device=device)
        zc = extract_zc(m, rec.X, device)
        np.savez_compressed(zc_path, zc=zc)

    obs = subject_recoverability(zc, rec.subject, uid, test_mask, cap=args.cap, rs=split)
    obs_max = float(np.nanmax([obs["linear"], obs["mlp"], obs["hgb"]]))
    null_max, null_by = perm_null_distribution(zc, rec.subject, uid, test_mask,
                                               cap=args.cap, n_perm=args.n_perm, rs=split)
    rec_d = {
        "split": split, "K": obs["K"], "chance": 1.0 / obs["K"],
        "observed": obs, "observed_max": obs_max, "observed_hgb": obs["hgb"],
        "null_max": [round(float(v), 5) for v in null_max],
        "null_hgb": [round(float(v), 5) for v in null_by["hgb"]],
        "n_perm": int(len(null_max)),
        "ceiling95_max": float(np.nanpercentile(null_max, 95)),
        "ceiling95_hgb": float(np.nanpercentile(null_by["hgb"], 95)),
    }
    tmp = cell.with_suffix(".tmp"); tmp.write_text(json.dumps(rec_d)); tmp.replace(cell)
    print(f"[done] split{split}: obs_max={obs_max:.4f} obs_hgb={obs['hgb']:.4f} "
          f"null_max95={rec_d['ceiling95_max']:.4f} null_max_mean={np.nanmean(null_max):.4f} "
          f"(n_perm={rec_d['n_perm']})")


def worker(gpu, splits, args, out):
    import torch
    device = "cpu"
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu); device = f"cuda:{gpu}"
    for s in splits:
        try:
            process_split(s, args, device, out)
        except Exception:
            print(f"[FAIL] split{s}:\n{traceback.format_exc()}")


def analyze(out, args):
    """Pool the per-split nulls, run the tests, write verdict.json and the figure."""
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted(out.glob("split*.json"), key=lambda p: int(p.stem[5:]))
    data = [json.loads(p.read_text()) for p in cells]
    metric = args.metric  # "max" (max over probes, the certificate quantity) or "hgb"
    obs = np.array([d[f"observed_{metric}"] for d in data])
    null_pooled = np.concatenate([np.array(d[f"null_{metric}"], float) for d in data])
    null_pooled = null_pooled[np.isfinite(null_pooled)]
    chance = float(np.mean([d["chance"] for d in data]))

    n = len(obs); obs_mean = float(obs.mean()); obs_sd = float(obs.std(ddof=1))
    tci = stats.t.ppf(0.975, n - 1) * obs_sd / np.sqrt(n)
    obs_lo, obs_hi = obs_mean - tci, obs_mean + tci
    null_mean = float(null_pooled.mean()); null_95 = float(np.percentile(null_pooled, 95))
    null_hi = float(np.percentile(null_pooled, 97.5))

    # (1) Permutation p-value: fraction of the pooled null at or above the observed
    # mean, with the +1 correction.
    p_perm = float((np.sum(null_pooled >= obs_mean) + 1) / (len(null_pooled) + 1))
    # Per split: is the observed value above that split's own null 95th percentile?
    # The count is tested against the 5% expected under the null (binomial).
    per_split_above = [d[f"observed_{metric}"] > np.percentile(np.array(d[f"null_{metric}"], float), 95)
                       for d in data]
    n_above = int(np.sum(per_split_above))
    p_binom = float(stats.binomtest(n_above, n, 0.05, alternative="greater").pvalue)
    # (2) One-sided Mann-Whitney U: observed per-split values vs pooled null values.
    mw = stats.mannwhitneyu(obs, null_pooled, alternative="greater")
    # Bootstrap 95% CI of the difference in means (observed minus null).
    rng = np.random.default_rng(0)
    diffs = [obs[rng.integers(0, n, n)].mean() - null_pooled[rng.integers(0, len(null_pooled), len(null_pooled))].mean()
             for _ in range(5000)]
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5]); d_mean = obs_mean - null_mean
    # (3) CI overlap: does the observed-mean 95% CI clear the null's 97.5th percentile?
    # Reported as a secondary descriptor, not a criterion: with few splits the
    # observed CI is wide from split heterogeneity and the max-over-probes null has a
    # heavy upper tail, so tail overlap alone does not contradict tests (1) and (2).
    ci_separated = obs_lo > null_hi

    # Distinguishability is decided by the two distributional tests (1) and (2).
    distinguishable = (p_perm < 0.05) and (mw.pvalue < 0.05)
    residual_vs_chance = obs_mean - chance
    residual_vs_ceiling = obs_mean - null_95
    if distinguishable:
        label = "NEAR-C1 (DG) with a small characterized residual"
    else:
        label = "CLEAN DG C1-analog (subject-recoverability indistinguishable from chance)"

    verdict = {
        "metric": metric, "n_splits": n, "n_perm_pooled": int(len(null_pooled)),
        "chance_1_over_K": chance,
        "observed_mean": obs_mean, "observed_ci95": [obs_lo, obs_hi],
        "observed_per_split": [round(float(x), 4) for x in obs],
        "null_mean": null_mean, "null_95pct": null_95, "null_975pct": null_hi,
        "perm_p_value": p_perm,
        "per_split_above_own_95pct": f"{n_above}/{n}", "binomial_p": p_binom,
        "mann_whitney_U": float(mw.statistic), "mann_whitney_p": float(mw.pvalue),
        "bootstrap_diff_mean": d_mean, "bootstrap_diff_ci95": [float(d_lo), float(d_hi)],
        "ci_separated_from_null": bool(ci_separated),
        "distinguishable_from_null": bool(distinguishable),
        "residual_above_chance": float(residual_vs_chance),
        "residual_above_null95_ceiling": float(residual_vs_ceiling),
        "label": label,
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))

    # Figure: pooled null histogram with chance, the null 95th percentile, and the
    # observed mean, CI and per-split values.
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.hist(null_pooled, bins=45, density=True, color="#9ecae1", edgecolor="white",
            label=f"permutation null (max-over-probes)\nn={len(null_pooled)}, mean={null_mean:.3f}")
    ax.axvline(chance, color="#636363", ls=":", lw=1.4, label=f"chance 1/K = {chance:.3f}")
    ax.axvline(null_95, color="#3182bd", ls="--", lw=1.6, label=f"null 95th pct = {null_95:.3f}")
    ax.axvspan(obs_lo, obs_hi, color="#fdae6b", alpha=0.35, label=f"observed 95% CI [{obs_lo:.3f},{obs_hi:.3f}]")
    ax.axvline(obs_mean, color="#e6550d", lw=2.2, label=f"observed mean = {obs_mean:.3f}")
    for x in obs:
        ax.plot(x, 0, marker="v", color="#e6550d", ms=8, clip_on=False)
    ax.set_xlabel(f"held-out subject-recoverability ({metric}-over-probes, balanced acc)")
    ax.set_ylabel("density")
    ax.set_title(f"FACED — subject-recoverability vs permutation null\n{label}  "
                 f"(perm p={p_perm:.3f}, MW p={mw.pvalue:.3f})", fontsize=10)
    ax.legend(fontsize=7.2, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig_path = out / "fig_perm.png"
    fig.savefig(fig_path, dpi=170); fig.savefig(out / "fig_perm.pdf")
    print(f"[fig] {fig_path}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_REPO / "results" / "faced_permtest"))
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=125, help="permutations per split (pooled = x8)")
    ap.add_argument("--cap", type=int, default=8000)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--metric", default="max", choices=["max", "hgb"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if not args.analyze_only:
        splits = list(range(args.splits))
        gpus = [int(g) for g in args.gpus.split(",")]
        if len(gpus) <= 1:
            worker(gpus[0], splits, args, out)
        else:
            import torch.multiprocessing as mp
            ctx = mp.get_context("spawn")
            shards = {g: [] for g in gpus}
            for i, s in enumerate(splits):
                shards[gpus[i % len(gpus)]].append(s)
            procs = [ctx.Process(target=worker, args=(g, shards[g], args, out)) for g in gpus]
            for p in procs: p.start()
            for p in procs: p.join()

    v = analyze(out, args)
    print("\n================ VERDICT ================")
    print(f"  metric               : {v['metric']}-over-probes")
    print(f"  observed mean [95%CI] : {v['observed_mean']:.4f} [{v['observed_ci95'][0]:.4f}, {v['observed_ci95'][1]:.4f}]  (n={v['n_splits']} splits)")
    print(f"  null: mean={v['null_mean']:.4f}  95th={v['null_95pct']:.4f}  97.5th={v['null_975pct']:.4f}  (n_perm pooled={v['n_perm_pooled']})")
    print(f"  chance 1/K           : {v['chance_1_over_K']:.4f}")
    print(f"  (1) permutation p     : {v['perm_p_value']:.4f}   [P(null >= observed mean)]")
    print(f"  (2) Mann-Whitney U/p  : U={v['mann_whitney_U']:.0f}  p={v['mann_whitney_p']:.4f}")
    print(f"      bootstrap diff-mean: {v['bootstrap_diff_mean']:.4f}  CI[{v['bootstrap_diff_ci95'][0]:.4f},{v['bootstrap_diff_ci95'][1]:.4f}]")
    print(f"  (3) CI separated?     : {v['ci_separated_from_null']}  (per-split {v['per_split_above_own_95pct']} above own 95th, binom p={v['binomial_p']:.4f})")
    print(f"  distinguishable       : {v['distinguishable_from_null']}   residual above chance = {v['residual_above_chance']:.4f}")
    print(f"  ==> LABEL: {v['label']}")


if __name__ == "__main__":
    main()
