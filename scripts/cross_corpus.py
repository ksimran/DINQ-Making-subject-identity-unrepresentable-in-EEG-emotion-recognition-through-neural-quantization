"""Cross-dataset experiment for one source -> target pair: dataset recoverability of the
recentred baseline and of DINQ, against a dataset-label permutation null.

Usage:
  python scripts/cross_corpus.py --pair seedv,faced --epochs 60 --gpu 0 --n-perm 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars optional
    def tqdm(x, **k):
        return x

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path when run directly

from dinq.features.covariance import CovarianceSet
from dinq.features.recenter import recentered_tangent_featureset
from dinq.eval.certificate import certify, _grouped_pred, _pca, _DEEP_PROBES
from dinq.eval.probe_suite import Thresholds
from dinq.eval.blockbootstrap import trial_group_ids
from dinq.models.minimal_dinq import DINQConfig
from dinq.eval.dinq_probe import train_minimal_dinq, extract_zc
from dinq.paths import CACHE_ROOT
from sklearn.metrics import balanced_accuracy_score

REPO = Path(__file__).resolve().parents[1]
SUBJ_OFFSET = 100_000  # keeps the second corpus's subject ids disjoint from the first


def _build_aligned(pair: tuple[str, str], subj_cap: int | None = None) -> tuple:
    """Pool two corpora on their common montage, recentered per subject.

    Returns (X, emotion y, corpus d, subject groups g, trial-group ids uid, meta).
    `subj_cap` keeps only the first N subjects per corpus (smoke tests); the subset
    is taken on the covariances before recentering, so the expensive Riemannian
    means are computed only for the kept subjects.
    """
    covsets = {c: CovarianceSet.load(Path(CACHE_ROOT) / "cov" / f"{c}_four.npz") for c in pair}
    ch = {c: tuple(covsets[c].info["channels"]) for c in pair}
    target = tuple(c for c in ch[pair[0]] if c in set(ch[pair[1]]))  # common channels, first-corpus order
    assert len(target) >= 20, f"suspiciously small common montage: {target}"

    Xs, emo, corp, subj, sess, tid = [], [], [], [], [], []
    for di, c in enumerate(pair):
        cs = covsets[c].subset_montage(target)
        if subj_cap is not None:
            keep = np.isin(cs.subject, np.unique(cs.subject)[:subj_cap])
            cs = cs.select(keep)
        fs = recentered_tangent_featureset(cs)
        Xs.append(fs.X)
        emo.append(fs.label.astype(np.int64))
        corp.append(np.full(fs.X.shape[0], di, np.int64))  # corpus = domain
        subj.append(fs.subject.astype(np.int64) + di * SUBJ_OFFSET)  # disjoint subject ids
        sess.append(fs.session.astype(np.int64) + di * SUBJ_OFFSET)
        tid.append(fs.trial_id.astype(np.int64))
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(emo); d = np.concatenate(corp); g = np.concatenate(subj)
    uid = trial_group_ids(g, np.concatenate(sess), np.concatenate(tid))
    meta = {"pair": list(pair), "common_channels": list(target), "in_dim": int(X.shape[1]),
            "n": int(X.shape[0]), "n_subjects": {pair[i]: int(np.unique(subj[i]).size) for i in range(2)},
            "corpus_windows": {pair[i]: int(corp[i].size) for i in range(2)}}
    return X, y, d, g, uid, meta


def load_aligned(pair: tuple[str, str], subj_cap: int | None = None) -> tuple:
    """Cached wrapper around `_build_aligned`.

    Recentering the full pool is the expensive step, so the result is cached and
    reused by the baseline, the permutation null and every DINQ config. Runs with
    `subj_cap` are not cached.
    """
    if subj_cap is not None:
        return _build_aligned(pair, subj_cap)
    cache = Path(CACHE_ROOT) / "tangent" / f"xcorpus_{pair[0]}_{pair[1]}_recentered.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"[cache] reusing {cache}", flush=True)
        return z["X"], z["y"], z["d"], z["g"], z["uid"], json.loads(str(z["meta"]))
    X, y, d, g, uid, meta = _build_aligned(pair)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, d=d, g=g, uid=uid, meta=np.array(json.dumps(meta)))
    print(f"[cache] wrote {cache}", flush=True)
    return X, y, d, g, uid, meta


def domain_perm_null(Z, d, groups, n_perm: int, rs: int = 0) -> dict:
    """Domain-recoverability ceiling from a permutation null.

    Corpus labels are permuted at subject granularity (each subject keeps a single
    corpus label; only the assignment is shuffled), the max-over-probes deep domain
    probe is refit under the same grouped CV, and the 95th percentile is reported.
    For a balanced binary domain the expected accuracy under H0 is about 0.5.
    """
    Zp = _pca(Z, rs)
    uniq = np.unique(groups)
    dom_of = np.array([int(d[groups == g][0]) for g in uniq])  # each subject's corpus
    rng = np.random.default_rng(rs)
    accs = []
    for _ in tqdm(range(n_perm), desc="domain perm-null", unit="perm"):
        perm = rng.permutation(dom_of)
        dp = np.empty_like(d)
        for gi, g in enumerate(uniq):
            dp[groups == g] = perm[gi]
        best = -1.0
        for mk in _DEEP_PROBES.values():
            p, _ = _grouped_pred(lambda mk=mk: mk(rs), Zp, dp, groups, rs=rs)
            best = max(best, float(balanced_accuracy_score(dp, p)))
        accs.append(best)
    accs = np.asarray(accs)
    return {"mean": float(accs.mean()), "p95": float(np.percentile(accs, 95)),
            "max": float(accs.max()), "n_perm": int(n_perm)}


def _cell_summary(cell) -> dict:
    """JSON-ready summary of a certificate cell (finite quantities, rounded to 4 dp)."""
    q = cell.quantities
    g = lambda k: {kk: round(float(v), 4) for kk, v in q.get(k, {}).items() if np.isfinite(v)}
    return {"category": cell.category, "ci_stable": cell.ci_stable,
            "task_target_acc": g("task_target_acc"),  # cross-corpus emotion transfer
            "domain_linear_acc": g("domain_linear_acc"),
            "domain_deep_acc": g("domain_deep_acc"),  # max-over-probes corpus residual
            "residual_domain_acc": g("residual_domain_acc"),
            "conditional_domain_acc": g("conditional_domain_acc"),
            "deep_probe_accs": cell.diagnostics.get("deep_probe_accs", {}),
            "probe_underpowered": cell.probe_underpowered}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=str, required=True, help="e.g. seedv,faced")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--quantizer", type=str, default="vq")
    ap.add_argument("--vq-codebook", type=int, default=512)
    ap.add_argument("--d-r", type=int, default=16)
    ap.add_argument("--contrast-on", type=str, default="u")
    ap.add_argument("--source", type=int, default=0, help="source corpus index (0=first in --pair)")
    ap.add_argument("--train-pool", type=str, default="source", choices=["source", "both"],
                    help="'source': train DINQ on source corpus only (honest zero-shot transfer, "
                         "target fully held out — the DG claim). 'both': transductive (uses target "
                         "emotion labels via SupCon; domain-invariance only, NOT a transfer claim).")
    ap.add_argument("--out", type=str, default="results/cross_corpus")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="seeds numpy + torch so the DINQ arm is reproducible")
    args = ap.parse_args()

    pair = tuple(s.strip() for s in args.pair.split(","))
    assert len(pair) == 2
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    cc = yaml.safe_load((REPO / "configs" / "thresholds_recentered.yaml").read_text())
    thr = Thresholds(**cc["thresholds_recentered"]["cross_corpus"])  # frozen cross-corpus cutoffs
    tgt = 1 - args.source

    print(f"[load] pair={pair} recentered tangent (recentering held fixed) ...", flush=True)
    X, y, d, g, uid, meta = load_aligned(pair)
    print(f"[load] common montage: {meta['in_dim']}-dim tangent ({len(meta['common_channels'])} ch), "
          f"n={meta['n']}, subjects={meta['n_subjects']}", flush=True)

    out = {"meta": meta, "source": pair[args.source], "target": pair[tgt],
           "seed": args.seed, "thresholds": "cross_corpus (frozen)"}

    print("[baseline] certifying recentered tangent (domain=corpus) ...", flush=True)
    base = certify(X, y, d, g, thr, target_domain=tgt, source_domain=args.source)
    out["baseline"] = _cell_summary(base)
    print(f"[baseline] {base.category}  domain_deep={out['baseline']['domain_deep_acc']}  "
          f"emotion_transfer={out['baseline']['task_target_acc']}", flush=True)

    print(f"[null] domain permutation-null ({args.n_perm} perms) ...", flush=True)
    out["domain_perm_null"] = domain_perm_null(X, d, g, args.n_perm)
    print(f"[null] domain ceiling p95={out['domain_perm_null']['p95']:.4f}", flush=True)

    if not args.baseline_only:
        cfg = DINQConfig(in_dim=meta["in_dim"], quantizer=args.quantizer,
                         vq_codebook=args.vq_codebook, d_r=args.d_r, contrast_on=args.contrast_on)
        tr = (d == args.source) if args.train_pool == "source" else np.ones(len(d), bool)
        print(f"[dinq] training MinimalDINQ ({args.quantizer}{args.vq_codebook}, d_r={args.d_r}, "
              f"contrast_on={args.contrast_on}) on {args.train_pool} "
              f"({pair[args.source] if args.train_pool=='source' else 'pooled'}, n={int(tr.sum())}); "
              f"target corpus {'held out' if args.train_pool=='source' else 'included'} ...", flush=True)
        m = train_minimal_dinq(X[tr], y[tr], cfg, epochs=args.epochs, device=device)
        zc = extract_zc(m, X, device)  # z_c for both corpora (target unseen with --train-pool source)
        print("[dinq] certifying z_c (domain=corpus) ...", flush=True)
        dinq = certify(zc, y, d, g, thr, target_domain=tgt, source_domain=args.source)
        out["dinq"] = _cell_summary(dinq)
        out["dinq"]["config"] = {"quantizer": args.quantizer, "vq_codebook": args.vq_codebook,
                                 "d_r": args.d_r, "contrast_on": args.contrast_on, "epochs": args.epochs,
                                 "train_pool": args.train_pool}
        print(f"[dinq] {dinq.category}  domain_deep={out['dinq']['domain_deep_acc']}  "
              f"emotion_transfer={out['dinq']['task_target_acc']}", flush=True)

    outdir = REPO / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"{pair[args.source]}_to_{pair[tgt]}.json"  # one file per direction
    fp.write_text(json.dumps(out, indent=2))
    print(f"[done] -> {fp}", flush=True)

    # Summary: corpus residual vs the null p95, and emotion transfer.
    bd = out["baseline"]["domain_deep_acc"].get("point")
    p95 = out["domain_perm_null"]["p95"]
    line = f"\nBASELINE corpus residual (domain_deep) = {bd}  vs null p95 {p95:.3f}"
    if not args.baseline_only:
        dd = out["dinq"]["domain_deep_acc"].get("point")
        bt = out["baseline"]["task_target_acc"].get("point")
        dt = out["dinq"]["task_target_acc"].get("point")
        line += (f"\nDINQ      corpus residual (domain_deep) = {dd}  (emotion transfer "
                 f"base {bt} -> dinq {dt})")
    print(line, flush=True)


if __name__ == "__main__":
    main()
