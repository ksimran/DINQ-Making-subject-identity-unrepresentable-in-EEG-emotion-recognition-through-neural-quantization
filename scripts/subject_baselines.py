"""Subject-axis reference baselines on FACED (recentring, random, task-only, subject GRL,
subject MMD), evaluated exactly as DINQ is.

Usage:
  python scripts/subject_baselines.py --corpus faced --splits 8 --epochs 60 --gpus 0,1
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

BASELINES = ["recentering", "random", "task_only", "grl_subject", "mmd"]


def _make_zc(mode, rec, train_mask, device, epochs, seed):
    """Return the (N, d) representation for `mode`; trained encoders see training subjects only."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from dinq.baselines.reference_encoders import MLPEncoder, grad_reverse, _mmd_rbf

    if mode == "recentering":
        return rec.X.astype(np.float32)

    def subject_mmd(h, sb):
        """Mean RBF-MMD between each subject's batch samples and the whole batch.

        Only subjects with at least 2 samples in the batch contribute.
        """
        terms = []
        for s in torch.unique(sb):
            m = sb == s
            if int(m.sum()) >= 2:
                terms.append(_mmd_rbf(h[m], h))
        return torch.stack(terms).mean() if terms else h.new_zeros(())

    from dinq.utils.seed import set_seed
    set_seed(seed)
    enc = MLPEncoder(rec.dim).to(device)
    if mode == "random":
        with torch.no_grad():
            X = torch.from_numpy(rec.X).float()
            return np.concatenate([enc(X[i:i + 1024].to(device)).cpu().numpy()
                                   for i in range(0, X.shape[0], 1024)]).astype(np.float32)

    y = rec.label.astype(np.int64)
    n_emo = int(len(np.unique(y)))
    head = nn.Linear(enc.out_dim, n_emo).to(device)
    # K-way subject adversary over the training subjects (grl_subject only).
    subj_tr = rec.subject[train_mask]
    su = {s: i for i, s in enumerate(np.unique(subj_tr))}
    adv = nn.Linear(enc.out_dim, len(su)).to(device) if mode == "grl_subject" else None

    Xtr = torch.from_numpy(rec.X[train_mask]).float()
    ytr = torch.from_numpy(y[train_mask]).long()
    str_ = torch.from_numpy(np.array([su[s] for s in subj_tr])).long()
    loader = DataLoader(TensorDataset(Xtr, ytr, str_), batch_size=256, shuffle=True, drop_last=True)
    params = list(enc.parameters()) + list(head.parameters()) + (list(adv.parameters()) if adv else [])
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    enc.train()
    for _ in range(epochs):
        for xb, yb, sb in loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            h = enc(xb)
            loss = ce(head(h), yb)
            if adv is not None:  # grl_subject: remove subject via gradient reversal
                loss = loss + ce(adv(grad_reverse(h, 1.0)), sb)
            elif mode == "mmd":  # subject-MMD marginal alignment
                loss = loss + subject_mmd(h, sb)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        X = torch.from_numpy(rec.X).float()
        return np.concatenate([enc(X[i:i + 1024].to(device)).cpu().numpy()
                               for i in range(0, X.shape[0], 1024)]).astype(np.float32)


def worker(gpu, cells, args, out):
    import torch
    from dinq.features.featureset import FeatureSet
    from dinq.eval.dinq_probe import subject_split, evaluate_representation
    dev = "cpu"
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu); dev = f"cuda:{gpu}"
    rec = FeatureSet.load(_REPO / "cache" / "tangent" / f"{args.corpus}_recentered_four.npz")
    for (mode, split) in cells:
        cp = out / "cells" / f"{mode}__split{split}.json"
        if cp.exists():
            print(f"[skip] {cp.name}"); continue
        try:
            tr, te, n_tr, n_te = subject_split(rec, split, args.test_frac)
            zc = _make_zc(mode, rec, tr, dev, args.epochs, seed=split)
            # The full null distribution is not saved per cell: the reference null is
            # the pooled faced_permtest one (n = 1000), and a 200-permutation K ~ 92
            # null would cost 600 multiclass fits per cell. The per-cell ceiling is
            # still computed when --n-perm > 0.
            r = evaluate_representation(zc, rec, tr, te, split, n_perm=args.n_perm, cap=args.cap,
                                        n_test=n_te, n_train=n_tr, save_null_dist=False,
                                        recov_pca_dim=args.recov_pca_dim)
            r["config"] = mode; r["split"] = split
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(".tmp"); tmp.write_text(json.dumps(r, indent=2)); tmp.replace(cp)
            print(f"[done] {mode} split{split}: emo={r['emotion_unseen']:.3f} "
                  f"recov(mlp/hgb held)={r['recov_heldout']['mlp']:.3f}/{r['recov_heldout']['hgb']:.3f} "
                  f"ceiling={r['ceiling_heldout']:.3f}")
        except Exception:
            print(f"[FAIL] {mode} split{split}:\n{traceback.format_exc()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="faced")
    ap.add_argument("--out", default=str(_REPO / "results" / "baselines_faced_subject"))
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--n-perm", type=int, default=0, help="0 = SKIP per-cell perm-null ceiling (authoritative null is faced_permtest; ceiling on K~92 HGB is ~20min/perm)")
    ap.add_argument("--recov-pca-dim", type=int, default=64, help="cap recov PCA dims (match DINQ z_c=64; makes high-dim recentering tractable)")
    ap.add_argument("--null-n-perm", type=int, default=200, help="perms for the saved null distribution (recentering)")
    ap.add_argument("--cap", type=int, default=8000, help="MATCH the DINQ headline sweep (8000) for like-for-like Table-MAIN")
    ap.add_argument("--gpus", default="0,1")
    args = ap.parse_args()
    out = Path(args.out); (out / "cells").mkdir(parents=True, exist_ok=True)

    modes = args.baselines.split(",")
    cells = [(m, s) for m in modes for s in range(args.splits)]
    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"[subj-baselines] {modes} x {args.splits} splits = {len(cells)} cells; gpus={gpus}")

    if len(gpus) <= 1:
        worker(gpus[0], cells, args, out)
    else:
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        shards = {g: [] for g in gpus}
        for i, c in enumerate(cells):
            shards[gpus[i % len(gpus)]].append(c)
        procs = [ctx.Process(target=worker, args=(g, shards[g], args, out)) for g in gpus]
        for p in procs: p.start()
        for p in procs: p.join()

    # Per-baseline summary over splits.
    cs = [json.loads(p.read_text()) for p in sorted((out / "cells").glob("*.json"))]
    by = {}
    for c in cs:
        by.setdefault(c["config"], []).append(c)
    rows = []
    for name, g in by.items():
        emo = np.array([c["emotion_unseen"] for c in g])
        rh = np.array([c["recov_heldout"]["hgb"] for c in g])
        rows.append({"config": name, "emotion_mean": float(np.nanmean(emo)),
                     "emotion_std": float(np.nanstd(emo)),
                     "recov_heldout_hgb": float(np.nanmean(rh)),
                     "recov_heldout_hgb_std": float(np.nanstd(rh)), "n_splits": len(g)})
    (out / "frontier.json").write_text(json.dumps(rows, indent=2))
    print(f"[frontier] {out/'frontier.json'}")


if __name__ == "__main__":
    main()
