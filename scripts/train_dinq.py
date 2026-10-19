"""Train Minimal DINQ on one subject split and certify z_c (checkpointed, resumable).

Usage:
  python scripts/train_dinq.py --corpus seedv --quantizer fsq --epochs 60
  python scripts/train_dinq.py --corpus seedv --quantizer vq --lambda-con 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from dinq.features.featureset import FeatureSet
from dinq.models.minimal_dinq import MinimalDINQ, DINQConfig
from dinq.eval.blockbootstrap import trial_group_ids
from dinq.utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="seedv", choices=["seedv", "faced"])
    ap.add_argument("--quantizer", default="fsq", choices=["fsq", "vq"])
    ap.add_argument("--fsq-levels", default="8,8,8,8")
    ap.add_argument("--vq-codebook", type=int, default=256)
    ap.add_argument("--lambda-con", type=float, default=1.0)
    ap.add_argument("--d-r", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.25, help="fraction of subjects held out for the emotion test")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rec = FeatureSet.load(_REPO / "cache" / "tangent" / f"{args.corpus}_recentered_four.npz")
    # Cross-subject split (same seeded shuffle as dinq_probe.subject_split): DINQ
    # trains on the training subjects only, so emotion is tested on subjects whose
    # data and labels it never saw. Subject-recoverability is measured on the
    # training subjects, to test whether the bottleneck removed subject identity
    # from the z_c it was trained to produce.
    subs = np.unique(rec.subject)
    rng = np.random.RandomState(args.seed); rng.shuffle(subs)
    n_test = max(2, int(round(len(subs) * args.test_frac)))
    test_subs = set(subs[:n_test].tolist())
    train_mask = ~np.isin(rec.subject, list(test_subs))
    test_mask = ~train_mask
    X = torch.from_numpy(rec.X[train_mask]).float()
    y = torch.from_numpy(rec.label[train_mask]).long()
    print(f"[data] {args.corpus} recentered tangent X={rec.X.shape}; DINQ trains on "
          f"{len(subs)-n_test} subjects, {n_test} held out for the emotion test")

    cfg = DINQConfig(in_dim=rec.dim, quantizer=args.quantizer,
                     fsq_levels=tuple(int(x) for x in args.fsq_levels.split(",")),
                     vq_codebook=args.vq_codebook, d_r=args.d_r, lambda_con=args.lambda_con)
    model = MinimalDINQ(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt = Path(args.ckpt) if args.ckpt else _REPO / "checkpoints" / f"dinq_{args.corpus}_{args.quantizer}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["epoch"]
        print(f"[resume] from epoch {start}")

    n = X.shape[0]
    for ep in range(start, args.epochs):
        model.train()
        perm = torch.randperm(n)
        agg = {"rec": 0, "con": 0, "vq": 0, "perp": 0}; nb = 0
        for i in range(0, n - args.batch + 1, args.batch):
            b = perm[i:i + args.batch]
            vb, yb = X[b].to(dev), y[b].to(dev)
            total, logs = model.loss(vb, yb)
            opt.zero_grad(); total.backward(); opt.step()
            for k in ("rec", "con", "vq"):
                agg[k] += logs[k]
            agg["perp"] += logs["perplexity"]; nb += 1
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep+1}/{args.epochs}  rec={agg['rec']/nb:.4f} con={agg['con']/nb:.3f} "
                  f"vq={agg['vq']/nb:.4f} perplexity={agg['perp']/nb:.1f}/{model.codebook_size}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1}, ckpt)

    # z_c for all windows; the encoder was fit on training subjects only.
    model.eval()
    Xall = torch.from_numpy(rec.X).float()
    zc = []
    with torch.no_grad():
        for i in range(0, Xall.shape[0], 1024):
            zc.append(model.encode_zc(Xall[i:i + 1024].to(dev)).cpu().numpy())
    zc = np.concatenate(zc).astype(np.float32)
    uid = trial_group_ids(rec.subject, rec.session, rec.trial_id)

    # Emotion: probe on trial-mean z_c, fit on training subjects and tested on
    # held-out subjects.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def agg(mask):
        u = uid[mask]; Z = zc[mask]; lab = rec.label[mask]
        return (np.array([Z[u == k].mean(0) for k in np.unique(u)]),
                np.array([lab[u == k][0] for k in np.unique(u)]))
    Xtr, ytr = agg(train_mask); Xte, yte = agg(test_mask)
    task_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    task_xsub = float(balanced_accuracy_score(yte, task_clf.fit(Xtr, ytr).predict(Xte)))

    # Subject-recoverability on the training subjects: trial-grouped CV on a fixed
    # random subset of up to 8000 windows, chance 1/K_train. The verdict requires
    # both the linear probe and max(MLP, HGB) to be below the near-chance cutoff.
    from dinq.eval.cross_subject import _subject_recoverability, _relabel
    zt, st, gt = zc[train_mask], rec.subject[train_mask], uid[train_mask]
    cap = np.random.default_rng(0).permutation(len(st))[:8000]
    recov = _subject_recoverability(zt[cap], _relabel(st[cap]), gt[cap], rs=0)
    K = len(np.unique(st)); chance = 1.0 / K
    lin, deep = recov["linear"], np.nanmax([recov["mlp"], recov["hgb"]])

    print("\n=== z_c cross-subject certificate ===")
    print(f"  quantizer={args.quantizer} codebook={model.codebook_size} d_r={args.d_r} lambda_con={args.lambda_con}")
    print(f"  emotion task on UNSEEN subjects = {task_xsub:.3f}  (floor 0.62)")
    print(f"  subject-recoverability on train (K={K}, chance={chance:.3f}): {recov}")
    print(f"  -> lin={lin:.3f} HGB-deep={deep:.3f}")
    near = chance + 0.10
    invariant = deep < near and lin < near
    verdict = ("null_task" if task_xsub < 0.62 else
               ("C1_analog_subject_invariant" if invariant else "residual_subject_entangled"))
    print(f"  VERDICT: {verdict}   (subject near-chance cutoff {near:.3f})")


if __name__ == "__main__":
    main()
