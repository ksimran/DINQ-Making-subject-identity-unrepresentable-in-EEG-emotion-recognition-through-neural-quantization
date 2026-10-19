"""Train and evaluate DINQ configurations over subject splits; one JSON per
(config, split), so runs resume.

Usage:
  python scripts/dinq_sweep.py --corpus seedv --splits 4 --epochs 50 --gpus 0,1
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

# Sweep grid: VQ codebook sizes, VQ-512 with the contrastive term on u, FSQ, and a
# continuous control.
CONFIGS = [
    {"name": "vq128", "quantizer": "vq", "vq_codebook": 128, "d_r": 16},
    {"name": "vq256", "quantizer": "vq", "vq_codebook": 256, "d_r": 16},
    {"name": "vq512", "quantizer": "vq", "vq_codebook": 512, "d_r": 16},
    {"name": "vq1024", "quantizer": "vq", "vq_codebook": 1024, "d_r": 16},
    {"name": "vq512_conU", "quantizer": "vq", "vq_codebook": 512, "d_r": 16, "contrast_on": "u"},
    {"name": "fsq512", "quantizer": "fsq", "fsq_levels": [8, 8, 8], "d_r": 16},
    # Continuous control: identity bottleneck with the same d_c/d_r as vq512, so
    # DINQ vs continuous separates the effect of quantization from that of the
    # autoencoder bottleneck.
    {"name": "continuous", "quantizer": "none", "d_c": 64, "d_r": 16},
]


def _git_rev() -> str:
    """Short HEAD hash, suffixed '-dirty' when tracked files differ from it."""
    import subprocess
    try:
        rev = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(_REPO), "diff", "--quiet", "HEAD"]).returncode != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def worker(gpu, cells, args, out):
    """Train and evaluate the given (config, split) cells on one GPU (or CPU)."""
    import torch
    from dinq.features.featureset import FeatureSet
    from dinq.models.minimal_dinq import DINQConfig
    from dinq.eval.dinq_probe import evaluate_split
    dev = "cpu"
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu); dev = f"cuda:{gpu}"
    rec = FeatureSet.load(_REPO / "cache" / "tangent" / f"{args.corpus}_recentered_four.npz")
    for (cfg_d, split) in cells:
        cell_path = out / "cells" / f"{cfg_d['name']}__split{split}.json"
        if cell_path.exists():
            print(f"[skip] {cell_path.name}"); continue
        try:
            # Seed each cell with torch.manual_seed(split), as faced_permtest does, so a
            # cell is reproducible (up to CUDA nondeterminism) and comparable to the
            # permutation-test z_c.
            torch.manual_seed(split); np.random.seed(split)
            cfg = DINQConfig(in_dim=rec.dim, **{k: v for k, v in cfg_d.items() if k != "name"})
            r = evaluate_split(rec, cfg, split_seed=split, test_frac=args.test_frac,
                               epochs=args.epochs, device=dev, n_perm=args.n_perm, cap=args.cap)
            r["config"] = cfg_d["name"]; r["split"] = split
            # Store the run settings and git revision in each cell, so the protocol
            # behind a result (epochs, cap, seed) can be verified from the cell itself.
            r["run"] = {"epochs": args.epochs, "cap": args.cap, "n_perm": args.n_perm,
                        "test_frac": args.test_frac, "seed": split, "git": _git_rev()}
            cell_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cell_path.with_suffix(".tmp"); tmp.write_text(json.dumps(r, indent=2)); tmp.replace(cell_path)
            print(f"[done] {cfg_d['name']} split{split}: emo={r['emotion_unseen']:.3f} "
                  f"recovHGB(train)={r['recov_train']['hgb']:.3f} recovHGB(held)={r['recov_heldout']['hgb']:.3f} "
                  f"ceiling={r['ceiling_train']:.3f}")
        except Exception:
            print(f"[FAIL] {cfg_d['name']} split{split}:\n{traceback.format_exc()}")


def aggregate(out):
    """Summarize the cells per config, print the frontier table and write frontier.json."""
    cells = [json.loads(p.read_text()) for p in sorted((out / "cells").glob("*.json"))]
    by = {}
    for c in cells:
        by.setdefault(c["config"], []).append(c)
    print(f"\n{'config':14}{'emo(unseen)':16}{'HGB recov train':18}{'HGB recov held':18}{'ceiling':10}{'clean C1?'}")
    rows = []
    for name, cs in by.items():
        emo = np.array([c["emotion_unseen"] for c in cs])
        rt = np.array([c["recov_train"]["hgb"] for c in cs])
        rh = np.array([c["recov_heldout"]["hgb"] for c in cs])
        ceil = np.array([c["ceiling_train"] for c in cs])
        clean = (np.nanmean(emo) > 0.62) and (np.nanmean(rt) <= np.nanmean(ceil))
        print(f"{name:14}{np.nanmean(emo):.3f}±{np.nanstd(emo):.3f}     "
              f"{np.nanmean(rt):.3f}±{np.nanstd(rt):.3f}      "
              f"{np.nanmean(rh):.3f}±{np.nanstd(rh):.3f}      "
              f"{np.nanmean(ceil):.3f}    {'YES' if clean else 'no'}")
        rows.append({"config": name, "emotion_mean": float(np.nanmean(emo)), "emotion_std": float(np.nanstd(emo)),
                     "recov_train_hgb": float(np.nanmean(rt)), "recov_heldout_hgb": float(np.nanmean(rh)),
                     "ceiling": float(np.nanmean(ceil)), "clean_c1": bool(clean), "n_splits": len(cs)})
    (out / "frontier.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[frontier] {out / 'frontier.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="seedv")
    ap.add_argument("--out", default=str(_REPO / "results" / "dinq_sweep"))
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--n-perm", type=int, default=15)
    ap.add_argument("--cap", type=int, default=8000, help="windows for subject-recov probes (raise for FACED's large K)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--configs", default="", help="comma-separated subset of config names")
    args = ap.parse_args()
    out = Path(args.out); (out / "cells").mkdir(parents=True, exist_ok=True)

    configs = CONFIGS if not args.configs else [c for c in CONFIGS if c["name"] in args.configs.split(",")]
    cells = [(c, s) for c in configs for s in range(args.splits)]
    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"[sweep] {len(configs)} configs x {args.splits} splits = {len(cells)} cells; gpus={gpus}")

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
    aggregate(out)


if __name__ == "__main__":
    main()
