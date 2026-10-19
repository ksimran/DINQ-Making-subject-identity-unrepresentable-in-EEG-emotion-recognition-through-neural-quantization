"""Cross-dataset reference baselines (DEAP <-> SEED-V): certificate cell and transfer
accuracy per (direction, model, seed); resumable, multi-GPU.

Usage:
  python scripts/run_baselines.py --input covariance --gpus 0,1
  python scripts/run_baselines.py --input de --models scmm,random,task_only,mmd,grl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

DIRECTIONS = [("deap->seedv", "deap", "seedv", 1, 0),
              ("seedv->deap", "seedv", "deap", 0, 1)]


def cap_by_subjects(pooled, cap: int, seed: int):
    """Subsample to about `cap` windows, keeping whole subjects (budget split by domain).

    Subject-grouped CV and the block bootstrap need intact subject groups, so whole
    subjects are dropped (never split) and every retained subject is kept complete.
    At least two subjects per domain are kept (when available) so GroupKFold and
    the bootstrap stay valid.
    """
    if pooled.n <= cap:
        return pooled
    rng = np.random.default_rng(seed)
    keep = np.zeros(pooled.n, dtype=bool)
    for dom in np.unique(pooled.domain):
        dmask = pooled.domain == dom
        subs = np.unique(pooled.subject[dmask]); rng.shuffle(subs)
        budget, got, chosen = int(cap * dmask.mean()), 0, []
        for s in subs:
            chosen.append(s); got += int((pooled.subject == s).sum())
            if got >= budget and len(chosen) >= 2:
                break
        keep |= np.isin(pooled.subject, chosen)
    return pooled.select(keep)


def _wrap(fs, z):
    """Wrap embeddings `z` in a FeatureSet carrying the metadata of `fs`."""
    from dinq.features.featureset import FeatureSet
    return FeatureSet(X=z.astype(np.float32), label=fs.label, subject=fs.subject,
                      domain=fs.domain, trial_id=fs.trial_id, session=fs.session,
                      feature_kind="repr", info=dict(fs.info))


def load_inputs(input_mode, subs):
    """Return {'deap': ..., 'seedv': ...} FeatureSets of DE or recentered tangent features."""
    from dinq.data.deap import load_deap_trials
    from dinq.data.seedv import load_seedv_de
    from dinq.features.de import deap_de_featureset
    if input_mode == "de":
        return {"deap": deap_de_featureset(load_deap_trials(subjects=subs), bands="four"),
                "seedv": load_seedv_de(subjects=subs, bands="four")}
    # Covariance input: per-subject Riemannian recentered tangent features (the
    # default for all methods). Recentering is label-free and split-independent,
    # so the recentered tangent is cached per corpus and reused.
    from dinq.features.covariance import CovarianceSet
    from dinq.features.recenter import recentered_tangent_featureset
    from dinq.features.featureset import FeatureSet
    from dinq.data.labels import SEEDV_SUBJECT_OFFSET
    cov = _REPO / "cache" / "cov"
    tan = _REPO / "cache" / "tangent"; tan.mkdir(parents=True, exist_ok=True)
    out = {}
    for corpus, off in (("deap", 0), ("seedv", SEEDV_SUBJECT_OFFSET)):
        cache_f = tan / f"{corpus}_recentered_four.npz"
        if cache_f.exists():
            fs = FeatureSet.load(cache_f)
        else:
            print(f"[recenter] building recentered tangent for {corpus} (one-time) ...")
            fs = recentered_tangent_featureset(CovarianceSet.load(cov / f"{corpus}_four.npz"))
            fs.save(cache_f)
        if subs is not None:
            fs = fs.select(np.isin(fs.subject, [x + off for x in subs]))
        out[corpus] = fs
    return out


def build_direction(input_mode, data, src_key, tgt_key, val_frac, seed):
    """Return (source_train, source_val, target) FeatureSets without leakage.

    Both inputs are pre-built FeatureSets (DE or recentered tangent), so the split
    is a whole-subject holdout on the source; the target is fully held out.
    """
    from dinq.data.splits import subject_holdout
    st, sv = subject_holdout(data[src_key], val_frac=val_frac, seed=seed)
    return st, sv, data[tgt_key]


def represent(mode, src_train, src_embed, tgt, cfg, device, seed, epochs, ref_epochs):
    """Train `mode` on src_train, then embed all source data (src_embed) and tgt.

    Embedding every source subject (not just source_val) gives the subject-grouped
    domain probe enough groups per domain; the encoder still trains only on
    src_train, so the source->target task-transfer test is unaffected.
    """
    from dinq.baselines.reference_encoders import train_reference, embed_encoder, dg_accuracy
    extras = {}
    if mode == "scmm":
        from dinq.baselines import scmm_train as st
        m = st.SCMM(cfg).to(device)
        st.pretrain(m, src_train, cfg, device, epochs=epochs, log_every=max(1, epochs // 4))
        rv = _wrap(src_embed, st.embed(m, src_embed, cfg, device))
        rt = _wrap(tgt, st.embed(m, tgt, cfg, device))
        extras["scmm_finetune_acc"] = _scmm_finetune_acc(m, cfg, tgt, device, seed)
        return rv, rt, extras
    enc, head = train_reference(mode, src_train, tgt, device=device, epochs=ref_epochs, seed=seed)
    extras["dg_acc"] = dg_accuracy(enc, head, tgt, device)
    return embed_encoder(enc, src_embed, device), embed_encoder(enc, tgt, device), extras


def _scmm_finetune_acc(model, cfg, tgt, device, seed, ft_epochs=50):
    """Target-finetuned SCMM balanced accuracy on a 30% whole-subject target holdout."""
    import torch
    from sklearn.metrics import balanced_accuracy_score
    from dinq.data.splits import subject_holdout
    from dinq.baselines import scmm_train as st
    tr, te = subject_holdout(tgt, val_frac=0.3, seed=seed)
    clf = st.finetune(model, cfg, tr, device, epochs=ft_epochs, tune_encoder=True)
    with torch.no_grad():
        h = model(st.featureset_to_scmm(te, cfg)[0].to(device), train_mode="ft")
        pred = clf(h).argmax(1).cpu().numpy()
    return float(balanced_accuracy_score(te.label, pred))


def run_cell(name, mode, seed, src_train, src_val, tgt, tdom, sdom, cfg, device, args, out):
    """Certify one (direction, model, seed) cell and write its JSON; skip if it exists."""
    from dinq.features.featureset import concat
    from dinq.eval.certificate import certify
    from diagnostic_framework.thresholds import DEFAULT_THRESHOLDS
    cell_path = out / "cells" / f"{name}__{mode}__{args.input}__seed{seed}.json"
    if cell_path.exists():
        print(f"[skip] {cell_path.name}"); return
    t0 = time.time()
    src_embed = concat([src_train, src_val])  # all source subjects for the domain probe
    rv, rt, extras = represent(mode, src_train, src_embed, tgt, cfg, device, seed,
                               args.epochs, args.ref_epochs)
    pooled = cap_by_subjects(concat([rv, rt]), args.cap, seed)
    # Subject-grouped CV and block bootstrap: trial-level groups would leak subject identity.
    groups = pooled.subject
    cell = certify(pooled.X, pooled.label, pooled.domain, groups, DEFAULT_THRESHOLDS,
                   target_domain=tdom, source_domain=sdom, n_block=args.n_block, rs=seed)
    rec = {"direction": name, "model": mode, "seed": seed, "input": args.input,
           "category": cell.category, "ci_stable": cell.ci_stable,
           "probe_underpowered": cell.probe_underpowered, "diagnostics": cell.diagnostics,
           "task_target_acc_dg": cell.quantities["task_target_acc"]["point"],
           "quantities": cell.quantities, "n_windows": cell.n_windows,
           "n_trials": cell.n_trials, "secs": round(time.time() - t0, 1), **extras}
    tmp = cell_path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec, indent=2)); tmp.replace(cell_path)
    flags = " UNDERPOWERED" if cell.probe_underpowered else ""
    print(f"[done] {name} {mode} seed{seed} [{args.input}]: {cell.category}{flags} "
          f"dom_lin={cell.quantities['domain_linear_acc']['point']:.2f} "
          f"resid={cell.quantities['residual_domain_acc']['point']:.2f} "
          f"task_dg={rec['task_target_acc_dg']:.2f} ({rec['secs']:.0f}s)")


def worker(gpu, pairs, args, models, out):
    """Run every model on the given (direction, seed) pairs on one GPU (or CPU)."""
    import torch
    from dinq.baselines.scmm import SCMMConfig
    device = "cpu"
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu); device = f"cuda:{gpu}"
    data = load_inputs(args.input, args.subs)
    cfg = SCMMConfig(in_channels=32, fea_length=4)
    for (name, sk, tk, tdom, sdom, seed) in pairs:
        src_train, src_val, tgt = build_direction(args.input, data, sk, tk, args.val_frac, seed)
        for mode in models:
            if args.input == "covariance" and mode == "scmm":
                print(f"[skip] {name} scmm: SCMM is DE-native (headline stays DE bar)"); continue
            try:
                run_cell(name, mode, seed, src_train, src_val, tgt, tdom, sdom, cfg, device, args, out)
            except Exception:
                print(f"[FAIL] {name} {mode} seed{seed}:\n{traceback.format_exc()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_REPO / "results" / "baselines"))
    ap.add_argument("--input", choices=["de", "covariance"], default="covariance")
    ap.add_argument("--models", default="random,task_only,mmd,grl,scmm")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ref-epochs", type=int, default=60)
    ap.add_argument("--n-block", type=int, default=200)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--subjects", type=int, default=0, help="0=all; else first-N per corpus (dev)")
    args = ap.parse_args()
    args.subs = list(range(1, args.subjects + 1)) if args.subjects else None

    out = Path(args.out); (out / "cells").mkdir(parents=True, exist_ok=True)
    models = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    pairs = [(n, sk, tk, td, sd, sd_) for sd_ in seeds for (n, sk, tk, td, sd) in DIRECTIONS]

    # Do not touch torch.cuda in the parent before spawning: initializing CUDA here
    # deadlocks the spawned workers. --gpus is trusted and each worker checks CUDA.
    gpus = [int(g) for g in args.gpus.split(",")]
    print(f"[sweep] input={args.input} models={models} pairs={len(pairs)} gpus={gpus}")

    if len(gpus) <= 1:
        worker(gpus[0], pairs, args, models, out)
    else:
        import torch.multiprocessing as mp
        ctx = mp.get_context("spawn")
        shards = {g: [] for g in gpus}
        for i, p in enumerate(pairs):
            shards[gpus[i % len(gpus)]].append(p)
        procs = [ctx.Process(target=worker, args=(g, shards[g], args, models, out)) for g in gpus]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()

    all_cells = [json.loads(p.read_text()) for p in sorted((out / "cells").glob("*.json"))]
    cells = [c for c in all_cells if c.get("input") == args.input]
    (out / f"results_{args.input}.json").write_text(json.dumps(cells, indent=2))
    print(f"\n[sweep] {len(cells)} {args.input} cells -> results_{args.input}.json")
    for c in cells:
        acc = c.get("dg_acc", c.get("scmm_finetune_acc"))
        print(f"  {c['direction']:14s} {c['model']:10s} {c['category']:22s} "
              f"dom_lin={c['quantities']['domain_linear_acc']['point']:.2f} "
              f"resid={c['quantities']['residual_domain_acc']['point']:.2f}"
              + (f" acc={acc:.2f}" if acc == acc else "")
              + (" [UNDERPOWERED]" if c.get("probe_underpowered") else ""))


if __name__ == "__main__":
    main()
