"""Run cross_corpus.py over dataset pairs and directions, one subprocess per GPU.

Usage:
  python scripts/cross_corpus_sweep.py --gpus 0,1 --epochs 60 --n-perm 30 --seed 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ["seedv,faced", "deap,seedv", "deap,faced"]  # main pair first; DEAP pairs have a weak label


def _run_phase(jobs, gpus, log_dir, py):
    """Run `jobs` (list of (pair, source)) across `gpus`, at most one job per GPU at a time.

    Returns (done, failed) lists of job labels.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = list(jobs)
    running: dict[int, tuple] = {}  # gpu -> (proc, logf, label)
    done, failed = [], []
    while queue or running:
        for gpu in gpus:  # fill idle GPUs
            if gpu not in running and queue:
                pair, source = queue.pop(0)
                a, b = pair.split(",")
                tgt = b if source == 0 else a
                src = a if source == 0 else b
                label = f"{src}_to_{tgt}"
                logf = open(log_dir / f"{label}.log", "w")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                cmd = [py, str(_REPO / "scripts" / "cross_corpus.py"),
                       "--pair", pair, "--source", str(source), "--gpu", "0",
                       "--epochs", str(ARGS.epochs), "--n-perm", str(ARGS.n_perm),
                       "--seed", str(ARGS.seed), "--quantizer", ARGS.quantizer,
                       "--vq-codebook", str(ARGS.vq_codebook), "--d-r", str(ARGS.d_r),
                       "--contrast-on", ARGS.contrast_on, "--train-pool", ARGS.train_pool]
                if ARGS.baseline_only:
                    cmd.append("--baseline-only")
                proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
                running[gpu] = (proc, logf, label)
                print(f"[launch] gpu{gpu}  {label}  (log: {log_dir.name}/{label}.log)", flush=True)
        for gpu, (proc, logf, label) in list(running.items()):  # reap finished jobs
            if proc.poll() is not None:
                logf.close()
                (done if proc.returncode == 0 else failed).append(label)
                print(f"[{'done' if proc.returncode == 0 else 'FAIL'}] gpu{gpu}  {label}"
                      f"{'' if proc.returncode == 0 else f' (rc={proc.returncode}, see log)'}", flush=True)
                del running[gpu]
        time.sleep(2)
    return done, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--pairs", type=str, default=";".join(DEFAULT_PAIRS),
                    help="';'-separated, e.g. 'seedv,faced;deap,seedv'")
    ap.add_argument("--directions", type=str, default="both", choices=["forward", "both"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantizer", type=str, default="vq")
    ap.add_argument("--vq-codebook", type=int, default=512)
    ap.add_argument("--d-r", type=int, default=16)
    ap.add_argument("--contrast-on", type=str, default="u")
    ap.add_argument("--train-pool", type=str, default="source", choices=["source", "both"])
    ap.add_argument("--baseline-only", action="store_true")
    global ARGS
    ARGS = ap.parse_args()

    gpus = [int(g) for g in ARGS.gpus.split(",")]
    pairs = [p.strip() for p in ARGS.pairs.split(";") if p.strip()]
    log_dir = _REPO / "results" / "cross_corpus" / "logs"
    py = sys.executable

    fwd = [(p, 0) for p in pairs]
    print(f"=== PHASE 1: forward ({len(fwd)} jobs) on gpus {gpus} — builds tangent caches ===", flush=True)
    d1, f1 = _run_phase(fwd, gpus, log_dir, py)

    d2, f2 = [], []
    if ARGS.directions == "both":
        rev = [(p, 1) for p in pairs]  # reverse jobs reuse the caches built by the forward phase
        print(f"\n=== PHASE 2: reverse ({len(rev)} jobs) — reuse cached tangents ===", flush=True)
        d2, f2 = _run_phase(rev, gpus, log_dir, py)

    print(f"\n=== SWEEP COMPLETE: {len(d1 + d2)} ok, {len(f1 + f2)} failed ===", flush=True)
    if f1 + f2:
        print("failed:", ", ".join(f1 + f2), flush=True)
    print("results -> results/cross_corpus/<source>_to_<target>.json", flush=True)


if __name__ == "__main__":
    main()
