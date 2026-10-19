"""Block bootstrap over whole trials or subjects, since windows within a trial are not independent."""

from __future__ import annotations

from typing import Callable

import numpy as np


def block_bootstrap_dict(
    statistic: Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, float]],
    Z: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 200,
    ci_level: float = 95.0,
    random_state: int = 0,
) -> dict[str, dict[str, float]]:
    """Block-bootstrap a dict-valued statistic by resampling whole groups.

    groups : (N,) integer group id per row (trial uid, or subject id for a
             subject-level bootstrap). Resampling is stratified by domain so each
             domain keeps its group count.
    Returns {key: {point, mean, std, ci_low, ci_high, n}} per statistic key.
    """
    Z = np.asarray(Z); y = np.asarray(y); d = np.asarray(d); groups = np.asarray(groups)
    point = statistic(Z, y, d)
    keys = list(point.keys())

    uniq = np.unique(groups)
    rows_of = {int(g): np.where(groups == g)[0] for g in uniq}
    grp_domain = {int(g): int(d[rows_of[int(g)][0]]) for g in uniq}
    by_domain: dict[int, list[int]] = {}
    for g in uniq:
        by_domain.setdefault(grp_domain[int(g)], []).append(int(g))
    by_domain = {dom: np.array(gs) for dom, gs in by_domain.items()}

    rng = np.random.default_rng(random_state)
    collected: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_resamples):
        parts = []
        for dom, gs in by_domain.items():
            chosen = rng.choice(gs, size=len(gs), replace=True)
            parts.extend(rows_of[int(g)] for g in chosen)
        idx = np.concatenate(parts)
        try:
            v = statistic(Z[idx], y[idx], d[idx])
        except Exception:  # degenerate resample (missing class) -> skip
            continue
        for k in keys:
            val = v.get(k, np.nan)
            if val is not None and np.isfinite(val):
                collected[k].append(float(val))

    alpha = (100.0 - ci_level) / 2.0
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        arr = np.asarray(collected[k], dtype=np.float64)
        if arr.size < max(10, n_resamples // 10):
            lo = hi = mean = std = float("nan")
        else:
            lo = float(np.percentile(arr, alpha))
            hi = float(np.percentile(arr, 100.0 - alpha))
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        pv = point[k]
        out[k] = {
            "point": float(pv) if pv is not None and np.isfinite(pv) else float("nan"),
            "mean": mean, "std": std, "ci_low": lo, "ci_high": hi, "n": int(arr.size),
        }
    return out


def trial_group_ids(subject: np.ndarray, session: np.ndarray, trial_id: np.ndarray) -> np.ndarray:
    """Stable integer uid per (subject, session, trial) for the block bootstrap."""
    key = (np.asarray(subject).astype(np.int64) * 1_000_000
           + np.asarray(session).astype(np.int64) * 1_000
           + np.asarray(trial_id).astype(np.int64))
    _, inv = np.unique(key, return_inverse=True)
    return inv.astype(np.int64)
