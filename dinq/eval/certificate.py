"""Probe certificate: subject-grouped CV probes that place an embedding set in a C1-C5
taxonomy cell (leakage safeguards are summarised in the README)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .probe_suite import classify, Thresholds

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars optional
    def tqdm(x, **k):
        return x
from diagnostic_framework.probes.p5_subspace import (
    subspace_probe, _class_mean_difference_directions, _orthonormal_basis,
)

MIN_STRATUM = 50  # matches the conditional probe's min_samples_per_stratum
PCA_VAR = 0.95  # matches the subspace probe's pca_variance
DEEP_MAX_ITER = 1500  # high enough for the deep probe to converge
CAPACITY_DELTA = 0.03  # linear-over-deep margin that flags probe_underpowered


def _linear():
    return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")


def _deep_mlp(rs):
    # Regularized and early-stopped: an unregularized (128, 128, 128) MLP scored
    # below chance under subject-grouped CV because it overfit the training subjects.
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=DEEP_MAX_ITER,
                      learning_rate_init=1e-3, solver="adam", early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=15, random_state=rs),
    )


def _deep_hgb(rs):
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
        min_samples_leaf=50, early_stopping=True, random_state=rs)


# domain_deep is the max over these probes (domain recoverability = max over
# probes), which is conservative: one underpowered probe cannot fake invariance.
_DEEP_PROBES = {"mlp": _deep_mlp, "hgb": _deep_hgb}


def _grouped_pred(make_clf, Z, labels, groups, n_folds=5, rs=0):
    """Row-aligned held-out predictions via StratifiedGroupKFold.

    Groups (subjects) never straddle train/test, and each fold keeps both label
    classes: subjects are domain-homogeneous, so plain GroupKFold can give a fold a
    single-domain train set, leaving LogisticRegression with one class. `k` is
    capped by the rarest class's group count so stratification is feasible.
    """
    labels = labels.astype(np.int64)
    groups = np.asarray(groups)
    per_class_groups = [len(np.unique(groups[labels == c])) for c in np.unique(labels)]
    k = int(min(n_folds, min(per_class_groups)))
    if k < 2:
        raise ValueError(f"need >= 2 subject-groups per class for grouped CV; got {per_class_groups}")
    gkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs)
    pred = np.full(labels.shape[0], -1, dtype=np.int64)
    iters = []
    for tr, te in gkf.split(Z, labels, groups):
        clf = make_clf()
        clf.fit(Z[tr], labels[tr])
        pred[te] = clf.predict(Z[te])
        est = clf.steps[-1][1] if hasattr(clf, "steps") else clf
        ni = getattr(est, "n_iter_", np.nan)
        iters.append(float(np.ravel(ni)[0]))  # LogReg n_iter_ is an array
    return pred, float(np.nanmean(iters)) if iters else float("nan")


def _pca(Z, rs):
    return PCA(n_components=PCA_VAR, svd_solver="full", random_state=rs).fit_transform(
        np.asarray(Z, dtype=np.float64))


def _precompute(Z, y, d, groups, target_domain, source_domain, rs):
    """Fit each probe once (grouped CV); return row-aligned predictions and diagnostics."""
    Z = np.asarray(Z, dtype=np.float64)
    Zp = _pca(Z, rs)  # PCA dimensionality control for the domain probes
    preds: dict[str, np.ndarray] = {}
    diag: dict[str, float] = {}

    # Task probe: a source-trained linear model predicts target rows (no CV needed;
    # source and target share no trials).
    src, tgt = d == source_domain, d == target_domain
    tp = np.full(y.shape[0], -1, np.int64)
    if src.sum() >= 5 and len(np.unique(y[src])) == 2:
        tp[tgt] = _linear().fit(Z[src], y[src]).predict(Z[tgt])
    preds["task"] = tp

    # Domain probe: linear, plus deep = max over {regularized MLP, tuned HGB}.
    preds["dom_lin"], _ = _grouped_pred(_linear, Zp, d, groups, rs=rs)
    diag["domain_linear_acc"] = float(balanced_accuracy_score(d, preds["dom_lin"]))
    deep_accs = {}
    best_pred, best_acc = None, -1.0
    for nm, mk in tqdm(_DEEP_PROBES.items(), total=len(_DEEP_PROBES),
                       desc="domain deep-probes (grouped CV)", unit="probe"):
        p, _ = _grouped_pred(lambda mk=mk: mk(rs), Zp, d, groups, rs=rs)
        a = float(balanced_accuracy_score(d, p))
        deep_accs[nm] = round(a, 3)
        if a > best_acc:
            best_pred, best_acc = p, a
    preds["dom_deep"] = best_pred  # max-over-probes predictions
    diag["deep_probe_accs"] = deep_accs

    # Subspace-residual probe: project the task class-mean subspace out of Zp, then
    # run a grouped-CV domain probe (raw accuracy).
    Uy = _orthonormal_basis(_class_mean_difference_directions(Zp, y))
    Zres = Zp - (Uy @ (Uy.T @ Zp.T)).T if Uy.shape[1] else Zp
    preds["residual"], _ = _grouped_pred(lambda: LogisticRegression(max_iter=1000, C=1.0),
                                         Zres, d, groups, rs=rs)

    # Conditional probe: grouped-CV domain probe per label stratum (balanced acc).
    # Skip a stratum that lacks >= 2 subjects of each domain (grouped CV infeasible).
    cond = np.full(d.shape[0], -1, np.int64)
    strata = []
    for lv in np.unique(y):
        m = y == lv
        if int(m.sum()) < MIN_STRATUM or len(np.unique(d[m])) < 2:
            continue
        try:
            cond[m], _ = _grouped_pred(_linear, Zp[m], d[m], groups[m], rs=rs)
            strata.append(lv)
        except ValueError:
            continue
    preds["cond"] = cond
    preds["_strata"] = np.array(strata)
    return preds, diag


def _metrics_on_idx(idx, y, d, preds, target_domain) -> dict[str, float]:
    di, yi = d[idx], y[idx]
    out = {}
    tp = preds["task"][idx]
    tmask = (di == target_domain) & (tp >= 0)
    out["task_target_acc"] = (balanced_accuracy_score(yi[tmask], tp[tmask])
                              if tmask.sum() and len(np.unique(yi[tmask])) == 2 else np.nan)
    out["domain_linear_acc"] = balanced_accuracy_score(di, preds["dom_lin"][idx])
    out["domain_deep_acc"] = balanced_accuracy_score(di, preds["dom_deep"][idx])
    out["residual_domain_acc"] = accuracy_score(di, preds["residual"][idx])
    accs = []
    for lv in preds["_strata"]:
        m = (yi == lv) & (preds["cond"][idx] >= 0)
        if m.sum() and len(np.unique(di[m])) >= 2:
            accs.append(balanced_accuracy_score(di[m], preds["cond"][idx][m]))
    out["conditional_domain_acc"] = float(np.mean(accs)) if accs else np.nan
    return out


def _block_ci(y, d, groups, preds, target_domain, n_block, rs, ci_level=95.0):
    uniq = np.unique(groups)
    rows_of = {int(g): np.where(groups == g)[0] for g in uniq}
    by_dom: dict[int, list[int]] = {}
    for g in uniq:
        by_dom.setdefault(int(d[rows_of[int(g)][0]]), []).append(int(g))
    by_dom = {k: np.array(v) for k, v in by_dom.items()}
    rng = np.random.default_rng(rs)
    keys = ["task_target_acc", "domain_linear_acc", "domain_deep_acc",
            "residual_domain_acc", "conditional_domain_acc"]
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_block):
        parts = []
        for gs in by_dom.values():
            parts.extend(rows_of[int(g)] for g in rng.choice(gs, size=len(gs), replace=True))
        idx = np.concatenate(parts)
        m = _metrics_on_idx(idx, y, d, preds, target_domain)
        for k in keys:
            if np.isfinite(m[k]):
                acc[k].append(m[k])
    a = (100.0 - ci_level) / 2.0
    ci = {}
    for k in keys:
        arr = np.asarray(acc[k])
        ci[k] = ((float(np.percentile(arr, a)), float(np.percentile(arr, 100 - a)))
                 if arr.size >= max(10, n_block // 10) else (np.nan, np.nan))
    return ci


@dataclass
class CertificateCell:
    category: str
    ci_stable: bool
    quantities: dict[str, dict[str, float]]
    borderline_tests: list[str]
    n_windows: int
    n_trials: int
    diagnostics: dict[str, float]  # domain_linear_acc, deep_probe_accs (mlp/hgb)
    probe_underpowered: bool

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "ci_stable": self.ci_stable,
                "quantities": self.quantities, "borderline_tests": self.borderline_tests,
                "n_windows": self.n_windows, "n_trials": self.n_trials,
                "diagnostics": self.diagnostics, "probe_underpowered": self.probe_underpowered}


def certify(Z, y, d, groups, thresholds: Thresholds,
            target_domain=1, source_domain=0, n_block=200, rs=0) -> CertificateCell:
    """Grouped-CV point estimates and group-block CIs -> C1-C5 cell, plus the capacity guard."""
    Z = np.asarray(Z); y = np.asarray(y).astype(np.int64)
    d = np.asarray(d).astype(np.int64); groups = np.asarray(groups)
    preds, diag = _precompute(Z, y, d, groups, target_domain, source_domain, rs)

    point = _metrics_on_idx(np.arange(Z.shape[0]), y, d, preds, target_domain)
    point["subspace_cos2_max"] = float(subspace_probe(Z, y, d, random_state=rs).cos2_max)
    ci = _block_ci(y, d, groups, preds, target_domain, n_block, rs)

    # domain_deep is already the max over deep probes. Flag if it still falls short
    # of the linear probe: a nonlinear probe should recover at least as much as a
    # linear one on separable data, so deep << linear means underpowered or overfit.
    underpowered = bool(point["domain_linear_acc"] - point["domain_deep_acc"] > CAPACITY_DELTA)

    quantities, probe_dict = {}, {}
    for k, (lo, hi) in ci.items():
        quantities[k] = {"point": point[k], "ci_low": lo, "ci_high": hi}
        probe_dict[k] = {"point": point[k], "ci_low": lo, "ci_high": hi}
    quantities["subspace_cos2_max"] = {"point": point["subspace_cos2_max"]}
    probe_dict["subspace_cos2_max"] = {"point": point["subspace_cos2_max"],
                                       "ci_low": point["subspace_cos2_max"],
                                       "ci_high": point["subspace_cos2_max"]}
    probe_dict["pareto_slope"] = {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    assign = classify(probe_dict, thresholds)
    return CertificateCell(category=assign.category, ci_stable=assign.ci_stable,
                           quantities=quantities, borderline_tests=assign.borderline_tests,
                           n_windows=int(Z.shape[0]), n_trials=int(len(np.unique(groups))),
                           diagnostics=diag, probe_underpowered=underpowered)
