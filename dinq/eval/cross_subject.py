"""Cross-subject certificate: emotion on unseen subjects and K-way subject
recoverability (max over probes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .certificate import _linear, _deep_mlp, _deep_hgb


def _cap_keep_all_subjects(subject, cap, rs):
    """Row mask subsampling to ~cap windows while keeping every subject present."""
    n = len(subject)
    if n <= cap:
        return np.ones(n, bool)
    rng = np.random.default_rng(rs)
    keep = np.zeros(n, bool)
    per = max(1, cap // len(np.unique(subject)))
    for s in np.unique(subject):
        idx = np.where(subject == s)[0]
        keep[rng.choice(idx, min(per, len(idx)), replace=False)] = True
    return keep


def _task_subject_grouped(X, y, subject, rs, n_folds=5):
    """Emotion balanced-acc, whole subjects held out (GroupKFold)."""
    k = min(n_folds, len(np.unique(subject)))
    pred = np.full(len(y), -1)
    for tr, te in GroupKFold(k).split(X, y, subject):
        pred[te] = _linear().fit(X[tr], y[tr]).predict(X[te])
    return float(balanced_accuracy_score(y, pred))


def _subject_recoverability(X, subject, groups, rs, n_folds=5):
    """Multi-class subject balanced-acc (chance 1/K), trial-grouped, max over probes."""
    # k is capped by the smallest subject's window count (not its trial count).
    k = min(n_folds, min(np.bincount(_relabel(subject))))
    k = max(2, min(n_folds, k))
    accs = {}
    for name, mk in (("linear", _linear), ("mlp", lambda: _deep_mlp(rs)), ("hgb", lambda: _deep_hgb(rs))):
        pred = np.full(len(subject), -1)
        try:
            sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs)
            for tr, te in sgkf.split(X, subject, groups):
                pred[te] = mk().fit(X[tr], subject[tr]).predict(X[te])
            accs[name] = float(balanced_accuracy_score(subject, pred))
        except Exception:
            accs[name] = float("nan")
    return accs


def _relabel(a):
    _, inv = np.unique(a, return_inverse=True)
    return inv


@dataclass
class CrossSubjectCell:
    task_acc: float
    task_acc_trial: float
    subject_recoverability: dict[str, float]  # linear / mlp / hgb (balanced acc)
    chance: float  # 1/K
    n_subjects: int
    verdict: str  # C1-analog / residual-entangled / null-task
    diagnostics: dict[str, Any]

    def to_dict(self):
        return self.__dict__


def cross_subject_certify(
    X, emotion, subject, trial_uid, task_floor=0.55, near_chance_mult=2.0,
    pca_var=0.95, cap=8000, rs=0, X_trialmean=None, trial_labels=None, trial_subject=None,
) -> CrossSubjectCell:
    """Run the cross-subject certificate on representation X.

    task_floor       : task balanced-accuracy floor; below it the verdict is null_task.
    near_chance_mult : subject recoverability is "near chance" if < mult * (1/K).
    cap              : subsample to ~cap windows while keeping every subject present.
    X_trialmean/...  : optional trial-aggregated features for a trial-level task accuracy.
    """
    X = np.asarray(X, float); subject = np.asarray(subject); emotion = np.asarray(emotion)
    groups = np.asarray(trial_uid)
    m = _cap_keep_all_subjects(subject, cap, rs)
    Xp = PCA(n_components=pca_var, svd_solver="full", random_state=rs).fit_transform(X[m])
    subj, emo, grp = subject[m], emotion[m], groups[m]
    K = len(np.unique(subj)); chance = 1.0 / K

    task = _task_subject_grouped(Xp, emo, subj, rs)
    recov = _subject_recoverability(Xp, _relabel(subj), grp, rs)

    # Optional trial-level task accuracy.
    task_trial = float("nan")
    if X_trialmean is not None:
        Xt = PCA(pca_var, svd_solver="full", random_state=rs).fit_transform(np.asarray(X_trialmean, float))
        task_trial = _task_subject_grouped(Xt, np.asarray(trial_labels), np.asarray(trial_subject), rs)

    deep_recov = np.nanmax([recov.get("mlp", np.nan), recov.get("hgb", np.nan)])
    lin_recov = recov.get("linear", np.nan)
    near = near_chance_mult * chance
    if task < task_floor:
        verdict = "null_task"
    elif lin_recov < near and deep_recov < near:
        verdict = "C1_analog_subject_invariant"
    else:
        verdict = "residual_subject_entangled"  # subject identity still recoverable
    return CrossSubjectCell(
        task_acc=task, task_acc_trial=task_trial, subject_recoverability=recov,
        chance=chance, n_subjects=K, verdict=verdict,
        diagnostics={"near_chance_thresh": near, "deep_recov": float(deep_recov),
                     "lin_recov": float(lin_recov), "n_windows": int(m.sum())},
    )
