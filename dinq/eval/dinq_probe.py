"""Train Minimal DINQ and evaluate z_c over subject splits: unseen-subject emotion,
subject recoverability and permutation nulls."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..models.minimal_dinq import MinimalDINQ, DINQConfig
from ..eval.blockbootstrap import trial_group_ids
from ..eval.certificate import _deep_hgb, _deep_mlp, _linear
from ..eval.cross_subject import _relabel

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars optional
    def tqdm(x, **k):
        return x


def train_minimal_dinq(X, y, cfg: DINQConfig, epochs=60, lr=1e-3, batch=256, device="cpu"):
    m = MinimalDINQ(cfg).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(X).float(), torch.from_numpy(y).long()
    n = Xt.shape[0]
    m.train()
    for _ in tqdm(range(epochs), desc="DINQ train", unit="ep", leave=False):
        perm = torch.randperm(n)
        for i in range(0, n - batch + 1, batch):
            b = perm[i:i + batch]
            total, _ = m.loss(Xt[b].to(device), yt[b].to(device))
            opt.zero_grad(); total.backward(); opt.step()
    return m


@torch.no_grad()
def extract_zc(m, X, device):
    m.eval()
    Xt = torch.from_numpy(X).float()
    return np.concatenate([m.encode_zc(Xt[i:i + 1024].to(device)).cpu().numpy()
                           for i in range(0, Xt.shape[0], 1024)]).astype(np.float32)


def _trial_agg(Z, lab, uid):
    """Trial-level mean of Z and the per-trial label (labels are constant within a trial)."""
    u, inv = np.unique(uid, return_inverse=True)
    sums = np.zeros((len(u), Z.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, Z.astype(np.float64))
    counts = np.bincount(inv, minlength=len(u))[:, None]
    yagg = np.empty(len(u), dtype=np.asarray(lab).dtype)
    yagg[inv] = lab
    return (sums / counts).astype(np.float32), yagg


def emotion_unseen(zc, label, uid, train_mask, test_mask):
    """Trial-level emotion probe: fit on training-subject z_c, test on held-out subjects."""
    Xtr, ytr = _trial_agg(zc[train_mask], label[train_mask], uid[train_mask])
    Xte, yte = _trial_agg(zc[test_mask], label[test_mask], uid[test_mask])
    if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    return float(balanced_accuracy_score(yte, clf.fit(Xtr, ytr).predict(Xte)))


def _grouped_multiclass(Z, labels, groups, make_clf, rs=0):
    """Held-out balanced accuracy of a multi-class probe under StratifiedGroupKFold."""
    labels = _relabel(labels)
    k = max(2, min(5, min(np.bincount(labels))))
    pred = np.full(len(labels), -1)
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=rs).split(Z, labels, groups):
        pred[te] = make_clf().fit(Z[tr], labels[tr]).predict(Z[te])
    return balanced_accuracy_score(labels, pred), labels, groups


def subject_recoverability(zc, subject, uid, mask, cap=8000, rs=0, pca_dim=None):
    """Max-over-probes multi-class subject recoverability on subjects in `mask`.

    Rows are subsampled to at most `cap` windows. `pca_dim` fixes the number of PCA
    components (e.g. 64 to match DINQ's z_c dimensionality) instead of keeping 0.95
    variance. This keeps a high-dimensional baseline (the raw recentered tangent)
    like-for-like with DINQ and tractable: HGB over K~92 classes takes ~20 min per
    fit on ~300 PCs and seconds on 64."""
    idx = np.where(mask)[0]
    if len(idx) > cap:
        idx = np.random.default_rng(rs).choice(idx, cap, replace=False)
    n_comp = min(pca_dim, zc.shape[1], len(idx)) if pca_dim else 0.95
    Z = PCA(n_comp, svd_solver="full", random_state=rs).fit_transform(zc[idx].astype(float))
    s, g = subject[idx], uid[idx]
    out = {}
    for nm, mk in (("linear", _linear), ("mlp", lambda: _deep_mlp(rs)), ("hgb", lambda: _deep_hgb(rs))):
        try:
            out[nm] = float(_grouped_multiclass(Z, s, g, mk, rs)[0])
        except Exception:
            out[nm] = float("nan")
    out["K"] = int(len(np.unique(s)))
    return out


def task_perm_null_ceiling(X, label, uid, train_mask, test_mask, n_perm=30, rs=0):
    """Task floor from a permutation null, symmetric with the subject ceiling.

    Emotion labels are shuffled at trial level and the same train-on-train /
    test-on-held-out emotion probe is rerun; the floor is the 95th percentile.
    Features are aggregated to trial level once, so only the shuffled labels
    change per permutation."""
    Xtr, ytr = _trial_agg(X[train_mask], label[train_mask], uid[train_mask])
    Xte, yte = _trial_agg(X[test_mask], label[test_mask], uid[test_mask])
    all_lab = np.concatenate([ytr, yte]); ntr = len(ytr)
    vals = []
    for r in tqdm(range(n_perm), desc="task-null", leave=False):
        perm = np.random.RandomState(rs * 1000 + r).permutation(all_lab)
        yp_tr, yp_te = perm[:ntr], perm[ntr:]
        if len(np.unique(yp_tr)) < 2 or len(np.unique(yp_te)) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        vals.append(float(balanced_accuracy_score(yp_te, clf.fit(Xtr, yp_tr).predict(Xte))))
    return float(np.percentile(vals, 95)) if vals else float("nan")


def perm_null_ceiling(zc, subject, uid, mask, cap=8000, n_perm=20, rs=0):
    """Subject-recoverability ceiling: 95th-percentile HGB balanced accuracy under
    trial-level subject-label shuffles (PCA fit once; only labels change)."""
    idx = np.where(mask)[0]
    if len(idx) > cap:
        idx = np.random.default_rng(rs).choice(idx, cap, replace=False)
    Z = PCA(0.95, svd_solver="full", random_state=rs).fit_transform(zc[idx].astype(float))
    s, g = _relabel(subject[idx]), uid[idx]
    trials = np.unique(g); tri_subj = np.array([s[g == t][0] for t in trials])
    null = []
    for r in tqdm(range(n_perm), desc="subj-null", leave=False):
        perm = np.random.RandomState(r).permutation(tri_subj)
        tmap = dict(zip(trials, perm)); s_perm = np.array([tmap[t] for t in g])
        try:
            null.append(_grouped_multiclass(Z, s_perm, g, lambda: _deep_hgb(rs), rs)[0])
        except Exception:
            pass
    null = np.array(null)
    return float(np.percentile(null, 95)) if len(null) else float("nan")


def perm_null_distribution(zc, subject, uid, mask, cap=8000, n_perm=200, rs=0,
                           probes=("linear", "mlp", "hgb"), pca_dim=None):
    """Permutation-null distribution of max-over-probes subject recoverability.

    Subject labels are shuffled at trial granularity (every window of a trial gets
    that trial's permuted label), and the statistic is the same max over probes as
    the observed recoverability. PCA is fit once; only labels change per
    permutation.

    Returns (null_max, null_by_probe): `null_max` is the per-permutation
    max-over-probes balanced accuracy (chance ~1/K); `null_by_probe[p]` holds the
    per-probe arrays. `pca_dim` caps the PCA to match the observed recoverability's
    dimensionality (e.g. 64); this is essential for a high-dimensional baseline,
    where PCA(0.95) makes HGB over K~92 classes take ~20 min per permutation.
    """
    idx = np.where(mask)[0]
    if len(idx) > cap:
        idx = np.random.default_rng(rs).choice(idx, cap, replace=False)
    n_comp = min(pca_dim, zc.shape[1], len(idx)) if pca_dim else 0.95
    Z = PCA(n_comp, svd_solver="full", random_state=rs).fit_transform(zc[idx].astype(float))
    s, g = _relabel(subject[idx]), uid[idx]
    trials = np.unique(g)
    tri_subj = np.array([s[g == t][0] for t in trials])
    mks = {"linear": _linear, "mlp": lambda: _deep_mlp(rs), "hgb": lambda: _deep_hgb(rs)}
    null_max = []
    null_by = {p: [] for p in probes}
    for r in tqdm(range(n_perm), desc="perm-null", leave=False):
        perm = np.random.RandomState(rs * 100000 + r).permutation(tri_subj)
        tmap = dict(zip(trials, perm))
        s_perm = np.array([tmap[t] for t in g])
        vals = []
        for p in probes:
            try:
                v = float(_grouped_multiclass(Z, s_perm, g, mks[p], rs)[0])
            except Exception:
                v = float("nan")
            null_by[p].append(v)
            vals.append(v)
        null_max.append(float(np.nanmax(vals)) if np.any(np.isfinite(vals)) else float("nan"))
    return np.array(null_max), {p: np.array(v) for p, v in null_by.items()}


def subject_split(rec, split_seed, test_frac=0.25):
    """Deterministic train/test subject masks for a split (shared by DINQ + baselines)."""
    subs = np.unique(rec.subject)
    rng = np.random.RandomState(split_seed); rng.shuffle(subs)
    n_test = max(2, int(round(len(subs) * test_frac)))
    test = set(subs[:n_test].tolist())
    test_mask = np.isin(rec.subject, list(test))
    return ~test_mask, test_mask, len(subs) - n_test, n_test


def evaluate_representation(zc, rec, train_mask, test_mask, split_seed, n_perm=20, cap=8000,
                            n_test=None, n_train=None, save_null_dist=False, null_n_perm=200,
                            recov_pca_dim=None):
    """Unseen-subject emotion, per-probe subject recoverability (train and held-out)
    and permutation-null ceilings for any representation `zc`.

    `zc` may be DINQ z_c, the recentering baseline (rec.X) or a reference encoder.
    `n_perm <= 0` skips the per-cell ceiling (the baseline sweep takes its null from
    faced_permtest). `recov_pca_dim` caps the recoverability PCA (see
    subject_recoverability). `save_null_dist` also stores the full held-out null
    distribution."""
    uid = trial_group_ids(rec.subject, rec.session, rec.trial_id)
    nan = float("nan")
    out = {
        "emotion_unseen": emotion_unseen(zc, rec.label, uid, train_mask, test_mask),
        "recov_train": subject_recoverability(zc, rec.subject, uid, train_mask, cap=cap, rs=split_seed, pca_dim=recov_pca_dim),
        "recov_heldout": subject_recoverability(zc, rec.subject, uid, test_mask, cap=cap, rs=split_seed, pca_dim=recov_pca_dim),
        "ceiling_train": (perm_null_ceiling(zc, rec.subject, uid, train_mask, cap=cap, n_perm=n_perm, rs=split_seed) if n_perm > 0 else nan),
        "ceiling_heldout": (perm_null_ceiling(zc, rec.subject, uid, test_mask, cap=cap, n_perm=n_perm, rs=split_seed) if n_perm > 0 else nan),
        "n_train_subjects": n_train, "n_test_subjects": n_test,
    }
    if save_null_dist:  # full held-out null distribution for the permutation-test figure
        nm, nby = perm_null_distribution(zc, rec.subject, uid, test_mask, cap=cap,
                                         n_perm=null_n_perm, rs=split_seed)
        out["null_heldout_max"] = [float(x) for x in nm]
        out["null_heldout_by_probe"] = {k: [float(x) for x in v] for k, v in nby.items()}
    return out


def evaluate_split(rec, cfg: DINQConfig, split_seed, test_frac=0.25, epochs=60,
                   device="cpu", n_perm=20, cap=8000, save_null_dist=False):
    """Train DINQ on training subjects; return evaluate_representation's results for its z_c."""
    train_mask, test_mask, n_train, n_test = subject_split(rec, split_seed, test_frac)
    m = train_minimal_dinq(rec.X[train_mask], rec.label[train_mask], cfg, epochs=epochs, device=device)
    zc = extract_zc(m, rec.X, device)
    return evaluate_representation(zc, rec, train_mask, test_mask, split_seed, n_perm=n_perm,
                                   cap=cap, n_test=n_test, n_train=n_train,
                                   save_null_dist=save_null_dist)
