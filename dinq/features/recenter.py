"""Label-free per-subject Riemannian recentring of band covariances (Zanini et al. 2018),
applied to every method before the tangent map."""

from __future__ import annotations

import numpy as np

from .covariance import CovarianceSet
from .featureset import FeatureSet

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars optional
    def tqdm(x, **k):
        return x


def _upper(logmats: np.ndarray) -> np.ndarray:
    """Frobenius-preserving upper-triangle vectorization (off-diags x sqrt(2))."""
    C = logmats.shape[-1]
    di = np.diag_indices(C)
    iu = np.triu_indices(C, 1)
    out = np.empty((logmats.shape[0], C * (C + 1) // 2), dtype=np.float64)
    out[:, :C] = logmats[:, di[0], di[1]]
    out[:, C:] = np.sqrt(2.0) * logmats[:, iu[0], iu[1]]
    return out


def recenter_covariances(covs: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """(N,B,C,C) -> (N,B,C,C), each group's band covariances recentered at identity.

    Sigma' = M^-1/2 Sigma M^-1/2, with M the group's per-band Frechet (Riemannian)
    mean. `groups` is the subject id in the standard pipeline.
    """
    from pyriemann.utils.mean import mean_riemann
    from pyriemann.utils.base import invsqrtm

    covs = np.asarray(covs, dtype=np.float64)
    out = np.empty_like(covs)
    B = covs.shape[1]
    uniq = np.unique(groups)
    for g in tqdm(uniq, desc="recenter (Frechet means)", unit="subj"):
        m = groups == g
        for b in range(B):
            Mi = invsqrtm(mean_riemann(covs[m, b]))
            out[m, b] = np.einsum("ij,njk,kl->nil", Mi, covs[m, b], Mi)
    return out


def recentered_tangent_featureset(cs: CovarianceSet, group_by: str = "subject") -> FeatureSet:
    """CovarianceSet -> per-group-recentered tangent FeatureSet (reference = identity).

    After recentering, every group sits at identity, so the tangent map is just
    logm at identity: no source-fit reference, hence no reference-offset artifact.
    """
    from pyriemann.utils.base import logm

    groups = cs.subject if group_by == "subject" else getattr(cs, group_by)
    rc = recenter_covariances(cs.covs, groups)
    B = rc.shape[1]
    feats = [_upper(np.stack([logm(w) for w in rc[:, b]]))
             for b in tqdm(range(B), desc="tangent (logm/band)", unit="band")]
    X = np.concatenate(feats, axis=1).astype(np.float32)
    return FeatureSet(
        X=X, label=cs.label, subject=cs.subject, domain=cs.domain,
        trial_id=cs.trial_id, session=cs.session, feature_kind="tangent_recentered",
        info={**cs.info, "repr": "tangent", "recenter": group_by},
    )
