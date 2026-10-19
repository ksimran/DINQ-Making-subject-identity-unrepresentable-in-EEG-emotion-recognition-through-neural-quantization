"""Covariance-to-tangent FeatureSets with the tangent reference fitted on source-train only."""

from __future__ import annotations

import numpy as np

from .covariance import CovarianceSet
from .tangent import BandTangent
from .featureset import FeatureSet


def _to_featureset(cs: CovarianceSet, bt: BandTangent) -> FeatureSet:
    return FeatureSet(
        X=bt.transform(cs.covs), label=cs.label, subject=cs.subject, domain=cs.domain,
        trial_id=cs.trial_id, session=cs.session, feature_kind="tangent",
        info={**cs.info, "repr": "tangent"},
    )


def covset_split_by_subject(cs: CovarianceSet, val_frac: float, seed: int):
    subs = np.unique(cs.subject)
    rng = np.random.RandomState(seed)
    rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * val_frac)))
    val = set(subs[:n_val].tolist())
    vmask = np.isin(cs.subject, list(val))
    return cs.select(~vmask), cs.select(vmask)


def tangent_direction(
    source_cov: CovarianceSet,
    target_cov: CovarianceSet,
    val_frac: float = 0.2,
    seed: int = 0,
    ref_cap: int = 3000,
) -> tuple[FeatureSet, FeatureSet, FeatureSet]:
    """Return (source_train, source_val, target) tangent FeatureSets.

    The BandTangent reference is fit on a capped subsample of source-train
    covariances only (leakage-safe), then applied to all three sets.
    """
    src_train, src_val = covset_split_by_subject(source_cov, val_frac, seed)
    ref = src_train.covs
    if ref.shape[0] > ref_cap:
        idx = np.random.RandomState(seed).choice(ref.shape[0], ref_cap, replace=False)
        ref = ref[idx]
    bt = BandTangent().fit(ref)
    return _to_featureset(src_train, bt), _to_featureset(src_val, bt), _to_featureset(target_cov, bt)
