"""Leakage-safe cross-corpus splits: the source corpus is split into train/val by whole
subjects and the target corpus is used only for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.featureset import FeatureSet, concat


def subject_holdout(
    fs: FeatureSet,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[FeatureSet, FeatureSet]:
    """Partition a FeatureSet into (train, val) by whole subjects."""
    subs = np.unique(fs.subject)
    rng = np.random.RandomState(seed)
    rng.shuffle(subs)
    n_val = max(1, int(round(len(subs) * val_frac)))
    val_subs = set(subs[:n_val].tolist())
    val_mask = np.isin(fs.subject, list(val_subs))
    return fs.select(~val_mask), fs.select(val_mask)


@dataclass
class CrossCorpusSplit:
    """One transfer direction, ready for training + probing."""

    direction: str  # e.g. 'deap->seedv'
    source_train: FeatureSet
    source_val: FeatureSet
    target_eval: FeatureSet

    def pooled_eval(self) -> FeatureSet:
        """Source(val)+target pooled representation for the domain probes.

        Uses source val (not train) so the domain probe sees held-out source
        windows, matching how the target side is unseen.
        """
        return concat([self.source_val, self.target_eval])


def make_cross_corpus(
    source: FeatureSet,
    target: FeatureSet,
    val_frac: float = 0.2,
    seed: int = 0,
) -> CrossCorpusSplit:
    """Build a leakage-safe cross-corpus split from two per-corpus FeatureSets."""
    if source.feature_kind != target.feature_kind or source.dim != target.dim:
        raise ValueError("source/target FeatureSets must share kind and dim")
    dsrc = int(np.unique(source.domain)[0])
    dtgt = int(np.unique(target.domain)[0])
    if dsrc == dtgt:
        raise ValueError("source and target share a domain id — not cross-corpus")
    tr, va = subject_holdout(source, val_frac=val_frac, seed=seed)
    name = {0: "deap", 1: "seedv"}
    split = CrossCorpusSplit(
        direction=f"{name.get(dsrc, dsrc)}->{name.get(dtgt, dtgt)}",
        source_train=tr,
        source_val=va,
        target_eval=target,
    )
    assert_no_subject_leakage(split)
    return split


def assert_no_subject_leakage(split: CrossCorpusSplit) -> None:
    """Raise on any subject overlap or cross-corpus contamination."""
    tr = set(split.source_train.subject.tolist())
    va = set(split.source_val.subject.tolist())
    te = set(split.target_eval.subject.tolist())
    if tr & va:
        raise AssertionError(f"subject leakage train/val: {sorted(tr & va)}")
    if tr & te or va & te:
        raise AssertionError("source/target subject-id overlap (offset not applied?)")
    tr_dom = set(split.source_train.domain.tolist())
    te_dom = set(split.target_eval.domain.tolist())
    if tr_dom & te_dom:
        raise AssertionError("source_train and target_eval share a domain id")
