"""FeatureSet: a feature matrix with row-aligned subject, domain, label, trial and session metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_META_KEYS = ("label", "subject", "domain", "trial_id", "session")


@dataclass
class FeatureSet:
    X: np.ndarray  # (N, D) float32
    label: np.ndarray  # (N,) int64, binary emotion (0/1)
    subject: np.ndarray  # (N,) int64, global id (SEED-V offset applied)
    domain: np.ndarray  # (N,) int64, 0=DEAP, 1=SEED-V
    trial_id: np.ndarray  # (N,) int64
    session: np.ndarray  # (N,) int64
    feature_kind: str  # e.g. 'tangent', 'tangent_recentered', 'de'
    info: dict = field(default_factory=dict)  # bands, channels, window, fs, ...

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        for k in _META_KEYS:
            v = getattr(self, k)
            if v.shape[0] != n:
                raise ValueError(f"{k} length {v.shape[0]} != N {n}")
        if self.X.dtype != np.float32:
            self.X = self.X.astype(np.float32)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    def select(self, mask: np.ndarray) -> "FeatureSet":
        """Row-subset by boolean mask or index array (keeps info)."""
        return FeatureSet(
            X=self.X[mask],
            label=self.label[mask],
            subject=self.subject[mask],
            domain=self.domain[mask],
            trial_id=self.trial_id[mask],
            session=self.session[mask],
            feature_kind=self.feature_kind,
            info=dict(self.info),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            label=self.label,
            subject=self.subject,
            domain=self.domain,
            trial_id=self.trial_id,
            session=self.session,
            feature_kind=self.feature_kind,
            info=np.array(json.dumps(self.info)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        d = np.load(path, allow_pickle=False)
        info = json.loads(str(d["info"])) if "info" in d.files else {}
        return cls(
            X=d["X"],
            label=d["label"],
            subject=d["subject"],
            domain=d["domain"],
            trial_id=d["trial_id"],
            session=d["session"],
            feature_kind=str(d["feature_kind"]),
            info=info,
        )


def concat(sets: list[FeatureSet]) -> FeatureSet:
    """Concatenate FeatureSets that share feature_kind and dim (e.g. per subject)."""
    if not sets:
        raise ValueError("nothing to concat")
    kind = sets[0].feature_kind
    dim = sets[0].dim
    for s in sets:
        if s.feature_kind != kind or s.dim != dim:
            raise ValueError("cannot concat FeatureSets of differing kind/dim")
    return FeatureSet(
        X=np.concatenate([s.X for s in sets], axis=0),
        label=np.concatenate([s.label for s in sets]),
        subject=np.concatenate([s.subject for s in sets]),
        domain=np.concatenate([s.domain for s in sets]),
        trial_id=np.concatenate([s.trial_id for s in sets]),
        session=np.concatenate([s.session for s in sets]),
        feature_kind=kind,
        info=dict(sets[0].info),
    )
