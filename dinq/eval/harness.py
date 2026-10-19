"""Cross-corpus harness: target task accuracy and certificate cell for both transfer directions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..features.featureset import FeatureSet, concat
from ..data.splits import make_cross_corpus
from .certificate import certify, CertificateCell
from .probe_suite import Thresholds


@dataclass
class DirectionResult:
    direction: str
    task_target_acc: float  # axis 1: cross-corpus task accuracy (bal-acc)
    category: str  # axis 2: taxonomy cell
    ci_stable: bool
    cell: CertificateCell

    def summary(self) -> dict[str, Any]:
        return {"direction": self.direction, "task_target_acc": self.task_target_acc,
                "category": self.category, "ci_stable": self.ci_stable}


@dataclass
class TwoAxisResult:
    model_name: str
    directions: dict[str, DirectionResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model_name,
                "directions": {k: {**v.summary(), "cell": v.cell.to_dict()}
                               for k, v in self.directions.items()}}


def evaluate_both_directions(
    deap_fs: FeatureSet,
    seedv_fs: FeatureSet,
    thresholds: Thresholds,
    model_name: str = "model",
    val_frac: float = 0.2,
    n_block: int = 200,
    seed: int = 0,
) -> TwoAxisResult:
    """Run deap->seedv and seedv->deap; return task accuracy + certificate per direction."""
    res = TwoAxisResult(model_name=model_name)
    # (source, target, target_domain, source_domain)
    plan = [(deap_fs, seedv_fs, 1, 0), (seedv_fs, deap_fs, 0, 1)]
    for src, tgt, tdom, sdom in plan:
        split = make_cross_corpus(src, tgt, val_frac=val_frac, seed=seed)  # name + leakage assert
        # Pool all source and target subjects so subject-grouped CV has enough
        # groups per domain (source_val alone can have too few subjects).
        pooled = concat([src, tgt])
        # Group by subject: subjects are domain-homogeneous, so trial-level grouping
        # would let the probe recover domain from subject identity.
        groups = pooled.subject
        cell = certify(pooled.X, pooled.label, pooled.domain, groups,
                       thresholds=thresholds, target_domain=tdom, source_domain=sdom,
                       n_block=n_block, rs=seed)
        res.directions[split.direction] = DirectionResult(
            direction=split.direction,
            task_target_acc=cell.quantities["task_target_acc"]["point"],
            category=cell.category,
            ci_stable=cell.ci_stable,
            cell=cell,
        )
    return res
