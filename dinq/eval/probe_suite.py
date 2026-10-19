"""Bridge to the companion framework's probes, with an adapter for a single binary emotion axis."""

from __future__ import annotations

import sys

import numpy as np

from .. import paths as P

if str(P.PAPER1_REPO) not in sys.path:
    sys.path.insert(0, str(P.PAPER1_REPO))

# Companion framework imports (unmodified).
from diagnostic_framework.data.embedding_loader import EmbeddingSet  # noqa: E402
from diagnostic_framework.probes.p1_task import task_probe  # noqa: E402
from diagnostic_framework.probes.p2_domain import domain_probe  # noqa: E402
from diagnostic_framework.probes.p5_subspace import subspace_probe_with_ci  # noqa: E402
from diagnostic_framework.probes.p6_conditional import (  # noqa: E402
    conditional_domain_probe_with_ci,
)
from diagnostic_framework.taxonomy import (  # noqa: E402
    classify,
    load_thresholds,
    Thresholds,
)

BINARY_AXIS = 0  # DINQ's single emotion axis lives in column 0.


def build_embedding_set(
    z: np.ndarray,
    y_binary: np.ndarray,
    subject: np.ndarray,
    domain: np.ndarray,
    model_name: str,
    split_name: str = "deap_seedv_cross_corpus",
) -> EmbeddingSet:
    """Wrap DINQ arrays as an EmbeddingSet with the binary label on axis 0.

    y_binary is duplicated into both columns so `es.axis(0)` (used by the probes)
    returns the pos/neg emotion label directly, with no median split.
    """
    z = np.asarray(z, dtype=np.float32)
    y = np.asarray(y_binary, dtype=np.int64).reshape(-1)
    yb = np.stack([y, y], axis=1)  # (N, 2), both cols = the binary emotion label
    return EmbeddingSet(
        z=z,
        y_cont=yb.astype(np.float32),
        y_binary=yb,
        subject=np.asarray(subject, dtype=np.int64),
        domain=np.asarray(domain, dtype=np.int64),
        model_name=model_name,
        split_name=split_name,
    )
