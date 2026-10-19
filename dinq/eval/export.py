"""Export z / z_c to .npz in the companion framework's probe format, plus trial group ids."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .blockbootstrap import trial_group_ids


def export_probe_npz(
    path: str | Path,
    z: np.ndarray,
    y_binary: np.ndarray,
    subject: np.ndarray,
    domain: np.ndarray,
    session: np.ndarray | None = None,
    trial_id: np.ndarray | None = None,
) -> None:
    """Write one model's evaluation embeddings in the probe .npz format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        z=np.asarray(z, np.float32),
        y=np.asarray(y_binary, np.int64).reshape(-1),
        subject=np.asarray(subject, np.int64),
        domain=np.asarray(domain, np.int64),
    )
    if session is not None and trial_id is not None:
        payload["group"] = trial_group_ids(subject, session, trial_id)
    np.savez_compressed(path, **payload)


def load_probe_npz(path: str | Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}
