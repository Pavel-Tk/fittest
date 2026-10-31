"""Prediction cache: the hand-off from the GPU stage to the CPU agent stage (DESIGN.md §9).

``data/frames.pkl`` holds labelled frames and is read only by the Oracle, the Skeptic and
final scoring. ``data/preds/{model}.npz`` holds one model's energies and forces for any set
of uids, with a ragged force layout (``forces_flat`` + ``offsets``) and JSON metadata.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trustswarm.data import Frame


def save_frames(path: str | Path, frames: list[Frame]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(list(frames), fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_frames(path: str | Path) -> list[Frame]:
    with Path(path).open("rb") as fh:
        frames = pickle.load(fh)
    if not all(isinstance(f, Frame) for f in frames):
        raise TypeError(f"{path} does not contain a list[Frame]")
    return frames


@dataclass
class Predictions:
    """One model's cached predictions, keyed by frame uid."""

    model: str
    energy: dict[str, float]
    forces: dict[str, np.ndarray]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def uids(self) -> list[str]:
        return list(self.energy)

    def __contains__(self, uid: object) -> bool:
        return uid in self.energy

    def __len__(self) -> int:
        return len(self.energy)

    def subset(self, uids: Iterable[str]) -> Predictions:
        keep = [u for u in uids if u in self.energy]
        return Predictions(self.model, {u: self.energy[u] for u in keep},
                           {u: self.forces[u] for u in keep}, dict(self.meta))


def save_predictions(path: str | Path, model: str, uids: list[str], energies: np.ndarray,
                     forces: list[np.ndarray], meta: dict[str, Any]) -> None:
    """Write ``uids``, ``energy`` (N,), ``forces_flat`` (Σn_i, 3), ``offsets`` (N+1), ``meta_json``."""
    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    if not (len(uids) == len(energies) == len(forces)):
        raise ValueError("uids, energies and forces must have the same length")
    if len(set(uids)) != len(uids):
        raise ValueError("duplicate uids")
    sizes = np.array([len(f) for f in forces], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    flat = (np.concatenate([np.asarray(f, dtype=np.float64).reshape(-1, 3) for f in forces])
            if forces else np.zeros((0, 3)))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, uids=np.array(uids, dtype=str), energy=energies, forces_flat=flat,
                        offsets=offsets, meta_json=np.array(json.dumps({"model": model, **meta})))


def load_predictions(path: str | Path) -> Predictions:
    with np.load(Path(path), allow_pickle=False) as z:
        uids = [str(u) for u in z["uids"]]
        energy = z["energy"]
        flat = z["forces_flat"]
        off = z["offsets"]
        meta = json.loads(str(z["meta_json"]))
    model = str(meta.pop("model", Path(path).stem))
    return Predictions(
        model=model,
        energy={u: float(e) for u, e in zip(uids, energy)},
        forces={u: flat[off[i]:off[i + 1]].copy() for i, u in enumerate(uids)},
        meta=meta,
    )


def prediction_path(preds_dir: str | Path, model: str) -> Path:
    return Path(preds_dir) / f"{model}.npz"


def load_models(preds_dir: str | Path, models: Sequence[str]) -> dict[str, Predictions]:
    """Load several models; raises FileNotFoundError naming *every* missing file."""
    missing = [str(prediction_path(preds_dir, m)) for m in models
               if not prediction_path(preds_dir, m).exists()]
    if missing:
        raise FileNotFoundError("missing prediction file(s): " + ", ".join(missing))
    return {m: load_predictions(prediction_path(preds_dir, m)) for m in models}
