"""Disagreement signal and error definitions (DESIGN.md §2-3). Pure numpy, no labels needed
for the disagreement functions; the error functions are only ever called where labels exist
(Oracle, Skeptic references, final scoring)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trustswarm.config import WRONG_E, WRONG_F


def force_spread(forces: np.ndarray) -> tuple[float, float]:
    """Cross-model force disagreement for one structure.

    ``forces``: (M, n, 3). Per-atom spread ``s_i = sqrt(mean_m ||F_m,i - F̄_i||²)``.
    Returns ``(σ_F, σ_F_max) = (mean_i s_i, max_i s_i)`` in eV/Å.
    """
    f = np.asarray(forces, dtype=float)
    if f.ndim != 3 or f.shape[-1] != 3:
        raise ValueError(f"forces must have shape (M, n, 3), got {f.shape}")
    if f.shape[0] < 2:
        raise ValueError("force_spread needs at least two models")
    dev = f - f.mean(axis=0, keepdims=True)
    s = np.sqrt(np.mean(np.sum(dev * dev, axis=-1), axis=0))
    return float(s.mean()), float(s.max())


def model_deviations(forces: np.ndarray) -> np.ndarray:
    """Per-model deviation from the ensemble mean, ``d_m = mean_i ||F_m,i - F̄_i||`` (M,)."""
    f = np.asarray(forces, dtype=float)
    dev = f - f.mean(axis=0, keepdims=True)
    return np.mean(np.linalg.norm(dev, axis=-1), axis=1)


def served_spread(forces: Mapping[str, np.ndarray], members: Sequence[str]) -> float:
    """σ_served: uncertainty of the answer actually served (a mean over ``members``).

    * ``|members| ≥ 2``: the members' own force spread, ``force_spread(F_members)[0]``.
    * one member c: its distance from the full committee mean F̄ over all models in
      ``forces``, ``sqrt(mean_i ||F_c,i − F̄_i||²)`` [eV/Å] — how far the served model sits
      from the collective.
    """
    if len(members) >= 2:
        return force_spread(np.stack([np.asarray(forces[m], dtype=float) for m in members]))[0]
    (c,) = members
    fbar = np.mean([np.asarray(f, dtype=float) for f in forces.values()], axis=0)
    dev = np.asarray(forces[c], dtype=float) - fbar
    return float(np.sqrt(np.mean(np.sum(dev * dev, axis=-1))))


def energy_spread(rel_energies: Sequence[float]) -> float:
    """σ_E = std_m(E_m - E_m,reactant): spread of *relative* energies across models."""
    e = np.asarray(rel_energies, dtype=float)
    return float(np.std(e)) if e.size >= 2 else 0.0


def mean_abs_force(F: np.ndarray) -> float:
    """mean_i |F_i| (used on the ensemble-mean forces F̄ as a context feature)."""
    return float(np.mean(np.linalg.norm(np.asarray(F, dtype=float), axis=-1)))


def max_abs_force(F: np.ndarray) -> float:
    """max_i |F_i|: the single-model heuristic uncertainty score (arm A)."""
    return float(np.max(np.linalg.norm(np.asarray(F, dtype=float), axis=-1)))


def force_rmse(F_hat: np.ndarray, F: np.ndarray) -> float:
    """sqrt(mean over atoms and xyz of (F̂ - F)²) [eV/Å]."""
    a, b = np.asarray(F_hat, dtype=float), np.asarray(F, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rel_e_err(E_hat: float, E_hat_reactant: float, E: float, E_reactant: float) -> float:
    """|(Ê - Ê_reactant) - (E - E_reactant)| [eV]; per-model constant offsets cancel."""
    return float(abs((E_hat - E_hat_reactant) - (E - E_reactant)))


def is_wrong(force_rmse_value: float, rel_e_err_value: float | None,
             wrong_f: float = WRONG_F, wrong_e: float = WRONG_E) -> bool:
    """wrong ⇔ force_rmse > WRONG_F or rel_e_err > WRONG_E (rel_e_err may be unknown)."""
    if force_rmse_value > wrong_f:
        return True
    return rel_e_err_value is not None and rel_e_err_value > wrong_e


@dataclass(frozen=True)
class FrameError:
    force_rmse: float
    rel_e_err: float | None

    def wrong(self, wrong_f: float = WRONG_F, wrong_e: float = WRONG_E) -> bool:
        return is_wrong(self.force_rmse, self.rel_e_err, wrong_f, wrong_e)

    def to_dict(self) -> dict[str, float | bool | None]:
        return {"force_rmse": self.force_rmse, "rel_e_err": self.rel_e_err, "wrong": self.wrong()}


def frame_error(F_hat: np.ndarray, F: np.ndarray, rel_hat: float | None,
                rel_true: float | None) -> FrameError:
    """Error of a prediction given *relative* energies (Ê-Ê_r and E-E_r) and forces."""
    rel = None if rel_hat is None or rel_true is None else float(abs(rel_hat - rel_true))
    return FrameError(force_rmse(F_hat, F), rel)
