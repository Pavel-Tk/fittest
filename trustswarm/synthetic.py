"""Synthetic stand-in for T1x + cached MLIP predictions (CPU smoke test, no GPU, no T1x).

What is built in, and why
-------------------------
* **Labels.** Small C/H/N/O "molecules" on 8-frame reaction paths (reactant, product, TS and
  path images). Reference energies/forces come from a pairwise Morse potential with
  analytic forces, so E and F are mutually consistent.
* **Model errors.** Each model's force error is a sum of
  (i) a *shared* component every model gets (hard chemistry nobody has learned — here
  disagreement cannot help, by design), and (ii) a *model-specific* systematic component
  whose amplitude depends on label-free features (bond stretch, composition), with a
  different blind spot per heterogeneous model. MACE-MPA-0 has a "confidently wrong"
  region (≥2 N atoms) — the planted-failure region.
* **Ensembles.** ``aimnet2_m0..3`` share AIMNet2's systematic error plus small member
  noise and ``aimnet2`` is their mean (as the shipped model is). ``mace_seed0..3`` likewise
  share MACE's systematic error. So seed ensembles disagree *less* where they are wrong than
  heterogeneous models do — that is the hypothesis encoded, not the ablation outcome:
  the outcome depends on how the agents spend the budget, on the shared component, and on
  noise, and is not fixed here.
* **Offsets.** Every model has a per-atom constant energy offset (a level-of-theory shift)
  that cancels in relative energies.
* **Fine-tune replay.** :func:`write_curated` simulates a LoRA fine-tune of MACE-MPA-0 on a
  given set of oracle-labelled uids (shrinking, not removing, its systematic error), with an
  acceptance report on 10% of those labels held out and ``train_uids`` recorded for the
  replay's provenance check — the same contract as the GPU script.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trustswarm.cache import save_frames, save_predictions
from trustswarm.config import CURATED_BASE_MODEL, CURATED_MODEL, WRONG_F
from trustswarm.data import (COVALENT_RADII, Frame, assign_splits, composition_counts,
                             hill_formula, max_bond_stretch)

ELEMENTS = np.array([1, 6, 7, 8])
ELEMENT_P = np.array([0.45, 0.30, 0.12, 0.13])
MORSE_D, MORSE_A = 1.2, 1.6  # eV, 1/Å
MODEL_OFFSET_PER_ATOM = {"mace_mpa0": -0.31, "aimnet2": 0.12, "orb": -0.05}
ENERGY_ERROR_SCALE = 0.25  # eV per (eV/Å of force-error amplitude × sqrt(atom))


@dataclass(frozen=True)
class SyntheticSpec:
    n_train: int = 20
    n_val: int = 80
    n_test: int = 200
    n_path: int = 5
    min_atoms: int = 5
    max_atoms: int = 12
    seed: int = 0


# --- reference potential -------------------------------------------------------------------

def morse_energy_forces(Z: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray]:
    r0 = np.array([COVALENT_RADII[int(z)] for z in Z])
    r0 = r0[:, None] + r0[None, :]
    diff = R[:, None, :] - R[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(len(Z), k=1)
    x = np.exp(-MORSE_A * (d - r0))
    energy = float(np.sum((MORSE_D * ((1 - x) ** 2 - 1))[iu]))
    dE_dd = 2 * MORSE_D * MORSE_A * (1 - x) * x
    np.fill_diagonal(dE_dd, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(d[..., None] > 0, diff / d[..., None], 0.0)
    forces = -np.sum(dE_dd[..., None] * unit, axis=1)
    return energy, forces


def _molecule(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Grow a tree-like molecule: each atom bonds to a random earlier one at ~covalent length."""
    Z = rng.choice(ELEMENTS, size=n, p=ELEMENT_P)
    Z[0] = 6
    R = np.zeros((n, 3))
    for i in range(1, n):
        for _ in range(100):
            j = rng.integers(i)
            v = rng.normal(size=3)
            bond = (COVALENT_RADII[int(Z[i])] + COVALENT_RADII[int(Z[j])]) * rng.uniform(0.97, 1.05)
            cand = R[j] + bond * v / np.linalg.norm(v)
            radii = np.array([COVALENT_RADII[int(z)] for z in Z[:i]]) + COVALENT_RADII[int(Z[i])]
            if np.all(np.linalg.norm(R[:i] - cand, axis=1) > 0.9 * radii):
                R[i] = cand
                break
        else:
            R[i] = R[i - 1] + np.array([1.5, 0.0, 0.0])
    return Z, R - R.mean(axis=0)


def _reaction_frames(rng: np.random.Generator, spec: SyntheticSpec, split: str,
                     idx: int) -> list[Frame]:
    n = int(rng.integers(spec.min_atoms, spec.max_atoms + 1))
    Z, R0 = _molecule(rng, n)
    moved = rng.choice(n, size=max(1, n // 3), replace=False)
    R1 = R0.copy()
    R1[moved] += rng.normal(scale=0.6, size=(len(moved), 3))
    bump = np.zeros_like(R0)
    bump[moved] = rng.normal(scale=0.35, size=(len(moved), 3))
    formula = hill_formula(Z)
    reaction = f"{formula}/rxn{idx:04d}"
    lams = [("reactant", 0.0), ("product", 1.0), ("ts", 0.5)]
    lams += [(f"path{k}", float(lam)) for k, lam in enumerate(np.sort(rng.uniform(0.05, 0.95, spec.n_path)))]
    frames = []
    for tag, lam in lams:
        R = (1 - lam) * R0 + lam * R1 + np.sin(np.pi * lam) * bump
        E, F = morse_energy_forces(Z, R)
        role = tag if tag in ("reactant", "product", "ts") else "path"
        frames.append(Frame(uid=f"{split}:{reaction}:{tag}", reaction=reaction, formula=formula,
                            split=split, role=role, Z=Z.astype(np.int64), R=R, E=E, F=F))
    return frames


def make_frames(spec: SyntheticSpec) -> list[Frame]:
    rng = np.random.default_rng(spec.seed)
    frames: list[Frame] = []
    idx = 0
    for split, n in (("train", spec.n_train), ("val", spec.n_val), ("test", spec.n_test)):
        for _ in range(n):
            frames.extend(_reaction_frames(rng, spec, split, idx))
            idx += 1
    return assign_splits(frames, seed=spec.seed)


# --- model error model ------------------------------------------------------------------------

def _features(f: Frame) -> dict[str, float]:
    c, h, nn, o = composition_counts(f.Z)
    return {"stretch": max(0.0, max_bond_stretch(f.Z, f.R) - 1.08), "nC": c, "nH": h,
            "nN": nn, "nO": o}


def _systematic_amplitude(model: str, x: dict[str, float]) -> float:
    s = min(x["stretch"], 1.5)
    if model == "mace_mpa0":   # materials-trained: worst on molecules; blind spot = N-rich
        return 0.14 + 0.30 * s + 0.40 * (x["nN"] >= 2)
    if model == "aimnet2":     # blind spot = O-rich
        return 0.06 + 0.15 * s + 0.30 * (x["nO"] >= 3)
    if model == "orb":         # blind spot = H-rich, stretched
        return 0.08 + 0.18 * s + 0.30 * (x["nH"] >= 7) * (s > 0.1)
    if model == CURATED_MODEL:  # LoRA on oracle labels: shrinks MACE's systematic error
        return 0.10 + 0.18 * s + 0.12 * (x["nN"] >= 2)
    raise KeyError(model)


def _shared_amplitude(x: dict[str, float]) -> float:
    """Error every model makes in the same direction (no disagreement signal at all)."""
    return 0.03 + 0.20 * min(x["stretch"], 1.5) ** 2


class _ErrorField:
    """Deterministic per-(frame, source) unit-RMS force noise and unit energy noise."""

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def __call__(self, uid: str, source: str, n: int) -> tuple[np.ndarray, float]:
        key = np.frombuffer(f"{uid}|{source}".encode(), dtype=np.uint8).astype(np.uint32)
        rng = np.random.default_rng([self._seed, *key.tolist()])
        return rng.normal(size=(n, 3)), float(rng.normal())


def _predict(frames: Sequence[Frame], field: _ErrorField, systematic: str, offset_model: str,
             member: str | None = None, member_scale: float = 0.0) -> tuple[np.ndarray, list[np.ndarray]]:
    energies, forces = [], []
    for f in frames:
        assert f.E is not None and f.F is not None
        x = _features(f)
        n = f.n_atoms
        xi_shared, eta_shared = field(f.uid, "shared", n)
        xi_sys, eta_sys = field(f.uid, systematic, n)
        a_shared, a_sys = _shared_amplitude(x), _systematic_amplitude(systematic, x)
        F = f.F + a_shared * xi_shared + a_sys * xi_sys
        E = (f.E + MODEL_OFFSET_PER_ATOM[offset_model] * n
             + ENERGY_ERROR_SCALE * np.sqrt(n) * (a_shared * eta_shared + a_sys * eta_sys))
        if member is not None:
            xi_m, eta_m = field(f.uid, member, n)
            amp = member_scale * (1.0 + 2.0 * min(x["stretch"], 1.5))
            F = F + amp * xi_m
            E = E + ENERGY_ERROR_SCALE * np.sqrt(n) * amp * eta_m
        energies.append(E)
        forces.append(F)
    return np.array(energies), forces


def write_synthetic(out_dir: str | Path, spec: SyntheticSpec = SyntheticSpec(),
                    include_mace_seeds: bool = True) -> dict[str, Path]:
    """Write ``frames.pkl`` and ``preds/*.npz`` under ``out_dir``; returns the paths."""
    out = Path(out_dir)
    frames = make_frames(spec)
    uids = [f.uid for f in frames]
    field = _ErrorField(spec.seed + 1)
    paths = {"frames": out / "frames.pkl"}
    save_frames(paths["frames"], frames)

    def write(model: str, E: np.ndarray, F: list[np.ndarray], **meta: object) -> None:
        paths[model] = out / "preds" / f"{model}.npz"
        save_predictions(paths[model], model, uids, E, F,
                         {"synthetic": True, "atoms_per_s": None, **meta})

    for model in ("mace_mpa0", "orb"):
        write(model, *_predict(frames, field, model, model))
    members = [_predict(frames, field, "aimnet2", "aimnet2", f"aimnet2_m{k}", 0.025) for k in range(4)]
    for k, (E, F) in enumerate(members):
        write(f"aimnet2_m{k}", E, F, ensemble_of="aimnet2")
    write("aimnet2", np.mean([E for E, _ in members], axis=0),
          [np.mean([Fs[i] for _, Fs in members], axis=0) for i in range(len(frames))],
          note="mean of aimnet2_m0..3")
    if include_mace_seeds:
        for k in range(4):
            write(f"mace_seed{k}", *_predict(frames, field, "mace_mpa0", "mace_mpa0",
                                            f"mace_seed{k}", 0.02), lora_seed=k)

    # No replay is written here: like the real pipeline, a fine-tune needs the Curator's
    # labels first (run the ablation, then :func:`write_curated` with the recorded uids).
    return paths


def write_curated(out_dir: str | Path, train_uids: Sequence[str],
                  spec: SyntheticSpec = SyntheticSpec()) -> Path:
    """Simulated offline LoRA fine-tune of MACE-MPA-0 on the given oracle-labelled uids.

    Mirrors ``scripts/finetune.py``: 10% of the given labels are held out for the acceptance
    report and the file records ``train_uids`` so the replay's provenance can be checked."""
    out = Path(out_dir)
    frames = make_frames(spec)
    by_uid = {f.uid: f for f in frames}
    labelled = sorted(u for u in train_uids if u in by_uid)
    if not labelled:
        raise ValueError("write_curated needs at least one known labelled uid")
    rng = np.random.default_rng(spec.seed + 2)
    held = set(rng.choice(labelled, size=max(1, len(labelled) // 10), replace=False).tolist())
    train = [u for u in labelled if u not in held]
    field = _ErrorField(spec.seed + 1)
    E_cur, F_cur = _predict(frames, field, CURATED_MODEL, CURATED_BASE_MODEL)
    E_base, F_base = _predict(frames, field, CURATED_BASE_MODEL, CURATED_BASE_MODEL)
    index = {f.uid: i for i, f in enumerate(frames)}

    def rmse(F_list: list[np.ndarray]) -> float:
        return float(np.mean([np.sqrt(np.mean((F_list[index[u]] - by_uid[u].F) ** 2)) for u in held]))

    base_rmse, cur_rmse = rmse(F_base), rmse(F_cur)
    path = out / "preds" / f"{CURATED_MODEL}.npz"
    save_predictions(path, CURATED_MODEL, [f.uid for f in frames], E_cur, F_cur,
                     {"synthetic": True, "base_model": CURATED_BASE_MODEL, "weights": CURATED_MODEL,
                      "train_uids": train, "holdout_uids": sorted(held),
                      "acceptance_report": {"verdict": "pass" if cur_rmse < base_rmse else "fail",
                                            "heldout_force_rmse_base": base_rmse,
                                            "heldout_force_rmse_finetuned": cur_rmse,
                                            "n_heldout": len(held), "wrong_f": WRONG_F,
                                            "note": "synthetic: simulated LoRA fine-tune"}})
    return path
