"""Frames, label stripping, reaction-level splits and label-free descriptors (DESIGN.md §1, §4).

The Transition1x HDF5 reader is *not* here: it lives in :mod:`trustswarm.t1x`
(``trustswarm.t1x.sample_split(path, split, n_reactions, n_path, seed) -> list[Frame]``),
which returns frames tagged with the *official* T1x split (``train``/``val``/``test``).
:func:`assign_splits` then maps those onto the TrustSwarm splits, always by reaction.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# --- Split names (DESIGN.md §1) ---------------------------------------------------------
POOL = "pool"
EVAL = "eval"
SKEPTIC_REF = "skeptic_ref"
PLANTED = "planted"
SEED_TRAIN = "seed_train"
TRUSTSWARM_SPLITS: tuple[str, ...] = (POOL, EVAL, SKEPTIC_REF, PLANTED, SEED_TRAIN)
OFFICIAL_SPLITS: tuple[str, ...] = ("train", "val", "test")

ROLES: tuple[str, ...] = ("reactant", "product", "ts", "path")

# Covalent radii [Å], Cordero et al., Dalton Trans. 2008.
COVALENT_RADII: Mapping[int, float] = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
}
ELEMENT_SYMBOLS: Mapping[int, str] = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl",
}
DESCRIPTOR_ELEMENTS: tuple[int, ...] = (6, 1, 7, 8)  # C, H, N, O


@dataclass
class Frame:
    """One structure on a reaction path. ``E``/``F`` are DFT labels, ``None`` once stripped."""

    uid: str
    reaction: str
    formula: str
    split: str
    role: str
    Z: np.ndarray
    R: np.ndarray
    E: float | None = None
    F: np.ndarray | None = None

    @property
    def n_atoms(self) -> int:
        return int(len(self.Z))

    @property
    def has_labels(self) -> bool:
        return self.E is not None or self.F is not None


def strip_labels(frame: Frame) -> Frame:
    """Copy of ``frame`` with the DFT labels removed (geometry arrays are copied too)."""
    return dataclasses.replace(frame, Z=frame.Z.copy(), R=frame.R.copy(), E=None, F=None)


def strip_all(frames: Iterable[Frame]) -> list[Frame]:
    return [strip_labels(f) for f in frames]


def hill_formula(Z: np.ndarray) -> str:
    """Hill-order chemical formula (C, H first, then alphabetical)."""
    counts = Counter(ELEMENT_SYMBOLS.get(int(z), f"Z{int(z)}") for z in Z)
    order = [s for s in ("C", "H") if s in counts] + sorted(s for s in counts if s not in ("C", "H"))
    return "".join(f"{s}{counts[s] if counts[s] > 1 else ''}" for s in order)


# --- Grouping / splits ------------------------------------------------------------------

def by_split(frames: Iterable[Frame]) -> dict[str, list[Frame]]:
    out: dict[str, list[Frame]] = defaultdict(list)
    for f in frames:
        out[f.split].append(f)
    return dict(out)


def by_reaction(frames: Iterable[Frame]) -> dict[str, list[Frame]]:
    out: dict[str, list[Frame]] = defaultdict(list)
    for f in frames:
        out[f.reaction].append(f)
    return dict(out)


def reactant_index(frames: Iterable[Frame]) -> dict[str, Frame]:
    """reaction -> its reactant frame (the anchor for relative energies)."""
    return {f.reaction: f for f in frames if f.role == "reactant"}


def split_reactions(reactions: Iterable[str], fraction: float,
                    rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """Partition reactions into (first ≈ fraction, rest). Order-independent, seeded."""
    names = sorted(set(reactions))
    perm = rng.permutation(len(names))
    n_first = int(round(fraction * len(names)))
    first = sorted(names[i] for i in perm[:n_first])
    rest = sorted(names[i] for i in perm[n_first:])
    return first, rest


def assign_splits(frames: Sequence[Frame], seed: int = 0,
                  n_seed_train_reactions: int | None = None) -> list[Frame]:
    """Map official T1x splits onto TrustSwarm splits, by reaction (DESIGN.md §1).

    test → pool / eval (50/50 by reaction); val → skeptic_ref / planted (50/50 **by formula**,
    see :func:`split_val_by_formula`); train → seed_train (optionally subsampled to
    ``n_seed_train_reactions``). Frames already carrying a TrustSwarm split are returned
    unchanged.
    """
    rng = np.random.default_rng(seed)
    official = by_split(frames)

    def reactions_of(split: str) -> list[str]:
        return sorted({f.reaction for f in official.get(split, [])})

    pool, eval_ = split_reactions(reactions_of("test"), 0.5, rng)
    train = reactions_of("train")
    if n_seed_train_reactions is not None and n_seed_train_reactions < len(train):
        train, _ = split_reactions(train, n_seed_train_reactions / len(train), rng)

    mapping: dict[tuple[str, str], str] = {}
    for source, names, target in (("test", pool, POOL), ("test", eval_, EVAL),
                                  ("train", train, SEED_TRAIN)):
        mapping.update({(source, r): target for r in names})
    out: list[Frame] = []
    for f in frames:
        if f.split in TRUSTSWARM_SPLITS or f.split == "val":
            out.append(f)
        elif (f.split, f.reaction) in mapping:
            out.append(dataclasses.replace(f, split=mapping[(f.split, f.reaction)]))
    return split_val_by_formula(out, seed)


def official_split(frame: Frame) -> str:
    """The official T1x split a frame came from (uids are ``"{official}:{reaction}:{tag}"``)."""
    if frame.split in OFFICIAL_SPLITS:
        return frame.split
    prefix = frame.uid.split(":", 1)[0]
    if prefix in OFFICIAL_SPLITS:
        return prefix
    if frame.split in (SKEPTIC_REF, PLANTED):
        return "val"
    return frame.split


def split_val_by_formula(frames: Sequence[Frame], seed: int = 0) -> list[Frame]:
    """(Re-)derive skeptic_ref / planted from the official ``val`` frames, 50/50 **by formula**.

    T1x val holds a handful of formulas with many reactions each, so a reaction-level split
    would leave near-duplicate geometries (same formula) on both sides and the Skeptic's kNN
    would find the planted cases' twins among its references. Deterministic in ``seed``;
    every other frame is returned unchanged. Predictions are per uid, so they stay valid.
    """
    val = [f for f in frames if official_split(f) == "val"]
    ref_formulas, _ = split_reactions({f.formula for f in val}, 0.5,
                                      np.random.default_rng([seed, 0xF0]))
    ref = set(ref_formulas)
    return [dataclasses.replace(f, split=SKEPTIC_REF if f.formula in ref else PLANTED)
            if official_split(f) == "val" else f for f in frames]


def arrival_rounds(frames: Sequence[Frame], n_rounds: int, seed: int = 0) -> dict[str, int]:
    """Stream schedule: uid -> round in which it arrives. Whole reactions arrive together
    (so the reactant is always present for relative energies), in a seeded order."""
    reactions = sorted({f.reaction for f in frames})
    perm = np.random.default_rng(seed).permutation(len(reactions))
    chunk_of = {reactions[j]: int(i * n_rounds // max(len(reactions), 1)) for i, j in enumerate(perm)}
    return {f.uid: chunk_of[f.reaction] for f in frames}


# --- Label-free descriptors (Skeptic, DESIGN.md §4) --------------------------------------

def composition_counts(Z: np.ndarray) -> np.ndarray:
    """Counts of (C, H, N, O)."""
    z = np.asarray(Z)
    return np.array([np.sum(z == e) for e in DESCRIPTOR_ELEMENTS], dtype=float)


def pairwise_distances(R: np.ndarray) -> np.ndarray:
    diff = R[:, None, :] - R[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def min_interatomic_distance(R: np.ndarray) -> float:
    if len(R) < 2:
        return float("inf")
    d = pairwise_distances(np.asarray(R, dtype=float))
    return float(np.min(d[np.triu_indices(len(R), k=1)]))


def max_bond_stretch(Z: np.ndarray, R: np.ndarray) -> float:
    """max over atoms of (nearest-neighbour distance / sum of covalent radii).

    ≈1 for an equilibrium bonded geometry; >1 flags a stretched or breaking bond (TS-like).
    """
    if len(Z) < 2:
        return 0.0
    d = pairwise_distances(np.asarray(R, dtype=float))
    np.fill_diagonal(d, np.inf)
    r = np.array([COVALENT_RADII.get(int(z), 0.75) for z in Z])
    nearest = np.argmin(d, axis=1)
    ratio = d[np.arange(len(Z)), nearest] / (r + r[nearest])
    return float(np.max(ratio))


def geometry_descriptor(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """[nC, nH, nN, nO, n_atoms, min distance, max bond-stretch ratio]."""
    return np.concatenate([
        composition_counts(Z),
        [float(len(Z)), min_interatomic_distance(R), max_bond_stretch(Z, R)],
    ])


def skeptic_descriptor(Z: np.ndarray, R: np.ndarray, mean_abs_fbar: float,
                       sigma_f: float) -> np.ndarray:
    """Full Skeptic descriptor: geometry + mean |F̄| + σ_F (all label-free)."""
    return np.concatenate([geometry_descriptor(Z, R), [mean_abs_fbar, sigma_f]])
