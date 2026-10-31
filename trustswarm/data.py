"""Frames, label stripping, reaction-level splits and label-free descriptors (DESIGN.md §1, §4).

The Transition1x HDF5 reader is *not* here: it lives in :mod:`trustswarm.t1x`
(``trustswarm.t1x.sample_split(path, split, n_reactions, n_path, seed) -> list[Frame]``),
which returns frames tagged with the *official* T1x split (``train``/``val``/``test``).
:func:`assign_splits` then maps those onto the TrustSwarm splits, always by reaction.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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

    test → pool / eval (50/50 by connected component of shared endpoint geometry); val →
    skeptic_ref / planted (by formula, components kept whole; see
    :func:`assign_leakfree_splits`); train → seed_train (optionally subsampled to
    ``n_seed_train_reactions``). Frames already carrying a TrustSwarm split are returned
    unchanged.
    """
    rng = np.random.default_rng(seed)
    official = by_split(frames)

    def reactions_of(split: str) -> list[str]:
        return sorted({f.reaction for f in official.get(split, [])})

    train = reactions_of("train")
    if n_seed_train_reactions is not None and n_seed_train_reactions < len(train):
        train, _ = split_reactions(train, n_seed_train_reactions / len(train), rng)

    mapping: dict[tuple[str, str], str] = {}
    for source, names, target in (("train", train, SEED_TRAIN),):
        mapping.update({(source, r): target for r in names})
    out: list[Frame] = []
    for f in frames:
        if f.split in TRUSTSWARM_SPLITS or f.split in ("val", "test"):
            out.append(f)
        elif (f.split, f.reaction) in mapping:
            out.append(dataclasses.replace(f, split=mapping[(f.split, f.reaction)]))
    return assign_leakfree_splits(out, seed)[0]


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


def geometry_hash(Z: np.ndarray, R: np.ndarray, decimals: int = 2) -> str:
    """Permutation- and rigid-motion-invariant fingerprint of a structure.

    Sorted atomic numbers + the sorted multiset of interatomic distances rounded to
    ``decimals`` Å. Identical geometries always collide; T1x reuses one reactant geometry
    across several reactions, which is exactly what this detects."""
    z = np.sort(np.asarray(Z, dtype=np.int64))
    d = pairwise_distances(np.asarray(R, dtype=float))[np.triu_indices(len(Z), k=1)]
    key = z.tobytes() + np.round(np.sort(d), decimals).astype(np.float32).tobytes()
    return hashlib.sha1(key).hexdigest()[:16]


ENDPOINT_ROLES: tuple[str, ...] = ("reactant", "product")


def reaction_components(frames: Sequence[Frame]) -> dict[str, int]:
    """reaction -> connected-component id; reactions are linked when they share any endpoint
    (reactant or product) geometry (:func:`geometry_hash`). Component ids are deterministic."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    for f in sorted(frames, key=lambda f: f.uid):
        parent.setdefault(f.reaction, f.reaction)
        if f.role not in ENDPOINT_ROLES:
            continue
        h = geometry_hash(f.Z, f.R)
        if h in owner:
            a, b = find(owner[h]), find(f.reaction)
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            owner[h] = f.reaction
    roots = sorted({find(r) for r in parent})
    index = {r: i for i, r in enumerate(roots)}
    return {r: index[find(r)] for r in parent}


def _split_components(frames: Sequence[Frame], component: Mapping[str, int], first: str,
                      second: str, rng: np.random.Generator,
                      preassigned: Mapping[str, str] | None = None) -> dict[str, str]:
    """reaction -> split. Whole components go to one side; ~50/50 by frame count. With
    ``preassigned`` (reaction -> split), a component takes its majority preassigned side."""
    sizes: dict[int, int] = defaultdict(int)
    for f in frames:
        sizes[component[f.reaction]] += 1
    comps = sorted(sizes)
    side: dict[int, str] = {}
    if preassigned is not None:
        votes: dict[int, Counter[str]] = defaultdict(Counter)
        for f in frames:
            votes[component[f.reaction]][preassigned[f.reaction]] += 1
        side = {c: max(sorted(votes[c]), key=lambda k: votes[c][k]) for c in comps}
    else:
        total, filled = sum(sizes.values()), 0
        for i in rng.permutation(len(comps)):
            c = comps[i]
            side[c] = first if filled < total / 2 else second
            filled += sizes[c] if side[c] == first else 0
    return {r: side[c] for r, c in component.items() if c in side}


def assign_leakfree_splits(frames: Sequence[Frame], seed: int = 0
                           ) -> tuple[list[Frame], dict[str, Any]]:
    """(Re-)derive pool/eval and skeptic_ref/planted so that no endpoint geometry is shared.

    * official test → pool / eval: by connected component of shared endpoint geometry
      (:func:`reaction_components`), ~50/50 by frame count, seeded.
    * official val → skeptic_ref / planted: by formula (as before), then any component that
      crosses the formula split is moved wholly to its majority side.
    Other frames are unchanged. Returns the new frames and a JSON-serialisable report.
    Deterministic in ``seed``; uids (hence cached predictions) are untouched.
    """
    test = [f for f in frames if official_split(f) == "test"]
    val = [f for f in frames if official_split(f) == "val"]
    split_of: dict[str, str] = {}
    report: dict[str, Any] = {"seed": seed, "hash": "sorted Z + sorted pair distances (0.01 Å)",
                              "linked_by_roles": list(ENDPOINT_ROLES)}
    if test:
        comp = reaction_components(test)
        split_of.update(_split_components(test, comp, POOL, EVAL, np.random.default_rng([seed, 0x7E])))
        report["test_components"] = len(set(comp.values()))
        report["test_reactions"] = len(comp)
    if val:
        comp = reaction_components(val)
        ref_formulas, _ = split_reactions({f.formula for f in val}, 0.5,
                                          np.random.default_rng([seed, 0xF0]))
        by_formula = {f.reaction: (SKEPTIC_REF if f.formula in set(ref_formulas) else PLANTED)
                      for f in val}
        merged = _split_components(val, comp, SKEPTIC_REF, PLANTED, np.random.default_rng(seed),
                                   preassigned=by_formula)
        report["val_components"] = len(set(comp.values()))
        report["val_reactions_moved_to_keep_components_whole"] = sum(
            merged[r] != by_formula[r] for r in by_formula)
        split_of.update(merged)
    out = [dataclasses.replace(f, split=split_of[f.reaction]) if f.reaction in split_of
           and official_split(f) in ("test", "val") else f for f in frames]
    report["frames_per_split"] = dict(sorted(Counter(f.split for f in out).items()))
    report["reactions_per_split"] = {k: len({f.reaction for f in out if f.split == k})
                                     for k in report["frames_per_split"]}
    report["shared_endpoint_geometries"] = {
        "pool|eval": len(shared_endpoint_hashes(out, POOL, EVAL)),
        "skeptic_ref|planted": len(shared_endpoint_hashes(out, SKEPTIC_REF, PLANTED))}
    report["near_duplicate_endpoints_max_dR_1e-3"] = {
        "pool|eval": near_duplicate_endpoints(out, POOL, EVAL),
        "skeptic_ref|planted": near_duplicate_endpoints(out, SKEPTIC_REF, PLANTED)}
    return out, report


def shared_endpoint_hashes(frames: Sequence[Frame], a: str, b: str) -> set[str]:
    """Endpoint geometry hashes present in both splits ``a`` and ``b`` (should be empty)."""
    def hashes(split: str) -> set[str]:
        return {geometry_hash(f.Z, f.R) for f in frames
                if f.split == split and f.role in ENDPOINT_ROLES}
    return hashes(a) & hashes(b)


def near_duplicate_endpoints(frames: Sequence[Frame], a: str, b: str, tol: float = 1e-3) -> int:
    """Independent of hashing: pairs of endpoint frames (same atom order) across splits ``a``
    and ``b`` whose coordinates agree to max|ΔR| < ``tol`` Å."""
    groups: dict[tuple[str, bytes], list[np.ndarray]] = defaultdict(list)
    for f in frames:
        if f.split == a and f.role in ENDPOINT_ROLES:
            groups[(a, np.asarray(f.Z).tobytes())].append(np.asarray(f.R, dtype=float))
    count = 0
    for f in frames:
        if f.split == b and f.role in ENDPOINT_ROLES:
            R = np.asarray(f.R, dtype=float)
            count += sum(1 for other in groups.get((a, np.asarray(f.Z).tobytes()), [])
                         if np.max(np.abs(other - R)) < tol)
    return count


def split_val_by_formula(frames: Sequence[Frame], seed: int = 0) -> list[Frame]:
    """Back-compat wrapper: the full leak-free re-split (see :func:`assign_leakfree_splits`)."""
    return assign_leakfree_splits(frames, seed)[0]


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
