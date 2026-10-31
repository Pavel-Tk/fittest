"""Transition1x HDF5 reader.

Layout (Schreiner et al., Sci. Data 2022; gitlab.com/matschreiner/T1x):

    /data/{formula}/{reaction}/    (/train, /val, /test are links; split is by formula)
        atomic_numbers            (n,)
        positions                 (frames, n, 3)   Å
        wB97x_6-31G(d).energy     (frames,)        eV
        wB97x_6-31G(d).forces     (frames, n, 3)   eV/Å
        reactant/ product/ transition_state/   same keys, one frame each

Path frames are the NEB images visited during the reference calculation. We
sample a fixed number per reaction, always keeping reactant, product and TS so
relative energies (E - E_reactant) can be formed within a reaction.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from fittest.data import Frame

ENERGY_KEY = "wB97x_6-31G(d).energy"
FORCES_KEY = "wB97x_6-31G(d).forces"
ENDPOINTS = {"reactant": "reactant", "product": "product", "transition_state": "ts"}


def _frame(group, idx: int | None, *, uid: str, reaction: str, formula: str,
           split: str, role: str, Z: np.ndarray) -> Frame:
    sl = slice(None) if idx is None else idx
    R = np.asarray(group["positions"][sl], dtype=np.float64).reshape(-1, 3)
    E = float(np.asarray(group[ENERGY_KEY][sl]).reshape(-1)[0])
    F = np.asarray(group[FORCES_KEY][sl], dtype=np.float64).reshape(-1, 3)
    return Frame(uid=uid, reaction=reaction, formula=formula, split=split,
                 role=role, Z=Z.astype(np.int64), R=R, E=E, F=F)


def iter_reactions(path: str | Path, split: str) -> Iterator[tuple[str, str]]:
    """Yield (formula, reaction) keys of one official split."""
    import h5py

    with h5py.File(path, "r") as f:
        for formula, grp in f[split].items():
            for rxn in grp:
                yield formula, rxn


def read_reaction(path_or_file, split: str, formula: str, rxn: str,
                  n_path: int, rng: np.random.Generator) -> list[Frame]:
    """Reactant, product, TS plus ``n_path`` NEB images sampled without replacement."""
    import h5py

    own = not isinstance(path_or_file, h5py.File)
    f = h5py.File(path_or_file, "r") if own else path_or_file
    try:
        g = f[split][formula][rxn]
        Z = np.asarray(g["atomic_numbers"][()])
        reaction = f"{formula}/{rxn}"
        frames = []
        for key, role in ENDPOINTS.items():
            if key in g:
                frames.append(_frame(g[key], None, uid=f"{split}:{reaction}:{role}",
                                     reaction=reaction, formula=formula,
                                     split=split, role=role, Z=Z))
        # Endpoints and TS are repeated inside the path arrays; never sample them twice.
        path_E = np.asarray(g[ENERGY_KEY][()])
        endpoint_E = np.array([fr.E for fr in frames])
        candidates = np.flatnonzero(~np.isclose(path_E[:, None], endpoint_E[None, :],
                                                rtol=0, atol=1e-6).any(axis=1))
        take = min(n_path, candidates.size)
        for i in np.sort(rng.choice(candidates, size=take, replace=False)):
            frames.append(_frame(g, int(i), uid=f"{split}:{reaction}:path{int(i)}",
                                 reaction=reaction, formula=formula, split=split,
                                 role="path", Z=Z))
        return frames
    finally:
        if own:
            f.close()


def sample_split(path: str | Path, split: str, n_reactions: int, n_path: int,
                 seed: int = 0) -> list[Frame]:
    """Sample ``n_reactions`` reactions of an official split, deterministic in ``seed``."""
    import h5py

    rng = np.random.default_rng(seed)
    keys = sorted(iter_reactions(path, split))
    chosen = [keys[i] for i in np.sort(rng.choice(len(keys), size=min(n_reactions, len(keys)), replace=False))]
    out: list[Frame] = []
    with h5py.File(path, "r") as f:
        for formula, rxn in chosen:
            out.extend(read_reaction(f, split, formula, rxn, n_path, rng))
    return out
