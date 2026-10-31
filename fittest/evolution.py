"""Evolutionary loop: survival of the fittest inside the swarm, at a fixed DFT budget.

One generation (all numbers are config values of :class:`EvoConfig`, fixed before the run):

1. **Triage.** The current population predicts the pool (cached predictions for the
   foundation models; an offspring predicts every pool + eval frame once, at birth). The
   Calibrator twin (:func:`fit_ledger` / :func:`recommend`) chooses what to serve; the
   population's force spread σ_F ranks unlabelled frames.
2. **Oracle.** ``labels_per_generation`` Transition1x look-ups per generation, split between
   the two component-disjoint halves of the pool (see *Breed / select split*): in each half
   ``1 - audit_fraction`` go to the highest-σ_F unlabelled frames and ``audit_fraction`` to
   uniform random audits of *trusted* frames (σ_F below the flag threshold τ, the Skeptic's
   exploration rule). Labels accumulate; the static control uses the identical procedure.
3. **Fitness** = marginal contribution of a member to the swarm, measured only on oracle
   labels that are *clean* for every member (no member trained on that frame's endpoint-
   geometry component):

   * answerer: ``Δ_ans(m) = err(P \\ m) - err(P)``, err = mean served error in wrongness
     units ``F_RMSE / WRONG_F + |ΔE_rel| / WRONG_E``; the served answer comes from a Calibrator
     ledger fitted out-of-fold (5-fold GroupKFold by reaction), so a member cannot look useful
     by overfitting the very labels it is scored on;
   * critic: ``Δ_crit(m) = AUROC(σ_F(P) → P wrong) - AUROC(σ_F(P \\ m) → P\\m wrong)``;
   * each Δ gets a paired reaction-bootstrap SE and 95 % CI; ``fitness = Δ_ans / SE_ans +
     Δ_crit / SE_crit`` (sum of the two z-scores); role = the larger of the two z-scores.
4. **Selection.** Repeatedly (fitness recomputed after every removal, so redundant twins are
   handled): a member is *eligible* for pruning when neither Δ has a CI lower bound above 0
   and its fitness ≤ ``prune_fitness_max``; the least fit eligible member dies while the
   population exceeds ``min_population``; then the least fit die while it exceeds
   ``max_population``.
5. **Breeding.** The Curator mutates the recipe of the fittest fine-tunable (MACE-family)
   members (energy:force loss weight, LoRA rank, learning rate, epochs, label subsample) and
   trains ``offspring_per_generation`` children on the *breed* labels (an offspring of an
   offspring starts from its parent's weights). Clones are rejected: Pearson correlation of
   the child's and a member's deviations from the population-mean force over pool atoms
   ``> clone_corr_max``, or median per-frame force RMS difference ``< clone_dist_min``.
6. **Gate.** On the *select* labels (component-disjoint from every child's training labels by
   construction, and re-asserted) a child must not be worse than its parent on force RMSE and
   on relative-energy MAE, and must pass the ALCHEMI acceptance report computed by the
   fine-tune job. Failing children are stillborn (logged, kept in the family tree).
7. **Hot-swap.** Survivors + gated, non-clone offspring are the next population; the ledger
   is re-fitted from the accumulated labels.

Breed / select split: the pool's endpoint-geometry components are split once (seeded) into
*breed* (offspring training) and *select* (fitness, gate, never trained on) parts, so selection
is never measured on frames a child was trained on. The Calibrator's ledger for a candidate uses
only labels clean for all of its members.

Evaluation never feeds back: eval labels are read only by :class:`EvalScorer`.

Everything here is numpy (no torch): offspring are produced by an ``OffspringFactory``
(torch fine-tune on the VM: ``scripts/evolve_offspring.py``; synthetic in tests).
"""

from __future__ import annotations

import dataclasses
import json
import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.stats import rankdata

from fittest import config
from fittest.ablation import _ci, auroc_mann_whitney, brier, expected_calibration_error
from fittest.blackboard import Blackboard
from fittest.cache import Predictions
from fittest.data import EVAL, POOL, Frame, reaction_components, reactant_index

BREED, SELECT = "breed", "select"
WRONG_UNITS = "F_RMSE/WRONG_F + |dE_rel|/WRONG_E"

BASE_RECIPE: dict[str, Any] = {"energy_weight": 10.0, "force_weight": 10.0, "lora_rank": 16,
                               "lora_alpha": 2.0, "lr": 1e-3, "epochs": 40, "batch_size": 16,
                               "label_fraction": 1.0}


@dataclass(frozen=True)
class EvoConfig:
    generations: int = 4
    labels_per_generation: int = 50
    select_label_share: float = 0.4
    """Share of each generation's labels spent in the select part of the pool."""
    select_frame_share: float = 0.35
    """Target share of pool frames (whole components) in the select part."""
    audit_fraction: float = config.AUDIT_FRACTION
    flag_quantile: float = config.TAU_FLOOR_QUANTILE
    min_population: int = 3
    max_population: int = 6
    offspring_per_generation: int = 3
    prune_fitness_max: float = 0.5
    """'Fitness ≈ 0 or negative': sum of z-scores at most this (and no CI above 0)."""
    clone_corr_max: float = 0.95
    clone_dist_min: float = 0.02
    n_bootstrap_fitness: int = 400
    n_bootstrap_eval: int = config.N_BOOTSTRAP
    cv_folds: int = 5
    seed: int = 0
    base_models: tuple[str, ...] = config.HETERO_MODELS
    finetunable: tuple[str, ...] = ("mace_mpa0",)
    wrong_f: float = config.WRONG_F
    wrong_e: float = config.WRONG_E
    breeding: bool = True
    selection: bool = True
    breeding_mode: str = "mutate"
    """'mutate' (varied recipes) or 'clone' (control: the parent's recipe, different seeds)."""
    selection_mode: str = "fitness"
    """'fitness' or 'random' (control: same number of deaths, victims chosen at random)."""

    @property
    def total_budget(self) -> int:
        return self.generations * self.labels_per_generation


# ============================================================ prediction store (label-free)
class PredStore:
    """Every model's predictions on a fixed frame list (pool + eval), as dense arrays.

    ``F[name]``: (A, 3) forces over the concatenated atoms; ``E[name]``: (N,) energies;
    ``rel[name]``: E(frame) - E(reactant of its reaction) (same composition, so per-element
    reference energies cancel)."""

    def __init__(self, frames: Sequence[Frame]) -> None:
        self.frames = list(frames)
        self.uids = [f.uid for f in self.frames]
        self.index = {u: i for i, u in enumerate(self.uids)}
        n = np.array([f.n_atoms for f in self.frames])
        self.n_atoms = n
        self.offsets = np.concatenate([[0], np.cumsum(n)])
        self.atom_frame = np.repeat(np.arange(len(n)), n)
        reactants = reactant_index(self.frames)
        missing = sorted({f.reaction for f in self.frames} - set(reactants))
        if missing:
            raise ValueError(f"{len(missing)} reactions have no reactant frame, e.g. {missing[:3]}")
        self.reactant_idx = np.array([self.index[reactants[f.reaction].uid] for f in self.frames])
        self.reaction = np.array([f.reaction for f in self.frames])
        self.split = np.array([f.split for f in self.frames])
        self.F: dict[str, np.ndarray] = {}
        self.E: dict[str, np.ndarray] = {}
        self.rel: dict[str, np.ndarray] = {}

    def add(self, name: str, p: Predictions) -> None:
        missing = [u for u in self.uids if u not in p]
        if missing:
            raise ValueError(f"{name}: no prediction for {len(missing)} frames, e.g. {missing[:3]}")
        self.E[name] = np.array([p.energy[u] for u in self.uids], dtype=float)
        self.F[name] = np.concatenate([np.asarray(p.forces[u], float).reshape(-1, 3)
                                       for u in self.uids])
        if len(self.F[name]) != self.offsets[-1]:
            raise ValueError(f"{name}: force array has the wrong number of atoms")
        self.rel[name] = self.E[name] - self.E[name][self.reactant_idx]

    def per_frame(self, per_atom: np.ndarray) -> np.ndarray:
        return np.add.reduceat(per_atom, self.offsets[:-1]) / self.n_atoms

    def sigma_F(self, models: Sequence[str]) -> np.ndarray:
        """σ_F per frame (DESIGN.md §3) for a set of models, vectorised over all frames."""
        if len(models) < 2:
            raise ValueError("σ_F needs at least two models")
        f = np.stack([self.F[m] for m in models])  # (M, A, 3)
        dev = f - f.mean(axis=0, keepdims=True)
        s = np.sqrt(np.mean(np.sum(dev * dev, axis=-1), axis=0))
        return self.per_frame(s)

    def mean_abs_fbar(self, models: Sequence[str]) -> np.ndarray:
        fbar = np.mean([self.F[m] for m in models], axis=0)
        return self.per_frame(np.linalg.norm(fbar, axis=-1))

    def force_rms_diff(self, a: str, b: str) -> np.ndarray:
        d = self.F[a] - self.F[b]
        return np.sqrt(self.per_frame(np.sum(d * d, axis=-1)) / 3.0)


# ============================================================ labels (oracle-private)
class LabelBook:
    """Holds the DFT labels of the frame list. Residual statistics are revealed only for
    indices in ``allowed`` (the Oracle's labelled set, or eval for final scoring)."""

    def __init__(self, store: PredStore) -> None:
        self.store = store
        if any(f.E is None or f.F is None for f in store.frames):
            raise ValueError("LabelBook needs labelled frames")
        self._F = np.concatenate([np.asarray(f.F, float) for f in store.frames])
        E = np.array([float(f.E) for f in store.frames])
        self._rel = E - E[store.reactant_idx]
        self._gram: dict[tuple[str, str], np.ndarray] = {}
        self._rel_res: dict[str, np.ndarray] = {}

    def _residual_gram(self, a: str, b: str) -> np.ndarray:
        key = (a, b) if a <= b else (b, a)
        if key not in self._gram:
            ra, rb = self.store.F[key[0]] - self._F, self.store.F[key[1]] - self._F
            self._gram[key] = np.add.reduceat(np.sum(ra * rb, axis=-1), self.store.offsets[:-1])
        return self._gram[key]

    def _rel_residual(self, m: str) -> np.ndarray:
        if m not in self._rel_res:
            self._rel_res[m] = self.store.rel[m] - self._rel
        return self._rel_res[m]

    def candidate_errors(self, models: Sequence[str], W: np.ndarray, idx: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(force RMSE, |ΔE_rel| error), each (len(idx), C), of the candidate means ``W``
        (C, M) over ``models`` on frames ``idx``. Exact, via per-frame residual Gram matrices:
        ||Σ_a w_a r_a||² = Σ_ab w_a w_b <r_a, r_b>."""
        idx = np.asarray(idx, dtype=int)
        M = len(models)
        G = np.empty((len(idx), M, M))
        for i in range(M):
            for j in range(i, M):
                G[:, i, j] = G[:, j, i] = self._residual_gram(models[i], models[j])[idx]
        sq = np.einsum("ca,fab,cb->fc", W, G, W) / (3.0 * self.store.n_atoms[idx])[:, None]
        R = np.stack([self._rel_residual(m)[idx] for m in models], axis=1)  # (F, M)
        return np.sqrt(np.maximum(sq, 0.0)), np.abs(R @ W.T)

    def model_errors(self, m: str, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f, e = self.candidate_errors([m], np.ones((1, 1)), idx)
        return f[:, 0], e[:, 0]

    def rel_true(self, idx: np.ndarray) -> np.ndarray:
        return self._rel[np.asarray(idx, int)]

    def forces_true(self, i: int) -> np.ndarray:
        o = self.store.offsets
        return self._F[o[i]:o[i + 1]]


# ============================================================ candidates + Calibrator twin
def candidate_matrix(models: Sequence[str]) -> tuple[list[tuple[str, ...]], np.ndarray, np.ndarray]:
    """All non-empty subsets of ``models`` (the Calibrator's candidates, fittest.candidates order:
    singletons in model order, then larger subsets), their (C, M) mean weights and the prior
    tie-break rank (full set first, then larger subsets, then model order)."""
    models = list(models)
    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(models) + 1):
        subsets.extend(combinations(models, size))
    W = np.zeros((len(subsets), len(models)))
    for c, s in enumerate(subsets):
        for m in s:
            W[c, models.index(m)] = 1.0 / len(s)
    order = sorted(range(len(subsets)), key=lambda c: (-len(subsets[c]), c))
    tiebreak = np.empty(len(subsets))
    tiebreak[order] = np.arange(len(subsets))
    return subsets, W, tiebreak


def candidate_name(subset: Sequence[str], population: Sequence[str]) -> str:
    if len(subset) == 1:
        return subset[0]
    if len(subset) == len(population):
        return config.CONSENSUS
    return f"mean({'+'.join(sorted(subset))})"


@dataclass
class Buckets:
    sigma_edges: np.ndarray
    fbar_edges: np.ndarray

    @property
    def n(self) -> int:
        return (len(self.sigma_edges) + 1) * (len(self.fbar_edges) + 1)

    def of(self, sigma: np.ndarray, fbar: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.sigma_edges, sigma, side="right")
        j = np.searchsorted(self.fbar_edges, fbar, side="right")
        return i * (len(self.fbar_edges) + 1) + j

    @classmethod
    def fit(cls, sigma: np.ndarray, fbar: np.ndarray) -> Buckets:
        q = lambda x, k: np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])  # noqa: E731
        return cls(q(sigma, config.SIGMA_BINS), q(fbar, config.FBAR_BINS))


@dataclass
class Ledger:
    alpha: np.ndarray  # (C, B)
    beta: np.ndarray
    g_alpha: np.ndarray  # (C,)
    g_beta: np.ndarray
    tiebreak: np.ndarray

    def p_correct(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)


def fit_ledger(wrong: np.ndarray, clean: np.ndarray, bucket: np.ndarray, n_buckets: int,
               tiebreak: np.ndarray, prior: tuple[float, float] = config.CORRECTNESS_PRIOR) -> Ledger:
    """Per (candidate, bucket) Beta posterior of correctness from labelled frames.

    ``wrong``, ``clean``: (L, C) — candidate c's verdict on labelled frame l, and whether that
    label may be used for c (no member of c trained on that frame's component)."""
    L, C = wrong.shape
    use = clean.astype(float)
    corr = (~wrong).astype(float) * use
    onehot = np.zeros((L, n_buckets))
    onehot[np.arange(L), bucket] = 1.0
    n = use.T @ onehot
    a = corr.T @ onehot
    return Ledger(alpha=prior[0] + a, beta=prior[1] + (n - a), g_alpha=prior[0] + corr.sum(0),
                  g_beta=prior[1] + (use.sum(0) - corr.sum(0)), tiebreak=tiebreak)


def recommend(ledger: Ledger, bucket: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(candidate index, trust = P(served correct | bucket)) per frame: argmax over candidates
    of P(c correct | bucket), ties by the global posterior, then the prior order (Calibrator)."""
    p = ledger.p_correct()  # (C, B)
    g = ledger.g_alpha / (ledger.g_alpha + ledger.g_beta)
    key = p + 1e-7 * g[:, None] - 1e-10 * ledger.tiebreak[:, None]
    best = np.argmax(key, axis=0)  # (B,)
    rec = best[bucket]
    return rec, p[rec, bucket]


# ============================================================ members + family tree
@dataclass
class Member:
    id: str
    family: str
    kind: str  # 'base' | 'offspring'
    generation_born: int
    parent: str | None = None
    recipe: dict[str, Any] = field(default_factory=dict)
    train_uids: list[str] = field(default_factory=list)
    train_components: set[int] = field(default_factory=set)
    gate: dict[str, Any] | None = None
    status: str = "alive"  # alive | dead | stillborn | clone
    died_generation: int | None = None
    death_reason: str | None = None
    fitness: dict[int, dict[str, Any]] = field(default_factory=dict)
    ckpt: str | None = None

    def tree_record(self) -> dict[str, Any]:
        last = self.fitness[max(self.fitness)] if self.fitness else None
        return {"id": self.id, "parent": self.parent, "family": self.family, "kind": self.kind,
                "generation": self.generation_born, "hyperparams": self.recipe,
                "n_train_labels": len(self.train_uids), "gate": self.gate, "status": self.status,
                "born": self.generation_born,
                "died": self.died_generation, "death_reason": self.death_reason,
                "fitness_per_generation": {str(g): v for g, v in sorted(self.fitness.items())},
                "role": None if last is None else last["role"], "ckpt": self.ckpt}


@dataclass
class OffspringRequest:
    id: str
    parent: Member
    recipe: dict[str, Any]
    train_uids: list[str]
    holdout_uids: list[str]
    seed: int


@dataclass
class OffspringResult:
    predictions: Predictions | None
    toolkit: dict[str, Any] = field(default_factory=dict)  # {'accepted': bool, ...}
    ckpt: str | None = None
    error: str | None = None
    wall_seconds: float | None = None


class OffspringFactory(Protocol):
    def __call__(self, requests: Sequence[OffspringRequest]) -> list[OffspringResult]: ...


# ============================================================ oracle (label budget)
class Oracle:
    """Reveals labels; keeps the labelled set and the budget. No other path to labels."""

    def __init__(self, store: PredStore, part: np.ndarray, budget: int) -> None:
        self.store, self.part, self.budget = store, part, budget
        self.labelled: list[int] = []
        self.generation_of: dict[int, int] = {}
        self.audit: set[int] = set()

    @property
    def spent(self) -> int:
        return len(self.labelled)

    def request(self, idx: Sequence[int], generation: int, audit: Sequence[int] = ()) -> None:
        idx = [int(i) for i in idx]
        if len(set(idx)) != len(idx) or set(idx) & set(self.labelled):
            raise ValueError("oracle request has duplicates or already-labelled frames")
        if any(self.store.split[i] != POOL for i in idx):
            raise ValueError("the oracle labels pool frames only")
        if self.spent + len(idx) > self.budget:
            raise ValueError(f"budget exceeded: {self.spent} + {len(idx)} > {self.budget}")
        self.labelled.extend(idx)
        self.generation_of.update({i: generation for i in idx})
        self.audit.update(int(i) for i in audit)

    def labelled_in(self, part: str) -> np.ndarray:
        return np.array([i for i in self.labelled if self.part[i] == part], dtype=int)


def split_pool(store: PredStore, select_share: float, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign each pool frame to BREED or SELECT by whole endpoint-geometry component (seeded
    greedy fill of SELECT up to ``select_share`` of pool frames). Non-pool frames: ''."""
    pool = [f for f in store.frames if f.split == POOL]
    comp = reaction_components(pool)
    sizes: dict[int, int] = {}
    for f in pool:
        sizes[comp[f.reaction]] = sizes.get(comp[f.reaction], 0) + 1
    comps = sorted(sizes)
    rng = np.random.default_rng([seed, 101])
    chosen: set[int] = set()
    cap = select_share * len(pool)
    for i in rng.permutation(len(comps)):
        c = comps[i]
        if sum(sizes[k] for k in chosen) + sizes[c] <= cap:
            chosen.add(c)
    part = np.array([(SELECT if comp[f.reaction] in chosen else BREED) if f.split == POOL else ""
                     for f in store.frames])
    component = np.array([comp.get(f.reaction, -1) if f.split == POOL else -1 for f in store.frames])
    report = {"n_components": len(comps), "select_components": sorted(int(c) for c in chosen),
              "n_select_frames": int((part == SELECT).sum()), "n_breed_frames": int((part == BREED).sum()),
              "select_reactions": int(len({f.reaction for f, p in zip(store.frames, part) if p == SELECT})),
              "breed_reactions": int(len({f.reaction for f, p in zip(store.frames, part) if p == BREED}))}
    return np.stack([part, component.astype(str)]), report


# ============================================================ statistics helpers
def _reaction_boot_indices(groups: np.ndarray, n: int, seed: Sequence[int]) -> list[np.ndarray]:
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng([*seed, zlib.crc32("\n".join(uniq).encode())])
    return [np.concatenate([members[g] for g in rng.integers(len(uniq), size=len(uniq))])
            for _ in range(n)]


def _auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    return auroc_mann_whitney(np.asarray(y, bool), np.asarray(s, float))


def group_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    uniq = np.unique(groups)
    k = min(k, len(uniq))
    rng = np.random.default_rng([seed, 7])
    perm = uniq[rng.permutation(len(uniq))]
    fold_of = {g: i % k for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    return [np.flatnonzero(f == i) for i in range(k)]


# ============================================================ the swarm view
class Swarm:
    """Label-free view of one population on the frame list + its Calibrator."""

    def __init__(self, store: PredStore, models: Sequence[str]) -> None:
        self.store, self.models = store, list(models)
        self.subsets, self.W, self.tiebreak = candidate_matrix(self.models)
        self.sigma = store.sigma_F(self.models)
        self.fbar = store.mean_abs_fbar(self.models)
        pool = store.split == POOL
        self.buckets = Buckets.fit(self.sigma[pool], self.fbar[pool])
        self.bucket = self.buckets.of(self.sigma, self.fbar)
        self.tau = float(np.quantile(self.sigma[pool], config.TAU_FLOOR_QUANTILE))

    def candidate_names(self) -> list[str]:
        return [candidate_name(s, self.models) for s in self.subsets]


def clean_matrix(subsets: Sequence[tuple[str, ...]], members: Mapping[str, Member],
                 component: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(L, C): label l usable for candidate c iff no member of c trained on its component."""
    out = np.ones((len(idx), len(subsets)), dtype=bool)
    comp = component[idx]
    for c, s in enumerate(subsets):
        for m in s:
            tc = members[m].train_components
            if tc:
                out[:, c] &= ~np.isin(comp, list(tc))
    return out


# ============================================================ evolution engine
class Evolution:
    def __init__(self, frames: Sequence[Frame], base_preds: Mapping[str, Predictions], cfg: EvoConfig,
                 factory: OffspringFactory | None = None, blackboard: Blackboard | None = None,
                 label: str = "evolved") -> None:
        self.cfg, self.factory, self.bb, self.label = cfg, factory, blackboard, label
        keep = [f for f in frames if f.split in (POOL, EVAL)]
        self.store = PredStore(keep)
        for m, p in base_preds.items():
            self.store.add(m, p)
        self.labels = LabelBook(self.store)
        partcomp, self.split_report = split_pool(self.store, cfg.select_frame_share, cfg.seed)
        self.part = partcomp[0]
        self.component = partcomp[1].astype(int)
        self.oracle = Oracle(self.store, self.part, cfg.total_budget)
        self.members: dict[str, Member] = {}
        for m in cfg.base_models:
            fam = "mace" if m.startswith("mace") else m.split("_")[0]
            self.members[m] = Member(id=m, family=fam, kind="base", generation_born=0,
                                     recipe=dict(BASE_RECIPE) if m in cfg.finetunable else {})
        self.population: list[str] = list(cfg.base_models)
        self.rng = np.random.default_rng([cfg.seed, 2026])
        self.history: list[dict[str, Any]] = []
        self.eval_states: list[dict[str, Any]] = []

    # --- logging -----------------------------------------------------------------------
    def post(self, kind: str, payload: dict[str, Any]) -> None:
        if self.bb is not None:
            self.bb.post(kind, self.label, _jsonable(payload))

    # --- 1-2. triage + oracle ------------------------------------------------------------
    def triage(self, g: int) -> dict[str, Any]:
        sw = Swarm(self.store, self.population)
        k_select = int(round(self.cfg.select_label_share * self.cfg.labels_per_generation))
        quota = {SELECT: k_select, BREED: self.cfg.labels_per_generation - k_select}
        labelled = set(self.oracle.labelled)
        picked, audits = [], []
        for part, k in quota.items():
            avail = np.array([i for i in np.flatnonzero(self.part == part) if i not in labelled])
            n_audit = int(round(self.cfg.audit_fraction * k))
            n_top = k - n_audit
            order = avail[np.lexsort((avail, -sw.sigma[avail]))]
            top = list(order[:n_top])
            rest = np.array([i for i in avail if i not in set(top) and sw.sigma[i] < sw.tau])
            if len(rest) < n_audit:  # not enough trusted frames: audit any remaining frame
                rest = np.array([i for i in avail if i not in set(top)])
            aud = list(self.rng.choice(rest, size=min(n_audit, len(rest)), replace=False)) if n_audit else []
            picked += top + aud
            audits += aud
        self.oracle.request(picked, g, audit=audits)
        info = {"generation": g, "population": list(self.population), "n_new_labels": len(picked),
                "n_audits": len(audits), "labels_total": self.oracle.spent, "tau": sw.tau,
                "new_labels": [self.store.uids[i] for i in picked]}
        self.post("triage", info)
        return info

    # --- calibrated serving ----------------------------------------------------------------
    def fitted_swarm(self, models: Sequence[str], label_idx: np.ndarray
                     ) -> tuple[Swarm, Ledger]:
        sw = Swarm(self.store, models)
        fw, fe = self.labels.candidate_errors(sw.models, sw.W, label_idx)
        wrong = (fw > self.cfg.wrong_f) | (fe > self.cfg.wrong_e)
        clean = clean_matrix(sw.subsets, self.members, self.component, label_idx)
        led = fit_ledger(wrong, clean, sw.bucket[label_idx], sw.buckets.n, sw.tiebreak)
        return sw, led

    # --- 3. fitness ----------------------------------------------------------------------------
    def clean_for(self, models: Sequence[str]) -> np.ndarray:
        idx = np.array(self.oracle.labelled, dtype=int)
        ok = np.ones(len(idx), bool)
        for m in models:
            tc = self.members[m].train_components
            if tc:
                ok &= ~np.isin(self.component[idx], list(tc))
        return idx[ok]

    def _oof_served(self, models: Sequence[str], idx: np.ndarray, folds: list[np.ndarray]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Out-of-fold served error (wrongness units), served-wrong flag and σ_F on ``idx``."""
        sw = Swarm(self.store, models)
        fw, fe = self.labels.candidate_errors(sw.models, sw.W, idx)
        wrong = (fw > self.cfg.wrong_f) | (fe > self.cfg.wrong_e)
        units = fw / self.cfg.wrong_f + fe / self.cfg.wrong_e
        b = sw.bucket[idx]
        err = np.empty(len(idx))
        w = np.empty(len(idx), bool)
        for test in folds:
            train = np.setdiff1d(np.arange(len(idx)), test)
            led = fit_ledger(wrong[train], np.ones_like(wrong[train]), b[train], sw.buckets.n, sw.tiebreak)
            rec, _ = recommend(led, b[test])
            err[test] = units[test, rec]
            w[test] = wrong[test, rec]
        return err, w, sw.sigma[idx]

    def contributions(self, population: Sequence[str], g: int) -> dict[str, dict[str, Any]]:
        idx = self.clean_for(population)
        out: dict[str, dict[str, Any]] = {}
        if len(idx) < 10 or len(population) < 3:
            for m in population:
                out[m] = {"fitness": 0.0, "role": None, "n_labels": int(len(idx)), "note": "too few clean labels"}
            return out
        groups = self.store.reaction[idx]
        folds = group_folds(groups, self.cfg.cv_folds, self.cfg.seed + g)
        boots = _reaction_boot_indices(groups, self.cfg.n_bootstrap_fitness, [self.cfg.seed, g, 31])
        err_all, w_all, s_all = self._oof_served(population, idx, folds)
        a_all = _auroc(w_all, s_all)
        a_all_b = np.array([_auroc(w_all[b], s_all[b]) or np.nan for b in boots])
        for m in population:
            rest = [x for x in population if x != m]
            err_r, w_r, s_r = self._oof_served(rest, idx, folds)
            d_ans = err_r - err_all
            ans_b = np.array([d_ans[b].mean() for b in boots])
            a_r = _auroc(w_r, s_r)
            a_r_b = np.array([_auroc(w_r[b], s_r[b]) or np.nan for b in boots])
            crit_b = a_all_b - a_r_b
            d_crit = (a_all - a_r) if a_all is not None and a_r is not None else 0.0
            se_a = float(np.nanstd(ans_b)) or 1e-9
            se_c = float(np.nanstd(crit_b)) if np.isfinite(crit_b).sum() > 10 else float("inf")
            se_c = se_c or 1e-9
            z_a, z_c = float(d_ans.mean()) / se_a, (d_crit / se_c if math.isfinite(se_c) else 0.0)
            out[m] = {"answerer": float(d_ans.mean()), "answerer_se": se_a, "answerer_ci": _ci(ans_b),
                      "critic": float(d_crit), "critic_se": se_c if math.isfinite(se_c) else None,
                      "critic_ci": _ci(crit_b), "z_answerer": z_a, "z_critic": z_c,
                      "fitness": z_a + z_c, "role": "answerer" if z_a >= z_c else "critic",
                      "n_labels": int(len(idx)), "n_reactions": int(len(np.unique(groups))),
                      "swarm_err_oof": float(err_all.mean()), "swarm_auroc_oof": a_all}
        return out

    @staticmethod
    def _eligible(c: Mapping[str, Any], cap: float) -> bool:
        lo_a = (c.get("answerer_ci") or [0.0, 0.0])[0]
        lo_c = (c.get("critic_ci") or [0.0, 0.0])[0]
        return lo_a <= 0 and lo_c <= 0 and c["fitness"] <= cap

    # --- 4. selection ------------------------------------------------------------------------
    def select(self, g: int) -> dict[str, Any]:
        log = []
        first = self.contributions(self.population, g)
        for m, c in first.items():
            self.members[m].fitness[g] = c
        current = first
        while len(self.population) > self.cfg.min_population and self.cfg.selection:
            elig = [m for m in self.population if self._eligible(current[m], self.cfg.prune_fitness_max)]
            over = len(self.population) > self.cfg.max_population
            if not elig and not over:
                break
            if self.cfg.selection_mode == "random":
                victim = str(self.rng.choice(sorted(elig or self.population)))
            else:
                victim = min(elig or self.population, key=lambda m: current[m]["fitness"])
            reason = ("no significant contribution, fitness ≈ 0 or negative" if victim in elig
                      else f"population cap {self.cfg.max_population}")
            mem = self.members[victim]
            mem.status, mem.died_generation, mem.death_reason = "dead", g, reason
            self.population.remove(victim)
            log.append({"victim": victim, "fitness": current[victim]["fitness"], "reason": reason})
            self.post("death", {"generation": g, "member": victim, "reason": reason,
                                "fitness": current[victim]})
            current = self.contributions(self.population, g)
        info = {"generation": g, "fitness": first, "deaths": log, "survivors": list(self.population),
                "fitness_after": current}
        self.post("fitness", info)
        return info

    # --- 5. breeding ------------------------------------------------------------------------------
    def mutate(self, parent: Member, k: int, g: int) -> dict[str, Any]:
        base = dict(parent.recipe or BASE_RECIPE)
        r = dict(base)
        if self.cfg.breeding_mode == "clone":
            return r
        rng = np.random.default_rng([self.cfg.seed, g, k, 404])
        ew = lambda x: float(np.clip(x, 1.0, 300.0))  # noqa: E731
        if k % 3 == 0:  # energy-weighted child: rel-E failed the gate last time
            r["energy_weight"] = ew(base["energy_weight"] * 3.0)
        elif k % 3 == 1:  # strongly energy-weighted, different capacity
            r["energy_weight"] = ew(base["energy_weight"] * 10.0)
            r["lora_rank"] = int(np.clip(base["lora_rank"] * rng.choice([0.5, 2.0]), 4, 64))
        else:  # same loss, different step size / schedule / label subsample
            r["lr"] = float(base["lr"] * rng.choice([0.5, 2.0]))
            r["epochs"] = int(np.clip(base["epochs"] + rng.choice([-10, 20]), 20, 80))
            r["label_fraction"] = 0.8
        return r

    def breed(self, g: int) -> dict[str, Any]:
        if not self.cfg.breeding or self.factory is None:
            return {"generation": g, "born": [], "note": "breeding disabled"}
        fam = [m for m in self.population if self.members[m].family == "mace"
               and (self.members[m].kind == "offspring" or m in self.cfg.finetunable)]
        fam.sort(key=lambda m: -self.members[m].fitness.get(g, {}).get("fitness", 0.0))
        if not fam:  # germline: the foundation checkpoint is always available
            fam = [m for m in self.cfg.finetunable]
            for m in fam:
                if m not in self.members:
                    raise KeyError(m)
        parents = [fam[0]] * self.cfg.offspring_per_generation
        if len(fam) > 1 and self.cfg.offspring_per_generation >= 3 and self.cfg.breeding_mode != "clone":
            parents[-1] = fam[1]
        breed_idx = self.oracle.labelled_in(BREED)
        hold_idx = self.oracle.labelled_in(SELECT)
        requests = []
        for k, pid in enumerate(parents):
            parent = self.members[pid]
            recipe = self.mutate(parent, k, g)
            idx = breed_idx
            if recipe.get("label_fraction", 1.0) < 1.0:
                rx = np.unique(self.store.reaction[idx])
                rng = np.random.default_rng([self.cfg.seed, g, k, 77])
                keep = set(rng.choice(rx, size=max(1, int(round(recipe["label_fraction"] * len(rx)))),
                                      replace=False))
                idx = np.array([i for i in idx if self.store.reaction[i] in keep], dtype=int)
            train_uids = [self.store.uids[i] for i in idx]
            hold_uids = [self.store.uids[i] for i in hold_idx]
            assert not set(self.component[idx]) & set(self.component[hold_idx]), "gate set leaks"
            cid = f"g{g + 1}_{chr(97 + k)}"
            requests.append(OffspringRequest(cid, parent, recipe, train_uids, hold_uids,
                                             seed=self.cfg.seed * 1000 + g * 10 + k))
        self.post("breed_request", {"generation": g, "requests": [
            {"id": r.id, "parent": r.parent.id, "recipe": r.recipe, "n_train": len(r.train_uids),
             "n_holdout": len(r.holdout_uids)} for r in requests]})
        results = self.factory(requests)
        born = []
        for req, res in zip(requests, results):
            born.append(self._admit(req, res, g))
        return {"generation": g, "born": born}

    def _admit(self, req: OffspringRequest, res: OffspringResult, g: int) -> dict[str, Any]:
        mem = Member(id=req.id, family="mace", kind="offspring", generation_born=g + 1,
                     parent=req.parent.id, recipe=req.recipe, train_uids=req.train_uids,
                     train_components={int(self.component[self.store.index[u]]) for u in req.train_uids},
                     ckpt=res.ckpt)
        self.members[mem.id] = mem
        if res.predictions is None:
            mem.status, mem.gate = "stillborn", {"passed": False, "error": res.error}
            self.post("birth", {"generation": g, "member": mem.id, "status": mem.status, "error": res.error})
            return {"id": mem.id, "status": mem.status, "error": res.error}
        self.store.add(mem.id, res.predictions)
        mem.gate = self.gate(mem, res)
        clone = self.clone_check(mem.id)
        mem.gate["clone_check"] = clone
        if not mem.gate["passed"]:
            mem.status = "stillborn"
        elif clone["is_clone"]:
            mem.status = "clone"
        else:
            self.population.append(mem.id)
        out = {"id": mem.id, "parent": mem.parent, "recipe": mem.recipe, "status": mem.status,
               "gate": mem.gate, "wall_seconds": res.wall_seconds}
        self.post("birth", {"generation": g, **out})
        return out

    # --- 6. gate ---------------------------------------------------------------------------------
    def gate(self, child: Member, res: OffspringResult) -> dict[str, Any]:
        hold = self.oracle.labelled_in(SELECT)
        if set(self.component[hold]) & child.train_components:
            raise AssertionError("gate hold-out shares a component with the child's training labels")
        parent = child.parent
        f_c, e_c = self.labels.model_errors(child.id, hold)
        f_p, e_p = self.labels.model_errors(parent, hold)
        n = self.store.n_atoms[hold]
        rmse = lambda f: float(np.sqrt(np.sum(f ** 2 * n) / n.sum()))  # noqa: E731  global over atoms
        checks = [
            {"name": "holdout_force_rmse_not_worse", "value": rmse(f_c), "limit": rmse(f_p),
             "passed": bool(rmse(f_c) <= rmse(f_p))},
            {"name": "holdout_rel_energy_mae_not_worse", "value": float(e_c.mean()),
             "limit": float(e_p.mean()), "passed": bool(e_c.mean() <= e_p.mean())},
        ]
        tk = res.toolkit or {}
        if "accepted" in tk:
            checks.append({"name": "alchemi_acceptance_report", "value": tk.get("worst_ratio"),
                           "limit": tk.get("max_ratio"), "passed": bool(tk["accepted"]),
                           "detail": tk.get("detail")})
        return {"passed": all(c["passed"] for c in checks), "checks": checks,
                "n_holdout": int(len(hold)), "n_holdout_reactions": int(len(np.unique(self.store.reaction[hold]))),
                "holdout_components_disjoint": True}

    def clone_check(self, cid: str) -> dict[str, Any]:
        pool = np.flatnonzero(self.store.split == POOL)
        atoms = np.isin(self.store.atom_frame, pool)
        pop = list(self.population)
        fbar = np.mean([self.store.F[m] for m in pop], axis=0)[atoms]
        dc = (self.store.F[cid][atoms] - fbar).ravel()
        rows = {}
        for m in pop:
            dm = (self.store.F[m][atoms] - fbar).ravel()
            r = float(np.corrcoef(dc, dm)[0, 1])
            d = float(np.median(self.store.force_rms_diff(cid, m)[pool]))
            rows[m] = {"corr": r, "median_force_diff": d}
        worst = max(rows, key=lambda m: rows[m]["corr"]) if rows else None
        is_clone = any(v["corr"] > self.cfg.clone_corr_max or v["median_force_diff"] < self.cfg.clone_dist_min
                       for v in rows.values())
        return {"is_clone": bool(is_clone), "closest": worst, "per_member": rows,
                "rule": f"corr of deviations from population-mean force > {self.cfg.clone_corr_max} "
                        f"or median per-frame force RMS diff < {self.cfg.clone_dist_min} eV/Å"}

    # --- serving on eval (for the scorer) ---------------------------------------------------
    def serve_eval(self) -> dict[str, Any]:
        idx = np.array(self.oracle.labelled, dtype=int)
        sw, led = self.fitted_swarm(self.population, idx)
        ev = np.flatnonzero(self.store.split == EVAL)
        rec, trust = recommend(led, sw.bucket[ev])
        names = sw.candidate_names()
        return {"population": list(self.population), "W": sw.W, "rec": rec, "trust": trust,
                "sigma": sw.sigma[ev], "tau": sw.tau, "eval_idx": ev,
                "served": {n: int((rec == c).sum()) for c, n in enumerate(names) if (rec == c).any()}}

    # --- loop ------------------------------------------------------------------------------------
    def run(self, scorer: EvalScorer | None = None) -> list[dict[str, Any]]:
        for g in range(self.cfg.generations):
            info: dict[str, Any] = {"generation": g, "triage": self.triage(g)}
            info["selection"] = self.select(g)
            state = self.serve_eval()
            if scorer is not None:
                scorer.record(f"{self.label}_g{g}", self, state)
            info["served_eval"] = state["served"]
            info["population_after_selection"] = list(self.population)
            if g < self.cfg.generations - 1:
                info["breeding"] = self.breed(g)
            info["population_next"] = list(self.population)
            self.history.append(info)
            self.post("generation", {k: v for k, v in info.items() if k != "triage"})
        return self.history

    def family_tree(self) -> list[dict[str, Any]]:
        return [m.tree_record() for m in self.members.values()]


# ============================================================ eval scoring (eval labels)
class EvalScorer:
    """Scores served answers on eval; keeps per-state reaction-bootstrap samples so any two
    states (generations, static control) can be compared paired. The only reader of eval labels."""

    METRICS = ("force_rmse", "rel_e_mae", "wrong_rate", "ece", "brier", "auroc", "trusted_fraction",
               "badly_wrong_trusted", "silent_failures")

    def __init__(self, n_boot: int, seed: int) -> None:
        self.n_boot, self.seed = n_boot, seed
        self.samples: dict[str, dict[str, np.ndarray]] = {}
        self.values: dict[str, dict[str, Any]] = {}
        self._boots: list[np.ndarray] | None = None

    def record(self, name: str, evo: Evolution, state: Mapping[str, Any]) -> dict[str, Any]:
        ev = state["eval_idx"]
        groups = evo.store.reaction[ev]
        if self._boots is None:
            self._boots = _reaction_boot_indices(groups, self.n_boot, [self.seed, 99])
        fw, fe = evo.labels.candidate_errors(state["population"], state["W"], ev)
        r = np.arange(len(ev))
        f = fw[r, state["rec"]]
        e = fe[r, state["rec"]]
        wrong = (f > evo.cfg.wrong_f) | (e > evo.cfg.wrong_e)
        p_wrong = 1.0 - state["trust"]
        trusted = p_wrong <= 0.5
        bad = f > config.SILENT_F
        flagged = state["sigma"] >= state["tau"]
        sig = state["sigma"]

        stats: dict[str, Callable[[np.ndarray], float | None]] = {
            "force_rmse": lambda i: float(f[i].mean()),
            "rel_e_mae": lambda i: float(e[i].mean()),
            "wrong_rate": lambda i: float(wrong[i].mean()),
            "ece": lambda i: expected_calibration_error(p_wrong[i], wrong[i]),
            "brier": lambda i: brier(p_wrong[i], wrong[i]),
            "auroc": lambda i: _auroc(wrong[i], sig[i]),
            "trusted_fraction": lambda i: float(trusted[i].mean()),
            "badly_wrong_trusted": lambda i: float((bad[i] & trusted[i]).sum() / max(bad[i].sum(), 1)),
            "silent_failures": lambda i: float((bad[i] & trusted[i] & ~flagged[i]).sum() / max(bad[i].sum(), 1)),
        }
        allidx = np.arange(len(ev))
        vals, samp = {}, {}
        for k, fn in stats.items():
            v = fn(allidx)
            s = np.array([np.nan if (x := fn(b)) is None else x for b in self._boots])
            vals[k] = {"value": v, "ci": _ci(s)}
            samp[k] = s
        vals["counts"] = {"n_eval": int(len(ev)), "n_reactions": int(len(np.unique(groups))),
                          "badly_wrong": int(bad.sum()), "badly_wrong_trusted": int((bad & trusted).sum()),
                          "silent": int((bad & trusted & ~flagged).sum())}
        vals["population"] = list(state["population"])
        vals["served"] = state["served"]
        vals["labels"] = int(evo.oracle.spent)
        self.values[name], self.samples[name] = vals, samp
        return vals

    def paired(self, a: str, b: str) -> dict[str, Any]:
        """a - b for every metric: point difference and paired 95 % CI (same resamples)."""
        out = {}
        for k in self.METRICS:
            d = self.samples[a][k] - self.samples[b][k]
            va, vb = self.values[a][k]["value"], self.values[b][k]["value"]
            ci = _ci(d)
            out[k] = {"diff": None if va is None or vb is None else va - vb, "ci": ci,
                      "significant": bool(ci is not None and (ci[0] > 0 or ci[1] < 0))}
        return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def jsonable(x: Any) -> Any:
    return _jsonable(x)


def static_config(cfg: EvoConfig) -> EvoConfig:
    """The static control: same triage/oracle procedure and budget, no breeding, no pruning."""
    return dataclasses.replace(cfg, breeding=False, selection=False)


def rank_fraction(x: np.ndarray) -> np.ndarray:
    return rankdata(x) / len(x)


__all__ = ["EvoConfig", "Evolution", "EvalScorer", "PredStore", "LabelBook", "Member",
           "OffspringRequest", "OffspringResult", "candidate_matrix", "fit_ledger", "recommend",
           "split_pool", "static_config", "jsonable", "BASE_RECIPE", "BREED", "SELECT"]
