"""Evolution run C: niche specialists (speciation) along a budget axis.

Builds on :mod:`fittest.evolution` (triage, oracle, Calibrator twin, fitness = marginal
contribution under the per-bucket routing, selection, clone check, eval scoring), which is
reused unchanged. What run C changes (pre-registered in ``results/evolution_c/PREREG.md``):

* **Niches** are fixed once, label-free, from the *base* swarm (MACE-MPA-0 + AIMNet2 + Orb):
  ``flagged`` = pool/eval frames with σ_F ≥ τ₀ (the 70th percentile of pool σ_F, the
  Calibrator's flag threshold, "TS-like"), ``trusted`` = the rest. Structure role is not known
  at query time, so it is never used.
* **Offspring** are head-only fine-tunes of AIMNet2, Orb-v3 and MACE-MPA-0, each trained only
  on the breed labels of one niche, with a reactant-relative energy loss
  (``scripts/evolve_c_offspring.py``). Per generation one child per (family, niche) with at least
  ``min_niche_labels`` labels. Parent = the fittest alive member of that family (base or
  offspring); a child of an offspring starts from its parent's head weights.
* **Gate** (``admission="gate"``): on the *select* labels of the child's niche (component-
  disjoint from every training label): force RMSE and reactant-relative ΔE MAE not worse than
  the parent; then the clone check of run A.
* **No-selection control** (``admission="all"``): every child that trained is admitted, no gate,
  no clone rejection, no fitness pruning; the population is the base models + the newest child
  of each (family, niche) lineage, and each lineage continues from its newest child.

Everything here is numpy; torch lives in the offspring script.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fittest.data import EVAL, POOL
from fittest.evolution import (BREED, SELECT, Evolution, EvoConfig, Member, OffspringRequest,
                               OffspringResult)

FLAGGED, TRUSTED = "flagged", "trusted"

FAMILY_OF_BASE = {"aimnet2": "aimnet2", "orb": "orb", "mace": "mace_mpa0"}

BASE_RECIPES: dict[str, dict[str, Any]] = {
    "aimnet2": {"lr": 1e-4, "energy_weight": 1.0, "force_weight": 1.0, "steps": 200, "reactions_per_batch": 4},
    "orb": {"lr": 1e-4, "energy_weight": 1.0, "force_weight": 1.0, "steps": 200, "reactions_per_batch": 4},
    "mace": {"lr": 1e-3, "energy_weight": 1.0, "force_weight": 1.0, "steps": 200, "reactions_per_batch": 4},
}


@dataclass(frozen=True)
class EvoCConfig(EvoConfig):
    families: tuple[str, ...] = ("aimnet2", "orb", "mace")
    niches: tuple[str, ...] = (FLAGGED,)
    niche_quantile: float = 0.70
    min_niche_labels: int = 10
    admission: str = "gate"
    """'gate' (evolved arm) or 'all' (no-selection control: admit every child, newest only)."""
    lr_mutation: tuple[float, ...] = (0.5, 2.0)
    energy_weight_mutation: tuple[float, ...] = (0.5, 2.0)


def budget_config(total: int, generations: int = 4, **kw: Any) -> EvoCConfig:
    """Same number of generations at every budget; per-generation labels = total / generations."""
    if total % generations:
        raise ValueError("budget must be divisible by the number of generations")
    return EvoCConfig(generations=generations, labels_per_generation=total // generations, **kw)


def static_c(cfg: EvoCConfig) -> EvoCConfig:
    return dataclasses.replace(cfg, breeding=False, selection=False)


def noselect_c(cfg: EvoCConfig) -> EvoCConfig:
    return dataclasses.replace(cfg, selection=False, admission="all")


@dataclass
class NicheRequest(OffspringRequest):
    family: str = ""
    niche: str = ""
    parent_ckpt: str | None = None
    predict_uids: list[str] = field(default_factory=list)


def niche_labels(sigma: np.ndarray, pool: np.ndarray, q: float) -> tuple[np.ndarray, float]:
    """Label-free niche of every frame from base-swarm σ_F: FLAGGED iff σ_F ≥ τ₀ (pool quantile)."""
    tau0 = float(np.quantile(sigma[pool], q))
    return np.where(sigma >= tau0, FLAGGED, TRUSTED), tau0


def mutate_recipe(parent_recipe: Mapping[str, Any] | None, family: str, cfg: EvoCConfig,
                  g: int, k: int) -> dict[str, Any]:
    """Generation-0 children of a base model use the family's base recipe; later children
    mutate their parent's recipe (lr and energy weight, ×0.5 or ×2, seeded)."""
    if not parent_recipe:
        return dict(BASE_RECIPES[family])
    r = {k_: v for k_, v in parent_recipe.items() if k_ in BASE_RECIPES[family]}
    rng = np.random.default_rng([cfg.seed, g, k, 505])
    r["lr"] = float(np.clip(r["lr"] * rng.choice(cfg.lr_mutation), 1e-6, 1e-2))
    r["energy_weight"] = float(np.clip(r["energy_weight"] * rng.choice(cfg.energy_weight_mutation), 0.1, 30.0))
    return r


class NicheEvolution(Evolution):
    cfg: EvoCConfig

    def __init__(self, frames, base_preds, cfg: EvoCConfig, factory=None, blackboard=None,
                 label: str = "evolved") -> None:
        super().__init__(frames, base_preds, cfg, factory, blackboard, label)
        for m in cfg.base_models:  # family names of run C
            self.members[m].family = "mace" if m.startswith("mace") else m.split("_")[0]
            self.members[m].recipe = {}
        pool = self.store.split == POOL
        self.sigma0 = self.store.sigma_F(list(cfg.base_models))
        self.niche, self.tau0 = niche_labels(self.sigma0, pool, cfg.niche_quantile)
        self.newest: dict[tuple[str, str], str] = {}
        self.niche_of_member: dict[str, str] = {}

    # --- parents --------------------------------------------------------------------------
    def parent_for(self, family: str, niche: str, g: int) -> Member:
        if self.cfg.admission == "all":
            pid = self.newest.get((family, niche), FAMILY_OF_BASE[family])
            return self.members[pid]
        alive = [m for m in self.population if self.members[m].family == family
                 and (self.members[m].kind == "base" or self.niche_of_member.get(m) == niche)]
        if not alive:
            return self.members[FAMILY_OF_BASE[family]]  # germline
        return self.members[max(alive, key=lambda m: (self.members[m].fitness.get(g, {}).get("fitness", 0.0),
                                                      self.members[m].generation_born))]

    # --- breeding -------------------------------------------------------------------------
    def breed(self, g: int) -> dict[str, Any]:
        if not self.cfg.breeding or self.factory is None:
            return {"generation": g, "born": [], "note": "breeding disabled"}
        breed_idx = self.oracle.labelled_in(BREED)
        hold_all = self.oracle.labelled_in(SELECT)
        predict_uids = list(self.store.uids)
        requests, skipped = [], []
        k = 0
        for fam in self.cfg.families:
            for niche in self.cfg.niches:
                idx = breed_idx[self.niche[breed_idx] == niche]
                if len(idx) < self.cfg.min_niche_labels:
                    skipped.append({"family": fam, "niche": niche, "n_labels": int(len(idx))})
                    continue
                parent = self.members[self.parent_for(fam, niche, g).id]
                recipe = mutate_recipe(parent.recipe if parent.kind == "offspring" else None, fam, self.cfg, g, k)
                hold = hold_all[self.niche[hold_all] == niche]
                assert not set(self.component[idx]) & set(self.component[hold_all]), "gate set leaks"
                cid = f"g{g + 1}_{fam}_{niche}"
                requests.append(NicheRequest(
                    id=cid, parent=parent, recipe=recipe, train_uids=[self.store.uids[i] for i in idx],
                    holdout_uids=[self.store.uids[i] for i in hold], seed=self.cfg.seed * 1000 + g * 10 + k,
                    family=fam, niche=niche, parent_ckpt=parent.ckpt if parent.kind == "offspring" else None,
                    predict_uids=predict_uids))
                k += 1
        self.post("breed_request", {"generation": g, "skipped": skipped, "requests": [
            {"id": r.id, "parent": r.parent.id, "family": r.family, "niche": r.niche, "recipe": r.recipe,
             "n_train": len(r.train_uids), "n_holdout": len(r.holdout_uids)} for r in requests]})
        results = self.factory(requests) if requests else []
        born = [self._admit(req, res, g) for req, res in zip(requests, results)]
        return {"generation": g, "born": born, "skipped": skipped}

    def _admit(self, req: OffspringRequest, res: OffspringResult, g: int) -> dict[str, Any]:
        assert isinstance(req, NicheRequest)
        mem = Member(id=req.id, family=req.family, kind="offspring", generation_born=g + 1,
                     parent=req.parent.id, recipe={**req.recipe, "niche": req.niche},
                     train_uids=req.train_uids,
                     train_components={int(self.component[self.store.index[u]]) for u in req.train_uids},
                     ckpt=res.ckpt)
        self.members[mem.id] = mem
        self.niche_of_member[mem.id] = req.niche
        if res.predictions is None:
            mem.status, mem.gate = "stillborn", {"passed": False, "error": res.error}
            self.post("birth", {"generation": g, "member": mem.id, "status": mem.status, "error": res.error})
            return {"id": mem.id, "status": mem.status, "error": res.error}
        self.store.add(mem.id, res.predictions)
        mem.gate = self.gate(mem, res)
        clone = self.clone_check(mem.id)
        mem.gate["clone_check"] = clone
        if self.cfg.admission == "all":
            old = self.newest.get((req.family, req.niche))
            if old is not None and old in self.population:
                self.population.remove(old)
                o = self.members[old]
                o.status, o.died_generation, o.death_reason = "dead", g, "replaced by newest child (no selection)"
            self.population.append(mem.id)
            self.newest[(req.family, req.niche)] = mem.id
        elif not mem.gate["passed"]:
            mem.status = "stillborn"
        elif clone["is_clone"]:
            mem.status = "clone"
        else:
            self.population.append(mem.id)
        out = {"id": mem.id, "parent": mem.parent, "family": req.family, "niche": req.niche,
               "recipe": mem.recipe, "status": mem.status, "gate": mem.gate, "wall_seconds": res.wall_seconds}
        self.post("birth", {"generation": g, **out})
        return out

    # --- gate: niche select labels ----------------------------------------------------------
    def gate(self, child: Member, res: OffspringResult) -> dict[str, Any]:
        hold_all = self.oracle.labelled_in(SELECT)
        niche = self.niche_of_member.get(child.id)
        hold = hold_all[self.niche[hold_all] == niche] if niche else hold_all
        scope = f"select labels in niche {niche}"
        if len(hold) < 5:
            hold, scope = hold_all, "all select labels (niche has < 5)"
        if set(self.component[hold]) & child.train_components:
            raise AssertionError("gate hold-out shares a component with the child's training labels")
        f_c, e_c = self.labels.model_errors(child.id, hold)
        f_p, e_p = self.labels.model_errors(child.parent, hold)
        n = self.store.n_atoms[hold]
        rmse = lambda f: float(np.sqrt(np.sum(f ** 2 * n) / n.sum()))  # noqa: E731
        checks = [
            {"name": "holdout_force_rmse_not_worse", "value": rmse(f_c), "limit": rmse(f_p),
             "passed": bool(rmse(f_c) <= rmse(f_p))},
            {"name": "holdout_rel_energy_mae_not_worse", "value": float(e_c.mean()),
             "limit": float(e_p.mean()), "passed": bool(e_c.mean() <= e_p.mean())},
        ]
        return {"passed": all(c["passed"] for c in checks), "checks": checks, "scope": scope,
                "n_holdout": int(len(hold)), "n_holdout_reactions": int(len(np.unique(self.store.reaction[hold]))),
                "holdout_components_disjoint": True}

    # --- niche report (label-free + per-niche eval served error for the summary) -------------
    def niche_report(self) -> dict[str, Any]:
        pool = self.store.split == POOL
        ev = self.store.split == EVAL
        return {"tau0": self.tau0, "quantile": self.cfg.niche_quantile,
                "pool": {n: int(((self.niche == n) & pool).sum()) for n in (FLAGGED, TRUSTED)},
                "eval": {n: int(((self.niche == n) & ev).sum()) for n in (FLAGGED, TRUSTED)}}


def served_by_niche(evo: NicheEvolution, state: Mapping[str, Any]) -> dict[str, Any]:
    """Served force RMSE / rel-E MAE on eval split by niche (descriptive; scorer-side only)."""
    ev = state["eval_idx"]
    fw, fe = evo.labels.candidate_errors(state["population"], state["W"], ev)
    r = np.arange(len(ev))
    f, e = fw[r, state["rec"]], fe[r, state["rec"]]
    out = {}
    for n in (FLAGGED, TRUSTED):
        m = evo.niche[ev] == n
        out[n] = {"n": int(m.sum()), "force_rmse": float(f[m].mean()) if m.any() else None,
                  "rel_e_mae": float(e[m].mean()) if m.any() else None}
    return out


__all__ = ["EvoCConfig", "NicheEvolution", "NicheRequest", "budget_config", "static_c", "noselect_c",
           "niche_labels", "mutate_recipe", "served_by_niche", "BASE_RECIPES", "FLAGGED", "TRUSTED"]
