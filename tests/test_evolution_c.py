"""Evolution run C (fittest.evolution_c): niche specialists, budget axis, no-selection control.
Synthetic frames and predictions, CPU, no torch."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fittest.cache import Predictions
from fittest.data import EVAL, POOL, Frame
from fittest.evolution import BREED, SELECT, EvalScorer, OffspringResult
from fittest.evolution_c import (BASE_RECIPES, FLAGGED, TRUSTED, NicheEvolution, budget_config,
                                 mutate_recipe, niche_labels, noselect_c, served_by_niche, static_c)

ROOT = Path(__file__).resolve().parents[1]
N_ATOMS = 5


def _frames(n_pool=48, n_eval=16, n_path=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for split, n in ((POOL, n_pool), (EVAL, n_eval)):
        for r in range(n):
            rx = f"{split}_rx{r:03d}"
            Z = np.array([6, 1, 1, 8, 7])
            base = rng.normal(size=(N_ATOMS, 3)) * 1.5
            e0 = rng.normal() * 5
            for k, role in enumerate(["reactant", "product", "ts"] + ["path"] * n_path):
                R = base + rng.normal(scale=0.3, size=base.shape)
                out.append(Frame(uid=f"{rx}_{k}", reaction=rx, formula="CH2NO", split=split, role=role,
                                 Z=Z, R=R, E=float(e0 + rng.normal() * 0.5), F=rng.normal(size=(N_ATOMS, 3))))
    return out


def _model(frames, name, f_noise, e_noise, seed, scale_by=None):
    rng = np.random.default_rng(seed)
    E, F = {}, {}
    for f in frames:
        s = f_noise if scale_by is None else f_noise * scale_by(f)
        F[f.uid] = f.F + rng.normal(scale=s, size=f.F.shape)
        E[f.uid] = f.E + 3.0 + rng.normal(scale=e_noise)
    return Predictions(name, E, F, {})


@pytest.fixture(scope="module")
def world():
    frames = _frames()
    hard = {f.uid: (3.0 if i % 4 == 0 else 1.0) for i, f in enumerate(frames)}
    preds = {"mace_mpa0": _model(frames, "mace_mpa0", 0.6, 0.4, 1),
             "aimnet2": _model(frames, "aimnet2", 0.15, 0.05, 2, lambda f: hard[f.uid]),
             "orb": _model(frames, "orb", 0.2, 0.08, 3, lambda f: hard[f.uid])}
    return frames, preds


def _cfg(total=96, **kw):
    return budget_config(total, 4, n_bootstrap_fitness=60, n_bootstrap_eval=60, select_frame_share=0.4,
                         min_niche_labels=3, **kw)


class SynthFactory:
    """aimnet2 children are good (low noise everywhere), orb children are bad (stillborn),
    mace children are copies of aimnet2 (clones)."""

    def __init__(self, frames, preds):
        self.frames, self.preds, self.calls = frames, preds, []

    def __call__(self, requests):
        out = []
        for r in requests:
            self.calls.append(r)
            if r.family == "aimnet2":
                p = _model(self.frames, r.id, 0.05, 0.02, 100 + len(self.calls))
            elif r.family == "orb":
                p = _model(self.frames, r.id, 2.0, 2.0, 200 + len(self.calls))
            else:
                src = self.preds["aimnet2"]
                p = Predictions(r.id, dict(src.energy), {u: v.copy() for u, v in src.forces.items()}, {})
            out.append(OffspringResult(p, ckpt=f"/ckpt/{r.id}.pt"))
        return out


def test_budget_config_scales_per_generation_labels():
    for B in (200, 400, 800):
        c = budget_config(B)
        assert c.generations == 4 and c.labels_per_generation * 4 == B and c.total_budget == B
    with pytest.raises(ValueError):
        budget_config(201)


def test_niches_are_label_free_quantiles():
    sigma = np.arange(100, dtype=float)
    pool = np.ones(100, bool)
    niche, tau = niche_labels(sigma, pool, 0.7)
    assert tau == pytest.approx(np.quantile(sigma, 0.7))
    assert (niche == FLAGGED).sum() == 30 and set(niche) == {FLAGGED, TRUSTED}


def test_mutation_base_recipe_then_bounded_seeded_changes():
    cfg = _cfg()
    assert mutate_recipe(None, "orb", cfg, 0, 0) == BASE_RECIPES["orb"]
    parent = dict(BASE_RECIPES["aimnet2"])
    a = mutate_recipe(parent, "aimnet2", cfg, 1, 0)
    assert a == mutate_recipe(parent, "aimnet2", cfg, 1, 0)  # deterministic
    assert a["lr"] / parent["lr"] in (0.5, 2.0)
    assert a["energy_weight"] / parent["energy_weight"] in (0.5, 2.0)
    assert a["steps"] == parent["steps"]


def test_evolved_specialists_gate_clone_and_niche_training(world):
    frames, preds = world
    fac = SynthFactory(frames, preds)
    evo = NicheEvolution(frames, preds, _cfg(), fac, label="evolved")
    evo.run(EvalScorer(60, 0))
    assert evo.oracle.spent == 96
    assert all(evo.store.split[i] == POOL for i in evo.oracle.labelled)
    fams = {r.family for r in fac.calls}
    assert fams == {"aimnet2", "orb", "mace"}
    for r in fac.calls:  # trained only on breed labels of its niche; gate on disjoint select labels
        idx = [evo.store.index[u] for u in r.train_uids]
        assert all(evo.part[i] == BREED and evo.niche[i] == r.niche for i in idx)
        hold = [evo.store.index[u] for u in r.holdout_uids]
        assert all(evo.part[i] == SELECT for i in hold)
        assert not set(evo.component[idx]) & set(evo.component[hold])
        assert not set(r.predict_uids) - set(evo.store.uids)
    tree = {m["id"]: m for m in evo.family_tree()}
    orb_kids = [m for m in tree.values() if m["family"] == "orb" and m["kind"] == "offspring"]
    assert orb_kids and all(m["status"] == "stillborn" for m in orb_kids)
    mace_kids = [m for m in tree.values() if m["family"] == "mace" and m["kind"] == "offspring"]
    assert mace_kids and all(m["status"] in ("clone", "stillborn") for m in mace_kids)
    aim = [m for m in tree.values() if m["family"] == "aimnet2" and m["kind"] == "offspring"]
    assert aim and aim[0]["gate"]["passed"] and aim[0]["hyperparams"]["niche"] == FLAGGED
    # a gated child of an offspring starts from the parent's checkpoint
    later = [r for r in fac.calls if r.family == "aimnet2" and r.parent.kind == "offspring"]
    assert all(r.parent_ckpt == r.parent.ckpt for r in later)
    assert len(evo.population) <= evo.cfg.max_population


def test_noselect_control_admits_all_and_keeps_newest(world):
    frames, preds = world
    fac = SynthFactory(frames, preds)
    evo = NicheEvolution(frames, preds, noselect_c(_cfg()), fac, label="noselect")
    evo.run(EvalScorer(60, 0))
    off = [m for m in evo.population if evo.members[m].kind == "offspring"]
    assert sorted(evo.members[m].family for m in off) == ["aimnet2", "mace", "orb"]  # stillborn/clone kept
    assert all(evo.members[m].generation_born == 3 for m in off)  # only the newest generation
    assert {"mace_mpa0", "aimnet2", "orb"} <= set(evo.population)
    later = [r for r in fac.calls if r.parent.kind == "offspring"]
    assert later and all(r.parent.generation_born == r.seed % 1000 // 10 for r in later)


def test_static_equals_evolved_without_factory(world):
    frames, preds = world
    sc = EvalScorer(60, 0)
    a = NicheEvolution(frames, preds, static_c(_cfg()), None, label="static")
    a.run(sc)
    b = NicheEvolution(frames, preds, dataclasses.replace(_cfg(), breeding=False), None, label="nobreed")
    b.run(sc)
    assert a.oracle.labelled == b.oracle.labelled
    assert sc.values["static_g3"]["force_rmse"]["value"] == pytest.approx(sc.values["nobreed_g3"]["force_rmse"]["value"])
    assert sc.paired("static_g3", "nobreed_g3")["force_rmse"]["diff"] == pytest.approx(0.0)
    st = a.serve_eval()
    by = served_by_niche(a, st)
    assert by[FLAGGED]["n"] + by[TRUSTED]["n"] == len(st["eval_idx"])


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_factory_cache_key_is_content_addressed(world, tmp_path):
    frames, preds = world
    drv = _load_script("evolve_c")
    fac = SynthFactory(frames, preds)
    evo = NicheEvolution(frames, preds, _cfg(), fac, label="x")
    evo.triage(0)
    evo.select(0)
    evo.breed(0)
    r = fac.calls[0]
    k1 = drv.GPUFactory.key(r)
    r2 = dataclasses.replace(r, id="other_name")
    assert drv.GPUFactory.key(r2) == k1  # run/child name does not matter
    r3 = dataclasses.replace(r, recipe={**r.recipe, "lr": r.recipe["lr"] * 2})
    assert drv.GPUFactory.key(r3) != k1


def test_verdict_rule():
    drv = _load_script("evolve_c")
    sig = lambda d: {"diff": d, "ci": [d - 0.01, d + 0.005], "significant": True}  # noqa: E731
    ns = lambda d: {"diff": d, "ci": [d - 0.1, d + 0.1], "significant": False}  # noqa: E731
    comp = {"final_vs_static": {"force_rmse": sig(-0.02), "rel_e_mae": ns(0.0), "ece": ns(0.0)},
            "final_vs_gen0": {"force_rmse": sig(-0.03), "rel_e_mae": ns(0.0)}}
    assert drv.verdict_for(comp, ["aimnet2", "g2_aimnet2_flagged"])["holds"]
    assert not drv.verdict_for(comp, ["aimnet2", "orb"])["holds"]  # no offspring
    comp["final_vs_static"]["rel_e_mae"] = sig(0.02)
    assert not drv.verdict_for(comp, ["g2_aimnet2_flagged"])["holds"]  # rel-E regression


def test_offspring_script_imports_without_torch():
    mod = _load_script("evolve_c_offspring")
    assert set(mod.HEAD_PREFIX) == {"aimnet2", "orb", "mace"}


def test_exploratory_knobs_default_off_and_explore_base_recipes():
    from fittest.evolution_c import BASE_RECIPES, budget_config, mutate_recipe
    cfg = budget_config(400)
    assert (cfg.children_per_niche, cfg.explore_base, cfg.gate_rel_e_tol, cfg.protect_base) == (1, False, 0.0, False)
    assert mutate_recipe(None, "aimnet2", cfg, 0, 0) == BASE_RECIPES["aimnet2"]  # pre-registered
    ex = budget_config(400, explore_base=True)
    ws = [mutate_recipe(None, "aimnet2", ex, 0, 0, sibling=j)["energy_weight"] for j in range(3)]
    assert ws == [3.0, 10.0, 1.0]
    assert mutate_recipe(None, "aimnet2", ex, 0, 0, sibling=3)["lr"] == BASE_RECIPES["aimnet2"]["lr"] * 0.5
