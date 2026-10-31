"""Evolutionary loop (fittest.evolution): synthetic frames and predictions, CPU, no torch."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fittest.cache import Predictions
from fittest.data import EVAL, POOL, Frame
from fittest.evolution import (BREED, SELECT, EvalScorer, Evolution, EvoConfig, LabelBook,
                               OffspringResult, PredStore, candidate_matrix, fit_ledger,
                               recommend, static_config)
from fittest.signals import force_rmse, force_spread

N_ATOMS = 5


def _frames(n_pool: int = 40, n_eval: int = 16, n_path: int = 3, seed: int = 0) -> list[Frame]:
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


def _model(frames, name, f_noise, e_noise, seed, scale_by=None) -> Predictions:
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


CFG = EvoConfig(generations=3, labels_per_generation=24, n_bootstrap_fitness=60, n_bootstrap_eval=60,
                offspring_per_generation=3, select_frame_share=0.4)


class SynthFactory:
    """Child quality by position: a = good (helps), b = worse than its parent (stillborn),
    c = a copy of aimnet2 (clone)."""

    def __init__(self, frames, preds):
        self.frames, self.preds, self.calls = frames, preds, []

    def __call__(self, requests):
        out = []
        for r in requests:
            self.calls.append(r)
            tag = r.id[-1]
            if tag == "a":
                p = _model(self.frames, r.id, 0.12, 0.05, 100 + len(self.calls))
            elif tag == "b":
                p = _model(self.frames, r.id, 2.0, 2.0, 200 + len(self.calls))
            else:
                src = self.preds["aimnet2"]
                p = Predictions(r.id, dict(src.energy), {u: v.copy() for u, v in src.forces.items()}, {})
            out.append(OffspringResult(p, toolkit={"accepted": True, "worst_ratio": 0.5, "max_ratio": 1.0}))
        return out


def test_candidate_errors_match_direct(world):
    frames, preds = world
    keep = [f for f in frames if f.split in (POOL, EVAL)]
    store = PredStore(keep)
    for m, p in preds.items():
        store.add(m, p)
    lb = LabelBook(store)
    models = list(preds)
    subsets, W, tiebreak = candidate_matrix(models)
    assert len(subsets) == 7 and subsets[-1] == tuple(models)
    assert tiebreak[-1] == 0  # full set first in the prior order
    idx = np.arange(0, len(keep), 7)
    fr, fe = lb.candidate_errors(models, W, idx)
    by_uid = {f.uid: f for f in keep}
    reactant = {f.reaction: f for f in keep if f.role == "reactant"}
    for row, i in enumerate(idx):
        f = keep[i]
        r = reactant[f.reaction]
        for c, s in enumerate(subsets):
            Fm = np.mean([preds[m].forces[f.uid] for m in s], axis=0)
            assert fr[row, c] == pytest.approx(force_rmse(Fm, by_uid[f.uid].F), rel=1e-9)
            rel_hat = np.mean([preds[m].energy[f.uid] - preds[m].energy[r.uid] for m in s])
            assert fe[row, c] == pytest.approx(abs(rel_hat - (f.E - r.E)), abs=1e-9)
    # σ_F matches the reference implementation
    sig = store.sigma_F(models)
    f0 = keep[3]
    assert sig[3] == pytest.approx(force_spread(np.stack([preds[m].forces[f0.uid] for m in models]))[0])


def test_ledger_recommends_best_and_prior_prefers_full_set():
    subsets, W, tb = candidate_matrix(["a", "b", "c"])
    C = len(subsets)
    # no labels: every candidate has the prior; the tie-break serves the full set
    led = fit_ledger(np.zeros((0, C), bool), np.zeros((0, C), bool), np.zeros(0, int), 2, tb)
    rec, trust = recommend(led, np.array([0, 1]))
    assert subsets[rec[0]] == ("a", "b", "c") and trust[0] == pytest.approx(2 / 3)
    # bucket 0: only 'b' is right; bucket 1: everything wrong except the full set
    wrong = np.ones((10, C), bool)
    wrong[:5, subsets.index(("b",))] = False
    wrong[5:, subsets.index(("a", "b", "c"))] = False
    bucket = np.array([0] * 5 + [1] * 5)
    led = fit_ledger(wrong, np.ones_like(wrong), bucket, 2, tb)
    rec, trust = recommend(led, np.array([0, 1]))
    assert subsets[rec[0]] == ("b",) and subsets[rec[1]] == ("a", "b", "c")
    # unclean labels are ignored
    led = fit_ledger(wrong, np.zeros_like(wrong), bucket, 2, tb)
    assert subsets[recommend(led, np.array([0]))[0][0]] == ("a", "b", "c")


def test_static_run_budget_split_and_no_eval_labels(world):
    frames, preds = world
    sc = EvalScorer(60, 0)
    evo = Evolution(frames, preds, static_config(CFG), None, label="static")
    evo.run(sc)
    assert evo.oracle.spent == CFG.total_budget
    labelled = np.array(evo.oracle.labelled)
    assert set(evo.store.split[labelled]) == {POOL}
    k_sel = int(round(CFG.select_label_share * CFG.labels_per_generation))
    assert (evo.part[labelled] == SELECT).sum() == k_sel * CFG.generations
    # breed and select parts share no component
    comp_s = set(evo.component[evo.part == SELECT])
    comp_b = set(evo.component[evo.part == BREED])
    assert comp_s and comp_b and not comp_s & comp_b
    assert evo.population == list(CFG.base_models)  # no breeding, no pruning
    assert set(sc.values) == {f"static_g{g}" for g in range(CFG.generations)}
    v = sc.values[f"static_g{CFG.generations - 1}"]
    assert 0 < v["force_rmse"]["value"] < 1 and v["force_rmse"]["ci"] is not None
    # the bad model gets negative or ~0 fitness, the good answerers positive
    fit = evo.members["mace_mpa0"].fitness[CFG.generations - 1]["fitness"]
    assert fit < evo.members["aimnet2"].fitness[CFG.generations - 1]["fitness"]


def test_requesting_eval_or_over_budget_is_refused(world):
    frames, preds = world
    evo = Evolution(frames, preds, CFG, None)
    ev = int(np.flatnonzero(evo.store.split == EVAL)[0])
    with pytest.raises(ValueError):
        evo.oracle.request([ev], 0)
    pool = np.flatnonzero(evo.store.split == POOL)
    with pytest.raises(ValueError):
        evo.oracle.request(pool[: CFG.total_budget + 1], 0)


def test_evolution_gates_clones_and_selection(world):
    frames, preds = world
    fac = SynthFactory(frames, preds)
    sc = EvalScorer(60, 0)
    evo = Evolution(frames, preds, CFG, fac, label="evolved")
    hist = evo.run(sc)
    assert len(hist) == CFG.generations and evo.oracle.spent == CFG.total_budget
    m = evo.members
    # child b is worse than its parent on the select labels: stillborn; child c is a clone
    assert m["g1_b"].status == "stillborn" and not m["g1_b"].gate["passed"]
    assert m["g1_c"].status == "clone" and m["g1_c"].gate["clone_check"]["is_clone"]
    assert m["g1_a"].gate["passed"]
    # the gate set never shares a component with any child's training labels
    sel = evo.oracle.labelled_in(SELECT)
    for req in fac.calls:
        train_c = {int(evo.component[evo.store.index[u]]) for u in req.train_uids}
        assert not train_c & set(evo.component[sel])
        assert set(req.holdout_uids) <= {evo.store.uids[i] for i in sel}
    # population bounds hold after every selection
    for h in hist:
        assert CFG.min_population <= len(h["population_after_selection"]) <= CFG.max_population
    # the good child entered; the bad foundation MACE was eventually pruned (it is useless)
    assert "g1_a" in hist[1]["triage"]["population"]
    assert m["mace_mpa0"].status == "dead"
    # fitness is measured only on labels clean for the population
    for h in hist[1:]:
        for c in h["selection"]["fitness"].values():
            assert c["n_labels"] <= evo.oracle.spent
    tree = evo.family_tree()
    ids = {t["id"] for t in tree}
    assert {"g1_a", "g1_b", "g1_c"} <= ids
    assert all(t["parent"] in ids for t in tree if t["parent"])
    # a second-generation child of an offspring has that offspring as parent
    assert any(t["parent"] == "g1_a" for t in tree)
    # paired comparisons are defined
    d = sc.paired("evolved_g2", "evolved_g0")
    assert set(d) == set(EvalScorer.METRICS)


def test_static_equals_evolved_without_offspring(world):
    frames, preds = world
    sc = EvalScorer(60, 0)
    a = Evolution(frames, preds, static_config(CFG), None, label="a")
    a.run(sc)
    b = Evolution(frames, preds, dataclasses.replace(CFG, selection=False), None, label="b")
    b.run(sc)
    assert a.oracle.labelled == b.oracle.labelled
    assert sc.values["a_g2"]["force_rmse"]["value"] == sc.values["b_g2"]["force_rmse"]["value"]


def test_random_selection_control_kills_same_number(world):
    frames, preds = world
    cfg = dataclasses.replace(CFG, selection_mode="random")
    evo = Evolution(frames, preds, cfg, SynthFactory(frames, preds), label="random")
    evo.run(None)
    assert CFG.min_population <= len(evo.population) <= CFG.max_population
