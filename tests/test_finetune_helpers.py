"""CPU-only checks of scripts/finetune.py's label-handling helpers (no torch, no GPU)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fittest.data import Frame

_SPEC = importlib.util.spec_from_file_location(
    "finetune_script", Path(__file__).resolve().parents[1] / "scripts" / "finetune.py")
ft = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ft)


def _frames(n_reactions: int = 20, per: int = 5, split: str = "pool") -> list[Frame]:
    rng = np.random.default_rng(0)
    Z = np.array([6, 1, 1, 1, 1])
    return [Frame(f"test:r{r}:p{i}", f"r{r}", "CH4", split, "path", Z, rng.normal(size=(5, 3)),
                  float(-100 + i), rng.normal(size=(5, 3)))
            for r in range(n_reactions) for i in range(per)]


@pytest.mark.parametrize("wrap", [lambda u: u, lambda u: {"uids": u},
                                  lambda u: [{"uid": x, "E": 0.0} for x in u],
                                  lambda u: {x: 1 for x in u}])
def test_read_uid_list_accepts_common_layouts(tmp_path: Path, wrap) -> None:
    uids = [f.uid for f in _frames()[:12]]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(wrap(uids)))
    assert ft.read_uid_list(path) == uids


def _with_reactants(n_reactions: int, share: dict[int, int]) -> list[Frame]:
    """Path frames plus one reactant per reaction; reaction r reuses the reactant of share[r]."""
    frames = _frames(n_reactions, 5)
    rng = np.random.default_rng(1)
    geo = {r: rng.normal(size=(5, 3)) * 2 for r in range(n_reactions)}
    for r in range(n_reactions):
        R = geo[share.get(r, r)]
        frames.append(Frame(f"test:r{r}:reactant", f"r{r}", "CH4", "pool", "reactant",
                            np.array([6, 1, 1, 1, 1]), R.copy(), -100.0, np.zeros((5, 3))))
    return frames


def test_holdout_is_component_disjoint() -> None:
    share = {1: 0, 2: 0, 5: 4, 9: 8, 13: 12}  # reactions sharing a reactant geometry
    frames = _with_reactants(20, share)
    labelled = [f for f in frames if f.role == "path"]
    for seed in range(5):
        train, hold, check = ft.holdout_by_component(labelled, frames, 0.15, seed)
        held = {f.reaction for f in hold}
        for r, src in share.items():  # linked reactions are never split across train/hold-out
            assert (f"r{r}" in held) == (f"r{src}" in held)
        assert not held & {f.reaction for f in train}
        assert 0.10 <= check["holdout_fraction"] <= 0.5 and check["shared_geometries_train_holdout"] == 0


def test_holdout_lifts_to_ten_percent_when_components_are_coarse() -> None:
    # 4 reactions, components {r0,r1,r2} and {r3}: 15 % cap fits nothing -> smallest lift (r3, 25 %)
    frames = _with_reactants(4, {1: 0, 2: 0})
    labelled = [f for f in frames if f.role == "path"]
    train, hold, check = ft.holdout_by_component(labelled, frames, 0.15, 0)
    assert {f.reaction for f in hold} == {"r3"} and check["holdout_fraction"] == 0.25
    with pytest.raises(ValueError, match="single"):
        one = _with_reactants(3, {1: 0, 2: 0})
        ft.holdout_by_component([f for f in one if f.role == "path"], one, 0.15, 0)


@pytest.mark.parametrize("split", ["eval", "skeptic_ref"])
def test_refuses_eval_and_skeptic_labels(split: str) -> None:
    frames = _frames(2, 2)
    frames[0] = dataclasses.replace(frames[0], split=split)
    with pytest.raises(ValueError, match="refusing"):
        ft.select_labelled(frames, [f.uid for f in frames])


def test_refuses_unknown_or_unlabelled() -> None:
    frames = _frames(2, 2)
    with pytest.raises(KeyError):
        ft.select_labelled(frames, ["nope"])
    frames[1] = dataclasses.replace(frames[1], E=None, F=None)
    with pytest.raises(ValueError, match="no E/F"):
        ft.select_labelled(frames, [f.uid for f in frames])


def test_error_metrics_relative_energy_cancels_offsets() -> None:
    frames = _frames(3, 4)
    E = np.array([f.E + 1000.0 * int(f.reaction[1:]) for f in frames])  # per-reaction offsets
    F = [f.F + 0.2 for f in frames]
    m = ft.error_metrics(frames, E, F)
    assert m["rel_energy_mae"] == pytest.approx(0.0, abs=1e-9)
    assert m["force_rmse"] == pytest.approx(0.2) and m["force_mae"] == pytest.approx(0.2)
    assert m["frac_force_wrong"] == 0.0


def test_geometry_leak_check_drops_frames_sharing_eval_geometry() -> None:
    frames = _frames(4, 3)
    shared = frames[0]  # an eval frame with the same geometry (shifted + permuted) as frames[0]
    perm = np.array([1, 0, 2, 4, 3])
    twin = dataclasses.replace(shared, uid="test:rE:p0", reaction="rE", split="eval",
                               Z=shared.Z[perm], R=shared.R[perm] + 3.0)
    endpoint = dataclasses.replace(frames[3], uid="test:r1:reactant", role="reactant")
    eval_endpoint = dataclasses.replace(endpoint, uid="test:rF:reactant", reaction="rF", split="eval")
    pool = frames[1:3] + [frames[0], endpoint] + frames[4:]
    kept, check = ft.exclude_eval_geometry(pool, pool + [twin, eval_endpoint])
    assert check["n_own_geometry_matches"] == 2  # frames[0] and the shared reactant itself
    assert {f.reaction for f in pool} - {f.reaction for f in kept} >= {"r1"}  # whole r1 via endpoint
    assert shared.uid not in {f.uid for f in kept}
    assert check["n_excluded"] == len(pool) - len(kept) and check["passed"]


def test_fittest_gates_require_both_metrics_not_worse() -> None:
    before = {"force_rmse": 0.6, "rel_energy_mae": 0.18}
    ok = ft.fittest_checks(before, {"force_rmse": 0.3, "rel_energy_mae": 0.15})
    assert all(c["passed"] for c in ok)
    bad = ft.fittest_checks(before, {"force_rmse": 0.3, "rel_energy_mae": 0.31})
    assert [c["passed"] for c in bad] == [True, False]
    nan = ft.fittest_checks(before, {"force_rmse": 0.3, "rel_energy_mae": float("nan")})
    assert nan[1]["passed"] is False and nan[1]["value"] is None
