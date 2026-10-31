import json

import numpy as np
import pytest

from fittest.blackboard import TICK, Blackboard, load_jsonl
from fittest.cache import (load_frames, load_models, load_predictions, save_frames,
                              save_predictions)
from fittest.data import Frame, strip_labels


def _frame(uid: str, n: int, labelled: bool = True) -> Frame:
    rng = np.random.default_rng(len(uid) + n)
    return Frame(uid=uid, reaction="CH4/rxn1", formula="CH4", split="pool", role="path",
                 Z=np.array([6] + [1] * (n - 1)), R=rng.normal(size=(n, 3)),
                 E=-100.0 if labelled else None, F=rng.normal(size=(n, 3)) if labelled else None)


def test_frames_round_trip(tmp_path) -> None:
    frames = [_frame("a", 3), _frame("bb", 5)]
    save_frames(tmp_path / "f.pkl", frames)
    back = load_frames(tmp_path / "f.pkl")
    assert [f.uid for f in back] == ["a", "bb"]
    np.testing.assert_array_equal(back[1].F, frames[1].F)
    stripped = strip_labels(back[0])
    assert stripped.E is None and stripped.F is None and back[0].E is not None


def test_predictions_round_trip_ragged(tmp_path) -> None:
    uids = ["x", "y", "z"]
    forces = [np.full((n, 3), float(n)) for n in (2, 5, 1)]
    save_predictions(tmp_path / "m.npz", "m", uids, np.array([1.0, 2.0, 3.0]), forces,
                     {"atoms_per_s": 123.0, "nested": {"a": [1, 2]}})
    p = load_predictions(tmp_path / "m.npz")
    assert p.model == "m" and p.uids == uids
    assert p.energy["y"] == 2.0
    for u, f in zip(uids, forces):
        np.testing.assert_array_equal(p.forces[u], f)
    assert p.meta == {"atoms_per_s": 123.0, "nested": {"a": [1, 2]}}
    assert p.subset(["z", "nope"]).uids == ["z"]


def test_save_predictions_validates(tmp_path) -> None:
    with pytest.raises(ValueError):
        save_predictions(tmp_path / "m.npz", "m", ["a", "a"], np.zeros(2), [np.zeros((1, 3))] * 2, {})
    with pytest.raises(ValueError):
        save_predictions(tmp_path / "m.npz", "m", ["a"], np.zeros(2), [np.zeros((1, 3))], {})


def test_load_models_names_every_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="orb.npz.*aimnet2.npz"):
        load_models(tmp_path, ["orb", "aimnet2"])


def test_blackboard_ordering_and_filters(tmp_path) -> None:
    with Blackboard(tmp_path / "bb.jsonl", clock=lambda: 0.0) as bb:
        a = bb.post("bid", "m1", {"uid": "x", "value": 0.5})
        b = bb.post("flag", "hook", {"uid": "x"})
        assert bb.tick() == 1
        c = bb.post("bid", "m2", {"uid": "y", "value": 0.1})
        assert (a, b, c) == (0, 1, 3)  # the tick is event 2
        assert [e.id for e in bb.read()] == [0, 1, 2, 3]
        assert [e.id for e in bb.read("bid")] == [0, 3]
        assert [e.id for e in bb.read(["bid", "flag"], since=1)] == [1, 3]
        assert [e.id for e in bb.read(round=0)] == [0, 1]
        assert [e.kind for e in bb.read(round=1)] == [TICK, "bid"]
        assert bb.read("bid", round=1)[0].author == "m2"
        events = bb.read()
    assert load_jsonl(tmp_path / "bb.jsonl") == events


def test_blackboard_rejects_non_json_payloads() -> None:
    bb = Blackboard()
    with pytest.raises(TypeError):
        bb.post("x", "a", {"arr": np.zeros(2)})
    with pytest.raises(ValueError):
        bb.post("x", "a", {"v": float("nan")})
    assert len(bb) == 0
    eid = bb.post("x", "a", {"v": 1})
    assert json.loads(bb.read()[eid].to_json())["payload"] == {"v": 1}
