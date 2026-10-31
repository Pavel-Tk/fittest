"""Trust service: verdict logic with fake model agents (no GPU, no labels)."""

import json

import numpy as np
import pytest

from trustswarm.service import TrustSnapshot, read_xyz, score

WATER = {"Z": [8, 1, 1], "R": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]}
METHANE = {"Z": [6, 1, 1, 1, 1],
           "R": [[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]}


def fake(scale: float, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)

    def predict(Z, R, batch):
        F = -scale * np.asarray(R) + noise * rng.standard_normal(np.shape(R))
        return {"energy": np.arange(int(batch.max()) + 1, dtype=float), "forces": F}

    return predict


@pytest.fixture
def snapshot():
    p = [[0.9, 0.9, 0.9], [0.7, 0.7, 0.7], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]
    return TrustSnapshot(models=["a", "b", "c"], tau=0.3, sigma_edges=[0.05, 0.15, 0.3],
                         fbar_edges=[0.5, 1.5],
                         p_correct={"a": p, "b": [[x - 0.1 for x in r] for r in p],
                                    "c": [[x + 0.05 for x in r] for r in p]})


def test_agreeing_models_are_trusted(snapshot):
    preds = {m: fake(1.0) for m in "abc"}
    res = score([WATER, METHANE], preds, snapshot)
    assert len(res) == 2
    for r in res:
        assert r["sigma_F"] == pytest.approx(0.0, abs=1e-12)
        assert not r["flagged"]
        assert r["recommended_model"] == "c"  # highest P(correct) in every bucket
        assert r["trust"] == pytest.approx(0.95)


def test_disagreement_flags_and_lowers_trust(snapshot):
    preds = {"a": fake(1.0), "b": fake(1.0, noise=1.0, seed=1), "c": fake(-1.0)}
    res = score([WATER], preds, snapshot)[0]
    assert res["sigma_F"] > snapshot.tau
    assert res["flagged"] and res["action"].startswith("send to oracle")
    assert res["bucket"][0] == 3 and res["trust"] < 0.5


def test_consensus_candidate_serves_ensemble_mean(snapshot):
    snapshot.candidates = ["a", "b", "c", "consensus"]
    snapshot.p_correct["consensus"] = [[0.99] * 3] * 4
    preds = {"a": fake(1.0), "b": fake(1.2), "c": fake(0.8)}
    res = score([WATER], preds, snapshot)[0]
    assert res["recommended_model"] == "consensus" and res["trust"] == pytest.approx(0.99)
    np.testing.assert_allclose(res["served_forces"], -np.asarray(WATER["R"]), atol=1e-12)


def test_subset_mean_candidate(snapshot):
    snapshot.candidates = ["a", "b", "c", "mean(a+b)"]
    snapshot.candidate_members = {"mean(a+b)": ["a", "b"]}
    snapshot.p_correct["mean(a+b)"] = [[0.98] * 3] * 4
    preds = {"a": fake(1.0), "b": fake(3.0), "c": fake(-5.0)}
    res = score([WATER], preds, snapshot)[0]
    assert res["recommended_model"] == "mean(a+b)" and res["served_members"] == ["a", "b"]
    np.testing.assert_allclose(res["served_forces"], -2.0 * np.asarray(WATER["R"]), atol=1e-12)


def test_bad_input_rejected(snapshot):
    with pytest.raises(ValueError):
        score([{"Z": [1, 1], "R": [[0, 0, 0]]}], {m: fake(1.0) for m in "abc"}, snapshot)


def test_snapshot_roundtrip(tmp_path, snapshot):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps({"models": snapshot.models, "tau": snapshot.tau,
                                "sigma_edges": snapshot.sigma_edges, "fbar_edges": snapshot.fbar_edges,
                                "p_correct": snapshot.p_correct, "created": "2026-10-31T00:00:00Z"}))
    snap = TrustSnapshot.load(path)
    assert snap.tau == 0.3 and snap.meta["created"].startswith("2026")


def test_read_xyz(tmp_path):
    path = tmp_path / "w.xyz"
    path.write_text("3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n3\nwater2\nO 0 0 0\nH 0.9 0 0\nH -0.2 0.9 0\n")
    systems = read_xyz(path)
    assert len(systems) == 2 and systems[0]["Z"] == [8, 1, 1]


def test_http_endpoint(snapshot):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from trustswarm.service import create_app

    client = TestClient(create_app(snapshot, {m: fake(1.0) for m in "abc"}))
    assert client.get("/health").json()["ok"] is True
    r = client.post("/trust", json={"systems": [dict(WATER, name="water")]})
    assert r.status_code == 200
    body = r.json()["results"][0]
    assert body["name"] == "water" and body["recommended_model"] == "c"
    assert client.post("/trust", json={"systems": [{"Z": [1], "R": [[0, 0, 0], [1, 0, 0]]}]}).status_code == 422
