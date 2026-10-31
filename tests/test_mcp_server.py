"""MCP server: tool schemas, parsing/limits, verdict payloads, logging policy (fake models, no GPU).

``test_defect_NN_*`` are regressions for the adversarial review of 2026-10-31 (16 defects).
"""

from __future__ import annotations

import json
import logging
import re
import socket
import threading
import time
from pathlib import Path

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")

import mcp.types as mt  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from fittest import __version__  # noqa: E402
from fittest import mcp_server as ms  # noqa: E402
from fittest import model_server  # noqa: E402
from fittest.service import TrustSnapshot, fake_predictors  # noqa: E402
from fittest.usage_log import CALL_COLUMNS, UsageLog  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WATER_XYZ = "3\nsecret-name-water\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"
WATER = "O 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0"
SECRET_COORD = "1.23456789"
SNAP = TrustSnapshot.load(ms.default_snapshot_path())


def make(tmp_path, models=None, mode="fake", preds=None, snap=SNAP, **kw):
    eng = ms.Engine(snap, mode)
    for m, p in (preds or fake_predictors(models or snap.models)).items():
        eng.add_model(m, p)
    ulog = UsageLog(tmp_path / "usage.sqlite", salt=b"t")
    return eng, ulog, ms.build_server(eng, ulog, **kw)


def call(server, tool, args, client_info=None):
    async def go():
        async with create_connected_server_and_client_session(server._mcp_server, client_info=client_info) as c:
            return await c.call_tool(tool, args)
    return anyio.run(go)


def payload(res):
    assert not res.isError, res.content[0].text
    return json.loads(res.content[0].text)


def noisy(level: float, seed: int = 0):
    """Predictors whose disagreement grows with `level` (model k gets k*level noise)."""
    def make_one(k):
        def predict(Z, R, batch):
            rng = np.random.default_rng(seed + k)
            R = np.asarray(R, float)
            c = np.zeros_like(R)
            for s in np.unique(batch):
                c[batch == s] = R[batch == s].mean(0)
            return {"energy": -100.0 * (k + 1) * np.ones(int(batch.max()) + 1),
                    "forces": -(R - c) + level * k * rng.standard_normal(R.shape)}
        return predict
    return {m: make_one(k) for k, m in enumerate(SNAP.models)}


@pytest.fixture(autouse=True)
def internal(monkeypatch):  # every test client is internal unless a test says otherwise
    monkeypatch.setenv("FITTEST_INTERNAL", "1")


# ------------------------------------------------------------------ parsing
def test_parse_formats():
    s = ms.parse_structures(WATER_XYZ + "\n" + WATER_XYZ)
    assert len(s) == 2 and s[0]["Z"] == [8, 1, 1] and s[0]["name"] == "secret-name-water"
    assert ms.parse_structures(WATER)[0]["Z"] == [8, 1, 1]
    d = ms.parse_structures([{"symbols": ["O", "H", "H"], "R": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]},
                             {"Z": [6, 1, 1, 1, 1], "positions": [[0, 0, 0], [0.63, 0.63, 0.63],
                                                                  [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63],
                                                                  [0.63, -0.63, -0.63]]},
                             WATER_XYZ])
    assert [len(x["Z"]) for x in d] == [3, 5, 3]
    js = ms.parse_structures(json.dumps({"Z": [8, 1, 1], "R": [0, 0, 0, 0.96, 0, 0, -0.24, 0.93, 0]}))
    assert js[0]["Z"] == [8, 1, 1] and np.shape(js[0]["R"]) == (3, 3)
    assert ms.parse_structures("2\n\nCl 0 0 0\nh 0 0 1.3")[0]["Z"] == [17, 1]


@pytest.mark.parametrize("bad, msg", [
    ("3\nx\nO 0 0 0\nH 1 0 0\n", "truncated"),
    ("2\nx\nQq 0 0 0\nH 1 0 0\n", "not an element symbol"),
    ("2\nx\nO 0 0 a\nH 1 0 0\n", "numbers"),
    ("2\nx\nO 0 0 0\nH 0 0 0.1\n", "overlapping"),
    ("2\nx\nO 0 0 0\nH 0 0 5000\n", "1000"),
    ("O nan 0 0\nH 0.96 0 0", "finite"),
    ({"Z": [8, 1], "R": [[0, 0, 0]]}, "positions"),
    ({"R": [[0, 0, 0]]}, "give 'Z'"),
    ("-3\nx\nO 0 0 0", "atom count"),
])
def test_parse_errors_are_per_structure(bad, msg):
    out = ms.parse_structures(bad)
    assert len(out) == 1 and "error" in out[0] and msg in out[0]["error"], out


@pytest.mark.parametrize("bad, msg", [("", "empty"), ("  \n ", "empty"), (42, "must be XYZ"), ([], "at least one")])
def test_call_level_errors(bad, msg):
    with pytest.raises(ms.InputError, match=msg):
        ms.parse_structures(bad)

def test_deeply_nested_json_is_a_clean_error():
    # CPython < 3.12 raises RecursionError in json.loads (-> InputError); 3.12+ parses it and the
    # nested list becomes one per-structure error. Either way: a clean error, never a crash.
    bad = "[" * 5000 + "]" * 5000
    try:
        out = ms.parse_structures(bad)
    except ms.InputError as exc:
        assert "nested too deeply" in str(exc)
    else:
        assert len(out) == 1 and "error" in out[0]



def test_limits():
    with pytest.raises(ms.InputError, match=f"more than {ms.MAX_STRUCTURES}"):
        ms.parse_structures(WATER_XYZ * (ms.MAX_STRUCTURES + 1))
    with pytest.raises(ms.InputError, match=f"more than {ms.MAX_STRUCTURES}"):
        ms.parse_structures([WATER_XYZ] * (ms.MAX_STRUCTURES + 1))
    big = {"Z": [6] * (ms.MAX_ATOMS + 1), "R": [[1.5 * i, 0, 0] for i in range(ms.MAX_ATOMS + 1)]}
    assert "limit" in ms.parse_structures(big)[0]["error"]
    assert "limit" in ms.parse_structures(f"{ms.MAX_ATOMS + 1}\nx\n")[0]["error"]


def test_domain_warning():
    assert ms.domain_warning([8, 1, 1], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]) is None
    w = ms.domain_warning([14, 1, 1, 1, 1], [[0, 0, 0], [1.5, 0, 0], [-1.5, 0, 0], [0, 1.5, 0], [0, -1.5, 0]])
    assert "Si" in w and "extrapolation" in w
    assert "Transition1x" in ms.domain_warning([6] * 9, [[1.5 * i, 0, 0] for i in range(9)])


# ------------------------------------------------------------------ tools
def test_tool_schemas_and_logging_disclosure(tmp_path):
    _, _, srv = make(tmp_path)

    async def go():
        async with create_connected_server_and_client_session(srv._mcp_server) as c:
            return (await c.list_tools()).tools
    tools = {t.name: t for t in anyio.run(go)}
    assert set(tools) == set(ms.TOOL_NAMES)
    assert tools["check_structures"].inputSchema["required"] == ["structures"]
    assert "include_forces" in tools["check_structures"].inputSchema["properties"]
    assert tools["explain"].inputSchema["required"] == ["structure"]
    assert "ctx" not in json.dumps([t.inputSchema for t in tools.values()])
    for t in tools.values():
        assert "logged" in t.description and "NOT stored" in t.description


def test_check_structures_payload(tmp_path):
    _, ulog, srv = make(tmp_path)
    out = payload(call(srv, "check_structures", {"structures": WATER_XYZ + WATER_XYZ}))
    assert out["n_structures"] == 2 and out["mode"] == "fake" and "FAKE" in out["warning"]
    r = out["results"][0]
    for k in ("verdict", "verdict_reasons", "trust_score", "trust_score_90ci", "labels_behind_score", "recommended",
              "served_energy_eV", "reference_energy_eV", "reference_energy_model", "max_force_eV_per_A",
              "mean_force_eV_per_A", "sigma_F", "tau", "per_model", "domain_warning"):
        assert k in r
    assert r["verdict"] in ("trust", "send_to_dft") and set(r["per_model"]) == set(SNAP.models)
    assert "served_forces_eV_per_A" not in r
    full = payload(call(srv, "check_structures", {"structures": WATER_XYZ, "include_forces": True}))
    assert np.shape(full["results"][0]["served_forces_eV_per_A"]) == (3, 3)
    row = ulog.rows()[0]
    assert row["n_structures"] == 2 and row["n_atoms"] == 6 and row["n_trust"] + row["n_dft"] == 2
    assert row["ok"] == 1 and row["internal"] == 1


def test_explain_payload(tmp_path):
    _, _, srv = make(tmp_path)
    out = payload(call(srv, "explain", {"structure": {"symbols": "O H H",
                                                      "R": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]}}))
    assert out["verdict"] in ("trust", "send_to_dft") and out["tau"] == pytest.approx(SNAP.tau, abs=1e-4)
    assert out["most_divergent_member"] in SNAP.models
    assert len(out["pairwise_force_rms_eV_per_A"]) == 3
    assert set(out["p_correct_in_bucket"]) <= set(SNAP.candidates)
    res = call(srv, "explain", {"structure": WATER_XYZ * 2})
    assert res.isError and "exactly one" in res.content[0].text


def test_partial_committee_operational(tmp_path):
    _, _, srv = make(tmp_path, models=["mace_mpa0", "aimnet2"], mode="local")
    out = payload(call(srv, "check_structures", {"structures": WATER_XYZ}))
    assert out["models_active"] == ["mace_mpa0", "aimnet2"] and not out["committee_complete"]
    assert "partial committee" in out["warning"]
    assert set(out["results"][0]["served_members"]) <= {"mace_mpa0", "aimnet2"}


def test_no_models_ready_is_not_input_error(tmp_path):
    eng = ms.Engine(SNAP, "local")
    ulog = UsageLog(tmp_path / "u.sqlite", salt=b"t")
    srv = ms.build_server(eng, ulog)
    res = call(srv, "check_structures", {"structures": WATER_XYZ})
    assert res.isError and "fewer than 2 model agents" in res.content[0].text
    assert ulog.rows()[-1]["error"] == "not_ready"


def test_swarm_status(tmp_path, monkeypatch):
    monkeypatch.setenv("FITTEST_RESULTS_DIR", str(tmp_path / "results"))
    _, _, srv = make(tmp_path)
    st = payload(call(srv, "swarm_status", {}))
    assert [a["agent"] for a in st["population"]["answerers"]] == SNAP.models
    assert st["population"]["critics"][0]["agent"] == "skeptic"
    assert st["family_tree"]["available"] is False and st["committee_complete"]
    assert "70th percentile" in st["calibration"]["tau_meaning"]
    assert st["calibration"]["committee_elements"][:5] == ["H", "B", "C", "N", "O"]


def test_logging_stores_no_structure_data(tmp_path, monkeypatch):
    monkeypatch.delenv("FITTEST_INTERNAL")
    _, ulog, srv = make(tmp_path)
    xyz = f"3\nsecret-name-water\nO {SECRET_COORD} 0 0\nH 2.2 0 0\nH 1.0 0.93 0\n"
    payload(call(srv, "check_structures", {"structures": xyz}))
    payload(call(srv, "explain", {"structure": xyz}))
    payload(call(srv, "check_structures", {"structures": "2\nx\nZzz 7.654321 0 0\nH 0 0 1\n"}))  # error entry
    payload(call(srv, "usage_stats", {}))
    ulog._db.execute("PRAGMA wal_checkpoint(FULL)")
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("usage.sqlite*"))
    for needle in (b"secret-name", SECRET_COORD.encode(), b"7.654321", b"Zzz"):
        assert needle not in raw
    assert all(tuple(r) == CALL_COLUMNS for r in ulog.rows())
    st = ulog.stats()
    assert st["users"] == 1 and st["total_calls"] == 4 and st["calls_per_tool"]["check_structures"]["calls"] == 2
    assert ulog.rows()[2]["n_failed"] == 1


def test_internal_env_excluded_from_users(tmp_path):
    _, ulog, srv = make(tmp_path)
    payload(call(srv, "check_structures", {"structures": WATER}))
    st = ulog.stats()
    assert st["users"] == 0 and st["internal_calls"] == 1 and st["internal_client_ids"] == 1


def test_rate_limit(tmp_path):
    from fittest.usage_log import TokenBucket

    _, ulog, srv = make(tmp_path, limiter=TokenBucket(capacity=2, rate=0.0))

    async def go():
        async with create_connected_server_and_client_session(srv._mcp_server) as c:
            return [await c.call_tool("swarm_status", {}) for _ in range(3)]
    res = anyio.run(go)
    assert [r.isError for r in res] == [False, False, True] and "rate limit" in res[2].content[0].text
    assert ulog.rows()[-1]["error"] == "rate_limited"


def test_subset_snapshot_drops_unavailable_candidates():
    sub = ms.subset_snapshot(SNAP, ["aimnet2", "orb"])
    assert sub.models == ["aimnet2", "orb"]
    assert "consensus" not in sub.candidates and "mean(aimnet2+orb)" in sub.candidates
    assert ms.subset_snapshot(SNAP, SNAP.models) is SNAP


def test_packaged_snapshot_loads():
    snap = TrustSnapshot.load(Path(ms.__file__).parent / "snapshots" / "trust_snapshot.json")
    assert snap.models == SNAP.models and snap.tau > 0


# ------------------------------------------------------------------ defect regressions
class _DummyBackend(model_server.Backend):
    level_of_theory = "dummy"

    def __init__(self, k, reject_z=None):
        self.k, self.reject_z = k, reject_z

    def predict(self, Z, R, batch, charge):
        if self.reject_z is not None and self.reject_z in set(np.asarray(Z).tolist()):
            raise ValueError(f"unsupported element {self.reject_z}")
        n = int(batch.max()) + 1
        return np.zeros(n), -np.asarray(R) * (1 + 0.05 * self.k)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def dummy_servers(monkeypatch):
    """Three real model_server.serve() loops (one connection at a time) with dummy backends."""
    ports = {m: _free_port() for m in SNAP.models}
    backends = {m: _DummyBackend(k, reject_z=9 if m == "aimnet2" else None) for k, m in enumerate(SNAP.models)}
    monkeypatch.setattr(model_server, "make_backend", lambda model, device, model_path=None: backends[model])
    for m, p in ports.items():
        threading.Thread(target=model_server.serve, args=(m, p, "cpu"), daemon=True).start()
    time.sleep(0.5)
    return ports


def _wait(pred, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_defect_01_model_rejection_does_not_poison_status(tmp_path, dummy_servers):
    eng = ms.Engine(SNAP, "hosted")
    ms.start_hosted_models(eng, dummy_servers, period=0.2)
    assert _wait(lambda: eng.healthy), eng.model_status
    ulog = UsageLog(tmp_path / "u.sqlite", salt=b"t")
    srv = ms.build_server(eng, ulog)
    out = payload(call(srv, "check_structures", {"structures": ["F 0 0 0\nH 0 0 0.92", WATER]}))
    bad, good = out["results"]
    assert bad["verdict"] == "send_to_dft" and bad["verdict_reasons"] == ["model_failed"] and "aimnet2" in bad["detail"]
    assert good["verdict"] in ("trust", "send_to_dft") and good["trust_score"] is not None
    assert all(s == "active" for s in eng.model_status.values()), eng.model_status
    assert payload(call(srv, "check_structures", {"structures": WATER}))["committee_complete"]


def test_defect_01_link_loss_reconnects_without_hanging(tmp_path, dummy_servers):
    eng = ms.Engine(SNAP, "hosted")
    ms.start_hosted_models(eng, dummy_servers, period=0.2)
    assert _wait(lambda: eng.healthy)
    rm = next(c for c in eng.closers if c.model == "orb")
    rm._client._conn.close()  # simulate a dropped link
    systems = ms.parse_structures(WATER)
    out, snap = eng.evaluate(systems)  # orb sits out this call, others answer
    assert "orb" not in snap.models and out[0]["kind"] == "ok"
    assert _wait(lambda: eng.healthy, 10), eng.model_status  # reconnected (old link closed first)
    out, snap = eng.evaluate(systems)
    assert snap.models == SNAP.models


def test_defect_01_handshake_timeout(dummy_servers):
    port = dummy_servers["mace_mpa0"]
    a = model_server.ModelClient("mace_mpa0", port).connect(timeout=5)
    t0 = time.time()
    with pytest.raises(TimeoutError):
        model_server.ModelClient("mace_mpa0", port).connect(timeout=2)  # server busy with `a`
    assert time.time() - t0 < 6
    a.close(shutdown=False)
    b = model_server.ModelClient("mace_mpa0", port).connect(timeout=5)  # free again
    assert b.info()["model"] == "mace_mpa0"
    b.close(shutdown=False)


def test_defect_01_health_reflects_models(tmp_path):
    import httpx

    eng = ms.Engine(SNAP, "hosted")
    for m, p in fake_predictors(SNAP.models[:2]).items():
        eng.add_model(m, p)
    eng.model_status["orb"] = "reconnecting (EOFError)"
    srv = ms.build_server(eng, UsageLog(tmp_path / "u.sqlite", salt=b"t"))
    from starlette.testclient import TestClient

    with TestClient(srv.streamable_http_app()) as c:
        r = c.get("/health")
    assert r.status_code == 503 and r.json()["ok"] is False and r.json()["status"] == "degraded"
    assert httpx  # imported for parity with the HTTP test


def test_defect_02_verdict_and_score_follow_one_rule(tmp_path):
    for level in (0.0, 0.2, 0.4, 0.6, 0.9, 1.5):
        _, _, srv = make(tmp_path, preds=noisy(level))
        frames = "".join(f"3\nf{i}\nO 0 0 0\nH {0.9 + 0.05 * i} 0 0\nH -0.24 {0.9 + 0.04 * i} 0\n" for i in range(6))
        out = payload(call(srv, "check_structures", {"structures": frames}))
        for r in out["results"]:
            assert r["trust_score"] < 1.0 and r["labels_behind_score"] >= 0
            lo, hi = r["trust_score_90ci"]
            assert lo <= r["trust_score"] <= hi
            if r["verdict"] == "trust":
                assert r["sigma_F"] < r["tau"] and r["trust_score"] >= ms.TRUST_MIN_SCORE and not r["verdict_reasons"]
            else:
                assert r["verdict_reasons"]
            if r["trust_score"] < ms.TRUST_MIN_SCORE or r["sigma_F"] >= r["tau"]:
                assert r["verdict"] == "send_to_dft"
        ex = payload(call(srv, "explain", {"structure": "3\nf0\nO 0 0 0\nH 0.9 0 0\nH -0.24 0.9 0\n"}))
        assert ex["trust_score"] == out["results"][0]["trust_score"]
        assert ex["p_correct_in_bucket"][ex["recommended"]]["posterior_mean"] == pytest.approx(ex["trust_score"], abs=1e-4)
        assert ex["labels_behind_score"] == ex["bucket"]["oracle_labels_in_bucket"]


def test_defect_02_tau_described_as_fixed():
    for text in (ms._TOOL_DESCRIPTIONS["check_structures"], ms.TAU_MEANING,
                 (REPO / "docs" / "MCP.md").read_text(encoding="utf-8")):
        assert "70th percentile" in text and "never moved" in text
    assert "Calibrator learned tau" not in (REPO / "fittest" / "mcp_server.py").read_text(encoding="utf-8")


def test_defect_02_posterior_never_one():
    for c in SNAP.candidates:
        for i in range(len(SNAP.sigma_edges) + 1):
            for j in range(len(SNAP.fbar_edges) + 1):
                p = ms.posterior(SNAP, c, (i, j))
                assert 0 < p["mean"] < 1 and p["n_labels"] >= 0


def test_defect_03_single_atom_and_vanishing_forces(tmp_path):
    _, _, srv = make(tmp_path)
    for single in ("H 0 0 0", "C 0 0 0", {"Z": [6], "R": [[0, 0, 0]]}):
        r = payload(call(srv, "check_structures", {"structures": single}))["results"][0]
        assert "single atom" in r["error"] and "trust_score" not in r
    zero = {m: (lambda Z, R, b: {"energy": np.zeros(int(b.max()) + 1), "forces": np.zeros((len(Z), 3))})
            for m in SNAP.models}
    _, _, srv = make(tmp_path, preds=zero)
    r = payload(call(srv, "check_structures", {"structures": WATER}))["results"][0]
    assert r["verdict"] == "send_to_dft" and any("forces vanish" in x for x in r["verdict_reasons"])


def test_defect_04_client_id_ignores_version_and_users_need_verdicts(tmp_path, monkeypatch):
    monkeypatch.delenv("FITTEST_INTERNAL")
    ulog = UsageLog(tmp_path / "id.sqlite", salt=b"s")
    assert ulog.client_id("claude-code", "1.2.3.4") == ulog.client_id("Claude-Code ", "1.2.3.4")
    _, ulog, srv = make(tmp_path)
    for v in ("3.0.0", "3.0.1", "9.9.9"):
        payload(call(srv, "usage_stats", {}, client_info=mt.Implementation(name="claude-code", version=v)))
        payload(call(srv, "swarm_status", {}, client_info=mt.Implementation(name="claude-code", version=v)))
    assert ulog.stats()["users"] == 0  # status/stats calls alone are not usage
    for v in ("3.0.0", "3.0.1", "9.9.9"):
        payload(call(srv, "check_structures", {"structures": WATER},
                     client_info=mt.Implementation(name="claude-code", version=v)))
    assert ulog.stats()["users"] == 1  # version changes do not mint users
    payload(call(srv, "check_structures", {"structures": "Hello 0 0 0\nH 1 0 0"},
                 client_info=mt.Implementation(name="only-errors", version="1")))
    assert ulog.stats()["users"] == 1  # a client whose structures all failed is not a user
    assert "check_structures or explain" in ulog.stats()["users_definition"]


@pytest.mark.parametrize("label", ["Hello", "Water", "Nonsense", "CA", "HE1", "Xx", "О"])
def test_defect_05_exact_element_symbols(label):
    r = ms.parse_structures(f"{label} 0 0 0\nH 1.5 0 0")[0]
    assert "error" in r and "not an element symbol" in r["error"]


def test_defect_05_pdb_style_digits_ok():
    assert ms.parse_structures("C1 0 0 0\nH12 1.1 0 0\ncl 0 1.8 0")[0]["Z"] == [6, 1, 17]


@pytest.mark.parametrize("xyz, els", [("Og 0 0 0\nH 1.6 0 0", "Og"), ("No 0 0 0\nH 1.6 0 0", "No"),
                                      ("Fe 0 0 0\nC 1.8 0 0\nO 2.95 0 0", "Fe"), ("Na 0 0 0\nCl 2.36 0 0", "Na"),
                                      ("He 0 0 0\nH 1.6 0 0", "He")])
def test_defect_06_unsupported_elements_never_trusted(tmp_path, xyz, els):
    _, _, srv = make(tmp_path)
    out = payload(call(srv, "check_structures", {"structures": xyz}))
    r = out["results"][0]
    assert r["verdict"] == "send_to_dft" and r["verdict_reasons"] == ["unsupported_elements"]
    assert r["trust_score"] is None and els in r["detail"]
    ex = payload(call(srv, "explain", {"structure": xyz}))
    assert ex["verdict"] == "send_to_dft" and ex["trust_score"] is None


def test_defect_07_energies_comparable(tmp_path):
    p = [[0.9, 0.1, 0.1]] + [[0.1, 0.1, 0.1]] * 3
    q = [[0.1, 0.1, 0.1]] + [[0.9, 0.9, 0.9]] * 3
    snap = TrustSnapshot(models=["a", "b", "c"], tau=10.0, sigma_edges=[1e-6, 2e-6, 3e-6], fbar_edges=[1e-6, 2e-6],
                         p_correct={"a": p, "b": q, "c": [[0.0] * 3] * 4}, counts={c: [[10] * 3] * 4 for c in "abc"})

    def pred(k):
        def f(Z, R, b):
            R = np.asarray(R, float)
            out = np.zeros_like(R)
            for s in np.unique(b):
                Rs = R[b == s]
                stretch = max(0.0, float(np.abs(Rs).max()) - 1.2)
                out[b == s] = -Rs * (1 + k * stretch)
            return {"energy": np.full(int(b.max()) + 1, -10.0 * (k + 1)), "forces": out}
        return f
    _, _, srv = make(tmp_path, preds={m: pred(k) for k, m in enumerate("abc")}, snap=snap)
    out = payload(call(srv, "check_structures", {"structures": [WATER, "O 0 0 0\nH 2.5 0 0\nH -0.24 0.93 0"]}))
    a, b = out["results"]
    assert a["recommended"] != b["recommended"] and "energy_warning" in out
    assert a["reference_energy_model"] == b["reference_energy_model"]


def test_defect_08_periodic_rejected():
    xyz = ('3\nLattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
           "O 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n")
    assert "periodic" in ms.parse_structures(xyz)[0]["error"]
    d = {"symbols": ["O", "H", "H"], "R": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], "pbc": [True, True, True]}
    assert "periodic" in ms.parse_structures(d)[0]["error"]
    assert "error" not in ms.parse_structures({**d, "pbc": False})[0]


def test_defect_09_random_authkeys(monkeypatch, tmp_path):
    assert model_server.AUTHKEY != b"fittest-local"
    seen = []

    class FakePopen:
        def __init__(self, cmd, stdin=None, stdout=None, stderr=None, env=None):
            seen.append(env["FITTEST_AUTHKEY"])
            self.pid = 1
    monkeypatch.setattr(model_server.subprocess, "Popen", FakePopen)
    c1 = model_server.spawn("x", _free_port(), log_dir=str(tmp_path))
    c2 = model_server.spawn("x", _free_port(), log_dir=str(tmp_path))
    assert len(set(seen)) == 2 and "fittest-local" not in seen
    assert c1.authkey == seen[0].encode() and c2.authkey == seen[1].encode()


def test_defect_10_systemd_hardening():
    unit = (REPO / "deploy" / "fittest-mcp.service").read_text()
    for d in ("User=fittest-mcp", "NoNewPrivileges=yes", "ProtectSystem=strict", "ProtectHome=read-only",
              "PrivateTmp=yes", "StateDirectory=fittest-mcp"):
        assert d in unit
    model = (REPO / "deploy" / "fittest-model@.service").read_text()
    for d in ("NoNewPrivileges=yes", "ProtectSystem=strict", "PrivateTmp=yes"):
        assert d in model


def test_defect_11_local_install_docs_honest():
    plain = re.compile(r"uvx --from git\+https://github\.com/Pavel-Tk/fittest fittest-mcp(?! --fake-models)")
    for text in ((REPO / "docs" / "MCP.md").read_text(encoding="utf-8"), ms.landing_html("u", ms.Engine(SNAP, "x"))):
        for line in text.splitlines():
            if plain.search(line):
                assert "no model" in line.lower() or "fake" in line.lower(), line


def test_defect_12_example_prompt_parses(tmp_path):
    coords = ms.EXAMPLE_PROMPT.split(": ", 1)[1]
    assert ms.parse_structures(coords)[0]["Z"] == [8, 1, 1]
    assert ms.parse_structures(coords + ".")[0]["Z"] == [8, 1, 1]
    assert ms.parse_structures("O 0 0 0; H 0.96 0 0; H -0.24 0.93 0")[0]["Z"] == [8, 1, 1]
    _, _, srv = make(tmp_path)
    assert payload(call(srv, "check_structures", {"structures": coords}))["results"][0]["formula"] == "H2O"
    assert coords in (REPO / "docs" / "MCP.md").read_text(encoding="utf-8")


def test_defect_13_every_failure_is_logged(tmp_path):
    _, ulog, srv = make(tmp_path)
    assert call(srv, "nonexistent_tool", {}).isError
    assert call(srv, "check_structures", {}).isError  # missing argument (schema)
    assert call(srv, "check_structures", {"structures": WATER, "include_forces": "maybe"}).isError
    assert call(srv, "check_structures", {"structures": 123}).isError
    assert call(srv, "check_structures", {"structures": "[" * 5000 + "]" * 5000}).isError
    errs = [(r["tool"], r["error"]) for r in ulog.rows()]
    assert errs[:4] == [("unknown", "unknown_tool"), ("check_structures", "invalid_arguments"),
                        ("check_structures", "invalid_arguments"), ("check_structures", "invalid_arguments")]
    assert errs[4] in (("check_structures", "input_error"), ("check_structures", "invalid_arguments"))
    assert all(r["ok"] == 0 for r in ulog.rows())


def test_defect_14_no_user_data_in_server_logs(tmp_path, caplog):
    def boom(Z, R, b):
        raise OSError(f"secret Z={list(Z)}")
    preds = dict(fake_predictors(SNAP.models))
    preds["orb"] = boom
    _, _, srv = make(tmp_path, preds=preds)
    with caplog.at_level(logging.DEBUG):
        payload(call(srv, "check_structures", {"structures": "3\nsecret-name\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0"}))
        call(srv, "check_structures", {"structures": "Zq9 0 0 0"})
    assert "secret" not in caplog.text and "Zq9" not in caplog.text and "Traceback" not in caplog.text
    caddy = (REPO / "deploy" / "Caddyfile").read_text()
    for h in ("X-Forwarded-For", "X-Real-IP", "Forwarded", "remote_ip", "client_ip"):
        assert h in caddy


def test_defect_15_links_versions_ports(tmp_path, monkeypatch):
    r = tmp_path / "results"
    (r / "evolution").mkdir(parents=True)
    (r / "evolution" / "family_tree.json").write_text(json.dumps({"best": "g3"}))
    monkeypatch.setenv("FITTEST_RESULTS_DIR", str(r))
    ft = ms.family_tree_summary()
    assert ft["available"] and "/blob/master/results/evolution/family_tree.json" in ft["github"]
    assert "/main/" not in json.dumps(ft)
    assert "6300" not in ms.__doc__ and "7300" in ms.__doc__
    _, _, srv = make(tmp_path)

    async def go():
        async with create_connected_server_and_client_session(srv._mcp_server) as c:
            return c
    # serverInfo comes from the lowlevel server's version
    assert srv._mcp_server.version == __version__


def test_defect_15_both_evolution_trees(tmp_path, monkeypatch):
    r = tmp_path / "results"
    for d, best in (("evolution", "g3"), ("evolution_c", "c7")):
        (r / d).mkdir(parents=True)
        (r / d / "family_tree.json").write_text(json.dumps({"best": best}))
    monkeypatch.setenv("FITTEST_RESULTS_DIR", str(r))
    _, _, srv = make(tmp_path)
    ft = payload(call(srv, "swarm_status", {}))["family_tree"]
    assert [t["tree"]["best"] for t in ft["trees"]] == ["g3", "c7"]


def test_defect_16_one_bad_structure_does_not_sink_the_batch(tmp_path):
    _, ulog, srv = make(tmp_path)
    out = payload(call(srv, "check_structures", {"structures": [WATER, "Hello 0 0 0\nH 1 0 0", WATER_XYZ]}))
    good1, bad, good2 = out["results"]
    assert "error" in bad and good1["verdict"] and good2["verdict"]
    assert out["summary"]["errors"] == 1 and ulog.rows()[-1]["n_failed"] == 1


# ------------------------------------------------------------------ HTTP transport end to end
def _serve(srv, port, proxy_headers=False):
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(srv.streamable_http_app(), host="127.0.0.1", port=port,
                                           log_level="warning", proxy_headers=proxy_headers, access_log=False))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    _wait(lambda: server.started, 10)
    return server, th


def test_http_transport_header_internal_and_stats(tmp_path, monkeypatch):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    monkeypatch.delenv("FITTEST_INTERNAL")
    eng = ms.Engine(SNAP, "fake")
    for m, p in fake_predictors(SNAP.models).items():
        eng.add_model(m, p)
    ulog = UsageLog(tmp_path / "u.sqlite", salt=b"t")
    port = _free_port()
    srv = ms.build_server(eng, ulog, transport="http", port=port)
    server, th = _serve(srv, port)
    try:
        async def go(headers, version="1", name="t"):
            async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp", headers=headers) as (r, w, _):
                async with ClientSession(r, w, client_info=mt.Implementation(name=name, version=version)) as s:
                    init = await s.initialize()
                    assert init.serverInfo.version == __version__
                    return await s.call_tool("check_structures", {"structures": WATER_XYZ})
        assert not anyio.run(go, {"X-Fittest-Internal": "1"}).isError
        st = httpx.get(f"http://127.0.0.1:{port}/stats").json()
        assert st["users"] == 0 and st["internal_calls"] == 1
        assert not anyio.run(go, {}).isError  # same client + IP is registered internal
        assert httpx.get(f"http://127.0.0.1:{port}/stats").json()["users"] == 0
        # defect 04: forged X-Forwarded-For on a non-proxied server does not mint users
        for i in range(3):
            assert not anyio.run(go, {"X-Forwarded-For": f"10.9.0.{i}"}, f"2.{i}", "ext").isError
        assert httpx.get(f"http://127.0.0.1:{port}/stats").json()["users"] == 1
        page = httpx.get(f"http://127.0.0.1:{port}/")
        assert page.status_code == 200 and "claude mcp add" in page.text and "logged" in page.text
        assert httpx.get(f"http://127.0.0.1:{port}/health").json()["ok"]
    finally:
        server.should_exit = True
        th.join(timeout=10)
