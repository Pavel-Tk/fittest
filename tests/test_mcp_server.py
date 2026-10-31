"""MCP server: tool schemas, parsing/limits, verdict payloads, logging policy (fake models, no GPU)."""

from __future__ import annotations

import json
import socket
import threading
import time

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from fittest import mcp_server as ms  # noqa: E402
from fittest.service import TrustSnapshot, fake_predictors  # noqa: E402
from fittest.usage_log import CALL_COLUMNS, UsageLog  # noqa: E402

WATER_XYZ = "3\nsecret-name-water\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"
SECRET_COORD = "1.23456789"
SNAP = TrustSnapshot.load(ms.default_snapshot_path())


def make(tmp_path, models=None, mode="fake"):
    eng = ms.Engine(SNAP, mode)
    for m, p in fake_predictors(models or SNAP.models).items():
        eng.add_model(m, p)
    ulog = UsageLog(tmp_path / "usage.sqlite", salt=b"t")
    return eng, ulog, ms.build_server(eng, ulog)


def call(server, tool, args):
    async def go():
        async with create_connected_server_and_client_session(server._mcp_server) as c:
            return await c.call_tool(tool, args)
    return anyio.run(go)


def payload(res):
    assert not res.isError, res.content[0].text
    return json.loads(res.content[0].text)


@pytest.fixture(autouse=True)
def internal(monkeypatch):  # every test client is internal unless a test says otherwise
    monkeypatch.setenv("FITTEST_INTERNAL", "1")


# ------------------------------------------------------------------ parsing
def test_parse_formats():
    s = ms.parse_structures(WATER_XYZ + "\n" + WATER_XYZ)
    assert len(s) == 2 and s[0]["Z"] == [8, 1, 1] and s[0]["name"] == "secret-name-water"
    bare = ms.parse_structures("O 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0")
    assert bare[0]["Z"] == [8, 1, 1]
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
    ("", "empty"),
    ("3\nx\nO 0 0 0\nH 1 0 0\n", "truncated"),
    ("2\nx\nQq 0 0 0\nH 1 0 0\n", "unknown element"),
    ("2\nx\nO 0 0 a\nH 1 0 0\n", "numbers"),
    ("2\nx\nO 0 0 0\nH 0 0 0.1\n", "overlapping"),
    ("2\nx\nO 0 0 0\nH 0 0 5000\n", "1000"),
    ({"Z": [8, 1], "R": [[0, 0, 0]]}, "positions"),
    ({"R": [[0, 0, 0]]}, "give 'Z'"),
    (42, "must be XYZ"),
    ([], "at least one"),
])
def test_parse_errors(bad, msg):
    with pytest.raises(ms.InputError, match=msg):
        ms.parse_structures(bad)


def test_limits():
    with pytest.raises(ms.InputError, match=f"more than {ms.MAX_STRUCTURES}"):
        ms.parse_structures(WATER_XYZ * (ms.MAX_STRUCTURES + 1))
    big = {"Z": [6] * (ms.MAX_ATOMS + 1), "R": [[1.5 * i, 0, 0] for i in range(ms.MAX_ATOMS + 1)]}
    with pytest.raises(ms.InputError, match="limit"):
        ms.parse_structures(big)
    with pytest.raises(ms.InputError, match="limit"):
        ms.parse_structures(f"{ms.MAX_ATOMS + 1}\nx\n")


def test_domain_warning():
    assert ms.domain_warning([8, 1, 1], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]) is None
    w = ms.domain_warning([14, 1, 1, 1, 1], [[0, 0, 0], [1.5, 0, 0], [-1.5, 0, 0], [0, 1.5, 0], [0, -1.5, 0]])
    assert "Si" in w and "extrapolation" in w
    n = 9
    w = ms.domain_warning([6] * n, [[1.5 * i, 0, 0] for i in range(n)])
    assert "Transition1x" in w


# ------------------------------------------------------------------ tools
def test_tool_schemas_and_logging_disclosure(tmp_path):
    _, _, srv = make(tmp_path)

    async def go():
        async with create_connected_server_and_client_session(srv._mcp_server) as c:
            return (await c.list_tools()).tools
    tools = {t.name: t for t in anyio.run(go)}
    assert set(tools) == {"check_structures", "explain", "swarm_status", "usage_stats"}
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
    for k in ("verdict", "trust_score", "recommended", "served_energy_eV", "max_force_eV_per_A",
              "mean_force_eV_per_A", "sigma_F", "tau", "per_model", "domain_warning"):
        assert k in r
    assert r["verdict"] in ("trust", "send_to_dft") and set(r["per_model"]) == set(SNAP.models)
    assert "served_forces_eV_per_A" not in r
    assert out["summary"]["trust"] + out["summary"]["send_to_dft"] == 2
    full = payload(call(srv, "check_structures", {"structures": WATER_XYZ, "include_forces": True}))
    assert np.shape(full["results"][0]["served_forces_eV_per_A"]) == (3, 3)
    row = ulog.rows()[0]
    assert row["n_structures"] == 2 and row["n_atoms"] == 6 and row["n_trust"] + row["n_dft"] == 2
    assert row["ok"] == 1 and row["internal"] == 1


def test_errors_are_clear_and_logged(tmp_path):
    _, ulog, srv = make(tmp_path)
    res = call(srv, "check_structures", {"structures": WATER_XYZ * 21})
    assert res.isError and "more than 20" in res.content[0].text
    res = call(srv, "explain", {"structure": WATER_XYZ * 2})
    assert res.isError and "exactly one" in res.content[0].text
    assert [r["error"] for r in ulog.rows()] == ["input_error", "input_error"]


def test_explain_payload(tmp_path):
    _, _, srv = make(tmp_path)
    out = payload(call(srv, "explain", {"structure": {"symbols": "O H H",
                                                      "R": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]}}))
    assert out["verdict"] in ("trust", "send_to_dft") and out["tau"] == pytest.approx(SNAP.tau, abs=1e-4)
    assert out["most_divergent_member"] in SNAP.models
    assert len(out["pairwise_force_rms_eV_per_A"]) == 3
    assert set(out["p_correct_in_bucket"]) <= set(SNAP.candidates)
    assert out["bucket"]["sigma_F_range"].startswith("[")


def test_partial_committee(tmp_path):
    _, _, srv = make(tmp_path, models=["mace_mpa0", "aimnet2"], mode="local")
    out = payload(call(srv, "check_structures", {"structures": WATER_XYZ}))
    assert out["models_active"] == ["mace_mpa0", "aimnet2"] and not out["committee_complete"]
    assert "partial committee" in out["warning"]
    assert set(out["results"][0]["served_members"]) <= {"mace_mpa0", "aimnet2"}


def test_no_models_ready(tmp_path):
    eng = ms.Engine(SNAP, "local")
    srv = ms.build_server(eng, UsageLog(tmp_path / "u.sqlite", salt=b"t"))
    res = call(srv, "check_structures", {"structures": WATER_XYZ})
    assert res.isError and "no model agent is ready" in res.content[0].text


def test_swarm_status_and_family_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("FITTEST_FAMILY_TREE", str(tmp_path / "missing.json"))
    _, _, srv = make(tmp_path)
    st = payload(call(srv, "swarm_status", {}))
    assert [a["agent"] for a in st["population"]["answerers"]] == SNAP.models
    assert st["population"]["critics"][0]["agent"] == "skeptic"
    assert st["family_tree"]["available"] is False and st["committee_complete"]
    (tmp_path / "tree.json").write_text(json.dumps({"generations": [{"id": 1}], "best": "mace_lora_g3"}))
    monkeypatch.setenv("FITTEST_FAMILY_TREE", str(tmp_path / "tree.json"))
    st = payload(call(srv, "swarm_status", {}))
    assert st["family_tree"]["available"] and st["family_tree"]["tree"]["best"] == "mace_lora_g3"


def test_logging_stores_no_structure_data(tmp_path, monkeypatch):
    monkeypatch.delenv("FITTEST_INTERNAL")
    _, ulog, srv = make(tmp_path)
    xyz = f"3\nsecret-name-water\nO {SECRET_COORD} 0 0\nH 2.2 0 0\nH 1.0 0.93 0\n"
    payload(call(srv, "check_structures", {"structures": xyz}))
    payload(call(srv, "explain", {"structure": xyz}))
    call(srv, "check_structures", {"structures": "2\nx\nZzz 7.654321 0 0\nH 0 0 1\n"})  # error echoes input
    payload(call(srv, "usage_stats", {}))
    ulog._db.execute("PRAGMA wal_checkpoint(FULL)")
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("usage.sqlite*"))
    for needle in (b"secret-name", SECRET_COORD.encode(), b"7.654321", b"Zzz"):
        assert needle not in raw
    assert all(tuple(r) == CALL_COLUMNS for r in ulog.rows())
    st = ulog.stats()
    assert st["users"] == 1 and st["total_calls"] == 4 and st["calls_per_tool"]["check_structures"]["calls"] == 2


def test_internal_env_excluded_from_users(tmp_path):
    _, ulog, srv = make(tmp_path)
    payload(call(srv, "swarm_status", {}))
    st = ulog.stats()
    assert st["users"] == 0 and st["internal_calls"] == 1 and st["internal_client_ids"] == 1


def test_rate_limit(tmp_path):
    from fittest.usage_log import TokenBucket

    eng = ms.Engine(SNAP, "fake")
    for m, p in fake_predictors(SNAP.models).items():
        eng.add_model(m, p)
    ulog = UsageLog(tmp_path / "u.sqlite", salt=b"t")
    srv = ms.build_server(eng, ulog, limiter=TokenBucket(capacity=2, rate=0.0))

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
    from pathlib import Path

    p = Path(ms.__file__).parent / "snapshots" / "trust_snapshot.json"
    snap = TrustSnapshot.load(p)
    assert snap.models == SNAP.models and snap.tau > 0


# ------------------------------------------------------------------ HTTP transport end to end
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_transport_header_internal_and_stats(tmp_path, monkeypatch):
    import httpx
    import uvicorn
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    monkeypatch.delenv("FITTEST_INTERNAL")
    eng = ms.Engine(SNAP, "fake")
    for m, p in fake_predictors(SNAP.models).items():
        eng.add_model(m, p)
    ulog = UsageLog(tmp_path / "u.sqlite", salt=b"t")
    port = _free_port()
    srv = ms.build_server(eng, ulog, transport="http", port=port)
    server = uvicorn.Server(uvicorn.Config(srv.streamable_http_app(), host="127.0.0.1", port=port,
                                           log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        async def go(headers):
            async with streamablehttp_client(f"http://127.0.0.1:{port}/mcp", headers=headers) as (r, w, _):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    return await s.call_tool("check_structures", {"structures": WATER_XYZ})
        assert not anyio.run(go, {"X-Fittest-Internal": "1"}).isError
        st = httpx.get(f"http://127.0.0.1:{port}/stats").json()
        assert st["users"] == 0 and st["internal_calls"] == 1
        # same client + IP is now registered internal even without the header
        assert not anyio.run(go, {}).isError
        st = httpx.get(f"http://127.0.0.1:{port}/stats").json()
        assert st["users"] == 0 and st["internal_calls"] == 2
        page = httpx.get(f"http://127.0.0.1:{port}/")
        assert page.status_code == 200 and "claude mcp add" in page.text and "logged" in page.text
        assert httpx.get(f"http://127.0.0.1:{port}/health").json()["ok"]
    finally:
        server.should_exit = True
        th.join(timeout=10)
