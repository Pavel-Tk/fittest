"""Call every Fittest MCP tool through an endpoint with the MCP Python client (marked internal).

    python deploy/verify_mcp.py                                   # public endpoint
    python deploy/verify_mcp.py --url http://127.0.0.1:8765/mcp

Also cross-checks the hosted verdicts on the held-out Transition1x frames of
``results/service_demo.xyz`` against ``results/service_demo.json`` (the GPU run).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = Path(__file__).resolve().parents[1]
WATER = "3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"


def text(res) -> str:
    return res.content[0].text if res.content else ""


async def run(url: str) -> dict:
    report: dict = {"url": url, "checks": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    async with streamablehttp_client(url, headers={"X-Fittest-Internal": "1"}, timeout=60,
                                     sse_read_timeout=300) as (r, w, _):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            check("initialize", init.serverInfo.name == "fittest", f"protocol {init.protocolVersion}")
            tools = {t.name for t in (await s.list_tools()).tools}
            check("list_tools", tools == {"check_structures", "explain", "swarm_status", "usage_stats"},
                  str(sorted(tools)))

            t0 = time.perf_counter()
            res = await s.call_tool("check_structures", {"structures": WATER})
            dt = time.perf_counter() - t0
            out = json.loads(text(res)) if not res.isError else {}
            r0 = (out.get("results") or [{}])[0]
            check("check_structures water", not res.isError and r0.get("verdict") in ("trust", "send_to_dft"),
                  f"verdict={r0.get('verdict')} trust={r0.get('trust_score')} rec={r0.get('recommended')} "
                  f"sigma_F={r0.get('sigma_F')} models={out.get('models_active')} {dt:.2f}s")

            demo_xyz = ROOT / "results" / "service_demo.xyz"
            demo_ref = ROOT / "results" / "service_demo.json"
            if demo_xyz.exists() and demo_ref.exists():
                ref = json.loads(demo_ref.read_text())
                t0 = time.perf_counter()
                res = await s.call_tool("check_structures", {"structures": demo_xyz.read_text()})
                dt = time.perf_counter() - t0
                rows = json.loads(text(res))["results"] if not res.isError else []
                agree = sum((row["verdict"] == "send_to_dft") == rf["flagged"] for row, rf in zip(rows, ref))
                dsig = max((abs(row["sigma_F"] - rf["sigma_F"]) for row, rf in zip(rows, ref)), default=None)
                report["t1x_parity"] = [{"uid": rf["uid"], "gpu_sigma_F": rf["sigma_F"], "cpu_sigma_F": row["sigma_F"],
                                         "gpu_flagged": rf["flagged"], "cpu_verdict": row["verdict"],
                                         "gpu_trust": rf["trust"], "cpu_trust": row["trust_score"],
                                         "dft_wrong@0.25": rf["wrong@0.25"]} for row, rf in zip(rows, ref)]
                check("check_structures T1x batch vs GPU run", len(rows) == len(ref) and agree == len(ref),
                      f"{len(rows)} frames, verdict agreement {agree}/{len(ref)}, max |d sigma_F| = "
                      f"{dsig if dsig is None else round(dsig, 4)}, {dt:.2f}s")

            res = await s.call_tool("explain", {"structure": {"symbols": ["O", "H", "H"],
                                                              "R": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]}})
            ex = json.loads(text(res)) if not res.isError else {}
            check("explain", not res.isError and "reason" in ex,
                  f"{ex.get('reason', text(res))[:160]} most_divergent={ex.get('most_divergent_member')}")

            res = await s.call_tool("check_structures", {"structures": "2\nx\nSi 0 0 0\nH 0 0 1.48\n"})
            dw = json.loads(text(res))["results"][0]["domain_warning"] if not res.isError else None
            check("domain warning (Si)", bool(dw), (dw or text(res))[:120])

            res = await s.call_tool("check_structures", {"structures": WATER * 21})
            check("limit error (21 structures)", res.isError and "more than 20" in text(res), text(res)[:100])

            res = await s.call_tool("swarm_status", {})
            st = json.loads(text(res)) if not res.isError else {}
            check("swarm_status", not res.isError and st.get("committee_complete"),
                  f"active={st.get('active_models')} family_tree={st.get('family_tree', {}).get('available')}")

            res = await s.call_tool("usage_stats", {})
            us = json.loads(text(res)) if not res.isError else {}
            check("usage_stats", not res.isError and "users" in us,
                  f"users={us.get('users')} internal_calls={us.get('internal_calls')} total={us.get('total_calls')}")
            report["usage_stats"] = us
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://fittest-mcp.westus2.cloudapp.azure.com/mcp")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = anyio.run(run, a.url)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    sys.exit(0 if all(c["ok"] for c in rep["checks"]) else 1)


if __name__ == "__main__":
    main()
