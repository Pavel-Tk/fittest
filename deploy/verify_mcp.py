"""Call every Fittest MCP tool through an endpoint with the MCP Python client (marked internal).

    python deploy/verify_mcp.py                                   # public endpoint
    python deploy/verify_mcp.py --url http://127.0.0.1:8765/mcp

Also re-scores the held-out Transition1x frames of ``results/service_demo.xyz`` and compares them
with ``results/service_demo.json`` (the GPU run, which knows the true DFT error of each frame).
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
sys.path.insert(0, str(ROOT))
WATER = "3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n"
PROMPT_COORDS = "O 0 0 0, H 0.96 0 0, H -0.24 0.93 0"  # the documented example prompt, verbatim
FE = "Fe 0 0 0\nC 1.8 0 0\nO 2.95 0 0"
PERIODIC = '3\nLattice="3 0 0 0 3 0 0 0 3" pbc="T T T"\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n'


def text(res) -> str:
    return res.content[0].text if res.content else ""


async def run(url: str) -> dict:
    from fittest import __version__

    report: dict = {"url": url, "checks": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    async with streamablehttp_client(url, headers={"X-Fittest-Internal": "1"}, timeout=60,
                                     sse_read_timeout=300) as (r, w, _):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            check("initialize", init.serverInfo.name == "fittest" and init.serverInfo.version == __version__,
                  f"protocol {init.protocolVersion}, server version {init.serverInfo.version}")
            tools = {t.name for t in (await s.list_tools()).tools}
            check("list_tools", tools == {"check_structures", "explain", "swarm_status", "usage_stats"},
                  str(sorted(tools)))

            t0 = time.perf_counter()
            res = await s.call_tool("check_structures", {"structures": PROMPT_COORDS})
            dt = time.perf_counter() - t0
            out = json.loads(text(res)) if not res.isError else {}
            r0 = (out.get("results") or [{}])[0]
            tau = r0.get("tau", 0.62)
            check("check_structures, example prompt verbatim (comma-separated water)",
                  not res.isError and r0.get("verdict") in ("trust", "send_to_dft") and r0.get("trust_score", 1) < 1,
                  f"verdict={r0.get('verdict')} trust={r0.get('trust_score')} ci={r0.get('trust_score_90ci')} "
                  f"labels={r0.get('labels_behind_score')} rec={r0.get('recommended')} sigma_F={r0.get('sigma_F')} "
                  f"models={out.get('models_active')} {dt:.2f}s")

            demo_xyz, demo_ref = ROOT / "results" / "service_demo.xyz", ROOT / "results" / "service_demo.json"
            if demo_xyz.exists() and demo_ref.exists():
                ref = json.loads(demo_ref.read_text())
                t0 = time.perf_counter()
                res = await s.call_tool("check_structures", {"structures": demo_xyz.read_text()})
                dt = time.perf_counter() - t0
                rows = json.loads(text(res))["results"] if not res.isError else []
                flag_agree = sum((row["sigma_F"] >= tau) == rf["flagged"] for row, rf in zip(rows, ref))
                rule_ok = all((row["verdict"] == "trust") == (row["sigma_F"] < tau and row["trust_score"] >= 0.5)
                              for row in rows)
                dsig = max((abs(row["sigma_F"] - rf["sigma_F"]) for row, rf in zip(rows, ref)), default=None)
                report["t1x_parity"] = [{"uid": rf["uid"], "gpu_sigma_F": rf["sigma_F"], "cpu_sigma_F": row["sigma_F"],
                                         "gpu_flagged": rf["flagged"], "cpu_verdict": row["verdict"],
                                         "cpu_trust_score": row["trust_score"], "labels": row["labels_behind_score"],
                                         "dft_wrong@0.25": rf["wrong@0.25"]} for row, rf in zip(rows, ref)]
                wrong_trusted = sum(row["verdict"] == "trust" and rf["wrong@0.25"] for row, rf in zip(rows, ref))
                wrong_old = sum((not rf["flagged"]) and rf["wrong@0.25"] for rf in ref)
                n_trust = sum(row["verdict"] == "trust" for row in rows)
                check("check_structures, 12 held-out T1x frames vs GPU run",
                      len(rows) == len(ref) and flag_agree == len(ref) and rule_ok,
                      f"sigma_F>=tau agreement {flag_agree}/{len(ref)}, max |d sigma_F| = "
                      f"{dsig if dsig is None else round(dsig, 4)}, verdict rule holds={rule_ok}, trusted {n_trust}/12, "
                      f"trusted-but-DFT-wrong {wrong_trusted} (old tau-only rule: {wrong_old}), {dt:.2f}s")

            res = await s.call_tool("explain", {"structure": {"symbols": ["O", "H", "H"],
                                                              "R": [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]}})
            ex = json.loads(text(res)) if not res.isError else {}
            check("explain (same score as check_structures)", not res.isError and "reason" in ex
                  and ex.get("trust_score") == r0.get("trust_score"),
                  f"{ex.get('reason', text(res))[:170]}")

            res = await s.call_tool("check_structures", {"structures": "2\nx\nSi 0 0 0\nH 0 0 1.48\n"})
            dw = json.loads(text(res))["results"][0].get("domain_warning") if not res.isError else None
            check("domain warning (SiH)", bool(dw), (dw or text(res))[:100])

            res = await s.call_tool("check_structures", {"structures": [FE, "H 0 0 0", "Hello 0 0 0\nH 1 0 0",
                                                                        PERIODIC, WATER]})
            rr = json.loads(text(res))["results"] if not res.isError else []
            ok = (len(rr) == 5 and rr[0].get("verdict_reasons") == ["unsupported_elements"]
                  and rr[0].get("trust_score") is None and all("error" in x for x in rr[1:4])
                  and rr[4].get("verdict") in ("trust", "send_to_dft"))
            check("per-structure handling: Fe unsupported | single atom | 'Hello' | periodic | water", ok,
                  " | ".join((x.get("detail") or x.get("error") or x.get("verdict") or "")[:40] for x in rr))

            res2 = await s.call_tool("check_structures", {"structures": WATER})
            committee = json.loads(text(res2)).get("committee_complete") if not res2.isError else None
            check("all model agents still active after unsupported/rejected inputs", committee is True,
                  f"committee_complete={committee}")

            res = await s.call_tool("check_structures", {"structures": WATER * 21})
            check("limit error (21 structures)", res.isError and "more than 20" in text(res), text(res)[:80])

            res = await s.call_tool("nonexistent_tool", {})
            check("unknown tool -> clean error", res.isError and "unknown tool" in text(res), text(res)[:80])

            res = await s.call_tool("swarm_status", {})
            st = json.loads(text(res)) if not res.isError else {}
            check("swarm_status", not res.isError and st.get("committee_complete")
                  and "never moved" in st.get("calibration", {}).get("tau_meaning", ""),
                  f"active={st.get('active_models')} family_tree={st.get('family_tree', {}).get('available')} "
                  f"committee_elements={''.join(st.get('calibration', {}).get('committee_elements', []))}")

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
