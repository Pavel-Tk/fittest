# Fittest MCP server

Ask the Fittest swarm, from any MCP client, whether an ML interatomic potential can be trusted
on your molecule or whether it needs DFT.

- **Hosted endpoint (streamable HTTP):** `https://fittest-mcp.westus2.cloudapp.azure.com/mcp`
- Landing page: <https://fittest-mcp.westus2.cloudapp.azure.com/> · public counts: [`/stats`](https://fittest-mcp.westus2.cloudapp.azure.com/stats) · [`/health`](https://fittest-mcp.westus2.cloudapp.azure.com/health)
- Code: `fittest/mcp_server.py`, `fittest/usage_log.py`, `deploy/`

## Connect

| Client | How |
|---|---|
| Claude Code | `claude mcp add --transport http fittest https://fittest-mcp.westus2.cloudapp.azure.com/mcp` |
| Claude Desktop / claude.ai | Settings → Connectors → Add custom connector → the URL above |
| Cursor | `~/.cursor/mcp.json`: `{"mcpServers": {"fittest": {"url": "https://fittest-mcp.westus2.cloudapp.azure.com/mcp"}}}` |
| Local, stdio | `claude mcp add fittest -- uvx --from git+https://github.com/Pavel-Tk/fittest fittest-mcp` |

Then ask, for example: *"Use fittest to check whether I can trust an ML potential on this water
molecule: O 0 0 0, H 0.96 0 0, H -0.24 0.93 0."*

### Local mode

```bash
uvx --from git+https://github.com/Pavel-Tk/fittest fittest-mcp --fake-models   # demo, no torch
uvx --from git+https://github.com/Pavel-Tk/fittest fittest-mcp                 # real models if installed
pip install "fittest[models] @ git+https://github.com/Pavel-Tk/fittest"        # MACE + AIMNet2 (nvalchemi-toolkit)
FITTEST_ORB_PYTHON=/path/to/orb-env/bin/python fittest-mcp                     # add Orb from its own env
```

Local mode runs over stdio on CPU (`--device cuda` if you have a GPU). It spawns one process per
installed model agent and says which are active. With only MACE + AIMNet2 (no Orb env) it serves a
**partial committee** and every response carries a warning: τ and the trust map were learned on the
full three-model committee, so the verdicts are approximate. With no models installed the tools
return a clear error that points to `--fake-models`, the `[models]` extra and the hosted server.
`--fake-models` gives CPU stand-in predictors whose verdicts mean nothing; use them for
integration only. Local mode logs only to `~/.fittest-mcp/usage.sqlite` and never reports anywhere.

## Tools

| Tool | Input | Returns |
|---|---|---|
| `check_structures` | `structures`: XYZ text (single or multi-frame; bare `El x y z` lines also work), one `{Z or symbols, R}` object, or a list mixing both (Å). `include_forces` (default false) | Per structure: `verdict` (`trust` / `send_to_dft`), `trust_score`, `recommended` model or blend (the Calibrator's choice) and its `served_members`, `served_energy_eV`, `max_force_eV_per_A` / `mean_force_eV_per_A`, `sigma_F` vs `tau`, `per_model` energy and max force, `domain_warning`. Full force arrays only with `include_forces`. |
| `explain` | one structure | σ_F vs τ, the Calibrator bucket (σ_F range, mean force range, oracle labels in it), P(correct) of every candidate in that bucket, each member's deviation from the committee mean, pairwise force RMS, the most divergent member, the three most uncertain atoms. |
| `swarm_status` | none | Population: answerers (MACE-MPA-0, AIMNet2, Orb-v3, with live status and reputation), critic (Skeptic), coordinators (Calibrator, Oracle, Curator); the ledger (reputations, oracle labels used, budget, global P(correct) per candidate); the calibration domain; and the evolution family tree when `results/evolution/family_tree.json` exists (otherwise `available: false`). |
| `usage_stats` | none | Distinct users, total / external / internal calls, calls per tool, first and last call time. |

**Limits:** at most 20 structures per call and 100 atoms per structure, non-periodic molecules,
coordinates within ±1000 Å, no atoms closer than 0.3 Å. A per-client token bucket allows a burst
of 20 calls, refilled at 1 call every 2 s, plus a per-IP bucket. Each request times out after
120 s, and model evaluations are serialized, so one user cannot wedge the box. Request bodies are
capped at 2 MB (Caddy and the server). Errors come back as MCP tool errors with a plain message.

**Domain.** The Calibrator learned from DFT oracle labels on Transition1x: gas-phase C/H/N/O
reactions with at most 7 heavy atoms, including transition states and NEB paths. A `domain_warning`
is added for other elements, larger molecules, isolated atoms or fragments, and unusually short
bonds. In those cases the trust score is an extrapolation.

## Logging policy

The tool descriptions, the server instructions and every `check_structures` response say this too:

- **Logged per call** (SQLite at `/home/azureuser/fittest-mcp-data/usage.sqlite` on the VM): UTC
  time, tool, transport, an anonymous client id, an internal flag, structure count, atom count,
  verdict counts, latency, ok/error, and an error *category* (`input_error`, `rate_limited`,
  `timeout` or the exception type, never the message, because messages can echo user input).
- **Client id** = `HMAC-SHA256(secret salt, client name | client version | remote IP)`, truncated
  to 16 hex characters. The salt is a random file on the VM (`fittest-mcp-data/salt`, mode 600,
  never committed), so ids cannot be reversed. **Raw IPs are never stored.** Caddy's access log
  drops the client IP and X-Forwarded-For fields.
- **Never stored:** structures, coordinates, element lists, structure names, results.
- **Users** = distinct client ids with at least one successful call, excluding internal test
  clients. A call is internal if it carries the header `X-Fittest-Internal: 1` (or, in local mode,
  the env var `FITTEST_INTERNAL=1`). The client id of such a call is also registered in the
  `internal_clients` table, so later calls from the same client and IP stay internal without the
  header. Every test we ran was marked internal.
- Caveats: two people with the same client version behind one NAT count as one user. One person
  on two networks or client versions counts as two.

## Deployment

The stack runs on an Azure VM (Ubuntu 24.04, 8 vCPU, 32 GB, CPU only):

- **Caddy 2.11** terminates TLS with Let's Encrypt for `fittest-mcp.westus2.cloudapp.azure.com`,
  redirects HTTP to HTTPS, and reverse-proxies to `127.0.0.1:8765` (flushes streaming responses
  immediately, 180 s response timeout).
- **`fittest-mcp.service`** runs the MCP server in its own torch-free venv (`~/envs/mcp`) from its
  own checkout `~/fittest-mcp-src`, which deploy resets to `origin/master`. The experiments'
  `~/fittest` is never touched.
- **`fittest-model@{mace_mpa0,aimnet2,orb}.service`** run warm CPU model agents on ports
  7300–7302 (`OMP_NUM_THREADS=2`, so the evolution experiment on the same VM isn't starved;
  Orb runs from `~/envs/orb`). The socket authkey is a random secret in `/etc/fittest-mcp/common.env`.
- Every unit has `Restart=always`. The MCP server reconnects to model agents that restart. While
  one is down it serves a partial committee with a warning, rather than failing.
- Deploy or update from the laptop with `bash deploy/deploy.sh`. It is idempotent; add
  `RESTART_MODELS=1` to also restart the model agents. Verify with `python deploy/verify_mcp.py`.

## Verification log (2026-10-31, all calls marked internal)

**(a) Public HTTPS endpoint, MCP Python client from the laptop** (`deploy/verify_mcp.py`): 9/9 pass.

| Check | Result |
|---|---|
| initialize | ok, protocol `2025-11-25`, server `fittest` |
| list_tools | `check_structures`, `explain`, `swarm_status`, `usage_stats` |
| check_structures, water | `trust`, trust 1.0, recommended `aimnet2`, σ_F = 0.378 (τ = 0.620), all 3 models live |
| check_structures, 12 held-out Transition1x frames (`results/service_demo.xyz`) vs the earlier GPU run (`results/service_demo.json`) | verdicts agree 12/12, trust scores identical, max \|Δσ_F\| = 0.0005 eV/Å (CPU vs GPU), 2.2 s for the batch |
| explain, water | reason + `most_divergent_member` returned |
| domain warning (SiH) | `elements Si are outside the calibration set (C/H/N/O)…` |
| limit (21 structures) | tool error `more than 20 structures in one call` |
| swarm_status | committee complete; family tree `available: false` (not published yet) |
| usage_stats | `users = 0`, all calls internal |

Resilience, live:

- Killed `fittest-model@orb` with SIGKILL. The next call was served by a flagged partial committee
  (`mace_mpa0`, `aimnet2`). About 40 s later systemd had restarted Orb, the server had reconnected,
  and the full committee was back.
- Killed `fittest-mcp` with SIGKILL. systemd restarted it, it reconnected to the warm model agents,
  and the next call ran in 0.9 s.
- 25 rapid `swarm_status` calls: 2 rate-limited (`retry in 1 s`).
- 6 concurrent clients × 12 Transition1x frames: all succeeded (serialized, 14.4 s wall time).

Privacy check: the first deploy left uvicorn's access log on, and it wrote client IPs into the
systemd journal. I found this during verification, turned the access log off (`access_log=False`),
redeployed and vacuumed the journal. Re-checked after another full verify run: the laptop's IP
appears 0 times in the `fittest-mcp` and `caddy` journals and 0 times in the Caddy access log.

**(b) Claude Code.** From a scratch directory I ran
`claude mcp add --transport http --scope local fittest https://fittest-mcp.westus2.cloudapp.azure.com/mcp --header "X-Fittest-Internal: 1"`.
`claude mcp list` showed it ✔ Connected. Then:
`claude -p "Use the fittest MCP tool check_structures to check whether I can trust an ML potential on this water molecule ..."`.
Claude answered: *"The verdict is **trust**, with a trust score of **1.0**, and Fittest recommends
**AIMNet2** for this water molecule. The force disagreement between the three models is sigma_F =
**0.378 eV/Å**, below the 0.620 eV/Å threshold, and there's no domain warning, so you don't need
DFT for this geometry."* `/stats` afterwards: `users = 0`, `internal_calls = 8`,
`internal_client_ids = 2`, so the call was counted as internal. The scratch config was then removed
(`claude mcp remove fittest --scope local`).

**(c) Local install.**

- Fresh uv cache: `uvx --from git+https://github.com/Pavel-Tk/fittest fittest-mcp` installed in
  68 s on Windows 11.
- `--fake-models` over stdio with the MCP Python client: all four tools ok, mode `fake` with the
  FAKE warning, log written only to the local data dir.
- Without models: clean error naming the missing models and the fixes.
- On Linux (VM core env) without an Orb env: detected and spawned MACE + AIMNet2, reported
  `orb: not installed`, served `mean(aimnet2+mace_mpa0)` with the partial-committee warning.

Test suite: `tests/test_mcp_server.py` and `tests/test_usage_log.py` cover tool schemas, parsing,
limits, the partial committee, the family tree, rate limiting, the HTTP transport with the
internal header, and that **no structure data reaches the database file** (raw bytes scanned for a
planted name and coordinates). Full suite: 123 passed, 2 skipped.

## Known limitations

- The MCP server evaluates one request at a time. Throughput is roughly one 12-frame batch per
  2 s on this CPU VM while the evolution experiment runs.
- Absolute energies are on each model's own reference (PBE for MACE/Orb, ωB97M for AIMNet2), so a
  blend's `served_energy_eV` is only meaningful for differences.
- On Windows, local mode with *real* models spawns model agents from background threads while
  stdio is reading. It was only exercised on Linux; the `--fake-models` path is verified on Windows.
