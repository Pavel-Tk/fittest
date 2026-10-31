# Fittest MCP server

Ask the Fittest swarm, from any MCP client, whether an ML interatomic potential can be trusted
on your molecule or whether it needs DFT.

- **Hosted endpoint (streamable HTTP):** `https://fittest-mcp.westus2.cloudapp.azure.com/mcp`
- Landing page: <https://fittest-mcp.westus2.cloudapp.azure.com/> · public counts: [`/stats`](https://fittest-mcp.westus2.cloudapp.azure.com/stats) · [`/health`](https://fittest-mcp.westus2.cloudapp.azure.com/health) (HTTP 503 unless all three model agents are live)
- Code: `fittest/mcp_server.py`, `fittest/usage_log.py`, `deploy/`

## Connect

| Client | How |
|---|---|
| Claude Code | `claude mcp add --transport http fittest https://fittest-mcp.westus2.cloudapp.azure.com/mcp` |
| Claude Desktop / claude.ai | Settings → Connectors → Add custom connector → the URL above |
| Cursor | `~/.cursor/mcp.json`: `{"mcpServers": {"fittest": {"url": "https://fittest-mcp.westus2.cloudapp.azure.com/mcp"}}}` |
| Local demo, no models | `claude mcp add fittest-demo -- uvx --from git+https://github.com/Pavel-Tk/fittest fittest-mcp --fake-models` |
| Local, real models | see [Local mode](#local-mode) |

Then ask, for example: *"Use fittest to check whether I can trust an ML potential on this water
molecule: O 0 0 0, H 0.96 0 0, H -0.24 0.93 0"*. Atoms may be separated by newlines, commas or
semicolons.

### Local mode

A plain `uvx --from git+…` install has **no model agents**. It only serves `--fake-models`, which
are CPU stand-ins whose verdicts mean nothing, or it reports that no models are installed. For real
verdicts, install into an environment that has the models and point your client at that
environment's `fittest-mcp`. This install is documented but has **not been tested end to end**
(fresh environment through to a verdict); the hosted server is the tested path.

```bash
uv venv fittest-env --python 3.12
uv pip install --python fittest-env/bin/python "fittest[models] @ git+https://github.com/Pavel-Tk/fittest" \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
claude mcp add fittest-local -- $PWD/fittest-env/bin/fittest-mcp          # MACE + AIMNet2
claude mcp add fittest-local -e FITTEST_ORB_PYTHON=/path/to/orb-env/bin/python -- $PWD/fittest-env/bin/fittest-mcp
```

The `[models]` extra installs MACE-MPA-0 and AIMNet2 through the ALCHEMI Toolkit. Orb-v3 needs its
own environment because orb-models pins conflicting dependencies; set `FITTEST_ORB_PYTHON` to that
environment's Python. With only MACE + AIMNet2 the server says so and serves a **partial
committee**: every response warns that the Calibrator's buckets and ledger were built on the full
three-model committee, so verdicts are approximate. Local mode runs over stdio on CPU (pass
`--device cuda` if you have a GPU), spawns one model process per installed model, each with its own
random socket key, and logs only to `~/.fittest-mcp/usage.sqlite`. It never reports anywhere.

## How a verdict is made (one rule)

1. Parse each structure on its own. A structure that can't be parsed gets its own `error` entry and
   the rest of the batch is still scored.
2. If any model agent does not support an element (AIMNet2 is the limit: the full committee covers
   **H B C N O F Si P S Cl As Se Br I**), the verdict is `send_to_dft` with reason
   `unsupported_elements` and no trust score. A partial committee is never trusted for this reason.
   If an agent rejects one structure, only that structure is affected (`model_failed`); the agent's
   status does not change.
3. Otherwise all agents predict forces. σ_F is their disagreement: the mean over atoms of the
   per-atom RMS deviation from the committee mean.
4. The Calibrator picks the candidate to serve (one model or a subset mean) by its ledger for the
   (σ_F, mean |F|) bucket.
5. **`trust` only if** the forces do not all vanish (max |F| < 0.05 eV/Å for every model means
   symmetry or exact equilibrium, where disagreement carries no information), **σ_F < τ**, and
   **trust_score ≥ 0.5**. Otherwise `send_to_dft`, with every failing condition in
   `verdict_reasons`. The same rule feeds `check_structures`, `explain` and the usage counts.

**τ is fixed.** It is the 70th percentile of σ_F on the round-0 pool (0.620 eV/Å). The Calibrator's
threshold rule never moved it during the run ([results/RESULTS.md](../results/RESULTS.md)), so τ is a
percentile rule, not something the swarm learned.

**trust_score** is the Calibrator's Beta posterior mean of P(served answer correct) for the served
candidate in the structure's bucket. It comes from DFT oracle labels on Transition1x, with prior
Beta(2, 1). It is never exactly 1. It is reported with `trust_score_90ci` (the 90% credible
interval) and `labels_behind_score` (DFT labels in that bucket). A bucket with 2 labels gives a wide
interval; read the score with it. "Correct" means force RMSE ≤ 0.25 eV/Å and relative energy
error ≤ 0.25.

**Energies.** `served_energy_eV` is the served model's (or blend's) total energy on its own
reference. Values from different models are not comparable, so when the served model differs within
a batch the response carries an `energy_warning`. `reference_energy_eV` comes from one fixed model
(AIMNet2, which is closest to the DFT level) for every structure in the batch; compare that field
across structures.

## Tools

| Tool | Input | Returns |
|---|---|---|
| `check_structures` | `structures`: XYZ text (single or multi-frame; bare atoms on lines or separated by `,`/`;`), one `{Z or symbols, R}` object, or a list mixing both (Å). `include_forces` (default false) | Per structure: `verdict` + `verdict_reasons`; `trust_score`, `trust_score_90ci`, `labels_behind_score`; `recommended` model or blend and `served_members`; `served_energy_eV`, `reference_energy_eV` (+ model); max/mean force; `sigma_F` vs `tau`; per-model energy and max force; `domain_warning`. Or `{index, error}` for a structure that could not be parsed. |
| `explain` | one structure | The same verdict, scored by the same rule, with the reason in words. Also: σ_F vs τ; the bucket (σ_F range, mean-force range, label count); the posterior P(correct) with interval for every candidate in it; each member's deviation from the committee mean; pairwise force RMS; the most divergent member; the three most uncertain atoms. |
| `swarm_status` | none | Answerers (MACE-MPA-0, AIMNet2, Orb-v3: live status, reputation), the critic (Skeptic), coordinators; the ledger; the calibration domain, τ, the verdict rule and the elements the committee supports; the evolution family trees (`results/evolution/family_tree.json`, `results/evolution_c/`) once they exist. |
| `usage_stats` | none | Users (definition below), total / external / internal calls, calls per tool, first and last call. |

**Input rules.**
- Element symbols must be exact (`C`, `Cl`, or lowercase `cl`). PDB-style trailing digits are fine
  (`C1`, `H12`). Ambiguous labels such as `CA` or `HE1`, and words such as `Hello`, are errors, not
  guesses. Atomic numbers are also accepted.
- 2 to 100 atoms per structure; a single atom has zero force by symmetry and is rejected.
- At most 20 structures per call.
- Isolated molecules only: `Lattice=`/`pbc="T…"` in an extended-XYZ comment, or `pbc`/`cell` in an
  object, is rejected.
- Coordinates within ±1000 Å; no two atoms closer than 0.3 Å.

**Operational limits.**
- Per-client token bucket (burst 20, then 1 call every 2 s) plus a per-IP bucket.
- 120 s timeout per request. Model calls time out after 300 s, and the socket handshake is bounded.
- Evaluations are serialized.
- Request bodies are capped at 2 MB.

**Domain.** The calibration comes from Transition1x: gas-phase C/H/N/O reactions with at most 7
heavy atoms, including transition states and NEB paths. A `domain_warning` is added for elements
outside C/H/N/O that the committee still supports, larger molecules, isolated atoms or fragments,
and unusually short bonds; there the trust score is an extrapolation.

## Logging policy

The tool descriptions, the server instructions and every `check_structures` response say this too:

- **Logged per call:** UTC time, tool (or `unknown`), transport, an anonymous client id, an internal
  flag, structure count, atom count, verdict counts, per-structure failures, latency, ok/error, and an
  error *category*. The categories are `input_error`, `invalid_arguments`, `unknown_tool`,
  `not_ready`, `rate_limited`, `busy`, `timeout`, or an exception type, never its message. Every
  call is logged: unknown tools and schema-validation failures go through the same dispatcher.
  The log is SQLite at `/var/lib/fittest-mcp/usage.sqlite`, owned by the unprivileged
  `fittest-mcp` user.
- **Client id** = `HMAC-SHA256(secret salt, client name | remote IP)`, truncated to 16 hex
  characters. The client *version* is deliberately not part of it, so upgrading or forging a
  version string does not create new ids. The salt is a random file (`/var/lib/fittest-mcp/salt`,
  mode 600, never committed), so ids cannot be reversed. Hosted, the IP comes from Caddy, which
  overwrites X-Forwarded-For. Local HTTP mode ignores X-Forwarded-For.
- **Raw IPs are not stored by the app or written to the access log.** uvicorn's access log is off.
  Caddy's access log drops the remote IP, client IP, X-Forwarded-For, X-Real-IP, Forwarded,
  Cf-Connecting-Ip and True-Client-Ip. Since 2026-10-31 19:11 UTC Caddy's error and unmapped-host
  logs (systemd journal) drop them too, via a filter on Caddy's default logger, and requests for any
  other host name (e.g. the bare IP) get a 404 and are not logged (`deploy/Caddyfile`). Before that,
  140 Caddy entries in the journal contained client IPs (9 distinct IPs: 2 error entries from 502s
  during restarts, 138 bare-IP requests, mostly scanners); they stay there until journal rotation.
- **Never stored or logged:** structures, coordinates, element lists, structure names, results. The
  server's own log lines (systemd journal) carry only the tool name and an error category or
  exception type. The SDK's protocol logger, which can quote raw input, is silenced. A test scans
  the raw database bytes for a planted structure name and coordinates, and another checks that a
  failing call leaves no input in the log records.
- **Users** = distinct non-internal client ids that received **at least one structure verdict from
  `check_structures` or `explain`**. Calls to `swarm_status`, `usage_stats`, and calls whose
  structures all failed to parse do not count. Internal = header `X-Fittest-Internal: 1` (local
  mode: env `FITTEST_INTERNAL=1`), after which that client id is also registered in
  `internal_clients`. Every test we ran was marked internal.
- **Caveats on the users count:**
  - People who share a public IP and client name merge into one user. That includes an office or
    venue Wi-Fi, and claude.ai / Claude Desktop custom connectors, which call from Anthropic's
    shared egress IPs, so all claude.ai users may count as one.
  - One person on several networks or clients counts several times.
  - Someone determined could still inflate the count by varying the client *name* from different
    IPs.

  Read it as a rough activity signal, not a count of people.

## Deployment

The stack runs on an Azure VM (Ubuntu 24.04, 8 vCPU, 32 GB, CPU only):

- **Caddy 2.11** terminates TLS with Let's Encrypt for `fittest-mcp.westus2.cloudapp.azure.com` and
  reverse-proxies to `127.0.0.1:8765`.
- **`fittest-mcp.service`**, the internet-facing process, runs as a dedicated system user
  `fittest-mcp` (no login, no sudo) with:
  - `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome=read-only`, `PrivateTmp`, `PrivateDevices`
  - no capabilities, restricted address families
  - write access only to `StateDirectory=/var/lib/fittest-mcp`

  It runs from its own torch-free venv and its own checkout (`~/fittest-mcp-src`, reset to
  `origin/master` on deploy). The experiments' `~/fittest` is never written; its `results/` dir is
  read for the family trees.
- **`fittest-model@{mace_mpa0,aimnet2,orb}.service`** are the warm CPU model agents on localhost
  ports 7300–7302 (`OMP_NUM_THREADS=2`). They run as the VM's admin user `azureuser` (not the
  `fittest-mcp` user), because the model weights are cached in `~azureuser/.cache`. They are
  sandboxed (`NoNewPrivileges`, so no sudo even as azureuser; `ProtectSystem=strict`,
  `ProtectHome=read-only`; writable only `~/.cache/{mace,aimnet,warp,cached_path}`, the weight and
  kernel caches the three models use) and use a random socket authkey from
  `/etc/fittest-mcp/common.env`. They are reachable only from localhost, not from the internet.
- The model-agent protocol pickles over a socket, so the authkey matters. There is no default key
  any more: `spawn()` gives each server its own random key.
- `Restart=always` on every unit. The MCP server keeps exactly one link per model agent: it closes
  the old link before reconnecting, bounds the handshake, and drops the link on a reply timeout.
  A model rejecting an input never changes its status. While an agent is down, calls are served by
  a flagged partial committee and `/health` returns 503.
- Deploy or update from the laptop with `bash deploy/deploy.sh`; it is idempotent. Add
  `RESTART_MODELS=1` to also restart the model agents. Verify with `python deploy/verify_mcp.py`.

## Verification log

### Round 2 (2026-10-31 afternoon): after the adversarial review (16 defects fixed)

Live run of `deploy/verify_mcp.py` against the hosted server after the redeploy, every call marked
internal: **12/12 checks pass.**

- `serverInfo` reports version 0.1.0. The example prompt, used verbatim, gives water `trust`, trust
  score 0.857 (90% CI 0.61–0.99, 4 labels behind it), served by AIMNet2, σ_F = 0.378 eV/Å.
- On the 12 held-out Transition1x frames of `results/service_demo.xyz`, σ_F ≥ τ agrees 12/12 with
  the GPU run (`results/service_demo.json`), max |Δσ_F| = 0.0005 eV/Å, and the verdict rule holds.
  7 of the 12 are trusted and none of the trusted frames is DFT-wrong; the old τ-only
  rule trusted 2 DFT-wrong frames.
- Fe gets `send_to_dft` (unsupported element). A single atom, `Hello` and a periodic input each get
  their own error entry, and the water in the same batch is still scored. All agents stay active
  afterwards. An unknown tool gives a clean error. The per-call structure limit holds (checked with 21 structures).
- The reviewer's scripts t2a/b/c, re-run: every input gives a clean per-structure or call error. In
  a scan of 60 acetaldehyde frames, 0 verdicts contradict the trust score, and the highest score is
  0.667.
- Known issue outside the server: the reviewer's `t3_scan.py` crashes on the new per-structure error
  entries, which have no `sigma_F`.

### Round 1 (2026-10-31 morning), superseded

The first deploy passed its own checks: the 9-step client run, 12/12 parity with the GPU run, Claude
Code connected, `uvx` installed. An adversarial review then found 16 defects:

1. One model-rejected element left every model in a permanent `error` state.
2. The verdict could contradict the trust score.
3. Single atoms were trusted.
4. The users count could be inflated.
5. Element prefixes were guessed.
6. Errors for unsupported elements were misleading.
7. Energies were not comparable within a batch.
8. Periodic input was accepted silently.
9. The model-server authkey was fixed.
10. The internet-facing unit was not hardened.
11. The local install line was misleading.
12. The example prompt failed verbatim.
13. Some failures were not logged.
14. Element numbers reached the journal.
15. The family-tree link, the server version and the documented ports were wrong.
16. One bad structure sank the whole batch.

All were fixed with a regression test each (`tests/test_mcp_server.py::test_defect_*`). Round 1 also
found and fixed uvicorn's access log writing client IPs to the journal.
