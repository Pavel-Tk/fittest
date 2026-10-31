# Handoff: fittest-mcp (2026-10-31, ~13:40 EST)

## State
- Live and healthy at https://fittest-mcp.westus2.cloudapp.azure.com/mcp (`/health` 200, all 3 model agents active).
- Commit bcce57c (fixes) is deployed. The commit that adds this file also updates `deploy/verify_mcp.py`
  (re-verification) and docs/MCP.md; those are not runtime code, so no redeploy is needed for them.
- Users = 0 before and after (all 171 calls so far are internal). The definition changed: a user now
  needs at least one structure verdict from check_structures/explain, and the client id is name + IP
  with no version.
- Tests: full suite 170 passed, 2 skipped (laptop, `.venv/Scripts/python -m pytest -q`).

## The 16 review defects: all fixed, deployed, with a regression test each (`tests/test_mcp_server.py::test_defect_NN_*`)
1. A model rejecting an input no longer changes its status. One link per agent, closed before reconnect;
   bounded handshake (15 s) and reply timeout (300 s); `/health` returns 503 unless all agents are active.
   The stuck state was cleared by restarting the agents.
2. One verdict rule: trust only if all agents support the elements, forces don't vanish, sigma_F < tau, and
   trust_score >= 0.5. The score is the Beta posterior (prior Beta(2,1)) with a 90% CI and
   `labels_behind_score`. tau is documented as the fixed round-0 70th percentile that never moved.
3. Single atoms are rejected; vanishing forces send the structure to DFT.
4. Client id = HMAC(name | IP), version ignored. Users require a verdict. X-Forwarded-For is trusted only
   behind Caddy. The claude.ai egress-IP caveat is documented.
5. Exact element symbols only (PDB trailing digits are fine; CA, HE1, Hello, Water are errors).
6. Per-model element support is checked up front (AIMNet2: H B C N O F Si P S Cl As Se Br I). Anything
   unsupported gets send_to_dft with no score.
7. New fields `reference_energy_eV` (one model for the whole batch) and `energy_warning`.
8. Periodic input (Lattice=/pbc, or pbc/cell in an object) is rejected per structure.
9. model_server has no fixed authkey: random per process, and a fresh one per spawn().
10. fittest-mcp runs as the dedicated `fittest-mcp` user with NoNewPrivileges, ProtectSystem=strict,
    ProtectHome=read-only, PrivateTmp and StateDirectory=/var/lib/fittest-mcp (systemd exposure 3.1).
    The model agents are sandboxed but still run as azureuser, because the model cache lives in
    ~/.cache. The usage DB and salt were migrated to /var/lib/fittest-mcp.
11. Docs say plainly that plain uvx is fake-models only. The command for a real-models env is
    documented, but it is NOT end-to-end tested (open item).
12. The example prompt works verbatim (comma- and semicolon-separated atoms).
13. Every call is logged through one dispatcher: unknown_tool, invalid_arguments, not_ready, busy,
    timeout, rate_limited.
14. No input or tracebacks reach the server logs; the SDK logger is silenced. Caddy now also drops
    X-Real-IP, Forwarded, Cf-Connecting-Ip and True-Client-Ip.
15. Links use /blob/master; serverInfo.version is the fittest version; docstring ports are 7300-7302.
    Family trees are read from results/evolution/ and results/evolution_c/ (none exist on the VM yet).
16. Per-structure errors; the rest of the batch is still scored.

## Open
- The real-models local install (`uv pip install "fittest[models] @ git+..."` into a fresh venv) has not
  been tested end to end.
- README "1.1 Hosted MCP server" / "1.2 Local" / "Privacy" sections still describe the old behaviour.
  They need updating from docs/MCP.md: users definition, client id without version, the verdict rule,
  the example output (trust score is now 0.857, not 1.0), local install wording, and the data path
  (/var/lib/fittest-mcp).
- The "Verification log, Round 2" section in docs/MCP.md is a stub; paste the results below into it.
- The reviewer's t3_scan.py crashes on the new per-structure error entries, which have no sigma_F.
  That is the script, not the server.

## Round-2 live verification (deploy/verify_mcp.py, all internal): 12/12 PASS
- serverInfo version 0.1.0. Example prompt verbatim: water trust 0.857 (90% CI 0.61-0.99, 4 labels),
  AIMNet2, sigma_F 0.378.
- 12 held-out T1x frames: sigma_F>=tau agrees 12/12 with the GPU run (max |d sigma_F| 0.0005). Verdict
  rule holds. 7/12 trusted, and 0 of the trusted frames are DFT-wrong (the old tau-only rule trusted 2
  DFT-wrong frames).
- Fe goes to send_to_dft (unsupported). Single atom, 'Hello' and periodic input each get their own error
  entry; the water in the same batch is still scored. All agents stay active afterwards. Unknown tool
  gives a clean error. The 21-structure limit holds.
- Reviewer scripts t2a/b/c re-run: all inputs give clean per-structure or call errors. In a scan of 60
  acetaldehyde frames, 0 verdicts contradict the score, and the max score is 0.667.

## Deploy from the VM itself (deploy.sh assumes the laptop)
```bash
cd ~/fittest-mcp-src && git fetch -q origin master && git reset -q --hard origin/master   # server checkout
mkdir -p ~/fittest-mcp-deploy && cp deploy/{Caddyfile,fittest-mcp.service,fittest-model@.service,fittest-mcp.target,remote_setup.sh} ~/fittest-mcp-deploy/
BRANCH=master RESTART_MODELS=0 bash ~/fittest-mcp-deploy/remote_setup.sh   # RESTART_MODELS=1 also restarts agents
```
(Edit code in your own checkout, push, then run the above. remote_setup.sh resets ~/fittest-mcp-src to
origin/master. It never touches ~/fittest, where the evolution runs.)
Quick restart: `sudo systemctl restart fittest-mcp`. Agents: `sudo systemctl restart fittest-model@{mace_mpa0,aimnet2,orb}`.
Logs: `sudo journalctl -u fittest-mcp -n 50`.

## Verify
```bash
curl -s https://fittest-mcp.westus2.cloudapp.azure.com/health     # ok:true, 200
curl -s https://fittest-mcp.westus2.cloudapp.azure.com/stats      # users = the pitch number
~/envs/mcp/bin/python ~/fittest-mcp-src/deploy/verify_mcp.py      # all PASS; calls marked internal
cd ~/fittest-mcp-src && ~/envs/mcp/bin/python -m pip show pytest >/dev/null 2>&1 || uv pip install --python ~/envs/mcp/bin/python pytest
~/envs/mcp/bin/python -m pytest -q tests/test_mcp_server.py tests/test_usage_log.py
```
