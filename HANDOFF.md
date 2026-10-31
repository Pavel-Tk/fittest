# HANDOFF: resume Fittest on the Azure VM (written Sat Oct 31, ~13:35 EST)

Deadline from Pavel: **Sat Oct 31, 19:00 EST** for everything; slides due Sun Nov 1, 13:00.

## Where things live
- Repo: `/home/azureuser/fittest` (GitHub `Pavel-Tk/fittest`, branch `master`). Run `git pull` first.
- Python: core env `/home/azureuser/envs/core/bin/python` (torch CPU, nvalchemi 0.3.0-rc, MACE, AIMNet2);
  Orb env `/home/azureuser/envs/orb/bin/python`; MCP env `~/envs/mcp` (it needs mcp<2). Pass `--device cpu` everywhere.
  See `/home/azureuser/READY.json`.
- Laptop-only material, copied to `~/handoff/`:
  - `notes_memory.md`: all project decisions; read it first.
  - `trustswarm_loop.html`: the plan page source.
  - `deckgen/gen.py` + `deck/project/`: the new deck generator and slide files.
  - `qr_path.txt`: the QR code for the last slide.
  - `client/`: the MCP review scripts.
- Artifacts (claude.ai, private):
  - Deck: https://claude.ai/artifact/3mLXUXYYzuKXJC7ThVGyYJ (Slides type; still shows the OLD 14-slide TrustSwarm deck).
  - Plan page: https://claude.ai/artifact/A4KEE6FRZhue4py331h9E5

## Status by workstream
1. **Rename → Fittest**: done. No name collisions found; the old GitHub URL redirects.
2. **MCP server**: live at https://fittest-mcp.westus2.cloudapp.azure.com/mcp (landing `/`, `/stats`, `/health`).
   All 16 adversarial-review defects are fixed and deployed (170 tests pass). Details and the on-VM deploy/verify steps
   are in `docs/HANDOFF_mcp.md`. Users = 0 so far, because every call has been an internal test.
   The Discord post that announces it is the last section of `docs/discord_post.md` (pavel-voice, unedited).
   Pavel posts it himself.
3. **Evolution**: see `HANDOFF_evolution.md` (repo root on the VM) and `results/evolution/`.
   - Run A (MACE-only LoRA offspring): no improvement. 5/6 children failed the gate on reaction energy, and the
     survivor was pruned.
   - Run B (AIMNet2 offspring, head-only fine-tune with a reaction-relative energy loss): running on this VM under nohup.
     Early sign: 2 of 3 children pass the gate (hold-out F-RMSE 0.564 → ~0.43), and selection pruned the parent.
     Eval scoring and SUMMARY.md are still to do.
   - Run C (niche specialists at budgets 200/400/800): code and pre-registration are pushed. The first Colab L4 attempt is
     INVALID (CUDA OOM children were counted as stillborn); partial logs are in `results/evolution_c/partial_run1/`.
     Early gate signals: AIMNet2 specialists fail on rel-E, and Orb specialists pass but the evolved swarm is worse
     (0.236 vs static 0.202 at budget 200). Colab is stopped. To finish on this VM with AIMNet2-only specialists
     (~1–1.5 h), see `results/evolution_c/HANDOFF.md`; it needs the `families=("aimnet2",)` change.
   - **Decision rule (pre-registered with Pavel):** the evolved swarm must beat the static swarm at the same DFT budget,
     with a paired CI excluding 0, after ≥3 generations. Report runs A/B/C side by side and never hide run A.
     Pavel wants to KEEP the flagship "Survival of the fittest, in materials science" as the name and frame. Only the
     evidence slide's headline follows the result.
4. **README**: rewritten, with a TL;DR. The `TODO(evolution)` markers wait on `results/evolution/SUMMARY.md`.
   The MCP sections need refreshing from `docs/MCP.md` because the trust rule and user definition changed.
   Known inconsistencies to fix: `results/RESULTS.md` and the figures still say "TrustSwarm" (regenerate them with
   `scripts/write_results.py` / `make_figures.py`); `docs/DESIGN.md` §1/§5/§10 are stale; and
   `docs/data_and_models.md` wrongly says "train formulas only".
5. **Deck redesign** (`~/handoff/deckgen/gen.py`):
   - Look: Instrument Serif + Geist + Geist Mono; ink #0E1310, bone #F1ECE1, ember #F26A21.
   - Done: cover, 01 abundant, 02 silent, 03 disagree, 05 wins (static evidence), 06 yours (QR + "Already used by X
     users"; set X in `deckgen/cfg.json` as {"users": N} from `/stats`).
   - To do: 04 "Let them compete" (family tree from results), a run-B/C-based variant of 05 if evolution holds, and a
     backup "what didn't work" slide.
   - Then publish to the deck artifact: read it first, write the new slide files plus `project/deck.json` (new
     order/sections/faces), set old slide files to null, and publish with `root` = the folder holding `project/`.
   - Every number must trace to a results file.
6. **Process rules**: use subagents; run one adversarial review per deliverable after its tests pass; report failures
   honestly. Any text written as Pavel must come from his pavel-voice model, which runs only on his laptop, so don't
   write in his voice here.
7. **Cost**: this VM (Standard_D8as_v5, westus2, rg `rg-fittest-eus`) must stay up to host the MCP. After the
   evolution runs finish, consider downsizing it (that restarts the VM; the services auto-start).

## Evolution jobs on the VM (fallback notes, in case HANDOFF_evolution.md is missing)
- Working copy: `/home/azureuser/fittest_evo` (a separate checkout, so `git pull` there only after the jobs finish).
- Run A, seed 1: `scripts/evolve.py ... --out results/evo_s1 --seeds 1`. Run A, seed 0: `results/evolution/`.
- Run B: `scripts/evolve.py ... --out results/evolution_aimnet --family aimnet2 --seeds 0,1 --controls newest --control-seeds 0,1`.
- Check progress: `pgrep -af evolve`, then `ls -t ~/fittest_evo/results/*/`, and see `generations.json` and `family_tree.json`
  in each out dir.
- When they finish: copy the out dirs into `~/fittest/results/`, write `results/evolution/SUMMARY.md` (runs A/B/C side by
  side, with the verdict under the decision rule), make the figures, then commit and push.
- The original `~/fittest` had an unmerged `scripts/finetune.py`; it was reset to origin/master, and the old file is
  backed up at `~/handoff/finetune.py.vm_unmerged_backup`.

- Update 13:55: run B (seeds 0,1 + newest control for both) FINISHED, outputs in `~/fittest_evoB/results/evolution_aimnet` (not scored yet). Run A seed 1 (PID 12933) still running, ETA ~18:40 UTC.
