# HANDOFF: evolutionary loop (Fittest). Written ~17:45 UTC Oct 31

Code (pushed): fittest/evolution.py, scripts/evolve.py, scripts/evolve_offspring.py (MACE LoRA child),
scripts/evolve_offspring_aimnet.py (AIMNet2 child), scripts/write_evolution_summary.py (tables + verdict.json),
scripts/make_evolution_figures.py (fig7/fig8), tests/test_evolution.py (8 pass), results/evolution/METHODS.md (methods + decision rule).

## Running on the VM (nohup, detached). Check with: ps -eo pid,etime,args | grep [e]volve
| run | clone dir | PID | log | out dir |
|---|---|---|---|---|
| Run A seed 1 (MACE offspring) | ~/fittest_evo | 12933 | ~/evo_runs/s1.log | ~/fittest_evo/results/evo_s1 |
| Run B seeds 0,1 + newest control (AIMNet2 offspring) | ~/fittest_evoB | 16927 | ~/evo_runs/B.log | ~/fittest_evoB/results/evolution_aimnet |
Child logs: <out>/jobs/<run>/<id>.log. Births/fitness: python3 ~/evo_runs/bbshow.py <out>/blackboard.jsonl
Run A s1: ~28 min per generation, at gen-2 breeding at 17:40 UTC, expect done ~18:40 UTC. Run B: ~3 min/generation, done ~18:00 UTC.

## Done
* Run A seed 0 (complete): ~/fittest_evo/results/evolution. 9 MACE children: 8 stillborn (hold-out rel-E worse than base
  MACE even at E:F 30:10 / 100:10); g1_c passed the gate but had fitness -2.4 and was pruned. Final population = base, so
  evolved == static. Flagship fails.
* Run B seed 0 + newest_s0 (complete): ~/fittest_evoB/results/evolution_aimnet. AIMNet2 child g1_a (head, lr 1e-4) passed
  the gate and replaced its parent aimnet2 (pruned): final pop mace_mpa0+orb+g1_a; later children stillborn on rel-E.
  Eval, final - static (same budget): F-RMSE -0.001 [-0.007,+0.005] n.s.; rel-E MAE -0.027 [-0.039,-0.014] *;
  wrong rate -0.061 [-0.103,-0.017] *. Final - gen0: F-RMSE -0.015 *, rel-E -0.029 *.
  Pre-registered rule (primary = served force RMSE, vs gen 0 AND vs static) => does NOT hold for run B s0 either; report
  the rel-E / wrong-rate gain honestly as a secondary result.

## Next steps
1. Wait for both PIDs to exit (`grep "done in" ~/evo_runs/*.log`; run A needs "[evolved_s1] done", run B "[newest_s1] done").
2. Merge run A seeds into one scorer: copy ~/fittest_evo/results/evo_s1/{jobs,preds,ckpt}/evolved_s1 into
   ~/fittest_evo/results/evolution/{jobs,preds,ckpt}/, delete results/evolution/blackboard.jsonl, then in ~/fittest_evo:
   `~/envs/core/bin/python scripts/evolve.py --frames data/frames.pkl --preds data/preds_base --out results/evolution --seeds 0,1 --threads 2 --parallel 3 --python ~/envs/core/bin/python`
   (finished children are reused via result.json, ~1 min).
3. For each out dir: `~/envs/core/bin/python scripts/write_evolution_summary.py --evo <out>` -> summary_tables.md, verdict.json
4. Figures: `python scripts/make_evolution_figures.py --evo <out> --seed 0` (writes results/fig/fig7_evolution, fig8_family_tree);
   run for run A, then for run B and rename (e.g. fig7b/fig8b).
5. Copy results into the repo (results/evolution/ = run A, results/evolution_aimnet/ = run B; skip preds/ and ckpt/), write
   results/evolution/SUMMARY.md: run A and run B side by side, run B labelled "pre-planned extension, run after run A's outcome
   was known"; verdict under the rule; run B rel-E secondary gain; newest-control comparison; seed spread; cite arm D 0.206.
   The run-B rel-E target uses the Oracle's free reactant reference (same contract as all arms), not extra labels.

## Known issues / caveats
* Harness has no Skeptic vetoes, so "badly wrong trusted" is ~40% (arm D: 0/96); same for static and evolved, so the comparison is fair.
* Toolkit acceptance gate tolerance 1.01 (E0-dominated energy/atom MAE ties at 1.0); not applicable to AIMNet2 (recorded n/a).
* Inherited-weight MACE children: PEFT strategy checkpoint skipped (no BaseSpec); merged model.pt saved and used.
* The VM ~/fittest clone has no GitHub push credentials. A `git pull --autostash` there left the VM agent's old local
  finetune.py edits in `git stash list` (stash@{0}: autostash) in ~/fittest (upstream already has the CPU guards).
