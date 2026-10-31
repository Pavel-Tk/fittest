# Run D: confirmation of "protect the base, soften the gate" (pre-registered)

Written and committed **before** any run D seed was run (Sat Oct 31, about 15:50 EST). Run D tests a design that
was chosen **after** looking at exploratory data. The pre-registered runs A, B and C failed the rule. Among the
exploratory variants, AIMNet2-only niche specialists with (i) base models never pruned and (ii) a gate that
tolerates up to 10 % worse reaction-relative energy than the parent held in 5 of 6 seeds (0–5) at budget 800 and
3 of 6 at budget 400 (`../evolution_c/expl_cpu_protect_gatetol/`). Because that design was picked from
those seeds, it needs new seeds to count as evidence. Run D supplies them.

## Fixed design
- Engine, data, splits, triage, oracle, niches, fitness, eval scorer: exactly run C (`../evolution_c/PREREG.md`),
  code at the commit that adds this file.
- Changes vs run C, and nothing else: `--protect-base` (base models MACE-MPA-0, AIMNet2 and Orb are never
  pruned) and `--gate-tol 0.1` (the child passes the rel-E gate check if its select-label MAE is ≤ 1.10 × its
  parent's; the force check is unchanged).
- Budget **800** (4 generations × 200 labels, 3 breeding generations). Arms: static, evolved.
- **New seeds 6, 7, 8, 9, 10, 11.** Seeds 0–5 are not part of run D.
- Two implementations, reported separately:
  - **D-cpu** (primary): AIMNet2 specialists only (`--families aimnet2`), CPU VM. This is the setting the design was
    selected in.
  - **D-gpu** (secondary): AIMNet2 + Orb + MACE specialists, Colab L4. Reported regardless of outcome; if it does not
    finish by 19:00 EST, it is reported as not finished, with whatever seeds completed.

## Decision rule (fixed now)
Per seed, the run-C rule (`verdict_for`): final-generation served force RMSE significantly below both static and
own generation 0 (paired 95 % reaction-bootstrap CI), offspring in the final population, no significant
regression vs static in rel-E MAE or ECE.

**The design is confirmed** iff, in D-cpu, the rule holds in **at least 5 of 6** new seeds **and no seed is
significantly worse than static** on served force RMSE. This deliberately relaxes the original
"every seed" rule. That relaxation is decided here, before the data, and it applies only to this confirmation. The
original flagship verdict (runs A, B and C under "every seed") is unchanged by run D whatever the outcome.

Also reported: the same rule under "every seed", per-seed deltas, and D-gpu.
