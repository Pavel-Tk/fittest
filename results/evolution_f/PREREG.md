# Run F: gate offspring on forces only (pre-registered)

Written and committed **before** any run F seed was run (Nov 1, about 00:25 UTC). Motivation: the exploratory
diagnosis (`../evolution/DIAGNOSIS.md`) found that in three-family GPU runs, evolved swarms with no AIMNet2
specialist in the final population were significantly worse than static in 7 of 8 final states, against 0 of 20
when one survived. AIMNet2 children are mostly rejected by the gate's reaction-relative-energy check on the small
select set: it passed 39 of 114 AIMNet2 children against 91 of 114 Orb children. On CPU, that check's sign agreed with
eval only 38 % of the time (r = −0.15), while the select-set force change predicted eval well (r = 0.87). The
hypothesis is that the rel-E check is noise that throws away the useful children. Run F removes it; nothing else
changes.

## Fixed design
- Exactly run C (`../evolution_c/PREREG.md`), code at the commit that adds this file, **except** that the gate's rel-E
  check is disabled via `--gate-tol 1e9`. The force check (child ≤ parent force RMSE on the niche's select
  labels) and the clone check are unchanged, base models can still be pruned, and run D's knobs are off.
- Budget 800, three families (AIMNet2 + Orb + MACE), Colab L4. Arms: static, evolved.
- **New seeds 24–35** (12 seeds). Seeds 0–23 are not part of run F.
- F-cpu (secondary): AIMNet2 specialists only, CPU VM, same seeds, if CPU time allows after run E.

## Decision rule (fixed now; same form as run E)
Per seed: Δ = evolved − static, served force RMSE on eval at the final generation. Seed-level aggregation with
`scripts/seed_level.py --arm evolved`. **The force-only gate is confirmed to fix the regression iff, with all 12
F-gpu seeds:**
1. the seed-level 95 % t-interval of mean Δ force RMSE lies entirely below 0, **and**
2. at most 1 of 12 seeds is significantly worse than static on force RMSE, **and**
3. the seed-level 95 % t-interval of mean Δ ECE does not lie entirely above 0.

If fewer than 12 seeds finish by 08:00 EST, the rule is applied to the finished seeds, and fewer than 8 counts as
inconclusive. Also reported: the original run C rule per seed and under every-seed aggregation, seed-level rel-E, and
how many AIMNet2 children survive the gate (from `family_tree.json`).

The flagship verdicts for runs A–D are unchanged by run F.
