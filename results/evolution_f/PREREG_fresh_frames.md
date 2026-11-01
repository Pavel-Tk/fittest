# Fresh-frame check for runs E and F (pre-registered addendum)

Written and committed **before** any run was scored on these frames (Nov 1, about 04:55 UTC).

**Why.** Every evolution run (A–F) is scored on the same eval split (1485 frames, 135 reactions). Eval was never
used inside any run. But the design of run F (force-only gate) came from a diagnosis that read earlier runs' eval
results, and run E's design came from exploratory eval results. New seeds do not remove that adaptivity to the
eval frames. This check scores the **same final swarms** on frames that no evolution run has read in any way.

**Fresh frames.** The leak-free split's `planted` + `skeptic_ref` frames: Transition1x **val** formulas, 2475 frames,
225 reactions (`results/split_report.json`). They are formula-disjoint from pool/eval, and the evolution engine has
never labelled, selected on or scored them. (Arm D of the static ablation used them for the Skeptic; no evolution run
did.) This is also an out-of-distribution test: the chemistry is new, not just the geometries.

**Procedure.** Re-run runs E-gpu, F-gpu, E-cpu and F-cpu with identical seeds, configs and code, plus an extra scorer
on the fresh frames. Children are content-addressed and cached, so the populations should be identical. Before
reading any fresh-frame number, check that the eval numbers reproduce the committed summaries, and report
any mismatch. The served answer per frame uses the same Calibrator ledger and routing as for eval.

**Decision rule** (same form as runs E/F, per run, on the fresh frames):
- **F holds on fresh frames** iff for F-gpu (12 seeds), the seed-level 95 % t-interval of mean Δ served force RMSE
  (evolved − static) lies entirely below 0 and at most 1 of 12 seeds is significantly worse.
- **E holds on fresh frames** iff the same holds for E-gpu no-selection − static.
- CPU versions are reported as secondary. If a rerun cannot reproduce the eval numbers, its fresh-frame result is
  reported as unreliable.
