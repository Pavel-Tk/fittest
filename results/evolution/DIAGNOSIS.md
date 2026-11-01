# Why does the evolved swarm sometimes get worse than static on forces? (diagnosis)

> **EXPLORATORY. Not pre-registered, post hoc, and it does not change any verdict in `SUMMARY.md`.** This is
> analysis only: saved run outputs plus CPU re-scoring of cached predictions. No new training. The hypotheses were
> formed after seeing the results, so they need a fresh pre-registered test before anyone relies on them.

**Question.** In run C-style runs, the evolved swarm's served force RMSE is sometimes significantly *worse* than
static's. GPU, three families: C b200 s0 +0.034, b400 s0 +0.019, b800 s1 +0.011; C extension s2 +0.038, s4
+0.017; D-gpu s9 +0.015, s10 +0.005. CPU, AIMNet2 only: C′ extension s5 +0.009. No-selection is never
significantly worse.

## Main finding: the harm follows the *absence of an AIMNet2 specialist*, and the rel-E gate causes that absence
Take every three-family (GPU) final state: 28 evolved, 8 no-selection. Group them by whether an AIMNet2
specialist is in the final population (`../fig/fig10_diagnosis_aimnet2_specialist.png`):

| group (GPU, three families, budgets 200–800) | runs | significantly worse than static | significantly better |
|---|---|---|---|
| evolved, **no** AIMNet2 specialist in final population | 8 | **7** | 1 (ablation_protect s1) |
| evolved, AIMNet2 specialist in final population | 20 | 0 | 18 |
| no-selection (admits every child, so always has one) | 8 | 0 | 7 |

Every one of the 7 harmed GPU runs ends with Orb and/or MACE specialists only. In all 7, all three AIMNet2
children were rejected by the gate, every time on the rel-E check. Those children had passed the force check, with
force RMSE on the select labels 10–40 % below their parent's (e.g. D-gpu s9: 0.370 vs 0.548, 0.367 vs 0.506,
0.355 vs 0.482 eV/Å; `../evolution_d/gpu/b800_s9/family_tree.json`). Over all GPU evolved runs, the rel-E check
passed **39 of 114 AIMNet2 children** and 91 of 114 Orb children. The force check passed 113 of 114 AIMNet2
children. Base AIMNet2 already has the lowest rel-E error, so its children rarely beat it on that check. Base Orb
has a much higher rel-E error, so Orb children pass easily. The harmed runs then serve mostly
`mean(aimnet2 + Orb specialist)` where static serves `mean(aimnet2+orb)`: 39–100 % of eval frames, from `states[*].served`
in each `generations.json`. Most of the harm is on *trusted* (low-σ) frames. Served force RMSE Δ vs static by niche
(`served_by_niche`), trusted / flagged: b200 s0 +0.041 / +0.011; ext s2 +0.042 / +0.027; D-gpu s9 +0.021 / −0.006; D-gpu s10 +0.010 / −0.009.
No-selection includes the newest AIMNet2 child by design, which explains why it is rarely worse.

This is an association across 36 states from 14 configurations, not a controlled test. With the base models
protected (D-gpu s9, s10), base Orb is available and `mean(aimnet2+orb)` is still a candidate. The ledger
nevertheless routes 39–46 % of eval frames to `mean(aimnet2+g3_orb_flagged)`. So the remaining harm is either
the Orb specialist being worse than base Orb on trusted frames, or the ledger picking the wrong candidate.
GPU child predictions were not saved, so these two cannot be separated here (see (a) and (d)).

## Hypotheses

### (a) Calibrator routing: the served candidate is worse than what the population could serve. **Supported where testable (CPU); GPU: can't tell**
- *GPU:* the harmed runs route 39–100 % of eval frames to one candidate, `mean(aimnet2 + Orb specialist)`. Static
  routes 70–100 % to `mean(aimnet2+orb)`. The routing is not more concentrated than in winning runs. b800 s0, for
  example, routes 80 % to `mean(g3_aimnet2+g3_orb)` and wins. Concentration alone does not mark harm. Without GPU
  child predictions, we cannot score the alternatives the ledger rejected.
- *CPU replay* (exact: replayed served RMSE matches the saved value to 1e-6 in all 36 states; `diagnosis/cpu_diag.json`):
  in the one CPU harmed run, C′ ext s5 (base AIMNet2 pruned), the ledger serves `g1_aimnet2_flagged` **alone**
  on 830 of 1485 frames, for a served RMSE of 0.233. The same population with the best candidate per bucket chosen
  in hindsight on eval would serve 0.195. A ledger that picks per bucket the lowest *force RMSE* on the same clean
  labelled frames, instead of the highest P(correct), would serve 0.197. Static serves 0.224. In this run the
  population was good enough, and the routing, not member quality, caused the harm.
- The P(correct) ledger leaves force accuracy unused in general. Served minus force-objective routing, CPU states:
  static mean +0.006 (max +0.021), evolved +0.008 (max +0.036), no-selection +0.005. The ledger's target is binary
  correctness (F-RMSE ≤ 0.25 **and** |ΔE_rel| ≤ 0.25), not force RMSE. In 4 of 7 harmed GPU runs the eval
  wrong-rate does not rise and rel-E improves (e.g. ext s4: wrong-rate −0.074 \*, rel-E −0.043 \*, forces +0.017 \*).
  That is the trade this objective allows. In the other three (b200 s0, ext s2, b800 s1) the wrong-rate
  rises too (+0.061 \*, +0.084 \*, +0.024 n.s.), so this objective mismatch is not the whole story.

### (b) Label allocation / triage drift. **Not supported as the cause (it is a real noise source)**
Test (`scripts/diagnose_labelswap.py`, `diagnosis/labelswap.json`): serve the **static population** with a ledger
fitted on the label set that the evolved or no-selection run actually acquired (from its blackboard triage events). This
reproduces static's own value exactly when given static's own labels. Over 102 evolved and no-selection states, the label set alone
moves static by −0.025 to +0.021 eV/Å (sd 0.009). In 23 % of states the move exceeds 0.01. This mechanism also
produced the run A s0 anomaly. In the 8 harmed runs, though, the evolved label set would have made static
**equal or better**: −0.025 to +0.006 (b200 s0 +0.006, b400 s0 −0.011, b800 s1 −0.025, ext s2 0.000, ext s4
−0.006, D-gpu s9 −0.001, s10 −0.004, C′ ext s5 −0.004). At fixed labels the population effect is therefore
*larger* than reported: about +0.009 to +0.038. Label overlap with static is 64–100 %, similar in harmed and
unharmed runs.

### (c) Specialists overfit the small gate/select set. **Partly supported: the rel-E check does not predict eval rel-E, and it rejects the best force children**
CPU, AIMNet2 children with predictions, de-duplicated (n = 45, from C′ ext b800, D-cpu, and protect+gate-tol b800):
- On eval, every child beats its parent on forces in the flagged niche (45/45, median −0.148 eV/Å) and almost every one does in the
  **trusted** niche (43/45, median −0.101). The select-set force change predicts the eval force change
  (sign agreement 89 %, r = 0.87).
- The select-set **rel-E** change does not predict eval rel-E: sign agreement 38 % (below chance), r = −0.15.
  Children whose select rel-E was worse than their parent's (n = 29) have the *largest* eval force gains: median
  −0.151 flagged, −0.108 trusted. Those whose select rel-E was not worse (n = 16) gain only −0.014 / −0.008. The
  rel-E check therefore mostly rejects the children that moved furthest, which are the most useful on forces. It
  decides on 15–170 frames from few reactions.
- GPU: no eval predictions. The gate records match the CPU picture: the rel-E check rejected AIMNet2 children with large select force gains (above).

### (d) Family effect: Orb/MACE vs AIMNet2 specialists. **Supported as an association; mechanism partly unresolved**
- AIMNet2 specialists are not "worse outside their niche". On CPU, 43 of 45 are better than their parent on trusted eval frames (above).
- What separates harmed from unharmed GPU runs is which families survive the gate (table above). The data cannot
  tell whether Orb specialists are themselves worse than base Orb on trusted frames: no Orb predictions were saved.
  The served-by-niche numbers are consistent with that (trusted +0.010 to +0.042 when `mean(aimnet2 + Orb
  specialist)` replaces `mean(aimnet2+orb)`). The one counter-example, ablation_protect s1 (Orb and MACE
  specialists only, −0.018 \*), serves `mean(aimnet2+g1_orb+orb)`, a three-way mean that keeps base Orb.

### Earlier claims, re-checked
- "Selection prunes base models": base models were protected in D-gpu s9 and s10, and those runs are still harmed.
  The C′ s5 harm involves a pruned base (AIMNet2), but in that run the routing is what is wrong (a). Not the cause in general.
- "Specialists are worse outside their niche": false for AIMNet2 (CPU, 43/45 better on trusted). Untestable for Orb or MACE.

## What this suggests (hypotheses for a pre-registered test, not findings)
1. Gate on forces only, or measure rel-E on a larger or pooled set. The rel-E check is the step that removes the
   AIMNet2 specialists, and on CPU it is anti-informative about eval rel-E.
2. Route on the target metric: a per-bucket force-RMSE ledger, or separate force and energy routing. On CPU this
   gains about 0.005–0.008 eV/Å on average and avoids the C′ s5 failure.
3. Save child eval predictions for GPU runs, so Orb and MACE specialist quality can be measured directly.

Sources: `../evolution_c/**/generations.json`, `family_tree.json` and `blackboard.jsonl` (local); `../evolution_d/{cpu,gpu}/`;
scripts `scripts/diagnose_labelswap.py`, `scripts/diagnose_cpu_replay.py` (needs the local CPU child cache in
`/home/azureuser/fittest_evoC/results/*/cache`, which is not in git), `scripts/diagnose_fig10.py`. Outputs: `diagnosis/labelswap.json`,
`diagnosis/cpu_diag.json`. GPU child predictions (.npz) are not available, so every GPU statement above comes from
saved aggregates, gate records and routing counts.
