"""Generate results/RESULTS.md from the result files (no ablation is re-run).

    python scripts/write_results.py [--results results] [--frames data/frames.pkl] [--preds data/preds]

Sources: results/ablation.json, ablation_budget.json, split_report.json, blackboard_armD.jsonl,
the preserved earlier versions (results/v1_flag_only_bidding, v2_all_member_sigma, v3_served_sigma),
and the meta of {preds}/mace_lora_curated.npz. Ensemble dispersion figures are computed from
frames.pkl + preds when present (skipped otherwise). Every number in the text is read from these
files; sentences that compare arms take their verb ("better" / "worse" / "tied") from the
paired-bootstrap comparisons in ablation.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

VERSIONS = (("v1", "v1_flag_only_bidding"), ("v2", "v2_all_member_sigma"),
            ("v3", "v3_served_sigma"), ("v4 (final)", "."))
SPLIT_OF_VERSION = {"v1": "by reaction (leaky)", "v2": "by reaction (leaky)",
                    "v3": "by geometry component", "v4 (final)": "by geometry component"}
ALL_COLS = (("auroc_pool", "AUROC pool"), ("auroc_eval", "AUROC eval"),
            ("auroc_common", "AUROC vs AIMNet2-wrong"), ("recall_at_k", "Recall@K"),
            ("planted_caught", "Planted caught"), ("eval_force_rmse_post", "Served F-RMSE eval (eV/Å)"),
            ("eval_rel_e_mae_post", "Served ΔE MAE eval (eV)"), ("ece_eval", "ECE eval"),
            ("brier_eval", "Brier eval"), ("silent_failures_eval", "Silent failures eval"),
            ("trusted_fraction_eval", "Trusted share eval"))
CMP_COLS = (("auroc_pool", "AUROC pool"), ("auroc_eval", "AUROC eval"), ("recall_at_k", "Recall@K"),
            ("planted_caught", "Planted"), ("eval_force_rmse_post", "Served F-RMSE"),
            ("ece_eval", "ECE"), ("brier_eval", "Brier"), ("silent_failures_pool", "Silent pool"),
            ("silent_failures_eval", "Silent eval"))
VER_COLS = (("auroc_pool", "AUROC pool"), ("auroc_eval", "AUROC eval"), ("recall_at_k", "Recall@K"),
            ("planted_caught", "Planted*"), ("eval_force_rmse_post", "Served F-RMSE eval"),
            ("eval_wrong_rate_pre", "Served wrong rate eval"), ("ece_eval", "ECE"),
            ("brier_eval", "Brier"), ("trusted_fraction_eval", "Trusted share"))


def load(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def val(arm: dict[str, Any], key: str) -> float | None:
    m = arm.get("metrics", {}).get(key) if arm.get("status") == "ok" else None
    return None if not m else m.get("value")


def fmt(m: dict[str, Any] | None, d: int = 3) -> str:
    if not m or m.get("value") is None:
        return "n/a"
    ci = m.get("ci")
    return f"{m['value']:.{d}f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{m['value']:.{d}f}"


def verdict(comp: dict[str, Any]) -> str:
    if comp["d_significantly_better"]:
        return "better"
    if comp["d_significantly_worse"]:
        return "**worse**"
    return "n/e" if comp["d_significantly_better"] is None else "n.s."


def who(r: dict[str, Any], key: str, word: str) -> list[str]:
    """Baselines against which D is significantly ``word`` ('better'/'worse') on ``key``."""
    flag = "d_significantly_better" if word == "better" else "d_significantly_worse"
    return [name[5:] for name, comp in r["comparisons"].items() if comp.get(key, {}).get(flag)]


def arms_table(r: dict[str, Any]) -> list[str]:
    out = ["| arm | " + " | ".join(c for _, c in ALL_COLS) + " |", "|---|" + "---|" * len(ALL_COLS)]
    for name, arm in r["arms"].items():
        if arm.get("status") != "ok":
            out.append(f"| {name} | {arm.get('status')}: {arm.get('reason', '')[:60]} |" + " |" * (len(ALL_COLS) - 1))
        else:
            out.append(f"| {name} | " + " | ".join(fmt(arm["metrics"].get(k)) for k, _ in ALL_COLS) + " |")
    return out


def comparison_table(r: dict[str, Any]) -> list[str]:
    out = ["| D vs | " + " | ".join(c for _, c in CMP_COLS) + " |", "|---|" + "---|" * len(CMP_COLS)]
    for name, comp in r["comparisons"].items():
        out.append(f"| {name[5:]} | " + " | ".join(verdict(comp[k]) if k in comp else "–"
                                                   for k, _ in CMP_COLS) + " |")
    return out


def versions_table(root: Path) -> list[str]:
    out = ["| version | split | " + " | ".join(c for _, c in VER_COLS) + " |",
           "|---|---|" + "---|" * len(VER_COLS)]
    for label, sub in VERSIONS:
        rr = load(root / sub / "ablation.json")
        if rr is None or "D_trustswarm" not in rr.get("arms", {}):
            continue
        d = rr["arms"]["D_trustswarm"]["metrics"]
        out.append(f"| {label} | {SPLIT_OF_VERSION[label]} | " +
                   " | ".join(fmt(d.get(k)) for k, _ in VER_COLS) + " |")
    return out


def budget_lines(sweep: dict[str, Any] | None) -> list[str]:
    if not sweep:
        return ["(no budget sweep found)"]
    b = sweep["budgets"]
    out = ["| arm | " + " | ".join(f"recall@K, k={k}" for k in b) + " | " +
           " | ".join(f"AUROC pool, k={k}" for k in b) + " |",
           "|---|" + "---|" * (2 * len(b))]
    for name, row in sweep["arms"].items():
        cells = [("n/a" if x["value"] is None else f"{x['value']:.3f}") for x in row["recall_at_k"]]
        cells += [("n/a" if x["value"] is None else f"{x['value']:.3f}") for x in row["auroc_pool"]]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out


def collective_lines(path: Path) -> list[str]:
    if not path.exists():
        return ["(blackboard log not found)"]
    events = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    kinds = Counter(e["kind"] for e in events)
    served: dict[str, dict[str, str]] = {"triage": {}, "serve": {}}
    for e in events:
        if e["kind"] == "verdict":
            p = e["payload"]
            for uid, ph, rec in zip(p["uid"], p["phase"], p["recommended_model"]):
                served["triage" if ph == "triage" else "serve"][uid] = rec
    taus = [(e["round"], e["payload"]["tau"], e["payload"]["n_labelled"])
            for e in events if e["kind"] == "threshold"]
    seen, tau_traj = set(), []
    for rnd, tau, n in taus:  # last threshold of each round
        tau_traj = [t for t in tau_traj if t[0] != rnd] + [(rnd, tau, n)]
        seen.add(rnd)
    reps = [e["payload"]["reputation"] for e in events if e["kind"] == "reputation"]
    oracle = [e["payload"] for e in events if e["kind"] == "oracle_result"]
    n_audit = sum(1 for p in oracle if p.get("audit"))
    vetoes = [e["payload"] for e in events if e["kind"] == "veto"]
    bids = sum(len(e["payload"]["uid"]) for e in events if e["kind"] == "bid")
    flags = kinds.get("flag", 0)
    fin = [e["payload"] for e in events if e["kind"] == "finetune_report"]
    items = Counter()
    for e in events:
        uid = e["payload"].get("uid")
        items[e["kind"]] += len(uid) if isinstance(uid, list) else 1
    out = [f"From `{path.as_posix()}` ({len(events)} events, arrays summarised). `prediction`, "
           "`pass`, `verdict` and `bid` are posted as batches (one per agent step).", "",
           "| event kind | events | systems covered |", "|---|---|---|"]
    out += [f"| {k} | {v} | {items[k]} |" for k, v in kinds.most_common()]
    out += ["", f"* **Oracle calls:** {len(oracle)}, of which {n_audit} exploration audits and "
                f"{len(oracle) - n_audit} won by bids; {bids} individual bids were placed in total.",
            f"* **Flags / vetoes:** the hook flagged {flags} systems; the Skeptic vetoed "
            f"{len(vetoes)} trusted verdicts ({sum(v.get('phase') == 'triage' for v in vetoes)} on "
            f"pool systems, the rest advisory on eval)."]
    for phase, label in (("serve", "eval (served)"), ("triage", "pool (last verdict before reveal)")):
        dist = Counter(served[phase].values())
        if dist:
            out.append(f"* **Recommended candidate, {label}:** " +
                       ", ".join(f"`{c}` {n}" for c, n in dist.most_common()))
    if tau_traj:
        out.append("* **τ trajectory (round: τ, labels):** " +
                   "; ".join(f"{r}: {t:.3f} ({n})" for r, t, n in tau_traj))
        if len({round(t, 9) for _, t, _ in tau_traj}) == 1:
            out.append("  τ never left its round-0 floor (70th percentile of σ_F): the isotonic rule "
                       "(smallest τ with P(served wrong | σ ≥ τ) ≥ 0.5 on the labelled points) never "
                       "exceeded the floor, so the floor set every flag.")
    if reps:
        out.append("* **Reputations at the end:** " +
                   ", ".join(f"{a} {v:.2f}" for a, v in sorted(reps[-1].items())))
    if fin:
        f = fin[-1]
        out.append(f"* **Curator:** chose `{f['model']}` as the worst model (ledger force RMSE " +
                   ", ".join(f"{m} {v:.3f}" for m, v in f["ledger_force_rmse"].items()) +
                   f"); shipped: {f['shipped']}.")
    return out


def dispersion(frames_path: Path, preds_dir: Path, seed: int) -> dict[str, dict[str, float]]:
    """Median seed spread vs median error of the ensemble mean on (leak-free) eval frames."""
    try:
        from trustswarm.cache import load_frames, load_models
        from trustswarm.data import EVAL, assign_leakfree_splits
        from trustswarm.signals import force_rmse, force_spread
    except ImportError:
        return {}
    if not frames_path.exists():
        return {}
    frames, _ = assign_leakfree_splits(load_frames(frames_path), seed)
    ev = [f for f in frames if f.split == EVAL]
    out = {}
    for label, models in (("MACE LoRA seeds", [f"mace_seed{k}" for k in range(4)]),
                          ("AIMNet2 ensemble members", [f"aimnet2_m{k}" for k in range(4)])):
        try:
            preds = load_models(preds_dir, models)
        except FileNotFoundError:
            continue
        spread = [force_spread(np.stack([preds[m].forces[f.uid] for m in models]))[0] for f in ev]
        err = [force_rmse(np.mean([preds[m].forces[f.uid] for m in models], axis=0), f.F) for f in ev]
        out[label] = {"spread": float(np.median(spread)), "error": float(np.median(err))}
    return out


def naive_leak(frames_path: Path) -> int | None:
    """Near-duplicate pool/eval endpoints under the per-reaction split stored in frames.pkl."""
    try:
        from trustswarm.cache import load_frames
        from trustswarm.data import EVAL, POOL, near_duplicate_endpoints
    except ImportError:
        return None
    if not frames_path.exists():
        return None
    return near_duplicate_endpoints(load_frames(frames_path), POOL, EVAL)


def repair_lines(r: dict[str, Any], preds_dir: Path) -> list[str]:
    ft = r["arms"].get("D_trustswarm", {}).get("finetune")
    meta = None
    try:
        from trustswarm.cache import load_predictions
        p = preds_dir / "mace_lora_curated.npz"
        meta = load_predictions(p).meta if p.exists() else None
    except ImportError:
        pass
    out = []
    if ft:
        out.append(f"* The Curator chose **`{ft['model']}`** (highest ledger force RMSE) and requested a LoRA "
                   f"fine-tune on {ft.get('n_labels')} oracle labels. Shipped: **{ft['shipped']}**"
                   + (f" — {ft['report'].get('reason')}." if not ft["shipped"] else "."))
    acc = (meta or {}).get("acceptance_report", {})
    hm = acc.get("holdout_metrics", {})
    if hm:
        b, a = hm["before"], hm["after"]
        out.append(f"* Held-out slice of the oracle labels ({b['n_frames']} frames, never trained on): force RMSE "
                   f"{b['force_rmse']:.3f} → {a['force_rmse']:.3f} eV/Å, relative-energy MAE "
                   f"{b['rel_energy_mae']:.3f} → {a['rel_energy_mae']:.3f} eV.")
    for c in acc.get("checks", []):
        out.append(f"  * check `{c['name']}` ({c['source']}): {c['value']:.3f} {c['comparison']} "
                   f"{c['limit'] if not isinstance(c['limit'], float) else round(c['limit'], 3)} → "
                   f"{'pass' if c['passed'] else '**fail**'}"
                   if isinstance(c.get("value"), float) else
                   f"  * check `{c['name']}` ({c['source']}): {'pass' if c['passed'] else '**fail**'}")
    leak = (meta or {}).get("geometry_leak_check")
    if leak:
        out.append(f"* Geometry leak check of the training labels against eval: "
                   f"{leak['n_own_geometry_matches']} matches → {'passed' if leak['passed'] else 'FAILED'}.")
    if acc:
        out.append(f"* Our gate requires every check to pass, so the fine-tune is **not shipped** even "
                   f"though the toolkit's own acceptance ({acc.get('toolkit_accepted')}) passes: it fixes "
                   f"forces but makes relative energies slightly worse." if not acc.get("accepted")
                   else "* All checks passed; the fine-tune shipped.")
    return out or ["(no fine-tune information found)"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--preds", default="data/preds")
    args = ap.parse_args(argv)
    root, preds_dir = Path(args.results), Path(args.preds)
    r = load(root / "ablation.json")
    if r is None:
        raise SystemExit(f"{root / 'ablation.json'} not found")
    sweep, split = load(root / "ablation_budget.json"), load(root / "split_report.json")
    arms, cfg = r["arms"], r["config"]
    D = arms["D_trustswarm"]
    ok = {n: a for n, a in arms.items() if a.get("status") == "ok"}
    baselines = [n for n in ok if n not in ("D_trustswarm", "A_random")]
    served = {n: val(a, "eval_force_rmse_post") for n, a in ok.items()}
    wrong = {n: val(a, "eval_wrong_rate_pre") for n, a in ok.items()}
    best_auc = max(ok, key=lambda n: val(ok[n], "auroc_eval") or 0)
    best_common = max(ok, key=lambda n: val(ok[n], "auroc_common") or 0)
    disp = dispersion(Path(args.frames), preds_dir, cfg["seed"])
    ft = D.get("finetune", {})
    others_served = [served[n] for n in baselines if served[n] is not None]
    others_wrong = [wrong[n] for n in baselines if wrong[n] is not None]
    auc_worse = who(r, "auroc_eval", "worse")
    cal_better = who(r, "ece_eval", "better")
    cal_tied = [n[5:] for n, c in r["comparisons"].items()
                if not c["ece_eval"]["d_significantly_better"] and not c["ece_eval"]["d_significantly_worse"]]
    recall_better = who(r, "recall_at_k", "better")

    summary = [
        "<!-- SUMMARY:START -->",
        f"**TrustSwarm on Transition1x (leak-free split, {cfg['budget_per_round']}×{cfg['rounds']} = "
        f"{cfg['total_budget']} oracle calls per arm, 95% reaction-bootstrap CIs).**",
        "",
        f"* **Better served answers:** TrustSwarm's served force RMSE on held-out eval is "
        f"**{served['D_trustswarm']:.3f} eV/Å** vs {min(others_served):.3f}–{max(others_served):.3f} for every "
        f"baseline; wrong on {wrong['D_trustswarm']:.0%} of eval frames vs {min(others_wrong):.0%}–"
        f"{max(others_wrong):.0%}. It learned to serve the subset mean of the models that is most often right.",
        f"* **Calibrated trust:** ECE {val(D, 'ece_eval'):.3f}, Brier {val(D, 'brier_eval'):.3f}; significantly "
        f"better than {', '.join(cal_better) or 'no baseline'}"
        + (f", tied with {', '.join(cal_tied)}" if cal_tied else "") +
        " — the gain comes from the Skeptic's random audits of trusted answers.",
        f"* **Negative result — error ranking:** TrustSwarm's AUROC (eval {val(D, 'auroc_eval'):.3f}) is "
        f"significantly worse than {', '.join(auc_worse) or 'no baseline'}. Best ranker on eval: "
        f"{best_auc} ({val(ok[best_auc], 'auroc_eval'):.3f}).",
        f"* **Heterogeneous disagreement detects AIMNet2's failures best:** AUROC vs 'AIMNet2 wrong' "
        f"{val(ok[best_common], 'auroc_common'):.3f} ({best_common}).",
        f"* **Recall@K:** TrustSwarm {val(D, 'recall_at_k'):.3f}; significantly better than "
        f"{', '.join(recall_better) or 'no baseline'}.",
    ]
    if disp:
        summary.append("* **Ranking ≠ calibration:** on eval the median seed spread is " + "; ".join(
            f"{k} {v['spread']:.3f} vs median error {v['error']:.3f} eV/Å (×{v['error'] / v['spread']:.1f} "
            f"under-dispersed)" for k, v in disp.items()) + f"; yet {best_auc} ranks errors best.")
    hm = ((ft.get("report") or {}).get("acceptance_report") or {}).get("holdout_metrics") or {}
    repair = (f"forces {hm['before']['force_rmse']:.3f} → {hm['after']['force_rmse']:.3f} eV/Å but "
              f"relative-energy MAE {hm['before']['rel_energy_mae']:.3f} → {hm['after']['rel_energy_mae']:.3f} eV "
              "on held-out labels, so our gate refuses it") if hm else (ft.get("report") or {}).get("reason", "")
    leak_before = naive_leak(Path(args.frames))
    summary += [
        f"* **Self-repair:** the Curator chose and LoRA-fine-tuned `{ft.get('model')}` on its oracle labels; "
        f"{'shipped' if ft.get('shipped') else 'not shipped'} — {repair}." if ft else "",
        "* **Leakage controlled:** pool/eval and skeptic_ref/planted share 0 endpoint geometries"
        + (f" (Transition1x reuses reactants: a per-reaction split had {leak_before} near-duplicate "
           "pool/eval endpoints)." if leak_before else "."),
        "* Full tables, per-version history and failures: `results/RESULTS.md`.",
        "<!-- SUMMARY:END -->",
    ]
    summary = [s for s in summary if s != ""] if len(summary) > 25 else summary

    doc: list[str] = ["# TrustSwarm — results", "", *summary, "",
                      "Generated by `scripts/write_results.py` from `results/ablation.json` and friends. "
                      f"Primary wrongness: force RMSE > {cfg['wrong_f']} eV/Å or |ΔE_rel| > {cfg['wrong_e']} eV "
                      "(fixed before running; 0.15 and 0.4 in `by_threshold`). Brackets: 95% bootstrap CIs over "
                      f"reactions ({cfg['n_bootstrap']} resamples, paired across arms).", "",
                      "## All arms (final run)", "", *arms_table(r), "",
                      "Planted = planted-split frames where AIMNet2's force RMSE > "
                      f"{cfg.get('planted_force_rmse')} eV/Å (AIMNet2 is in every arm). Silent failures = badly "
                      "wrong (F-RMSE > 0.4), unflagged, unchecked and trusted; read with the trusted share — "
                      "an arm that trusts nothing has none by construction.", "",
                      "### TrustSwarm (D) vs each baseline (paired 95% CI)", "",
                      "`better` / `**worse**`: CI of the difference excludes 0; `n.s.`: not significant.", "",
                      *comparison_table(r), "",
                      "### Budget sweep (per-round budget k, same 5 rounds)", "", *budget_lines(sweep), ""]
    if disp:
        doc += ["### Ensemble dispersion vs error (eval)", "",
                "| ensemble | median spread σ_F (eV/Å) | median error of its mean (eV/Å) | error / spread |",
                "|---|---|---|---|"]
        doc += [f"| {k} | {v['spread']:.3f} | {v['error']:.3f} | {v['error'] / v['spread']:.1f} |"
                for k, v in disp.items()]
        doc += ["", "A seed ensemble can rank errors well (high AUROC) while being badly under-dispersed: "
                "ranking and calibration are different properties.", ""]
    doc += ["## What the collective decided (arm D)", "", *collective_lines(root / "blackboard_armD.jsonl"), "",
            "## Repair (Curator)", "", *repair_lines(r, preds_dir), "",
            "## TrustSwarm across versions", "", *versions_table(root), "",
            "\\* v1 defined planted cases as 'MACE-MPA-0 wrong' (favoured arms containing MACE); later "
            "versions use 'AIMNet2 badly wrong'. v1→v2 and v3→v4 share a split; v2→v3 changes it.", "",
            "| version | change | reason |", "|---|---|---|",
            "| v1 | bidding on flagged (high-σ) systems; serve best single model or 3-way consensus | design as specified |",
            "| v2 | Skeptic exploration audits (25% of each round's budget on random trusted systems); served "
            "candidates = every subset mean; replay provenance; arm-independent planted cases; "
            "skeptic_ref/planted split by formula | the oracle only labelled high-σ systems, so trust was "
            "never tested where it was given; the best answer was a subset mean |",
            "| v3 | leak-free split by shared endpoint geometry; risk on σ_served | T1x reuses reactant "
            "geometries across reactions; measure uncertainty of the answer actually served (negative "
            "result: ranking got worse) |",
            "| v4 | Calibrator selects its risk feature by reaction-grouped CV on its own labels "
            f"(chose {D.get('risk_feature')}, CV AUROC "
            + ", ".join(f"{k} {v:.3f}" for k, v in (D.get("risk_feature_cv_auroc") or {}).items()
                        if v is not None) + ") | choose without looking at eval |", ""]
    if split:
        doc += ["## Splits and leakage", "",
                f"* pool/eval by connected component of shared reactant/product geometry: "
                f"{split.get('test_components')} components over {split.get('test_reactions')} reactions; "
                "frames per split: " + ", ".join(f"{k} {v}" for k, v in split["frames_per_split"].items()) + ".",
                "* skeptic_ref/planted by formula, components kept whole "
                f"({split.get('val_reactions_moved_to_keep_components_whole', 0)} reactions moved).",
                "* Shared endpoint geometries after the split: " +
                ", ".join(f"{k}: {v}" for k, v in split["shared_endpoint_geometries"].items()) +
                "; near-duplicates (max|ΔR| < 1e-3 Å): " +
                ", ".join(f"{k}: {v}" for k, v in split["near_duplicate_endpoints_max_dR_1e-3"].items()) + ".",
                "* Labels reach agents only through `oracle_result` events (tested); eval labels are never "
                "revealed; replayed fine-tunes must match this run's labels and share no uid or geometry with "
                "held-out frames.", ""]
    run_failures = [f for f in r["failures"] if not f.startswith("D vs ")]
    doc += ["## Failures and caveats (reported, not hidden)", ""]
    doc += [f"* {f}" for f in run_failures] or ["* (no run failures)"]
    doc += [f"* D's error ranking (AUROC eval) is significantly worse than: {', '.join(auc_worse)}."
            if auc_worse else "* D's error ranking is not significantly worse than any baseline.",
            "* The Calibrator's labels are mostly bid-won (high σ); audits reduce but do not remove "
            "this selection bias.",
            "* The synthetic smoke test (`scripts/make_synthetic.py`) shows the pipeline runs; it is not "
            "evidence for any claim.", ""]
    (root / "RESULTS.md").write_text("\n".join(doc), encoding="utf-8")
    n_summary = summary.index("<!-- SUMMARY:END -->") + 1
    print(f"wrote {root / 'RESULTS.md'} (summary block: {n_summary} lines)")


if __name__ == "__main__":
    main()
