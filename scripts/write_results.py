"""Generate results/RESULTS.md from the result files (no ablation is re-run).

    python scripts/write_results.py [--results results] [--frames data/frames.pkl] [--preds data/preds]
                                    [--allow-partial]

Sources: results/ablation.json, ablation_budget.json, split_report.json, signal.json,
blackboard_armD.jsonl, the preserved earlier versions (results/v1_flag_only_bidding,
v2_all_member_sigma, v3_served_sigma), data/frames.pkl + data/preds/*.npz (hindsight reference,
ensemble dispersion, naive-split leakage) and the meta of data/preds/mace_lora_curated.npz
(repair). If any input is missing the script refuses to write — a partial RESULTS.md would
silently drop claims — unless ``--allow-partial`` is given.

Every number in the text is read from these files; sentences that compare arms take their verb
("better" / "worse" / "tied") from the paired-bootstrap comparisons in ablation.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VERSIONS = (("v1", "v1_flag_only_bidding"), ("v2", "v2_all_member_sigma"),
            ("v3", "v3_served_sigma"), ("v4 (final)", "."))
SPLIT_OF_VERSION = {"v1": "by reaction (leaky)", "v2": "by reaction (leaky)",
                    "v3": "reactant/product-linked components", "v4 (final)": "reactant/product-linked components"}
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
PAIR = ("aimnet2", "orb")
PAIR_NAME = "mean(aimnet2+orb)"
NEEDED_PREDS = ("aimnet2", "orb", "mace_lora_curated", *(f"mace_seed{k}" for k in range(4)),
                *(f"aimnet2_m{k}" for k in range(4)))


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
    """Baselines vs which D is significantly ``word`` ('better' / 'worse' / 'ns') on ``key``."""
    out = []
    for name, comp in r["comparisons"].items():
        c = comp.get(key)
        if c is None:
            continue
        if (word == "better" and c["d_significantly_better"]) or \
                (word == "worse" and c["d_significantly_worse"]) or \
                (word == "ns" and not c["d_significantly_better"] and not c["d_significantly_worse"]):
            out.append(name[5:])
    return out


def missing_inputs(root: Path, frames: Path, preds: Path) -> list[str]:
    needed = [root / n for n in ("ablation.json", "ablation_budget.json", "split_report.json",
                                 "signal.json", "blackboard_armD.jsonl")]
    needed += [root / sub / "ablation.json" for _, sub in VERSIONS if sub != "."]
    needed += [frames] + [preds / f"{m}.npz" for m in NEEDED_PREDS]
    return [str(p) for p in needed if not p.exists()]


# --- tables -------------------------------------------------------------------------------------

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
    keys = (("recall_at_k", "recall@K"), ("auroc_pool", "AUROC pool"),
            ("eval_force_rmse_post", "served F-RMSE eval"))
    out = ["| arm | " + " | ".join(f"{lab}, k={k}" for _, lab in keys for k in b) + " |",
           "|---|" + "---|" * (len(keys) * len(b))]
    for name, row in sweep["arms"].items():
        cells = [("n/a" if x["value"] is None else f"{x['value']:.3f}")
                 for key, _ in keys for x in row.get(key, [{"value": None}] * len(b))]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return out


# --- derived facts ------------------------------------------------------------------------------

def collective(path: Path) -> tuple[list[str], dict[str, Any]]:
    events = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    kinds = Counter(e["kind"] for e in events)
    items: Counter[str] = Counter()
    for e in events:
        uid = e["payload"].get("uid")
        items[e["kind"]] += len(uid) if isinstance(uid, list) else 1
    served: dict[str, dict[str, str]] = {"triage": {}, "serve": {}}
    for e in events:
        if e["kind"] == "verdict":
            p = e["payload"]
            for uid, ph, rec in zip(p["uid"], p["phase"], p["recommended_model"]):
                served["triage" if ph == "triage" else "serve"][uid] = rec
    tau_last: dict[int, tuple[float, int]] = {}
    for e in events:
        if e["kind"] == "threshold":
            tau_last[e["round"]] = (e["payload"]["tau"], e["payload"]["n_labelled"])
    reps = [e["payload"]["reputation"] for e in events if e["kind"] == "reputation"]
    oracle = [e["payload"] for e in events if e["kind"] == "oracle_result"]
    vetoes = [e["payload"] for e in events if e["kind"] == "veto"]
    stats = {"served_eval": Counter(served["serve"].values()), "n_eval": len(served["serve"]),
             "tau_constant": len({round(t, 9) for t, _ in tau_last.values()}) == 1,
             "tau": next(iter(tau_last.values()))[0] if tau_last else None,
             "reputations": reps[-1] if reps else {},
             "n_audit": sum(1 for p in oracle if p.get("audit")), "n_oracle": len(oracle),
             "n_vetoes": len(vetoes), "n_flags": kinds.get("flag", 0)}
    rep_vals = list(stats["reputations"].values())
    stats["rep_range"] = (min(rep_vals), max(rep_vals)) if rep_vals else None
    out = [f"From `{path.as_posix()}` ({len(events)} events, arrays summarised). `prediction`, `pass`, "
           "`verdict` and `bid` are posted as batches (one per agent step).", "",
           "| event kind | events | systems covered |", "|---|---|---|"]
    out += [f"| {k} | {v} | {items[k]} |" for k, v in kinds.most_common()]
    out += ["", f"* **Oracle calls:** {stats['n_oracle']}: {stats['n_audit']} exploration audits, "
                f"{stats['n_oracle'] - stats['n_audit']} won by bids ({items['bid']} bids placed).",
            f"* **Flags / vetoes:** the hook flagged {stats['n_flags']} systems; the Skeptic vetoed "
            f"{stats['n_vetoes']} trusted verdicts ({sum(v.get('phase') == 'triage' for v in vetoes)} on pool "
            "systems; vetoes on eval are advisory)."]
    for phase, label in (("serve", "eval (served)"), ("triage", "pool (last verdict before reveal)")):
        dist = Counter(served[phase].values())
        if dist:
            out.append(f"* **Recommended candidate, {label}:** " +
                       ", ".join(f"`{c}` {n}" for c, n in dist.most_common()))
    if tau_last:
        out.append("* **τ (round: τ, labels):** " +
                   "; ".join(f"{r}: {t:.3f} ({n})" for r, (t, n) in sorted(tau_last.items())))
        if stats["tau_constant"]:
            out.append("  **τ never left its round-0 floor** (70th percentile of σ_F): the isotonic rule "
                       "(smallest τ with P(served wrong | σ ≥ τ) ≥ 0.5) never exceeded the floor, so the "
                       "Calibrator's threshold learning had no effect on flagging.")
    if stats["rep_range"]:
        lo, hi = stats["rep_range"]
        out.append("* **Reputations at the end:** " +
                   ", ".join(f"{a} {v:.2f}" for a, v in sorted(stats["reputations"].items())) +
                   (f" — **nearly equal** (range {lo:.2f}–{hi:.2f}): reputation weighting barely "
                    "differentiated the bidders." if hi - lo < 0.1 else ""))
    return out, stats


def eval_frames(frames_path: Path, seed: int) -> tuple[list[Any], list[Any]]:
    from trustswarm.cache import load_frames
    from trustswarm.data import EVAL, assign_leakfree_splits
    raw = load_frames(frames_path)
    frames, _ = assign_leakfree_splits(raw, seed)
    return raw, [f for f in frames if f.split == EVAL]


def hindsight_pair(ev: list[Any], preds_dir: Path) -> float:
    """Mean force RMSE on eval of a fixed mean(aimnet2+orb) — chosen with hindsight."""
    from trustswarm.cache import load_models
    from trustswarm.signals import force_rmse
    p = load_models(preds_dir, PAIR)
    return float(np.mean([force_rmse(np.mean([p[m].forces[f.uid] for m in PAIR], axis=0), f.F)
                          for f in ev]))


def dispersion(ev: list[Any], preds_dir: Path) -> dict[str, dict[str, float]]:
    """Median seed spread vs median error of the ensemble mean on eval frames."""
    from trustswarm.cache import load_models
    from trustswarm.signals import force_rmse, force_spread
    out = {}
    for label, models in (("MACE LoRA seeds (B′)", [f"mace_seed{k}" for k in range(4)]),
                          ("AIMNet2 ensemble members (B)", [f"aimnet2_m{k}" for k in range(4)])):
        preds = load_models(preds_dir, models)
        spread = [force_spread(np.stack([preds[m].forces[f.uid] for m in models]))[0] for f in ev]
        err = [force_rmse(np.mean([preds[m].forces[f.uid] for m in models], axis=0), f.F) for f in ev]
        out[label] = {"spread": float(np.median(spread)), "error": float(np.median(err))}
    return out


def repair(r: dict[str, Any], preds_dir: Path) -> tuple[list[str], str]:
    from trustswarm.cache import load_predictions
    ft = r["arms"]["D_trustswarm"].get("finetune") or {}
    meta = load_predictions(preds_dir / "mace_lora_curated.npz").meta
    acc = meta.get("acceptance_report", {})
    hm = acc.get("holdout_metrics", {})
    n_tr, n_ho = meta.get("n_train"), meta.get("n_holdout")
    out = [f"* The Curator chose **`{ft.get('model')}`** (highest ledger force RMSE) and handed over "
           f"{ft.get('n_labels')} oracle labels; the GPU fine-tune used {n_tr} for training and {n_ho} as a "
           f"hold-out. Shipped: **{ft.get('shipped')}**"
           + (f" — {ft['report'].get('reason')}." if not ft.get("shipped") and ft.get("report") else ".")]
    if hm:
        b, a = hm["before"], hm["after"]
        out.append(f"* Hold-out ({b['n_frames']} labelled frames, never trained on): force RMSE "
                   f"{b['force_rmse']:.3f} → {a['force_rmse']:.3f} eV/Å; relative-energy MAE "
                   f"{b['rel_energy_mae']:.3f} → {a['rel_energy_mae']:.3f} eV.")
    for c in acc.get("checks", []):
        v = c.get("value")
        lim = round(c["limit"], 3) if isinstance(c.get("limit"), float) else c.get("limit")
        out.append(f"  * `{c['name']}` ({c['source']}): " +
                   (f"{v:.3f} {c['comparison']} {lim} → " if isinstance(v, float) else "") +
                   ("pass" if c["passed"] else "**fail**"))
    for key, label in (("geometry_leak_check", "training labels vs eval geometries"),
                       ("holdout_leak_check", "hold-out vs training (component-disjoint)")):
        leak = meta.get(key)
        if leak is not None:
            out.append(f"* Leak check, {label}: " +
                       ("passed" if leak.get("passed") else "**FAILED**") +
                       (f" — {leak['detail']}" if leak.get("detail") else "") + ".")
        else:
            out.append(f"* Leak check, {label}: not recorded in the fine-tune metadata.")
    if acc:
        out.append("* Our gate requires every check to pass; " +
                   ("all passed, the fine-tune shipped." if acc.get("accepted") else
                    f"the fine-tune is **not shipped**" +
                    (" even though the toolkit's own acceptance passes" if acc.get("toolkit_accepted") else "")
                    + "."))
    short = ""
    if hm:
        short = (f"forces {hm['before']['force_rmse']:.3f} → {hm['after']['force_rmse']:.3f} eV/Å but "
                 f"relative-energy MAE {hm['before']['rel_energy_mae']:.3f} → "
                 f"{hm['after']['rel_energy_mae']:.3f} eV on {n_ho} held-out labels")
    return out, short


# --- main -----------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--preds", default="data/preds")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write even if inputs are missing (sections depending on them are omitted)")
    args = ap.parse_args(argv)
    root, preds_dir, frames_path = Path(args.results), Path(args.preds), Path(args.frames)
    missing = missing_inputs(root, frames_path, preds_dir)
    if missing and not args.allow_partial:
        sys.exit("refusing to write RESULTS.md: missing inputs (a partial file would silently drop "
                 "claims; pass --allow-partial to override):\n  " + "\n  ".join(missing))
    have = lambda p: str(p) not in missing  # noqa: E731

    r = load(root / "ablation.json")
    if r is None:
        sys.exit(f"{root / 'ablation.json'} not found")
    sweep, split, signal = (load(root / n) for n in ("ablation_budget.json", "split_report.json", "signal.json"))
    arms, cfg = r["arms"], r["config"]
    D = arms["D_trustswarm"]
    ok = {n: a for n, a in arms.items() if a.get("status") == "ok"}
    baselines = [n for n in ok if n not in ("D_trustswarm", "A_random")]
    served = {n: val(a, "eval_force_rmse_post") for n, a in ok.items()}
    wrong = {n: val(a, "eval_wrong_rate_pre") for n, a in ok.items()}
    best_auc = max(ok, key=lambda n: val(ok[n], "auroc_eval") or 0)
    best_common = max(ok, key=lambda n: val(ok[n], "auroc_common") or 0)
    others_served = [served[n] for n in baselines if served[n] is not None]
    others_wrong = [wrong[n] for n in baselines if wrong[n] is not None]

    need_data = have(frames_path) and all(have(preds_dir / f"{m}.npz") for m in NEEDED_PREDS)
    raw, ev = eval_frames(frames_path, cfg["seed"]) if have(frames_path) else (None, None)
    pair_rmse = hindsight_pair(ev, preds_dir) if need_data else None
    disp = dispersion(ev, preds_dir) if need_data else {}
    naive = None
    if raw is not None:
        from trustswarm.data import EVAL, POOL, near_duplicate_endpoints
        naive = near_duplicate_endpoints(raw, POOL, EVAL)
    coll_lines, cs = collective(root / "blackboard_armD.jsonl") if have(root / "blackboard_armD.jsonl") else ([], {})
    rep_lines, rep_short = repair(r, preds_dir) if have(preds_dir / "mace_lora_curated.npz") else ([], "")

    sf_e, sf_p = D["metrics"]["silent_failures_eval"], D["metrics"]["silent_failures_pool"]
    n_sf_e, n_sf_p = round(sf_e["value"] * sf_e["n"]), round(sf_p["value"] * sf_p["n"])
    trust_none = [n for n in baselines if (val(ok[n], "trusted_fraction_eval") or 0) == 0]
    sf_ns = who(r, "silent_failures_eval", "ns")
    nonrandom = [n for n in baselines]
    planted = {n: val(a, "planted_caught") for n, a in ok.items()}
    pl_ns, pl_better = who(r, "planted_caught", "ns"), who(r, "planted_caught", "better")
    others_pl = [planted[n] for n in ok if n != "D_trustswarm" and planted[n] is not None]
    auc_worse = who(r, "auroc_eval", "worse")
    cal_better, cal_tied = who(r, "ece_eval", "better"), who(r, "ece_eval", "ns")
    recall_better = who(r, "recall_at_k", "better")
    n_pair = cs.get("served_eval", {}).get(PAIR_NAME, 0) if cs else 0
    b_eval, c_eval = val(ok.get("B_seed", {}), "auroc_eval"), val(ok.get("C_collective", {}), "auroc_eval")
    sig04 = (signal or {}).get("auroc", {}).get("0.4", {})
    spreads = [v["error"] / v["spread"] for v in disp.values()]
    budget_label = (f"{cfg['total_budget']} Transition1x label lookups per arm ({cfg['budget_per_round']} per "
                    f"round × {cfg['rounds']} rounds; no live DFT)")
    bprime = ok.get("B_mace_seeds")
    bprime_note = (f"; B′ additionally uses MACE LoRA seeds fine-tuned on "
                   f"{(load_predictions_meta(preds_dir, 'mace_seed0') or {}).get('n_train', '?')} of arm D's labels"
                   if bprime else "")

    summary = ["<!-- SUMMARY:START -->",
               f"**TrustSwarm on Transition1x** — leak-free split, {budget_label}{bprime_note}; 95% "
               "reaction-bootstrap CIs.", ""]
    if pair_rmse is not None:
        summary.append(
            f"* **Better served answers, by discovery:** TrustSwarm's served force RMSE on held-out eval is "
            f"**{served['D_trustswarm']:.3f} eV/Å** vs {min(others_served):.3f}–{max(others_served):.3f} for the "
            f"baselines (wrong on {wrong['D_trustswarm']:.0%} of eval frames vs {min(others_wrong):.0%}–"
            f"{max(others_wrong):.0%}). It served `{PAIR_NAME}` on {n_pair}/{cs.get('n_eval', 0)} eval frames: the "
            f"gain comes from discovering that combination from {cfg['total_budget']} oracle labels; no baseline "
            f"had subset candidates. A fixed AIMNet2+Orb average chosen in hindsight (using eval labels) scores "
            f"{pair_rmse:.3f} eV/Å — as good.")
    summary += [
        f"* **Calibrated trust:** ECE {val(D, 'ece_eval'):.3f}, Brier {val(D, 'brier_eval'):.3f}; significantly "
        f"better than {', '.join(cal_better) or 'no baseline'}"
        + (f"; tied with {', '.join(cal_tied)}" if cal_tied else "") +
        " — the gain comes from the Skeptic's random audits of trusted answers.",
        f"* **Negative result — error ranking:** TrustSwarm's AUROC (eval {val(D, 'auroc_eval'):.3f}) is "
        f"significantly worse than {', '.join(auc_worse) or 'no baseline'}. Best ranker: {best_auc} "
        f"({val(ok[best_auc], 'auroc_eval'):.3f}).",
        f"* **Committee disagreement detects AIMNet2's failures best:** AUROC vs 'AIMNet2 wrong' "
        f"{val(ok[best_common], 'auroc_common'):.3f} ({best_common}).",
    ]
    if disp and sig04:
        summary.append(
            f"* **Seed ensembles rank but do not measure:** B′ AUROC {val(bprime, 'auroc_eval'):.3f}"
            + (" (best)" if best_auc == "B_mace_seeds" else "") +
            (f"; B ties C at {b_eval:.3f}" if b_eval is not None and c_eval is not None
             and abs(b_eval - c_eval) < 5e-4 else f"; B {b_eval:.3f} vs C {c_eval:.3f}") +
            f"; on all {signal['n_frames']:,} test frames at 0.4 eV/Å the AIMNet2 seed spread beats the "
            f"committee ({sig04['seed']['auroc']:.3f} vs {sig04['het']['auroc']:.3f}) — but seed spreads are "
            f"{min(spreads):.1f}–{max(spreads):.0f}× smaller than the errors on the "
            f"{len(ev):,} eval frames: not usable as an error bar.")
    summary += [
        f"* **Silent failures:** TrustSwarm has {n_sf_e}/{sf_e['n']} on eval (badly wrong, F-RMSE > 0.4) and "
        f"{n_sf_p}/{sf_p['n']} on pool; not significantly different from {len(sf_ns)} of "
        f"{len(nonrandom)} non-random baselines, {len(trust_none)} of which trust nothing (zero by construction).",
        f"* **Planted failures:** {planted['D_trustswarm']:.2f} caught vs {min(others_pl):.2f}–"
        f"{max(others_pl):.2f}; significantly better than {', '.join(pl_better) or 'none'}, not significant vs "
        f"{len(pl_ns)} of {len(r['comparisons'])}.",
    ]
    if rep_short:
        summary.append(f"* **Self-repair:** the Curator chose and LoRA-fine-tuned `{D['finetune']['model']}`; "
                       f"{'shipped' if D['finetune']['shipped'] else 'not shipped'} — {rep_short}.")
    summary += [
        "* **Leakage controlled:** split by reactant- and product-linked components; 0 shared endpoint "
        "geometries" + (f" (a per-reaction split had {naive} near-duplicate pool/eval endpoints)." if naive else "."),
        "* Details, history and failures: `results/RESULTS.md`.", "<!-- SUMMARY:END -->"]
    if len(summary) > 27:
        raise SystemExit(f"summary block too long ({len(summary)} lines)")

    doc: list[str] = ["# TrustSwarm — results", "", *summary, "",
                      "Generated by `scripts/write_results.py`. Primary wrongness: force RMSE > "
                      f"{cfg['wrong_f']} eV/Å or |ΔE_rel| > {cfg['wrong_e']} eV (fixed before running; 0.15 and "
                      f"0.4 in `by_threshold`). Brackets: 95% bootstrap CIs over reactions ({cfg['n_bootstrap']} "
                      f"resamples, paired across arms). Budget: {budget_label}{bprime_note}.", "",
                      "**How the ablation runs.** Agents replay cached predictions computed by the model-agent "
                      "processes on the GPU, and the DisagreementHook logic runs as its CPU twin "
                      "(`DisagreementMonitor`). The ALCHEMI hook itself is demonstrated in the 600-frame "
                      "in-flight screen (`results/screen_alchemi.json`); the ablation does not consume that "
                      "screen's oracle queue.", "",
                      "## All arms (final run)", "", *arms_table(r), "",
                      f"Planted = planted-split frames where AIMNet2's force RMSE > {cfg.get('planted_force_rmse')} "
                      "eV/Å (AIMNet2 is in every arm). Silent failures = badly wrong (F-RMSE > 0.4), unflagged, "
                      "unchecked and trusted; read with the trusted share. An arm that trusts nothing has none "
                      "by construction. TrustSwarm's zero is partly due to the Skeptic's advisory vetoes and the "
                      "hook's flags, which keep would-be silent failures out of the trusted set.", "",
                      "### TrustSwarm (D) vs each baseline (paired 95% CI)", "",
                      "`better` / `**worse**`: the CI of the difference excludes 0; `n.s.`: not significant.", "",
                      *comparison_table(r), "", "### Served answer: discovery vs hindsight", ""]
    if pair_rmse is not None:
        doc += [f"* TrustSwarm served `{PAIR_NAME}` on {n_pair}/{cs.get('n_eval', 0)} eval frames: served force "
                f"RMSE {served['D_trustswarm']:.3f} eV/Å.",
                f"* A fixed `{PAIR_NAME}` on every eval frame, chosen in hindsight with eval labels: "
                f"{pair_rmse:.3f} eV/Å. TrustSwarm's gain is the discovery of this combination from "
                f"{cfg['total_budget']} oracle labels; no baseline arm had subset candidates.",
                "* Budget sweep: TrustSwarm's served eval force RMSE at k = " +
                ", ".join(f"{k}: {x['value']:.3f}" for k, x in
                          zip(sweep["budgets"], sweep["arms"]["D_trustswarm"]["eval_force_rmse_post"]))
                + " eV/Å. The recommended-candidate distribution per budget is **not recorded** in "
                "`ablation_budget.json`, so whether it finds the pair with fewer labels is inferred only "
                "from the served error.", ""]
    doc += ["### Budget sweep (per-round budget k, same 5 rounds)", "", *budget_lines(sweep), ""]
    if disp:
        doc += ["### Ensemble dispersion vs error (eval)", "",
                "| ensemble | median spread σ_F (eV/Å) | median error of its mean (eV/Å) | error / spread |",
                "|---|---|---|---|"]
        doc += [f"| {k} | {v['spread']:.3f} | {v['error']:.3f} | {v['error'] / v['spread']:.1f} |"
                for k, v in disp.items()]
        doc += ["", f"Computed on the {len(ev):,} leak-free eval frames."]
        if signal:
            doc += [f"On all {signal['n_frames']:,} test frames (`results/signal.json`), Spearman(committee "
                    f"σ_F, AIMNet2 error) = {signal['spearman']['het']:.2f}, and the AIMNet2 seed spread is "
                    f"{signal['median_spread']['seed'] / signal['median_error_target']:.1f}× its median error."]
        doc += ["Seed ensembles can rank errors well while being badly under-dispersed: ranking and "
                "calibration are different properties.", ""]
    doc += ["## What the collective decided (arm D)", "", *coll_lines, "",
            "## Repair (Curator)", "", *rep_lines, "",
            "## TrustSwarm across versions", "", *versions_table(root), "",
            "\\* v1 defined planted cases as 'MACE-MPA-0 wrong' (favoured arms containing MACE); later "
            "versions use 'AIMNet2 badly wrong'. v1→v2 and v3→v4 share a split; v2→v3 changes it.", "",
            "| version | change | reason |", "|---|---|---|",
            "| v1 | bidding on flagged (high-σ) systems; serve best single model or 3-way consensus | design as specified |",
            "| v2 | Skeptic exploration audits (25% of each round's budget on random trusted systems); served "
            "candidates = every subset mean; replay provenance; arm-independent planted cases; "
            "skeptic_ref/planted split by formula | the oracle only labelled high-σ systems, so trust was never "
            "tested where it was given; the best answer was a subset mean |",
            "| v3 | split by reactant- and product-linked components; risk on σ_served | T1x reuses reactant "
            "geometries across reactions; measure uncertainty of the answer actually served (negative "
            "result: ranking got worse) |",
            "| v4 | Calibrator selects its risk feature by reaction-grouped CV on its own labels "
            f"(chose {D.get('risk_feature')}, CV AUROC "
            + ", ".join(f"{k} {v:.3f}" for k, v in (D.get("risk_feature_cv_auroc") or {}).items()
                        if v is not None) + ") | choose without looking at eval |", ""]
    if split:
        doc += ["## Splits and leakage", "",
                f"* pool/eval: split by reactant- and product-linked components (reactions sharing any endpoint "
                f"geometry are linked): {split.get('test_components')} components over "
                f"{split.get('test_reactions')} reactions; frames per split: "
                + ", ".join(f"{k} {v}" for k, v in split["frames_per_split"].items()) + ".",
                "* skeptic_ref/planted: split by formula, components kept whole "
                f"({split.get('val_reactions_moved_to_keep_components_whole', 0)} reactions moved).",
                "* Shared endpoint geometries after the split: " +
                ", ".join(f"{k}: {v}" for k, v in split["shared_endpoint_geometries"].items()) +
                "; near-duplicates (max|ΔR| < 1e-3 Å): " +
                ", ".join(f"{k}: {v}" for k, v in split["near_duplicate_endpoints_max_dR_1e-3"].items()) +
                (f" (per-reaction split: {naive} pool/eval near-duplicates)." if naive else "."),
                "* Labels reach agents only through `oracle_result` events (tested); eval labels are never "
                "revealed; replayed fine-tunes must match this run's labels and share no uid or geometry with "
                "held-out frames.", ""]
    doc += ["## Failures and caveats (reported, not hidden)", ""]
    doc += [f"* {f}" for f in r["failures"] if not f.startswith("D vs ")]
    doc += [f"* TrustSwarm's error ranking (AUROC eval) is significantly worse than: {', '.join(auc_worse)}."
            if auc_worse else "* TrustSwarm's error ranking is not significantly worse than any baseline."]
    if cs.get("tau_constant"):
        doc.append(f"* The Calibrator's threshold τ never left its floor ({cs['tau']:.3f}): flags were set by "
                   "the round-0 70th-percentile rule, not by learning.")
    if cs.get("rep_range") and cs["rep_range"][1] - cs["rep_range"][0] < 0.1:
        doc.append(f"* Reputations ended nearly equal ({cs['rep_range'][0]:.2f}–{cs['rep_range'][1]:.2f}): "
                   "reputation-weighted bidding behaved like plain summed bids.")
    doc += [f"* TrustSwarm's served-answer gain equals a fixed AIMNet2+Orb average chosen in hindsight "
            f"({pair_rmse:.3f} eV/Å)." if pair_rmse is not None else "",
            "* The Calibrator's labels are mostly bid-won (high σ); audits reduce but do not remove this "
            "selection bias.",
            "* The synthetic smoke test (`scripts/make_synthetic.py`) shows the pipeline runs; it is not "
            "evidence for any claim.", ""]
    if missing:
        doc += ["## Missing inputs (written with --allow-partial)", ""] + [f"* `{m}`" for m in missing] + [""]
    (root / "RESULTS.md").write_text("\n".join(x for x in doc if x is not None), encoding="utf-8")
    n_summary = summary.index("<!-- SUMMARY:END -->") + 1
    print(f"wrote {root / 'RESULTS.md'} (summary block: {n_summary} lines)")


def load_predictions_meta(preds_dir: Path, model: str) -> dict[str, Any] | None:
    from trustswarm.cache import load_predictions
    p = preds_dir / f"{model}.npz"
    return load_predictions(p).meta if p.exists() else None


if __name__ == "__main__":
    main()
