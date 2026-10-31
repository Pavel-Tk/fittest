"""Tables + decision-rule verdict for results/evolution/SUMMARY.md (generated part).

    python scripts/write_evolution_summary.py [--evo results/evolution]

Writes ``results/evolution/summary_tables.md`` (included verbatim by SUMMARY.md) and
``results/evolution/verdict.json``.

Decision rule (fixed before the real run): the flagship "Survival of the fittest" holds iff,
in every seed run, the evolved swarm's served force RMSE on eval is significantly lower
(paired 95 % reaction-bootstrap CI of the difference below 0) both (i) at the final generation
vs its own generation 0 and (ii) at the final generation vs the static swarm at the same total
DFT budget, over >= 3 generations, with offspring alive in the final population, and without
a significant regression vs static in served rel-E MAE or ECE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = [("force_rmse", "served F-RMSE (eV/Å)"), ("rel_e_mae", "served ΔE MAE (eV)"),
        ("wrong_rate", "wrong rate"), ("ece", "ECE"), ("brier", "Brier"), ("auroc", "AUROC σ_F"),
        ("trusted_fraction", "trusted"), ("badly_wrong_trusted", "badly-wrong trusted"),
        ("silent_failures", "silent (trusted, unflagged)")]


def fmt(m):
    if m is None or m.get("value") is None:
        return "n/a"
    ci = m.get("ci")
    return f"{m['value']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{m['value']:.3f}"


def fmt_d(d):
    if d is None or d.get("diff") is None:
        return "n/a"
    ci = d.get("ci")
    s = f"{d['diff']:+.3f}"
    if ci:
        s += f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" + (" *" if d["significant"] else "")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evo", default="results/evolution")
    a = ap.parse_args()
    evo = Path(a.evo)
    gen = json.loads((evo / "generations.json").read_text())
    tree = json.loads((evo / "family_tree.json").read_text())
    G = gen["config"]["generations"]
    st = gen["states"]
    lines = []
    runs = sorted({k.rsplit("_g", 1)[0] for k in st})
    for run in runs:
        lines += [f"### {run}", "", "| gen | labels | population | " + " | ".join(n for _, n in KEYS) + " |",
                  "|" + "---|" * (3 + len(KEYS))]
        for g in range(G):
            s = st.get(f"{run}_g{g}")
            if s is None:
                continue
            lines.append(f"| {g} | {s['labels']} | {', '.join(s['population'])} | "
                         + " | ".join(fmt(s[k]) for k, _ in KEYS) + " |")
        lines.append("")
    lines += ["### Paired differences (a − b), 95 % reaction-bootstrap CI over eval; * = CI excludes 0", "",
              "| comparison | " + " | ".join(n for _, n in KEYS) + " |", "|" + "---|" * (1 + len(KEYS))]
    for run, c in gen["comparisons"].items():
        lines.append(f"| {run} final − own gen 0 | " + " | ".join(fmt_d(c['final_vs_gen0'][k]) for k, _ in KEYS) + " |")
        lines.append(f"| {run} final − static (same budget) | "
                     + " | ".join(fmt_d(c['final_vs_static_same_budget'][k]) for k, _ in KEYS) + " |")
        for g, d in sorted(c["per_generation_vs_static"].items(), key=lambda x: int(x[0])):
            lines.append(f"| {run} g{g} − static g{g} | " + " | ".join(fmt_d(d[k]) for k, _ in KEYS) + " |")
    lines.append("")
    for run, members in tree.items():
        lines += [f"### Family tree: {run}", "", "| id | parent | born | recipe (E:F, rank, lr, epochs, labels) | "
                  "n train | gate | status | died | role | fitness by generation |", "|---|---|---|---|---|---|---|---|---|---|"]
        for m in members:
            hp = m.get("hyperparams") or {}
            rec = (f"{hp.get('energy_weight', '')}:{hp.get('force_weight', '')}, r{hp.get('lora_rank', '')}, "
                   f"{hp.get('lr', '')}, {hp.get('epochs', '')}, {hp.get('label_fraction', '')}"
                   if m["kind"] == "offspring" else "foundation")
            gate = m.get("gate")
            if gate and gate.get("checks"):
                gate_s = "; ".join(f"{c['name'].replace('holdout_', '').replace('_not_worse', '')} "
                                   f"{c['value']:.3f}≤{c['limit']:.3f} {'ok' if c['passed'] else 'FAIL'}"
                                   for c in gate["checks"] if c.get("value") is not None and c.get("limit") is not None)
            else:
                gate_s = "—" if gate is None else f"error: {str(gate.get('error'))[:60]}"
            fit = ", ".join(f"g{g}: {v['fitness']:+.1f} ({v.get('role') or '—'})"
                            for g, v in m["fitness_per_generation"].items())
            lines.append(f"| {m['id']} | {m['parent'] or '—'} | {m['born']} | {rec} | {m['n_train_labels']} | "
                         f"{gate_s} | {m['status']} | {m['died'] if m['died'] is not None else '—'} | "
                         f"{m['role'] or '—'} | {fit} |")
        lines.append("")
    (evo / "summary_tables.md").write_text("\n".join(lines), encoding="utf-8")

    verdicts = {}
    for run, c in gen["comparisons"].items():
        if not run.startswith("evolved"):
            continue
        final = st[f"{run}_g{G - 1}"]
        offspring_alive = [m for m in final["population"] if m.startswith("g")]
        v0, vs = c["final_vs_gen0"], c["final_vs_static_same_budget"]
        better = lambda d: bool(d["significant"] and d["diff"] < 0)  # noqa: E731
        worse = lambda d: bool(d["significant"] and d["diff"] > 0)  # noqa: E731
        verdicts[run] = {
            "generations": G, "offspring_in_final_population": offspring_alive,
            "force_better_than_gen0": better(v0["force_rmse"]),
            "force_better_than_static": better(vs["force_rmse"]),
            "no_rel_e_regression_vs_static": not worse(vs["rel_e_mae"]),
            "no_ece_regression_vs_static": not worse(vs["ece"]),
        }
        verdicts[run]["holds"] = bool(G >= 3 and offspring_alive and all(
            verdicts[run][k] for k in ("force_better_than_gen0", "force_better_than_static",
                                       "no_rel_e_regression_vs_static", "no_ece_regression_vs_static")))
    out = {"rule": __doc__.split("Decision rule (fixed before the real run): ")[1].strip(),
           "per_run": verdicts, "flagship_holds": bool(verdicts) and all(v["holds"] for v in verdicts.values())}
    (evo / "verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
