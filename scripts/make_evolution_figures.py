"""Figures for the evolutionary loop, deck style of scripts/make_figures.py.

    .venv/Scripts/python scripts/make_evolution_figures.py [--evo results/evolution] [--seed 0]

  results/fig/fig7_evolution.{png,svg}    served eval error per generation: evolved vs static
  results/fig/fig8_family_tree.{png,svg}  every member: parent, generation, fate
Numbers drawn are read from results/evolution/generations.json and family_tree.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location("mkfig", os.path.join(ROOT, "scripts", "make_figures.py"))
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)  # palette, rcParams, helpers
plt = mk.plt

STATUS_COLOR = {"alive": mk.ORANGE, "dead": mk.GRAYBAR, "stillborn": "#FFFFFF", "clone": "#FFFFFF"}
FAMILY_COLOR = {"mace": mk.MODEL_COLOR["mace_mpa0"], "aimnet2": mk.MODEL_COLOR["aimnet2"],
                "orb": mk.MODEL_COLOR["orb"]}


def fig7(gen, seed: str):
    st = gen["states"]
    G = gen["config"]["generations"]
    ev = [st[f"evolved_s{seed}_g{g}"] for g in range(G)]
    sc = [st[f"static_s{seed}_g{g}"] for g in range(G)]
    labels = [s["labels"] for s in ev]
    fig = mk.new_fig()
    comp = gen["comparisons"][f"evolved_s{seed}"]["final_vs_static_same_budget"]
    d = comp["force_rmse"]
    verdict = "better than" if d["significant"] and d["diff"] < 0 else (
        "worse than" if d["significant"] else "not distinguishable from")
    mk.takeaway(fig, f"Evolved swarm is {verdict} the static swarm",
                f"Served force RMSE on held-out eval, same DFT budget per generation; 95% paired CI")
    axes = [fig.add_axes([0.08, 0.17, 0.40, 0.58]), fig.add_axes([0.58, 0.17, 0.40, 0.58])]
    for ax, key, ylabel in ((axes[0], "force_rmse", "served force RMSE (eV/Å)"),
                            (axes[1], "rel_e_mae", "served ΔE_rel MAE (eV)")):
        for runs, col, name, off in ((sc, mk.GRAYBAR, "static", -0.06), (ev, mk.ORANGE, "evolved", 0.06)):
            x = [g + off for g in range(G)]
            y = [r[key]["value"] for r in runs]
            lo = [r[key]["value"] - r[key]["ci"][0] for r in runs]
            hi = [r[key]["ci"][1] - r[key]["value"] for r in runs]
            ax.errorbar(x, y, yerr=[lo, hi], color=col, lw=2.2, marker="o", ms=7, capsize=0,
                        elinewidth=2, label=name, zorder=3)
        ax.set_xticks(range(G))
        ax.set_xticklabels([f"g{g}\n{n}" for g, n in zip(range(G), labels)])
        ax.set_ylabel(ylabel)
        mk.style_axis(ax)
    axes[0].legend(frameon=False, loc="upper right")
    mk.footnote(fig, "x: generation and cumulative Transition1x labels. Static = same triage/oracle, no breeding or "
                     "pruning. CIs: reaction bootstrap over eval.")
    mk.save(fig, "fig7_evolution")
    return {"evolved": [r["force_rmse"]["value"] for r in ev], "static": [r["force_rmse"]["value"] for r in sc]}


def fig8(tree, seed: str, G: int):
    members = tree[f"evolved_s{seed}"]
    fig = mk.new_fig()
    alive = [m["id"] for m in members if m["status"] == "alive"]
    mk.takeaway(fig, "Family tree of the swarm", f"final population: {', '.join(alive)}")
    ax = fig.add_axes([0.06, 0.12, 0.90, 0.68])
    by_gen: dict[int, list] = {}
    for m in members:
        by_gen.setdefault(m["generation"], []).append(m)
    pos = {}
    for g, ms in by_gen.items():
        for i, m in enumerate(ms):
            pos[m["id"]] = (g, i - (len(ms) - 1) / 2)
    for m in members:
        if m["parent"] in pos:
            (x0, y0), (x1, y1) = pos[m["parent"]], pos[m["id"]]
            ax.plot([x0, x1], [y0, y1], color=mk.GRID, lw=1.6, zorder=1)
    for m in members:
        x, y = pos[m["id"]]
        edge = FAMILY_COLOR.get(m["family"], mk.MUTED)
        face = STATUS_COLOR[m["status"]] if m["status"] != "alive" else edge
        ls = "--" if m["status"] in ("stillborn", "clone") else "-"
        ax.scatter([x], [y], s=520, color=face, edgecolor=edge, linewidth=2.2, linestyle=ls, zorder=3)
        hp = m.get("hyperparams") or {}
        sub = f"E:F {hp['energy_weight']:g}:{hp['force_weight']:g}" if m["kind"] == "offspring" and hp else ""
        fate = {"alive": "", "dead": f"† g{m['died']}", "stillborn": "stillborn", "clone": "clone"}[m["status"]]
        ax.text(x + 0.12, y, f"{m['id']}\n{sub} {fate}".strip(), va="center", ha="left", fontsize=9.5,
                color=mk.INK if m["status"] == "alive" else mk.MUTED)
    ax.set_xlim(-0.3, G - 0.2)
    ax.set_xticks(range(G))
    ax.set_xticklabels([f"generation {g}" for g in range(G)])
    ax.set_yticks([])
    mk.style_axis(ax, ygrid=False)
    mk.footnote(fig, "Filled: alive at the end; grey: pruned (no significant marginal contribution); dashed: "
                     "stillborn (failed the gate) or rejected clone.")
    mk.save(fig, "fig8_family_tree")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evo", default=os.path.join(ROOT, "results", "evolution"))
    ap.add_argument("--seed", default="0")
    a = ap.parse_args()
    gen = json.load(open(os.path.join(a.evo, "generations.json")))
    tree = json.load(open(os.path.join(a.evo, "family_tree.json")))
    nums = {"fig7": fig7(gen, a.seed)}
    fig8(tree, a.seed, gen["config"]["generations"])
    with open(os.path.join(a.evo, "fig_numbers.json"), "w") as fh:
        json.dump(nums, fh, indent=1)


if __name__ == "__main__":
    main()
