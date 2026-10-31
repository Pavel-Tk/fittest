"""Figures for the evolutionary loop, deck style of scripts/make_figures.py.

    .venv/Scripts/python scripts/make_evolution_figures.py [--evo results/evolution] [--seed 0] [--run-label A]

  results/fig/fig7_evolution.{png,svg}    served eval error per generation: evolved vs static
  results/fig/fig8_family_tree.{png,svg}  every member: parent, generation, fate
Numbers drawn are read from <evo>/generations.json and family_tree.json. The fig7 headline follows
<evo>/verdict.json: it says "beats the static swarm" only if the pre-registered rule holds for that seed run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location("mkfig", os.path.join(ROOT, "scripts", "make_figures.py"))
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)  # palette, rcParams, helpers
plt = mk.plt

STATUS_COLOR = {"alive": mk.ORANGE, "dead": mk.GRAYBAR, "stillborn": "#FFFFFF", "clone": "#FFFFFF"}
FAMILY_COLOR = {"mace": mk.MODEL_COLOR["mace_mpa0"], "aimnet2": mk.MODEL_COLOR["aimnet2"],
                "orb": mk.MODEL_COLOR["orb"]}


def _fit_width(fig, t, right=0.975, min_size=9.0):
    """Shrink a figure text until it ends inside the canvas (right edge at `right` of the width)."""
    r = fig.canvas.get_renderer()
    W = fig.bbox.width
    while t.get_window_extent(r).x1 > right * W and t.get_fontsize() > min_size:
        t.set_fontsize(t.get_fontsize() - 0.5)


def _title(fig, text, sub=None):
    """mk.takeaway, but shrunk to fit the canvas width."""
    n0 = len(fig.texts)
    mk.takeaway(fig, text, sub)
    for t in fig.texts[n0:]:
        _fit_width(fig, t)


def _footnote(fig, lines, width=106) -> float:
    """Wrapped footnote at the bottom; returns its top edge as a figure fraction (to place axes above it)."""
    text = "\n".join(textwrap.fill(ln, width) for ln in lines)
    t = fig.text(0.035, 0.022, text, ha="left", va="bottom", fontsize=10.5, color=mk.MUTED, linespacing=1.3)
    _fit_width(fig, t)
    return t.get_window_extent(fig.canvas.get_renderer()).y1 / fig.bbox.height


def _signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", "\u2212")


def _judge(d: dict) -> str:
    if not d["significant"]:
        return "n.s."
    return "better, sig." if d["diff"] < 0 else "worse, sig."


def _prefix(run_label, seed: str) -> str:
    return f"Run {run_label}, seed {seed}" if run_label else f"Seed {seed}"


def _failed_criteria(v: dict) -> list[str]:
    out = []
    if v.get("generations", 0) < 3:
        out.append("fewer than 3 generations")
    if not v.get("offspring_in_final_population"):
        out.append("no offspring alive at the end")
    if not v.get("force_better_than_gen0"):
        out.append("forces not sig. better than gen 0")
    if not v.get("force_better_than_static"):
        out.append("forces not sig. better than static")
    if not v.get("no_rel_e_regression_vs_static"):
        out.append("rel. energy regresses vs static")
    if not v.get("no_ece_regression_vs_static"):
        out.append("ECE regresses vs static")
    return out


def fig7(gen, verdict, seed: str, run_label=None):
    st = gen["states"]
    G = gen["config"]["generations"]
    ev = [st[f"evolved_s{seed}_g{g}"] for g in range(G)]
    sc = [st[f"static_s{seed}_g{g}"] for g in range(G)]
    labels = [s["labels"] for s in ev]
    fig = mk.new_fig()
    fig.canvas.draw()
    comp = gen["comparisons"][f"evolved_s{seed}"]["final_vs_static_same_budget"]
    v = verdict["per_run"][f"evolved_s{seed}"]
    holds = bool(v["holds"])  # the pre-registered rule for this run; the only basis for a "better" headline
    head = ("evolved swarm beats the static swarm" if holds
            else "evolution does not beat the static swarm")
    f, e = comp["force_rmse"], comp["rel_e_mae"]
    sub = (f"Final gen vs static: forces {_signed(f['diff'])} eV/\u00c5 ({_judge(f)}), "
           f"relative energy {_signed(e['diff'])} eV ({_judge(e)})")
    _title(fig, f"{_prefix(run_label, seed)}: {head}", sub)
    rule = ("Pre-registered rule holds for this run." if holds else
            "Pre-registered rule does not hold: " + "; ".join(_failed_criteria(v)) + ".")
    top = _footnote(fig, [rule, "x: generation / cumulative Transition1x labels. Static = same triage/oracle and "
                                "DFT budget, no breeding or pruning. CIs: 95% paired reaction bootstrap over eval."])
    y0 = top + 0.15  # room for the two-line generation/labels tick labels
    axes = [fig.add_axes([0.105, y0, 0.375, 0.745 - y0]), fig.add_axes([0.605, y0, 0.375, 0.745 - y0])]
    for ax, key, ylabel in ((axes[0], "force_rmse", "served force RMSE (eV/\u00c5)"),
                            (axes[1], "rel_e_mae", "served \u0394E_rel MAE (eV)")):
        for runs, col, name, off in ((sc, mk.GRAYBAR, "static", -0.06), (ev, mk.ORANGE, "evolved", 0.06)):
            x = [g + off for g in range(G)]
            y = [r[key]["value"] for r in runs]
            lo = [r[key]["value"] - r[key]["ci"][0] for r in runs]
            hi = [r[key]["ci"][1] - r[key]["value"] for r in runs]
            ax.errorbar(x, y, yerr=[lo, hi], color=col, lw=2.2, marker="o", ms=7, capsize=0,
                        elinewidth=2, label=name, zorder=3)
        ax.set_xticks(range(G))
        ax.set_xticklabels([f"g{g}\n{n}" for g, n in zip(range(G), labels)])
        ax.set_ylabel(ylabel, fontsize=12.5)
        mk.style_axis(ax)
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=2, fontsize=11.5,
                   borderaxespad=0.0, handlelength=1.6, columnspacing=1.2)
    mk.save(fig, "fig7_evolution")
    return {"evolved": [r["force_rmse"]["value"] for r in ev], "static": [r["force_rmse"]["value"] for r in sc]}


def fig8(tree, seed: str, G: int, run_label=None):
    members = tree[f"evolved_s{seed}"]
    fig = mk.new_fig()
    fig.canvas.draw()
    alive = [m["id"] for m in members if m["status"] == "alive"]
    _title(fig, f"{_prefix(run_label, seed)}: family tree of the swarm",
           f"final population: {', '.join(alive)}")
    ax = fig.add_axes([0.035, 0.22, 0.93, 0.60])
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
        ew = hp.get("energy_weight", hp.get("rel_weight")) if hp else None  # MACE: energy_weight; AIMNet2: rel_weight
        sub = (f"E:F {ew:g}:{hp.get('force_weight', 1.0):g}" if m["kind"] == "offspring" and hp and ew is not None
               else "")
        fate = {"alive": "", "dead": f"\u2020 g{m['died']}", "stillborn": "stillborn", "clone": "clone"}[m["status"]]
        second = " ".join(p for p in (sub, fate) if p)
        ax.text(x + 0.11, y, f"{m['id']}\n{second}" if second else m["id"], va="center", ha="left",
                fontsize=9.5, color=mk.INK if m["status"] == "alive" else mk.MUTED)
    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 0.45, max(ys) + 0.45)
    ax.set_xlim(-0.25, G - 1 + 0.75)
    ax.set_xticks(range(G))
    ax.set_xticklabels([f"generation {g}" for g in range(G)], fontsize=12.5)
    ax.set_yticks([])
    mk.style_axis(ax, ygrid=False)
    _footnote(fig, ["Filled: alive at the end; grey: pruned (no significant marginal contribution);",
                    "dashed: stillborn (failed the gate) or rejected clone. \u2020 gN: pruned at generation N."])
    mk.save(fig, "fig8_family_tree")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evo", default=os.path.join(ROOT, "results", "evolution"))
    ap.add_argument("--seed", default="0")
    ap.add_argument("--run-label", default=None, help="run name for the titles, e.g. A (MACE offspring) or B (AIMNet2)")
    a = ap.parse_args()
    gen = json.load(open(os.path.join(a.evo, "generations.json")))
    tree = json.load(open(os.path.join(a.evo, "family_tree.json")))
    verdict = json.load(open(os.path.join(a.evo, "verdict.json")))
    nums = {"fig7": fig7(gen, verdict, a.seed, a.run_label)}
    fig8(tree, a.seed, gen["config"]["generations"], a.run_label)
    with open(os.path.join(a.evo, "fig_numbers.json"), "w") as fh:
        json.dump(nums, fh, indent=1)


if __name__ == "__main__":
    main()
