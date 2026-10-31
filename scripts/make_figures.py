"""Slide-quality figures for the Fittest pitch (16:9, 1600x900 px PNG + SVG).

Reads only repo data:
  data/frames.pkl, data/preds/{mace_mpa0,aimnet2,aimnet2_m0..m3,orb,mace_lora_curated}.npz,
  results/neb.json, results/ablation.json

    .venv/Scripts/python scripts/make_figures.py      # -> results/fig/fig{1..6}_*.{png,svg} + numbers.json

Splits are the leak-free ones (``assign_leakfree_splits(frames, seed=0)``, as the ablation uses).
Figures 1-2 use all held-out T1x test frames (pool ∪ eval); figures 4 and 6 use eval only;
figure 3 reads the batched-NEB results; figure 5 reads whatever results/ablation.json holds now;
figure 6 is skipped (with a message) if the fine-tune predictions or their acceptance report are
missing. Every number drawn is computed here and written to results/fig/numbers.json.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from fittest.cache import load_frames, load_predictions  # noqa: E402
from fittest.data import assign_leakfree_splits  # noqa: E402
from fittest.config import WRONG_F  # noqa: E402
from fittest.signals import force_spread  # noqa: E402

OUT = os.path.join(ROOT, "results", "fig")

# ---- deck palette -----------------------------------------------------------------------
BG = "#F7F6F2"
INK = "#14213D"
MUTED = "#5E636B"
GRID = "#D9D6CE"
ORANGE = "#C4561B"   # disagreement / collective
BLUE = "#1F5FA8"     # truth / DFT
GRAYBAR = "#BDB8AD"  # de-emphasised bars
MODEL_COLOR = {"mace_mpa0": "#8A6D3B", "aimnet2": "#1F5FA8", "orb": "#6B4C9A",
               "seed": "#7FA6D6", "het": "#C4561B"}
MODEL_NAME = {"mace_mpa0": "MACE-MPA-0", "aimnet2": "AIMNet2", "orb": "Orb"}
SHORT_NAME = {"mace_mpa0": "MACE", "aimnet2": "AIMNet2", "orb": "Orb"}
# AIMNet2 blue vs Orb purple fail the colour-blind separation check, so every model also
# gets its own marker shape and a direct label.
MODEL_MARKER = {"mace_mpa0": "s", "aimnet2": "o", "orb": "D"}

HET = ["mace_mpa0", "aimnet2", "orb"]
SEED = ["aimnet2_m0", "aimnet2_m1", "aimnet2_m2", "aimnet2_m3"]
ROLES = ["reactant", "product", "path", "ts"]
ROLE_LABEL = {"reactant": "Reactant", "product": "Product", "path": "Path", "ts": "Transition\nstate"}

W_IN, H_IN, DPI = 8.0, 4.5, 200  # 1600 x 900 px

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "lines.solid_capstyle": "round",
    "svg.fonttype": "none",
})


# ---- helpers -----------------------------------------------------------------------------
def new_fig():
    return plt.figure(figsize=(W_IN, H_IN), dpi=DPI)


def style_axis(ax, ygrid=True, xgrid=False):
    ax.tick_params(length=0, pad=6)
    ax.yaxis.grid(ygrid)
    ax.xaxis.grid(xgrid)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def takeaway(fig, text, sub=None):
    fig.text(0.035, 0.955, text, ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
    if sub:
        fig.text(0.035, 0.875, sub, ha="left", va="top", fontsize=13, color=MUTED)


def footnote(fig, text):
    fig.text(0.035, 0.022, text, ha="left", va="bottom", fontsize=10.5, color=MUTED, linespacing=1.3)


def check_layout(fig, name):
    """Warn about text that leaves the canvas or overlaps other text (the validator can't see layout)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        texts += [t for t, v in zip(ax.get_xticklabels(), ax.get_xticks()) if x0 - 1e-9 <= v <= x1 + 1e-9]
        texts += [t for t, v in zip(ax.get_yticklabels(), ax.get_yticks()) if y0 - 1e-9 <= v <= y1 + 1e-9]
        if ax.get_legend():
            texts += list(ax.get_legend().get_texts())
    for lg in fig.legends:
        texts += list(lg.get_texts())
    boxes = [(t, t.get_window_extent(r)) for t in texts if t.get_visible() and t.get_text().strip()]
    ok = True
    for t, b in boxes:
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            print(f"  [{name}] off-canvas: {t.get_text()!r}")
            ok = False
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                print(f"  [{name}] overlap: {boxes[i][0].get_text()!r} <-> {boxes[j][0].get_text()!r}")
                ok = False
    print(f"  [{name}] layout {'OK' if ok else 'has issues'}")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    check_layout(fig, name)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=DPI)
    fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def load_all():
    frames, split_report = assign_leakfree_splits(load_frames(os.path.join(ROOT, "data", "frames.pkl")), seed=0)
    test = [f for f in frames if f.split in ("pool", "eval")]  # held-out T1x test reactions
    preds = {m: load_predictions(os.path.join(ROOT, "data", "preds", f"{m}.npz")) for m in HET + SEED}
    return test, preds, split_report


# ---- figure 1: error by role ------------------------------------------------------------------
def fig1(test, preds, nums):
    roles = np.array([f.role for f in test])
    err = {m: np.array([rmse(preds[m].forces[f.uid], f.F) for f in test]) for m in HET}
    stats = {m: {r: np.percentile(err[m][roles == r], [25, 50, 75]) for r in ROLES} for m in HET}
    n_role = {r: int((roles == r).sum()) for r in ROLES}
    ratio = stats["aimnet2"]["ts"][1] / stats["aimnet2"]["reactant"][1]

    fig = new_fig()
    ax = fig.add_axes([0.105, 0.30, 0.70, 0.515])
    style_axis(ax)
    x = np.arange(len(ROLES))
    off = {"mace_mpa0": -0.2, "aimnet2": 0.0, "orb": 0.2}
    for m in HET:
        q = np.array([stats[m][r] for r in ROLES])
        xs = x + off[m]
        ax.plot(xs, q[:, 1], color=MODEL_COLOR[m], lw=1.4, alpha=0.45, zorder=2)
        ax.vlines(xs, q[:, 0], q[:, 2], color=MODEL_COLOR[m], lw=2.2, zorder=3)
        ax.scatter(xs, q[:, 1], s=95, marker=MODEL_MARKER[m], color=MODEL_COLOR[m],
                   edgecolor=BG, linewidth=2, zorder=4)

    # direct labels right of the TS column (legend + identity), nudged apart with leader lines
    lab_y = {m: stats[m]["ts"][1] for m in HET}
    order = sorted(HET, key=lambda m: lab_y[m])
    ys = [lab_y[m] for m in order]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.085)
    for m, yv in zip(order, ys):
        ax.plot([x[-1] + off[m] + 0.07, x[-1] + 0.42], [lab_y[m], yv], color=GRID, lw=1, zorder=1)
        ax.scatter([x[-1] + 0.47], [yv], s=60, marker=MODEL_MARKER[m], color=MODEL_COLOR[m],
                   edgecolor=BG, linewidth=1.5, zorder=4, clip_on=False)
        ax.text(x[-1] + 0.56, yv, MODEL_NAME[m], va="center", ha="left", fontsize=13, color=INK)

    # AIMNet2 TS / reactant callout in the empty lower-right region
    ax.text(2.30, 0.215, "AIMNet2 at the TS:", ha="left", va="top", fontsize=13.5, color=INK)
    ax.text(2.30, 0.13, f"{ratio:.1f}× its reactant error", ha="left", va="top", fontsize=14.5,
            fontweight="bold", color=INK)
    ax.annotate("", xy=(x[-1] - 0.03, stats["aimnet2"]["ts"][0] - 0.01), xytext=(2.9, 0.225),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, mutation_scale=14, shrinkA=0, shrinkB=2))

    ax.set_xticks(x, [ROLE_LABEL[r] for r in ROLES])
    for lbl, r in zip(ax.get_xticklabels(), ROLES):
        lbl.set_color(INK if r == "ts" else MUTED)
        lbl.set_fontweight("bold" if r == "ts" else "normal")
    ax.set_xlim(-0.45, len(ROLES) - 0.55)
    ax.set_ylim(0, 0.9)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.set_ylabel("Force error (eV/Å)")

    takeaway(fig, "Errors climb toward the transition state for every model",
             "Median per-frame force error; whiskers = interquartile range")
    footnote(fig, f"Held-out Transition1x test: {len({f.reaction for f in test})} reactions, {len(test)} frames "
                  f"({n_role['ts']} TS, {n_role['path']} path images).\n"
                  "Level of theory differs: models trained on PBE (MACE-MPA-0, Orb) or ωB97M-D3 (AIMNet2);\n"
                  "DFT labels are ωB97x/6-31G(d). The errors include that gap.")
    save(fig, "fig1_ts_failure")
    nums["fig1"] = {
        "n_frames": len(test), "n_reactions": len({f.reaction for f in test}), "n_by_role": n_role,
        "median_force_rmse_by_role": {MODEL_NAME[m]: {r: round(float(stats[m][r][1]), 3) for r in ROLES} for m in HET},
        "iqr_by_role": {MODEL_NAME[m]: {r: [round(float(stats[m][r][0]), 3), round(float(stats[m][r][2]), 3)]
                                        for r in ROLES} for m in HET},
        "ts_over_reactant": {MODEL_NAME[m]: round(float(stats[m]["ts"][1] / stats[m]["reactant"][1]), 2) for m in HET},
    }


# ---- figure 2: spread vs error ----------------------------------------------------------------
def fig2(test, preds, nums):
    y_err = np.array([rmse(preds["aimnet2"].forces[f.uid], f.F) for f in test])
    s_het = np.array([force_spread(np.stack([preds[m].forces[f.uid] for m in HET]))[0] for f in test])
    s_seed = np.array([force_spread(np.stack([preds[m].forces[f.uid] for m in SEED]))[0] for f in test])
    res = {}
    for key, s in (("het", s_het), ("seed", s_seed)):
        res[key] = {"median_spread_over_error": float(np.median(s / y_err)),
                    "ratio_of_medians": float(np.median(s) / np.median(y_err)),
                    "spearman": float(spearmanr(s, y_err)[0]),
                    "median_spread": float(np.median(s))}

    nb = 14  # quantile bins of the error, equal frame counts
    edges = np.quantile(y_err, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.digitize(y_err, edges[1:-1]), 0, nb - 1)
    xc = np.array([np.median(y_err[idx == b]) for b in range(nb)])

    fig = new_fig()
    ax = fig.add_axes([0.10, 0.255, 0.865, 0.56])
    style_axis(ax, ygrid=True, xgrid=True)
    ax.spines["bottom"].set_visible(False)
    xlim, ylim = 0.9, 1.1
    ax.plot([0, ylim], [0, ylim], color=MUTED, lw=1.3, ls=(0, (5, 4)), zorder=2)
    ax.text(0.74, 0.755, "spread = error", color=MUTED, fontsize=12.5, ha="left", va="bottom", rotation=45,
            rotation_mode="anchor", transform_rotates_text=True)
    for s, col in ((s_het, MODEL_COLOR["het"]), (s_seed, MODEL_COLOR["seed"])):
        ax.scatter(y_err, s, s=4, color=col, alpha=0.10, linewidths=0, zorder=1, rasterized=True)
        q = np.array([np.percentile(s[idx == b], [25, 50, 75]) for b in range(nb)])
        ax.fill_between(xc, q[:, 0], q[:, 2], color=col, alpha=0.22, linewidth=0, zorder=3)
        ax.plot(xc, q[:, 1], color=col, lw=2.6, zorder=4)
        ax.scatter(xc, q[:, 1], s=34, color=col, edgecolor=BG, linewidth=1.5, zorder=5)
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_xticks(np.arange(0.2, 0.81, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("AIMNet2 force error vs DFT (eV/Å)")
    ax.set_ylabel("Spread σ_F (eV/Å)")

    # direct labels next to each cloud (text in ink; the swatch carries identity)
    bbox = dict(boxstyle="square,pad=0.15", facecolor=BG, edgecolor="none", alpha=0.85)

    def label_block(x0, y0, col, name, r):
        ax.scatter([x0 + 0.008], [y0 - 0.035], s=110, marker="s", color=col, zorder=6, clip_on=False)
        ax.text(x0 + 0.03, y0, name, fontsize=13.5, fontweight="bold", va="top", ha="left", bbox=bbox, zorder=7)
        ax.text(x0 + 0.03, y0 - 0.085,
                f"spread ≈ {r['median_spread_over_error']:.1f} × error,  ρ = {r['spearman']:.2f}",
                fontsize=13, va="top", ha="left", color=INK, bbox=bbox, zorder=7)

    label_block(0.012, 1.07, MODEL_COLOR["het"], "Independent models: MACE-MPA-0 + AIMNet2 + Orb", res["het"])
    label_block(0.465, 0.205, MODEL_COLOR["seed"], "AIMNet2 seed ensemble ×4", res["seed"])

    takeaway(fig, f"Seed spread is {res['seed']['median_spread_over_error']:.1f}× the error; "
                  f"independent, {res['het']['median_spread_over_error']:.1f}×",
             "Lines: median spread in 14 equal-count error bins; bands: IQR; dots: frames")
    footnote(fig, f"Held-out Transition1x test, {len(test)} frames; axes clipped to the bulk of the data; ρ = Spearman.\n"
                  "Ratio = per-frame median of σ_F ÷ AIMNet2 force RMSE; below 1 = overconfident.")
    save(fig, "fig2_spread_vs_error")
    nums["fig2"] = {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in res.items()}
    nums["fig2"]["median_aimnet2_force_rmse"] = round(float(np.median(y_err)), 3)
    nums["fig2"]["n_frames"] = len(test)


# ---- figure 3: NEB bands ----------------------------------------------------------------------
def fig3(nums):
    with open(os.path.join(ROOT, "results", "neb.json"), encoding="utf-8") as fh:
        neb = json.load(fh)
    R = neb["reactions"]
    rx = {r["reaction_id"]: r for r in R}
    picks = [("C3HNO2/rxn0109", "Hit"), ("C2H2N2O/rxn3122", "Miss")]

    # 8-reaction summary, computed from the per-reaction entries
    n = len(R)
    rho = np.array([r["spearman_sigma_vs_mean_force_rmse"] for r in R])
    near = sum(abs(r["argmax_sigma_dft_band"] - r["dft_ts_index"]) <= 1 for r in R)
    at_ts = sum(r["argmax_sigma_dft_band"] == r["dft_ts_index"] for r in R)
    same_peak = sum(r["argmax_sigma_dft_band"] == r["argmax_mean_force_rmse_dft_band"] for r in R)
    n_conv = sum(bool(r["neb_converged"]) for r in R)
    n_img = len(R[0]["sigma_dft_band"])
    pooled = float(spearmanr(np.concatenate([r["sigma_dft_band"] for r in R]),
                             np.concatenate([r["mean_force_rmse_dft_band"] for r in R]))[0])

    fig = new_fig()
    axes = [fig.add_axes([0.085, 0.245, 0.28, 0.44]), fig.add_axes([0.405, 0.245, 0.28, 0.44])]
    vals = np.concatenate([np.r_[rx[k]["sigma_dft_band"], rx[k]["mean_force_rmse_dft_band"]] for k, _ in picks])
    ylo, yhi = 0.5, np.ceil(vals.max() * 10) / 10 + 0.15
    for ax, (rid, tag) in zip(axes, picks):
        r = rx[rid]
        sig = np.array(r["sigma_dft_band"])
        er = np.array(r["mean_force_rmse_dft_band"])
        xs = np.arange(len(sig))
        ts = r["dft_ts_index"]
        style_axis(ax)
        ax.axvspan(ts - 0.45, ts + 0.45, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        ax.text(ts, yhi - 0.02, "DFT TS", ha="center", va="top", fontsize=12, color=INK)
        for arr, col in ((er, BLUE), (sig, ORANGE)):
            ax.plot(xs, arr, color=col, lw=2.4, zorder=3)
            ax.scatter(xs, arr, s=38, color=col, edgecolor=BG, linewidth=1.5, zorder=4)
            k = int(np.argmax(arr))
            ax.scatter([k], [arr[k]], s=170, facecolor="none", edgecolor=col, linewidth=1.8, zorder=5)
        ax.set_xticks(range(len(sig)))
        ax.set_xticklabels([str(i) if i in (0, ts, len(sig) - 1) else "" for i in range(len(sig))])
        ax.set_xlim(-0.5, len(sig) - 0.5)
        ax.set_ylim(ylo, yhi)
        ax.set_yticks(np.arange(0.6, yhi - 0.05, 0.2))
        ax.set_xlabel("NEB image")
        ax.set_title(f"{tag}: {rid.split('/')[1]}", loc="left", fontsize=14, fontweight="bold", pad=27)
        ax.text(0.0, 1.03, f"ρ(σ_F, error) = {r['spearman_sigma_vs_mean_force_rmse']:.2f}",
                transform=ax.transAxes, fontsize=12.5, color=MUTED, va="bottom")
    axes[0].set_ylabel("eV/Å")
    axes[1].tick_params(labelleft=False)
    r0 = rx[picks[0][0]]
    axes[0].text(8.5, r0["sigma_dft_band"][8] + 0.05, "σ_F", color=INK, fontsize=13, ha="center", va="bottom")
    axes[0].text(8.5, r0["mean_force_rmse_dft_band"][8] + 0.04, "error", color=INK, fontsize=13,
                 ha="center", va="bottom")

    handles = [Line2D([0], [0], color=ORANGE, lw=2.6, marker="o", ms=6, mec=BG),
               Line2D([0], [0], color=BLUE, lw=2.6, marker="o", ms=6, mec=BG),
               Line2D([0], [0], color=MUTED, lw=0, marker="o", ms=11, mfc="none", mec=MUTED)]
    fig.legend(handles, ["Committee spread σ_F", "Mean force error vs DFT", "Peak"],
               loc="upper left", bbox_to_anchor=(0.025, 0.905), ncol=3, frameon=False,
               handlelength=1.6, columnspacing=1.4, fontsize=13)

    sx = 0.725
    fig.text(sx, 0.795, f"All {n} NEB reactions", fontsize=14, fontweight="bold", va="top")
    stats = [(f"{np.median(rho):.2f}", "median ρ(σ_F, error)"),
             (f"{near}/{n}", "σ_F peak ≤ 1 image\nfrom the DFT TS"),
             (f"{same_peak}/{n}", "σ_F peak = error peak"),
             (f"{pooled:.2f}", f"pooled ρ, {n * n_img} images")]
    yy = 0.725
    for big, small in stats:
        fig.text(sx, yy, big, fontsize=18, fontweight="bold", va="top", color=INK)
        fig.text(sx, yy - 0.065, small, fontsize=12, va="top", color=MUTED, linespacing=1.1)
        yy -= 0.13 + 0.045 * small.count("\n")

    takeaway(fig, "Disagreement tracks the true error along the path")
    footnote(fig, f"σ_F: MACE-MPA-0 + AIMNet2 + Orb on the {n_img}-image DFT band. Error: mean of their force RMSE\n"
                  f"vs DFT. ρ: Spearman over a reaction's images. {n_conv}/{n} AIMNet2-driven NEBs converged.")
    save(fig, "fig3_neb")
    nums["fig3"] = {
        "picks": {rid: {"spearman": round(rx[rid]["spearman_sigma_vs_mean_force_rmse"], 3),
                        "dft_ts_index": rx[rid]["dft_ts_index"],
                        "argmax_sigma": rx[rid]["argmax_sigma_dft_band"],
                        "argmax_error": rx[rid]["argmax_mean_force_rmse_dft_band"]} for rid, _ in picks},
        "n_reactions": n, "median_spearman": round(float(np.median(rho)), 3),
        "per_reaction_spearman": {r["reaction_id"]: round(r["spearman_sigma_vs_mean_force_rmse"], 3) for r in R},
        "sigma_peak_within_1_of_ts": near, "sigma_peak_at_ts": at_ts, "sigma_peak_eq_error_peak": same_peak,
        "pooled_spearman": round(pooled, 3), "n_converged": n_conv,
    }


# ---- figure 4: best subset --------------------------------------------------------------------
def fig4(test, preds, nums):
    ev = [f for f in test if f.split == "eval"]
    cands = {SHORT_NAME[m] if m != "mace_mpa0" else "MACE-MPA-0": [m] for m in HET}
    cands["AIMNet2 seeds ×4"] = SEED
    for k in (2, 3):
        for sub in itertools.combinations(HET, k):
            cands[" + ".join(SHORT_NAME[m] for m in sub)] = list(sub)
    rows = []
    for name, ms in cands.items():
        e = np.array([rmse(np.mean([preds[m].forces[f.uid] for m in ms], axis=0), f.F) for f in ev])
        rows.append((name, float((e > WRONG_F).mean()), float(np.median(e))))
    rows.sort(key=lambda r: r[1], reverse=True)  # barh draws bottom-up -> best on top
    best = "AIMNet2 + Orb"
    b = next(r for r in rows if r[0] == best)
    a2 = next(r for r in rows if r[0] == "AIMNet2")
    a3 = next(r for r in rows if r[0] == "MACE + AIMNet2 + Orb")

    fig = new_fig()
    axA = fig.add_axes([0.31, 0.135, 0.38, 0.60])
    axB = fig.add_axes([0.745, 0.135, 0.215, 0.60])
    y = np.arange(len(rows))
    for ax, ci, fmt, xm in ((axA, 1, "{:.0%}", 1.0), (axB, 2, "{:.2f}", 0.56)):
        style_axis(ax, ygrid=False, xgrid=True)
        ax.spines["bottom"].set_visible(False)
        v = [r[ci] for r in rows]
        ax.barh(y, v, height=0.6, color=[ORANGE if r[0] == best else GRAYBAR for r in rows], zorder=2)
        for yi, vi, r in zip(y, v, rows):
            txt = fmt.format(vi)
            if txt == "100%" and vi < 1:  # don't round 99.7% up to 100%
                txt = f"{vi:.1%}"
            ax.text(vi + xm * 0.02, yi, txt, va="center", ha="left", fontsize=13,
                    color=INK, fontweight="bold" if r[0] == best else "normal")
        ax.set_xlim(0, xm * 1.2)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(y)
    axA.set_yticklabels([r[0] for r in rows])
    for lbl, r in zip(axA.get_yticklabels(), rows):
        lbl.set_color(INK if r[0] == best else MUTED)
        lbl.set_fontweight("bold" if r[0] == best else "normal")
    axB.set_yticklabels([])
    axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100%"])
    axB.set_xticks([0, 0.2, 0.4])
    axA.set_title(f"Frames wrong (RMSE > {WRONG_F} eV/Å)", loc="left", fontsize=13.5, pad=10)
    axB.set_title("Median RMSE (eV/Å)", loc="left", fontsize=13.5, pad=10)

    takeaway(fig, f"AIMNet2 + Orb cuts wrong frames from {a2[1]:.0%} to {b[1]:.0%}",
             f"Adding MACE back undoes it ({a3[1]:.0%}). Committee = per-atom mean of member forces.")
    footnote(fig, f"Held-out eval split (leak-free): {len(ev)} frames, {len({f.reaction for f in ev})} reactions. "
                  "Force RMSE vs DFT. Sorted by fraction wrong.")
    save(fig, "fig4_best_subset")
    nums["fig4"] = {"n_frames": len(ev), "n_reactions": len({f.reaction for f in ev}), "wrong_f": WRONG_F,
                    "rows_best_first": [{"name": r[0], "frac_wrong": round(r[1], 3), "median_force_rmse": round(r[2], 3)}
                                        for r in rows[::-1]]}


# ---- figure 5: ablation -----------------------------------------------------------------------
ARMS = ["A_single", "A_random", "B_seed", "B_mace_seeds", "C_collective", "C_audit", "D_fittest"]
ARM_LABEL = {"A_single": "A  AIMNet2 alone", "A_random": "A  random triage", "B_seed": "B  AIMNet2 seeds",
             "B_mace_seeds": "B′ MACE seeds ×4",
             "C_collective": "C  collective", "C_audit": "C  collective + audit", "D_fittest": "D  Fittest"}
ARM_COLOR = {"A_single": "#A8A397", "A_random": "#DAD6CC", "B_seed": "#A8A397", "B_mace_seeds": "#A8A397",
             "C_collective": "#A8A397",
             "C_audit": "#A8A397", "D_fittest": ORANGE}


def _verdict(cmp, metric, others):
    """'better' if D is significantly better than every other arm, 'worse' if significantly worse than
    any, else 'mixed'. Uses the ablation's own paired-bootstrap comparisons."""
    res = [cmp.get(f"D_vs_{a}", {}).get(metric) for a in others]
    res = [r for r in res if r]
    if not res:
        return "unknown"
    if any(r.get("d_significantly_worse") for r in res):
        return "worse"
    if all(r.get("d_significantly_better") for r in res):
        return "better"
    return "tied"


def fig5(nums):
    path = os.path.join(ROOT, "results", "ablation.json")
    if not os.path.exists(path):
        print("  [fig5_ablation] skipped: results/ablation.json missing")
        return
    with open(path, encoding="utf-8") as fh:
        ab = json.load(fh)
    arms = [a for a in ARMS if ab["arms"].get(a, {}).get("status") == "ok" and "metrics" in ab["arms"][a]]
    if "D_fittest" not in arms:
        print("  [fig5_ablation] skipped: D_fittest has no metrics")
        return
    M = {a: ab["arms"][a]["metrics"] for a in arms}
    cmp = ab.get("comparisons", {})
    others = [a for a in arms if a != "D_fittest"]
    panels = [("eval_force_rmse_post", "Served error ↓\n(eV/Å)", "{:.2f}", 0.0),
              ("auroc_eval", "AUROC ↑\n(from chance)", "{:.2f}", 0.5),
              ("ece_eval", "ECE ↓\n(calibration)", "{:.2f}", 0.0)]

    fig = new_fig()
    lefts, widths = [0.255, 0.475, 0.685], [0.19, 0.18, 0.16]
    rows = arms[::-1]  # barh draws bottom-up: first arm on top
    y = np.arange(len(rows))
    for (key, title, fmt, base), l, w in zip(panels, lefts, widths):
        ax = fig.add_axes([l, 0.215, w, 0.50])
        style_axis(ax, ygrid=False, xgrid=True)
        ax.spines["bottom"].set_visible(False)
        v = np.array([M[a][key]["value"] for a in rows])
        lo = np.array([M[a][key]["ci"][0] for a in rows])
        hi = np.array([M[a][key]["ci"][1] for a in rows])
        ax.barh(y, v - base, left=base, height=0.6, color=[ARM_COLOR[a] for a in rows], zorder=2)
        ax.hlines(y, lo, hi, color=INK, lw=1.4, zorder=3)
        for yy in y:
            ax.vlines([lo[yy], hi[yy]], yy - 0.12, yy + 0.12, color=INK, lw=1.4, zorder=3)
        top = base + (hi.max() - base) * 1.42
        for yy, a in enumerate(rows):
            ax.text(hi[yy] + (top - base) * 0.03, yy, fmt.format(v[yy]), va="center", ha="left", fontsize=12.5,
                    color=INK, fontweight="bold" if a == "D_fittest" else "normal")
        ax.set_xlim(base, top)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(y)
        if key == "auroc_eval":
            ax.set_xticks([0.5, 0.7, 0.9])
        else:
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.set_title(title, loc="left", fontsize=13, pad=10, linespacing=1.15)
        if l == lefts[0]:
            ax.set_yticklabels([ARM_LABEL[a] for a in rows])
            for lbl, a in zip(ax.get_yticklabels(), rows):
                lbl.set_color(INK if a == "D_fittest" else MUTED)
                lbl.set_fontweight("bold" if a == "D_fittest" else "normal")
                lbl.set_fontsize(13)
        else:
            ax.set_yticklabels([])
    # trusted share column (fraction of eval answers the arm marks as trusted)
    tx = 0.885
    fig.text(tx, fig.axes[0].title.get_window_extent(fig.canvas.get_renderer()).y0 / fig.bbox.height,
             "Trusted\n(share)", fontsize=13, ha="left", va="bottom", linespacing=1.15)
    ax0 = fig.axes[0]
    for yy, a in enumerate(rows):
        tf = M[a].get("trusted_fraction_eval", {}).get("value")
        ypix = ax0.transData.transform((0, yy))[1]
        yfig = fig.transFigure.inverted().transform((0, ypix))[1]
        fig.text(tx, yfig, "–" if tf is None else f"{tf:.0%}", fontsize=12.5, va="center", ha="left",
                 color=INK, fontweight="bold" if a == "D_fittest" else "normal")

    d = M["D_fittest"]
    vr, vc, va = (_verdict(cmp, k, others) for k in ("eval_force_rmse_post", "ece_eval", "auroc_eval"))
    p_err = {"better": "serves better answers", "tied": "serves answers on par", "worse": "serves worse answers"}.get(vr, "")
    p_cal = {"better": "calibrated trust", "tied": "calibrated trust", "worse": "poorer calibration"}.get(vc, "")
    p_auc = {"worse": "it ranks its own errors worse", "tied": "it ranks its own errors on par",
             "better": "and ranks its own errors best"}.get(va, "")
    best_other_auc = [M[a]["auroc_eval"]["value"] for a in others if a != "A_random"]
    takeaway(fig, f"Fittest {p_err} and {p_cal}",
             f"…but {p_auc}: AUROC {d['auroc_eval']['value']:.2f} vs "
             f"{min(best_other_auc):.2f}–{max(best_other_auc):.2f}"
             if va == "worse" else (p_auc[0].upper() + p_auc[1:] if p_auc else None))
    data = ab.get("data", {})
    footnote(fig, f"Held-out eval (leak-free): {data.get('n_eval', '?')} frames, {data.get('n_eval_reactions', '?')} reactions; "
                  f"{ab['config'].get('total_budget', '?')} oracle labels per arm. Whiskers: 95% CI.\n"
                  "AUROC: does the arm's warning score rank its own wrong answers? ECE: calibration of its P(wrong).\n"
                  "Trusted: share of eval answers the arm serves as trusted.")
    save(fig, "fig5_ablation")
    nums["fig5"] = {"source_schema": ab.get("schema"), "n_eval": data.get("n_eval"),
                    "verdict_vs_other_arms": {"eval_force_rmse_post": vr, "ece_eval": vc, "auroc_eval": va},
                    "arms": {a: {k: {"value": round(M[a][k]["value"], 3), "ci": [round(c, 3) for c in M[a][k]["ci"]]}
                                 for k in ("eval_force_rmse_post", "auroc_eval", "ece_eval", "trusted_fraction_eval",
                                           "eval_wrong_rate_post") if k in M[a]} for a in arms}}


# ---- figure 6: fine-tune repair ---------------------------------------------------------------
CMP_SYM = {"<=": "≤", ">=": "≥"}

def fig6(test, preds, nums):
    path = os.path.join(ROOT, "data", "preds", "mace_lora_curated.npz")
    if not os.path.exists(path):
        print("  [fig6_repair] skipped: data/preds/mace_lora_curated.npz missing")
        return
    cur = load_predictions(path)
    rep_ = cur.meta.get("acceptance_report")
    if not rep_ or "holdout_metrics" not in rep_:
        print("  [fig6_repair] skipped: no acceptance_report/holdout_metrics in the npz meta")
        return
    base_m = cur.meta.get("base_model", "mace_mpa0")
    base = preds.get(base_m) or load_predictions(os.path.join(ROOT, "data", "preds", f"{base_m}.npz"))
    used = set(cur.meta.get("train_uids", [])) | set(cur.meta.get("holdout_uids", []))
    seen_rxn = {f.reaction for f in test if f.uid in used}
    ev_all = [f for f in test if f.split == "eval" and f.uid in cur and f.uid in base]
    # never score on a reaction the fine-tune trained or held out on (any of its frames)
    ev = [f for f in ev_all if f.reaction not in seen_rxn]
    n_rxn_excl = len({f.reaction for f in ev_all if f.reaction in seen_rxn})
    roles = np.array([f.role for f in ev])
    e_base = np.array([rmse(base.forces[f.uid], f.F) for f in ev])
    e_cur = np.array([rmse(cur.forces[f.uid], f.F) for f in ev])
    hb, ha = rep_["holdout_metrics"]["before"], rep_["holdout_metrics"]["after"]
    verdict = str(rep_.get("verdict", "?")).upper()
    shipped = bool(cur.meta.get("shipped", rep_.get("accepted", False)))

    fig = new_fig()
    # left: acceptance report on the fine-tune's own hold-out
    axL = fig.add_axes([0.075, 0.23, 0.32, 0.43])
    style_axis(axL)
    mets = [("force_rmse", "Force RMSE\n(eV/Å)"), ("rel_energy_mae", "Rel. energy\nMAE (eV)")]
    xL = np.arange(len(mets))
    wB = 0.34
    for k, (key, _) in enumerate(mets):
        for j, (vals, col) in enumerate(((hb, MODEL_COLOR["mace_mpa0"]), (ha, ORANGE))):
            v = vals.get(key)
            if v is None:
                continue
            xx = xL[k] + (j - 0.5) * (wB + 0.04)
            axL.bar(xx, v, width=wB, color=col, zorder=2)
            axL.text(xx, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=12.5, color=INK)
    axL.set_xticks(xL, [m[1] for m in mets])
    axL.set_ylim(0, max(hb.get("force_rmse", 0), ha.get("force_rmse", 0), hb.get("rel_energy_mae", 0),
                        ha.get("rel_energy_mae", 0)) * 1.25)
    axL.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axL.set_title(f"Fine-tune hold-out (n = {hb.get('n_frames', '?')})", loc="left", fontsize=13.5, pad=10)

    # right: base vs fine-tuned on leak-free eval, by role
    axR = fig.add_axes([0.53, 0.23, 0.43, 0.43])
    style_axis(axR)
    xr = np.arange(len(ROLES))
    for j, (e, col, mk, name) in enumerate(((e_base, MODEL_COLOR["mace_mpa0"], "s", "MACE-MPA-0"),
                                             (e_cur, ORANGE, "o", "+ LoRA fine-tune"))):
        q = np.array([np.percentile(e[roles == r], [25, 50, 75]) if (roles == r).any() else [np.nan] * 3
                      for r in ROLES])
        xs = xr + (j - 0.5) * 0.26
        axR.vlines(xs, q[:, 0], q[:, 2], color=col, lw=2.2, zorder=3)
        axR.scatter(xs, q[:, 1], s=90, marker=mk, color=col, edgecolor=BG, linewidth=2, zorder=4)
    axR.set_xticks(xr, ["Reactant", "Product", "Path", "TS"])
    axR.set_xlim(-0.55, len(ROLES) - 0.45)
    axR.set_ylim(0, max(np.nanpercentile(e_base, 80), np.nanpercentile(e_cur, 80)) * 1.25)
    axR.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axR.set_title(f"Eval force RMSE, eV/Å (n = {len(ev)})", loc="left", fontsize=13.5, pad=10)
    handles = [Line2D([0], [0], color=MODEL_COLOR["mace_mpa0"], lw=0, marker="s", ms=10),
               Line2D([0], [0], color=ORANGE, lw=0, marker="o", ms=10)]
    fig.legend(handles, ["MACE-MPA-0 (before)", "LoRA on oracle labels (after)"], loc="upper left",
               bbox_to_anchor=(0.025, 0.835), ncol=2, frameon=False, fontsize=13, handletextpad=0.4,
               columnspacing=1.6)

    checks = rep_.get("checks", [])
    failed = [c for c in checks if not c.get("passed", True)]
    chk_txt = f"{len(failed)} of {len(checks)} checks failed" if failed else "all checks passed"
    med_b, med_c = float(np.median(e_base)), float(np.median(e_cur))
    f_dir = "cuts" if ha["force_rmse"] < hb["force_rmse"] else "raises"
    gate = "passes its gate" if shipped else "fails its gate"
    takeaway(fig, f"Fine-tune {f_dir} force error, but {gate}" if not shipped or f_dir == "raises"
             else f"Fine-tune {f_dir} force error and {gate}")
    e_b, e_a = hb.get("rel_energy_mae"), ha.get("rel_energy_mae")
    fig.text(0.035, 0.875, f"Gate verdict: {verdict}" + (f"; hold-out rel. energy MAE {e_b:.2f} → {e_a:.2f} eV"
                                                          if e_b is not None and e_a is not None else ""),
             ha="left", va="top", fontsize=13, color=MUTED)
    footnote(fig, f"Gate {'passed' if shipped else 'failed (not shipped)'}{': ' + chk_txt if chk_txt else ''}.\n"
                  f"Hold-out shares no geometry with training. Eval: leak-free held-out reactions. Whiskers = IQR.")
    save(fig, "fig6_repair")
    nums["fig6"] = {"verdict": verdict, "shipped": shipped, "checks": checks,
                    "holdout_before": hb, "holdout_after": ha,
                    "eval_n": len(ev), "eval_frames_excluded": len(ev_all) - len(ev),
                    "eval_reactions_excluded": n_rxn_excl,
                    "eval_median_force_rmse": {"base": round(med_b, 3), "finetuned": round(med_c, 3)},
                    "eval_median_by_role": {r: {"base": round(float(np.median(e_base[roles == r])), 3),
                                                "finetuned": round(float(np.median(e_cur[roles == r])), 3)}
                                            for r in ROLES if (roles == r).any()}}


def main():
    import argparse

    argparse.ArgumentParser(description="Rebuild results/fig/* from data/frames.pkl, data/preds/*.npz "
                            "and results/*.json (data/ comes from scripts/predict_pool.py; not in git).").parse_args()
    if not os.path.exists(os.path.join(ROOT, "data", "frames.pkl")):
        raise SystemExit("data/frames.pkl not found: run scripts/predict_pool.py on a GPU first "
                         "(see docs/REPRODUCE.md). The committed figures in results/fig/ are the reference.")
    test, preds, split_report = load_all()
    nums: dict = {"split_report": split_report}
    fig1(test, preds, nums)
    fig2(test, preds, nums)
    fig3(nums)
    fig4(test, preds, nums)
    fig5(nums)
    fig6(test, preds, nums)
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(nums, fh, indent=2, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
