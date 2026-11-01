"""EXPLORATORY diagnosis figure (results/evolution/DIAGNOSIS.md): served force RMSE Δ vs static for every
three-family (GPU) run C-style evolved / no-selection final state, grouped by whether an AIMNet2
specialist is in the final population. Reads only saved generations.json files."""
import json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
GPU = ["evolution_c/b200", "evolution_c/b400", "evolution_c/b800", "evolution_c/b800_s1",
       "evolution_c/expl_gpu_ext_seeds2-5/b800", "evolution_c/expl_gpu_ext_seeds2-5/b800_s4",
       "evolution_c/expl_gpu_ablation_children3/b800", "evolution_c/expl_gpu_ablation_explore/b800",
       "evolution_c/expl_gpu_ablation_gatetol/b800", "evolution_c/expl_gpu_ablation_protect/b800",
       "evolution_c/expl_gpu_v2_combined/b400", "evolution_c/expl_gpu_v2_combined/b800",
       "evolution_d/gpu/b800", "evolution_d/gpu/b800_s9"]
groups = {0: [], 1: [], 2: []}
for d in GPU:
    J = json.loads((R / d / "generations.json").read_text())
    G = J["config"]["generations"]
    for name, c in J["comparisons"].items():
        if name.startswith("static"):
            continue
        pop = J["states"][f"{name}_g{G-1}"]["population"]
        has_aim = any(re.match(r"g\d_aimnet2_", m) for m in pop)
        f = c["final_vs_static"]["force_rmse"]
        k = 2 if name.startswith("noselect") else int(has_aim)
        groups[k].append((f["diff"], f["ci"], f["significant"]))
labels = ["evolved, no AIMNet2\nspecialist in final pop.", "evolved, AIMNet2\nspecialist in final pop.",
          "no-selection\n(always has one)"]
colors = ["#eb6834", "#2a78d6", "#1baf7a"]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.axhline(0, color="#52514e", lw=1)
for k, pts in groups.items():
    pts = sorted(pts)
    n = len(pts)
    for i, (dv, ci, sig) in enumerate(pts):
        x = k + (i - (n - 1) / 2) * min(0.035, 0.6 / max(n, 1))
        ax.plot([x, x], ci, color=colors[k], lw=1, alpha=0.6)
        ax.plot(x, dv, "o", ms=6, mfc=colors[k] if sig else "white", mec=colors[k], mew=1.5)
    worse = sum(1 for d, _, s in pts if s and d > 0)
    ax.text(k, 0.052, f"{worse} of {n} worse", ha="center", fontsize=9, color="#0b0b0b")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.06, 0.06)
ax.set_ylabel("served force RMSE, final − static [eV/Å]")
ax.grid(axis="y", alpha=0.3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("EXPLORATORY: three-family runs (GPU), budget 200–800; filled = paired 95% CI excludes 0", fontsize=8.5)
fig.tight_layout()
out = R / "fig" / "fig10_diagnosis_aimnet2_specialist.png"
fig.savefig(out, dpi=150)
print("wrote", out, {k: len(v) for k, v in groups.items()})
