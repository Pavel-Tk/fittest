"""Numbers of the README / deck storyline that no other summary file holds, recomputed from committed results.

    python scripts/story_numbers.py            # writes results/story_numbers.json and prints it

* critic vs answerer value of each foundation model: every fitness evaluation of a base model in every
  evolution run (``family_tree.json``, ``fitness_per_generation``; the invalid ``partial_run1`` and the
  ``fresh_*`` reruns, which duplicate their originals, are excluded);
* run F (GPU, 12 seeds): the family trees (children bred, their fates, foundation models retired) and the
  absolute served force RMSE of the static and evolved councils on eval and on the fresh frames, against
  the best static-study baseline on the same eval frames (``results/ablation.json``).
"""
import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
F_GPU = ["evolution_f/fresh_gpu1/b800", "evolution_f/fresh_gpu1/b800_b",
         "evolution_f/fresh_gpu2/b800", "evolution_f/fresh_gpu2/b800_b"]


def _p(*a):
    return os.path.join(ROOT, *a)


def critic_vs_answerer():
    vals = defaultdict(list)
    for f in sorted(glob.glob(_p("**", "family_tree.json"), recursive=True)):
        rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
        if "partial_run1" in rel or "/fresh_" in "/" + rel:
            continue
        for arm, members in json.load(open(f)).items():
            if not arm.startswith("evolved"):
                continue
            for m in members:
                if m["kind"] == "base":
                    vals[m["id"]].extend((m.get("fitness_per_generation") or {}).values())
    return {fam: {"n_checks": len(v),
                  "critic_delta_auroc_mean": round(st.mean(x["critic"] for x in v), 4),
                  "critic_ci_above_0": sum(1 for x in v if x["critic_ci"] and x["critic_ci"][0] > 0),
                  "answerer_delta_mean": round(st.mean(x["answerer"] for x in v), 3)}
            for fam, v in sorted(vals.items())}


def run_f_lineages():
    c, retired = Counter(), Counter()
    for d in F_GPU:
        for members in json.load(open(_p(d.replace("fresh_", ""), "family_tree.json"))).values():
            for m in members:
                if m["kind"] == "base":
                    c["foundation_models"] += 1
                    retired[m["family"]] += m["status"] == "dead"
                else:
                    c["children"] += 1
                    c[{"alive": "in_final_council", "dead": "pruned"}.get(m["status"], m["status"])] += 1
    out = dict(c)
    out["foundation_models_retired"] = sum(retired.values())
    out["retired_by_family"] = dict(retired)
    return out


def run_f_served():
    rows = defaultdict(list)
    for d in F_GPU:
        g = json.load(open(_p(d, "generations.json")))
        for split, key in (("eval", "states"), ("fresh", "states_fresh")):
            s = g[key]
            for seed in sorted({k.split("_")[1] for k in s}):
                rows[split].append((s[f"static_{seed}_g3"]["force_rmse"]["value"],
                                    s[f"evolved_{seed}_g3"]["force_rmse"]["value"]))
    out = {}
    for split, r in rows.items():
        stat, evo = st.mean(x[0] for x in r), st.mean(x[1] for x in r)
        out[split] = {"seeds": len(r), "static_force_rmse": round(stat, 4), "evolved_force_rmse": round(evo, 4),
                      "evolved_vs_static_pct": round(100 * (evo / stat - 1), 1)}
    return out


def best_baseline_eval():
    """Served force RMSE on eval of the static-study baselines (their answers use no labels)."""
    arms = json.load(open(_p("ablation.json")))["arms"]
    return {n: round(arms[n]["metrics"]["eval_force_rmse_post"]["value"], 4) for n in ("A_single", "B_seed")}


if __name__ == "__main__":
    res = {"critic_vs_answerer": critic_vs_answerer(), "run_f_gpu_lineages": run_f_lineages(),
           "run_f_gpu_served": run_f_served()}
    base = best_baseline_eval()
    res["static_study_baselines_eval"] = base
    if "B_seed" in base:
        e = res["run_f_gpu_served"]["eval"]["evolved_force_rmse"]
        res["run_f_gpu_served"]["eval"]["evolved_vs_best_baseline_pct"] = round(100 * (e / base["B_seed"] - 1), 1)
    json.dump(res, open(_p("story_numbers.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
