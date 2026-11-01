"""Generate Fittest deck slides (Slides artifact format) into ../deck/project/slides."""
import json, random, pathlib, sys
HERE = pathlib.Path(__file__).parent
OUT = HERE.parent / "deck" / "project" / "slides"
INK, INK2, BONE, BONE2 = "#0E1310", "#18201B", "#F1ECE1", "#E6DFD0"
EMBER, EMBER_D, TEAL = "#F26A21", "#B9420E", "#6CC3B5"
MUTE_D, MUTE_L, GRID_L = "#8F9A92", "#5F6760", "#B9B2A3"
SERIF = "'Instrument Serif', Georgia, serif"
SANS = "Geist, 'Helvetica Neue', Arial, sans-serif"
MONO = "'Geist Mono', 'Courier New', monospace"
CFG = json.loads((HERE / "cfg.json").read_text()) if (HERE / "cfg.json").exists() else {}


def sec(id_, bg, fg, body, notes, pad="128px 128px 160px", extra=""):
    return (f'<section id="{id_}" data-transition="fade" style="background:{bg};color:{fg};font-family:{SANS};'
            f'padding:{pad};display:flex;flex-direction:column;gap:48px{extra}">\n{body}\n<aside>{notes}</aside>\n</section>\n')


def kicker(n, txt, color):
    return (f'<p style="font-family:{MONO};font-size:24px;letter-spacing:4px;text-transform:uppercase;'
            f'color:{color}">{n} · {txt}</p>')


def footer(txt, color):
    return (f'<p style="position:absolute;left:128px;bottom:64px;width:1664px;font-family:{MONO};'
            f'font-size:24px;color:{color}">{txt}</p>')


def write(id_, html):
    (OUT / f"{id_}.html").write_text(html, encoding="utf-8")


# ---------- cover
def cover():
    rnd = random.Random(7)
    lines = []

    def branch(x, y, depth, spread):
        if depth == 0:
            alive = rnd.random() < 0.45
            c = EMBER if alive else "#3A453E"
            lines.append(f'<circle cx="{x}" cy="{y}" r="{8 if alive else 5}" fill="{c}"/>')
            return
        for dy in (-spread, spread):
            nx, ny = x + 130, y + dy
            lines.append(f'<path d="M{x} {y}V{ny}H{nx}" fill="none" stroke="#3E4A42" stroke-width="3"/>')
            branch(nx, ny, depth - 1, spread / 2)

    branch(20, 300, 4, 140)
    svg = (f'<svg aria-label="A branching family tree: a few lineages survive (orange), most die out" '
           f'viewBox="0 0 560 600" style="position:absolute;left:1240px;top:220px;width:560px;height:600px">'
           f'{"".join(lines)}</svg>')
    body = f'''{svg}
<p style="font-family:{MONO};font-size:24px;letter-spacing:4px;text-transform:uppercase;color:{MUTE_D}">ScienceClaw 2026 · NVIDIA ALCHEMI challenge</p>
<div style="display:flex;flex-direction:column;gap:24px">
<h1 style="font-family:{SERIF};font-size:320px;font-weight:400;line-height:0.9;letter-spacing:-6px;color:{BONE}">Fittest</h1>
<p style="font-family:{SERIF};font-style:italic;font-size:64px;line-height:1.1;color:{EMBER};width:1100px">Survival of the fittest, in materials science.</p>
</div>
<div style="display:flex;justify-content:space-between;align-items:end">
<p style="font-size:28px;color:#C9C3B6">Pavel Tkachyk</p>
<p style="font-family:{MONO};font-size:26px;color:#C9C3B6">github.com/Pavel-Tk/fittest</p>
</div>'''
    notes = ("Talking points: Fittest is a swarm of machine-learned interatomic potentials competing for a fixed DFT budget. "
             "DFT goes where the models disagree; models that add something to the swarm survive; offspring are bred on the DFT "
             "labels and must pass a gate. It runs on the NVIDIA ALCHEMI Toolkit and anyone can call it through one MCP server. "
             "Breeding is experimental; the pre-registered test says it does not yet beat the static swarm; the live service "
             "runs the static swarm.")
    write("cover", sec("cover", INK, BONE, body, notes, pad="128px", extra=";justify-content:space-between"))


# ---------- 01 abundant
def abundant():
    cols, rows, step = 88, 64, 10
    assert cols * rows == 5632
    picks = random.Random(2026).sample(range(cols * rows), 200)
    rowpaths = "".join(f"M5 {5 + r * step}H{cols * step}" for r in range(rows))
    dots = (f'<path d="{rowpaths}" stroke="{GRID_L}" stroke-width="3.6" stroke-linecap="round" '
            f'stroke-dasharray="0 {step}" fill="none"/>')
    em = "".join(f'<circle cx="{5 + (i % cols) * step}" cy="{5 + (i // cols) * step}" r="4.2"/>' for i in picks)
    svg = (f'<svg aria-label="5,632 grey dots, one per structure; 200 orange dots, one per DFT label" '
           f'viewBox="0 0 {cols * step} {rows * step}" style="width:{cols * step}px;height:{rows * step}px">'
           f'{dots}<g fill="{EMBER_D}">{em}</g></svg>')
    body = f'''{kicker("01", "The problem", EMBER_D)}
<div style="display:flex;gap:56px;align-items:flex-start">
<div style="width:736px;display:flex;flex-direction:column;gap:40px">
<h2 style="font-family:{SERIF};font-size:104px;font-weight:400;line-height:1.0;letter-spacing:-2px">Intelligence is abundant.<br><i style="color:{EMBER_D}">Ground truth is not.</i></h2>
<p style="font-size:30px;line-height:1.4;color:#2E3631">Three independent models answered all <b>5,632</b> structures. The swarm could afford <b style="color:{EMBER_D}">200</b> DFT labels.</p>
</div>
{svg}
</div>
{footer("Each dot: one Transition1x structure in this study · orange: one DFT label (the budget per arm, 3.6%)", MUTE_L)}'''
    notes = ("Talking points: machine-learned potentials made predictions nearly free; the reference calculation (DFT) and lab "
             "time did not get cheaper. In this study three foundation models with different architectures and training sets predicted every structure, and every "
             "arm of the experiment got the same 200 DFT labels. The question is where to spend them, and which answers to trust "
             "without one. Labels are Transition1x lookups, not live DFT.")
    write("abundant", sec("abundant", BONE, INK, body, notes))


# ---------- 02 silent
def silent():
    vals = [("Reactant", 0.22), ("Path", 0.287), ("TS", 0.461), ("Product", 0.251)]
    W, H, base, scale = 940, 470, 400, 360 / 0.6
    bw, gap, x0 = 150, 70, 110
    parts = []
    for t in (0, 0.2, 0.4, 0.6):
        y = base - t * scale
        parts.append(f'<line x1="{x0 - 20}" x2="{W}" y1="{y}" y2="{y}" stroke="#2C3630" stroke-width="2"/>')
        parts.append(f'<text x="{x0 - 32}" y="{y + 9}" text-anchor="end" font-family="Geist Mono, monospace" '
                     f'font-size="24" fill="{MUTE_D}">{t:.1f}</text>')
    for i, (lab, v) in enumerate(vals):
        x = x0 + 20 + i * (bw + gap)
        h = v * scale
        c = EMBER if lab == "TS" else "#4E5C53"
        parts.append(f'<rect x="{x}" y="{base - h}" width="{bw}" height="{h}" fill="{c}"/>')
        parts.append(f'<text x="{x + bw / 2}" y="{base - h - 16}" text-anchor="middle" font-family="Geist Mono, monospace" '
                     f'font-size="28" fill="{BONE}">{v:.2f}</text>')
        parts.append(f'<text x="{x + bw / 2}" y="{base + 44}" text-anchor="middle" font-family="Geist, sans-serif" '
                     f'font-size="26" fill="#C9C3B6">{lab}</text>')
    svg = (f'<svg aria-label="AIMNet2 median force error by structure role: reactant 0.22, path 0.29, transition state 0.46, '
           f'product 0.25 eV per angstrom" viewBox="0 0 {W} {H}" style="width:{W}px;height:{H}px">{"".join(parts)}</svg>')
    body = f'''{kicker("02", "Why it is hard", EMBER)}
<h2 style="font-family:{SERIF};font-size:96px;font-weight:400;line-height:1.05;letter-spacing:-2px;width:1500px">Models fail silently <i style="color:{EMBER}">where reactions are decided.</i></h2>
<div style="display:flex;gap:80px;align-items:flex-end">
<div style="width:620px;display:flex;flex-direction:column;gap:16px">
<p style="font-family:{SERIF};font-size:220px;line-height:0.9;color:{EMBER}">2.1×</p>
<p style="font-size:30px;line-height:1.4;color:#D8D2C4">AIMNet2's force error at the transition state vs. at the reactant: the one geometry that sets the barrier.</p>
</div>
{svg}
</div>
{footer("Median force RMSE, eV/Å · AIMNet2 · 287 held-out Transition1x reactions · results/fig/numbers.json (fig1)", MUTE_D)}'''
    notes = ("Talking points: the best single model here, AIMNet2, is 2.1 times worse at transition states than at reactants, "
             "and it gives no error bar. Transition states decide reaction rates, so a screen that trusts it is wrong "
             "exactly where it matters. MACE and Orb degrade too (about 1.3 times).")
    write("silent", sec("silent", INK, BONE, body, notes))


# ---------- 03 disagree
def disagree():
    scale = 700 / 0.3
    e, s = 0.278, 0.008
    barsA = (f'<svg aria-label="Error 0.278 versus spread 0.008 eV per angstrom, drawn to scale" viewBox="0 0 760 120" '
             f'style="width:760px;height:120px">'
             f'<rect x="0" y="8" width="{e * scale:.0f}" height="36" fill="#2E3631"/>'
             f'<text x="{e * scale + 14:.0f}" y="36" font-family="Geist Mono, monospace" font-size="24" fill="#2E3631">error</text>'
             f'<rect x="0" y="72" width="{max(s * scale, 3):.0f}" height="36" fill="{EMBER_D}"/>'
             f'<text x="{max(s * scale, 3) + 14:.0f}" y="100" font-family="Geist Mono, monospace" font-size="24" '
             f'fill="{EMBER_D}">spread</text></svg>')
    x = lambda a: (a - 0.5) / 0.5 * 700
    barsB = (f'<svg aria-label="AUROC scale from 0.5 (chance) to 1.0 with a marker at 0.917" viewBox="0 0 760 120" '
             f'style="width:760px;height:120px">'
             f'<rect x="0" y="40" width="700" height="12" fill="#CFC7B6"/>'
             f'<rect x="0" y="40" width="{x(0.917):.0f}" height="12" fill="{EMBER_D}"/>'
             f'<circle cx="{x(0.917):.0f}" cy="46" r="16" fill="{EMBER_D}"/>'
             f'<text x="0" y="100" font-family="Geist Mono, monospace" font-size="24" fill="{MUTE_L}">0.5 = chance</text>'
             f'<text x="700" y="100" text-anchor="end" font-family="Geist Mono, monospace" font-size="24" fill="{MUTE_L}">1.0</text></svg>')

    def card(top, h3, big, bigc, txt, svg):
        return f'''<div style="flex:1;display:flex;flex-direction:column;gap:20px;border-top:4px solid {top};padding:32px 0 0 0">
<h3 style="font-size:32px;font-weight:600">{h3}</h3>
<p style="font-family:{SERIF};font-size:160px;line-height:0.9;color:{bigc}">{big}</p>
<p style="font-size:28px;line-height:1.4;color:#2E3631;width:760px">{txt}</p>
{svg}
</div>'''
    body = f'''{kicker("03", "The signal", EMBER_D)}
<h2 style="font-family:{SERIF};font-size:96px;font-weight:400;line-height:1.05;letter-spacing:-2px;width:1600px">Disagreement is free data. <i style="color:{EMBER_D}">Independent models also get its size right.</i></h2>
<div style="display:flex;gap:96px">
{card(INK, "Copies of one model", "36×", INK, "Four LoRA seeds of MACE rank errors well but under-report them: their spread is 36× smaller than their error (AIMNet2 seeds: 2.4×).", barsA)}
{card(EMBER_D, "Three independent models", "0.92", EMBER_D, "Their spread flags where AIMNet2 is wrong (AUROC 0.917) on 1,485 held-out structures.", barsB)}
</div>
{footer("Force RMSE vs member spread, eV/Å (0.278 vs 0.008); AUROC vs “AIMNet2 wrong” · eval split · results/RESULTS.md", MUTE_L)}'''
    notes = ("Talking points: when models are genuinely different, their disagreement is an error signal that costs no labels. "
             "MACE, AIMNet2 and Orb have different architectures and training sets; their spread ranks AIMNet2's "
             "failures at AUROC 0.92. Fine-tuned copies of one model share its mistakes: four LoRA seeds of MACE disagree 36 times "
             "less than they are wrong. So diversity in the population is not decoration; it is the sensor.")
    write("disagree", sec("disagree", BONE, INK, body, notes))


# ---------- 06 yours
def yours():
    n, path = (HERE.parent / "qr_path.txt").read_text().split("\n", 1)
    n = int(n)
    b = 3
    qr = (f'<svg aria-label="QR code linking to fittest-mcp.westus2.cloudapp.azure.com" viewBox="{-b} {-b} {n + 2 * b} {n + 2 * b}" '
          f'style="width:400px;height:400px"><rect x="{-b}" y="{-b}" width="{n + 2 * b}" height="{n + 2 * b}" fill="{BONE}"/>'
          f'<path d="{path}" fill="{INK}" shape-rendering="crispEdges"/></svg>')
    users = CFG.get("users")
    if isinstance(users, int) and users >= 1:
        users_line = f'Already used by <span style="color:{EMBER}">{users:,}</span> user{"s" if users != 1 else ""}.'
    elif users == 0:
        users_line = f'Open now. <span style="color:{EMBER}">Be the first user.</span>'
    else:
        users_line = "Open now."
    body = f'''{kicker("06", "Use it", EMBER)}
<h2 style="font-family:{SERIF};font-size:120px;font-weight:400;line-height:1.0;letter-spacing:-2px">Survival of the fittest, <i style="color:{EMBER}">now yours.</i></h2>
<div style="display:flex;gap:96px;align-items:center">
<div style="flex:1;display:flex;flex-direction:column;gap:36px">
<p style="font-size:32px;line-height:1.4;color:#D8D2C4;width:1000px">One MCP server any agent can call. Send a structure; get the answer to trust, or “send this one to DFT”.</p>
<p style="font-family:{MONO};font-size:34px;color:{EMBER}">fittest-mcp.westus2.cloudapp.azure.com/mcp</p>
<!-- user count: https://fittest-mcp.westus2.cloudapp.azure.com/stats "users" via deckgen/cfg.json = {users} -->
<p style="font-family:{SERIF};font-size:88px;line-height:1.0;color:{BONE}">{users_line}</p>
</div>
{qr}
</div>
{footer("github.com/Pavel-Tk/fittest · users = distinct anonymous clients that got ≥1 structure verdict, our own tests excluded", MUTE_D)}'''
    notes = ("Talking points: the swarm is live as a remote MCP endpoint; add it to Claude Code, Claude Desktop or Cursor with "
             "one line, or run it locally on CPU. check_structures returns which model or blend to trust and when to call DFT; "
             "swarm_status shows the population and the family tree. Every call is logged without storing structures; the user "
             "count on this slide is read from that log (usage_stats).")
    write("yours", sec("yours", INK, BONE, body, notes))


# ---------- 05 wins (static swarm, same budget) — numbers from results/RESULTS.md (all arms, final run)
ARMS = [("One model (AIMNet2)", 0.317, 0.30, 0.33), ("AIMNet2 + random flags", 0.317, 0.30, 0.33),
        ("MACE LoRA seeds", 0.304, 0.29, 0.32), ("Committee", 0.293, 0.28, 0.31),
        ("Committee + audits", 0.293, 0.28, 0.31), ("AIMNet2 seed ensemble", 0.286, 0.28, 0.30),
        ("Fittest swarm", 0.206, 0.20, 0.22),
        # hindsight reference, not an arm: fixed mean(aimnet2+orb) chosen with eval labels; RESULTS.md gives no CI
        ("AIMNet2+Orb average|hindsight, uses eval labels", 0.202, None, None)]


def wins(kick="05", title=None, notes=None):
    W, rowh, x0, x1 = 1040, 54, 380, 1000
    lo, hi = 0.15, 0.35
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    parts = []
    for t in (0.15, 0.20, 0.25, 0.30, 0.35):
        parts.append(f'<line x1="{X(t):.0f}" x2="{X(t):.0f}" y1="0" y2="{len(ARMS) * rowh}" stroke="#D6CEBD" stroke-width="2"/>')
        parts.append(f'<text x="{X(t):.0f}" y="{len(ARMS) * rowh + 36}" text-anchor="middle" font-family="Geist Mono, monospace" '
                     f'font-size="24" fill="{MUTE_L}">{t:.2f}</text>')
    for i, (name, v, a, b) in enumerate(ARMS):
        y = i * rowh + rowh / 2
        me = name == "Fittest swarm"
        if a is None:   # hindsight reference: grey, hollow, point only
            n1, n2 = name.split("|")
            parts.append(f'<text x="0" y="{y - 4}" font-family="Geist, sans-serif" font-size="24" fill="{MUTE_L}">{n1}</text>'
                         f'<text x="0" y="{y + 24}" font-family="Geist, sans-serif" font-size="24" font-style="italic" fill="{MUTE_L}">{n2}</text>')
            parts.append(f'<circle cx="{X(v):.0f}" cy="{y}" r="10" fill="{BONE}" stroke="{MUTE_L}" stroke-width="3"/>')
            continue
        c = EMBER_D if me else "#3C4640"
        parts.append(f'<text x="0" y="{y + 9}" font-family="Geist, sans-serif" font-size="26" font-weight="{600 if me else 400}" fill="{c}">{name}</text>')
        parts.append(f'<line x1="{X(a):.0f}" x2="{X(b):.0f}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="4"/>')
        parts.append(f'<circle cx="{X(v):.0f}" cy="{y}" r="{13 if me else 10}" fill="{c}"/>')
    svg = (f'<svg aria-label="Served force error on held-out reactions with 95% intervals: Fittest swarm 0.206; baselines 0.286 to 0.317; a fixed AIMNet2 plus Orb average chosen in hindsight 0.202 eV per angstrom" '
           f'viewBox="0 0 {W} {len(ARMS) * rowh + 50}" style="width:{W}px;height:{len(ARMS) * rowh + 50}px">{"".join(parts)}</svg>')
    title = title or f'The static swarm wins <i style="color:{EMBER_D}">on the same DFT budget.</i>'
    stat = lambda big, txt: f'''<div style="display:flex;flex-direction:column;gap:8px">
<p style="font-family:{SERIF};font-size:88px;line-height:0.9;color:{EMBER_D}">{big}</p>
<p style="font-size:24px;line-height:1.35;color:#2E3631">{txt}</p></div>'''
    body = f'''<!-- sources: arms, CIs, −28%, 0/96, 71%: /home/azureuser/fittest/results/RESULTS.md (ablation.json);
hindsight AIMNet2+Orb average 0.202: RESULTS.md "Served answer: discovery vs hindsight" (no CI recorded anywhere; point only);
served mean(aimnet2+orb) 1373 and aimnet2 112 of 1485 eval frames (so MACE never served): RESULTS.md "Recommended candidate, eval (served)" -->
{kicker(kick, "The evidence", EMBER_D)}
<h2 style="font-family:{SERIF};font-size:96px;font-weight:400;line-height:1.05;letter-spacing:-2px;width:1600px">{title}</h2>
<div style="display:flex;gap:80px;align-items:flex-start">
{svg}
<div style="flex:1;display:flex;flex-direction:column;gap:28px">
{stat("−28%", "served force error vs. the best baseline (0.206 vs 0.286 eV/Å), significant — as good as the best fixed blend chosen in hindsight (0.202); the swarm found it from 200 labels.")}
{stat("0 / 96", "badly wrong answers trusted, while trusting 71% of all answers")}
</div>
</div>
<p style="font-family:{SERIF};font-size:40px;line-height:1.1;color:{INK}">Survival of the fittest answer: the swarm served <i style="color:{EMBER_D}">AIMNet2+Orb</i> on 1,373 of 1,485 frames, and never MACE.</p>
{footer("Force RMSE, eV/Å · 135 held-out reactions · 95% paired CIs · 200 Transition1x DFT lookups per arm · RESULTS.md", MUTE_L)}'''
    notes = notes or ("Talking points: every arm got the same 200 DFT labels. The swarm's served answers have 28 percent lower force "
                      "error than the best baseline. The honest reason: from those labels the Calibrator discovered that the AIMNet2 + Orb "
                      "average is the answer to serve; fixing that average in hindsight scores the same (0.202). It trusts 71 percent of "
                      "answers and none of the 96 badly wrong ones; calibration (ECE 0.08) comes from the Skeptic's random audits. "
                      "What it does worse: ranking its own leftover errors (AUROC 0.73 vs 0.80 to 0.87 for simpler scores).")
    write("wins", sec("wins", BONE, INK, body, notes, extra=";gap:40px"))


# ---------- evolution numbers (copied from results files; verify_evolution() re-checks them when the repo is present)
RES = pathlib.Path(CFG.get("results_dir", "/home/azureuser/fittest/results"))
A_TREE = "results/evolution/family_tree.json"                 # run A (MACE LoRA offspring), seeds 0,1
A_SUM = "results/evolution/summary_tables.md"
A_VERDICT = "results/evolution/verdict.json"
EVO_SUMMARY = "results/evolution/SUMMARY.md"
B_TREE = "results/evolution_aimnet/family_tree.json"          # run B, lineage evolved_s1
B_SUM = "results/evolution_aimnet/summary_tables.md"
B_VERDICT = "results/evolution_aimnet/verdict.json"
C_SUM = "results/evolution_c/summary.json"                     # run C as pre-registered (GPU, AIMNet2+Orb+MACE)
EVO = {
    # slide 04: run B evolved_s1 (family_tree.json ["evolved_s1"])
    "labels_per_gen": [50, 100, 150, 200],          # summary_tables.md, evolved_s1 "labels" column
    "fit_aimnet2_g2": -1.18,                        # aimnet2 fitness_per_generation["2"].fitness, died 2
    "fit_g1a_g2": -0.84,                            # g1_a fitness_per_generation["2"].fitness, died 2
    "fit_g2b_g3": 8.09,                             # g2_b fitness_per_generation["3"].fitness, alive
    # backup slide
    "A_children": 18, "A_stillborn": 15,                                # run A family_tree.json, seeds 0+1
    "A_holds": {0: False, 1: False},                                    # run A verdict.json per_run
    "A_s1_dF": -0.011, "A_s1_dRelE": 0.056,                             # run A summary_tables.md evolved_s1 final − static
    "B_holds": {0: False, 1: True},                                     # verdict.json per_run
    "B_s1_dF": -0.024, "B_s0_dF": -0.001,                               # summary_tables.md "final − static" F-RMSE
    "C_holds": {200: (False, True), 400: (False, False), 800: (True, False)},   # run C summary.json verdict.per_run
    "C_worse_F": (3, 0.011, 0.034),     # run C: n runs significantly WORSE than static on forces, min/max diff (eV/Å)
    # secondary outcome: served rel-E MAE, evolved final − static, over the 10 pre-registered seed runs (A 2, B 2, C 6)
    "relE_lower": 9, "relE_runs": 10, "relE_range": (-0.007, -0.044),
    "held_runs": 3,                     # pre-registered seed runs where the rule held: B s1, C b200 s1, C b800 s0
    # run D (evolution_d/cpu|gpu/summary.json), seeds 6..11; PREREG.md: confirmed iff >= 5 of 6 on CPU and none worse
    "D_cpu_holds": (True, True, False, True, False, True), "D_needed": 5, "D_gpu_held": 4, "D_gpu_worse": 2,
    "C_children": 54, "C_stillborn": 36, "C_stillborn_relE": 35,      # run C b*/family_tree.json evolved_* (GPU, pre-registered)
}


def verify_evolution():
    """Re-derive every evolution number above from the results files; raise if any drifted."""
    if not RES.exists():
        print("results dir not found; skipping evolution number check")
        return
    ft = json.loads((RES / "evolution_aimnet/family_tree.json").read_text())["evolved_s1"]
    by = {n["id"]: n for n in ft}
    fit = lambda i, g: round(by[i]["fitness_per_generation"][str(g)]["fitness"], 2)
    assert by["aimnet2"]["died"] == 2 and by["g1_a"]["died"] == 2 and by["g2_b"]["status"] == "alive"
    assert fit("aimnet2", 2) == EVO["fit_aimnet2_g2"] and fit("g1_a", 2) == EVO["fit_g1a_g2"]
    assert fit("g2_b", 3) == EVO["fit_g2b_g3"]
    assert [(n["id"], n["parent"], n["status"]) for n in ft[3:]] == [
        ("g1_a", "aimnet2", "dead"), ("g1_b", "aimnet2", "clone"), ("g1_c", "aimnet2", "stillborn"),
        ("g2_a", "g1_a", "stillborn"), ("g2_b", "g1_a", "alive"), ("g2_c", "aimnet2", "stillborn"),
        ("g3_a", "g2_b", "clone"), ("g3_b", "g2_b", "clone"), ("g3_c", "g2_b", "stillborn")]
    v = json.loads((RES / "evolution_aimnet/verdict.json").read_text())
    assert v["per_run"]["evolved_s0"]["holds"] is False and v["per_run"]["evolved_s1"]["holds"] is True
    assert v["flagship_holds"] is False
    md = (RES / "evolution_aimnet/summary_tables.md").read_text()
    assert "| evolved_s0 final − static (same budget) | -0.001 [-0.007, +0.005] | -0.027 [-0.039, -0.014] * | -0.061 [-0.103, -0.017] *" in md
    assert "| evolved_s1 final − static (same budget) | -0.024 [-0.031, -0.016] *" in md
    for g, lab in enumerate(EVO["labels_per_gen"]):
        assert f"| {g} | {lab} | " in md
    cs = json.loads((RES / C_SUM[len("results/"):]).read_text())
    c = cs["verdict"]
    for bud, (h0, h1) in EVO["C_holds"].items():
        k1 = f"b{bud}/evolved_s1" if bud != 800 else "b800_s1/evolved_s1"
        assert c["per_run"][f"b{bud}/evolved_s0"]["holds"] is h0 and c["per_run"][k1]["holds"] is h1
        assert c["per_budget"][str(bud)]["holds"] is False
    assert c["evolution_holds_any_budget"] is False
    c_runs = [x for bb in cs["budgets"].values() for r, x in bb["runs"].items() if r.startswith("evolved")]
    assert len(c_runs) == 6
    worse = [x["vs_static"]["force_rmse"]["diff"] for x in c_runs
             if x["vs_static"]["force_rmse"]["significant"] and x["vs_static"]["force_rmse"]["diff"] > 0]
    assert (len(worse), round(min(worse), 3), round(max(worse), 3)) == EVO["C_worse_F"], worse
    a = json.loads((RES / A_TREE[len("results/"):]).read_text())
    kids = [n for k in ("evolved_s0", "evolved_s1") for n in a[k] if n["parent"]]
    assert len(kids) == EVO["A_children"] and sum(n["status"] == "stillborn" for n in kids) == EVO["A_stillborn"]
    av = json.loads((RES / A_VERDICT[len("results/"):]).read_text())
    assert all(av["per_run"][f"evolved_s{i}"]["holds"] is h for i, h in EVO["A_holds"].items()) and not av["flagship_holds"]
    amd = (RES / A_SUM[len("results/"):]).read_text()
    assert "| evolved_s1 final − static (same budget) | -0.011 [-0.017, -0.005] * | +0.056 [+0.044, +0.068] * |" in amd
    # secondary: rel-E of evolved final − static, all 10 pre-registered seed runs
    rel = []
    for txt in (amd, md):
        for i in (0, 1):
            row = next(l for l in txt.splitlines() if l.startswith(f"| evolved_s{i} final − static"))
            cell = row.split("|")[3].strip()
            rel.append((float(cell.split()[0]), cell.endswith("*")))
    rel += [(x["vs_static"]["rel_e_mae"]["diff"], x["vs_static"]["rel_e_mae"]["significant"]) for x in c_runs]
    lower = [d for d, sig in rel if d < 0 and sig]
    assert len(rel) == EVO["relE_runs"] and len(lower) == EVO["relE_lower"]
    assert (round(max(lower), 3), round(min(lower), 3)) == EVO["relE_range"], lower
    held = sum(av["per_run"][k]["holds"] for k in av["per_run"]) + sum(v["per_run"][k]["holds"] for k in v["per_run"]) \
        + sum(c["per_run"][k]["holds"] for k in c["per_run"])
    assert held == EVO["held_runs"], held

    kids_c = [n for f in sorted(RES.glob("evolution_c/b*/family_tree.json"))
              for k, v in json.loads(f.read_text()).items() if k.startswith("evolved") for n in v if n["parent"]]
    sb = [n for n in kids_c if n["status"] == "stillborn"]
    sb_rel = [n for n in sb if any(not ch["passed"] and "rel_energy" in ch["name"] for ch in n["gate"]["checks"])]
    assert (len(kids_c), len(sb), len(sb_rel)) == (EVO["C_children"], EVO["C_stillborn"], EVO["C_stillborn_relE"])
    for kind in ("cpu", "gpu"):
        dv = json.loads((RES / f"evolution_d/{kind}/summary.json").read_text())
        pr = {int(k.split("_s")[-1]): v["holds"] for k, v in dv["verdict"]["per_run"].items()}
        worse = sum(x["vs_static"]["force_rmse"]["significant"] and x["vs_static"]["force_rmse"]["diff"] > 0
                    for bb in dv["budgets"].values() for r, x in bb["runs"].items() if r.startswith("evolved"))
        if kind == "cpu":
            assert tuple(pr[s] for s in range(6, 12)) == EVO["D_cpu_holds"] and worse == 0, (pr, worse)
        else:
            assert sum(pr.values()) == EVO["D_gpu_held"] and len(pr) == 6 and worse == EVO["D_gpu_worse"], (pr, worse)
    assert "at least 5 of 6" in (RES / "evolution_d/PREREG.md").read_text()
    print("evolution numbers verified against", RES)


# ---------- 04 compete: real family tree, run B seed 1 (AIMNet2 lineage)
def compete():
    G, END = [190, 440, 690, 940], 1150
    LINE, DEAD, MUTE = "#3E4A42", "#55635A", "#8F9A92"
    p = []
    # label with an ink backing so gridlines never run through it (start-anchored, 24px mono ≈ 14.5 px/char)
    TB = lambda x, y, s, c: (f'<rect x="{x - 8}" y="{y - 26}" width="{len(s) * 14.5 + 16:.0f}" height="34" fill="{INK}"/>'
                             + T(x, y, s, c))
    T = lambda x, y, s, c=MUTE, a="start", sz=24, f="Geist Mono, monospace": (
        f'<text x="{x}" y="{y}" text-anchor="{a}" font-family="{f}" font-size="{sz}" fill="{c}">{s}</text>')
    for g, (x, lab) in enumerate(zip(G, EVO["labels_per_gen"])):
        p.append(T(x, 24, f"gen {g}", BONE, "middle"))
        p.append(T(x, 54, f"{lab} DFT", MUTE, "middle"))
        p.append(f'<line x1="{x}" x2="{x}" y1="74" y2="590" stroke="#232C26" stroke-width="2"/>')
    hl = lambda x1, x2, y, c, w=4: f'<path d="M{x1} {y}H{x2}" stroke="{c}" stroke-width="{w}" fill="none"/>'
    elbow = lambda xv, y1, ys, xn, c: "".join(f'<path d="M{xv} {y1}V{y}H{xn}" stroke="{c}" stroke-width="3" fill="none"/>' for y in ys)
    dot = lambda x, y, c, r=10: f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>'
    hollow = lambda x, y: f'<circle cx="{x}" cy="{y}" r="9" fill="{INK}" stroke="{MUTE}" stroke-width="3"/>'
    cross = lambda x, y, c=EMBER_D: (f'<path d="M{x - 11} {y - 11}L{x + 11} {y + 11}M{x + 11} {y - 11}L{x - 11} {y + 11}" '
                                     f'stroke="{c}" stroke-width="5" stroke-linecap="round"/>')
    bar = lambda x, y: f'<path d="M{x} {y - 18}V{y + 18}" stroke="{BONE}" stroke-width="6" stroke-linecap="round"/>'
    # founders that are not in this lineage
    for name, y in (("MACE", 110), ("Orb", 160)):
        p += [T(0, y + 8, name, MUTE, sz=26, f="Geist, sans-serif"), hl(G[0], END, y, LINE), dot(G[0], y, MUTE, 8), dot(END, y, MUTE, 8)]
    # AIMNet2 founder -> pruned at gen 2
    yA, y2c, y1a, y1b, y1c, y2a, y2b = 230, 290, 340, 400, 460, 400, 460
    p += [T(0, yA + 8, "AIMNet2", BONE, sz=26, f="Geist, sans-serif"), hl(G[0], G[2], yA, EMBER), dot(G[0], yA, EMBER)]
    p.append(elbow(G[1] - 50, yA, [y1a, y1b, y1c], G[1], LINE))
    p.append(f'<path d="M{G[1] - 50} {yA}V{y1a}H{G[1]}" stroke="{EMBER}" stroke-width="4" fill="none"/>')
    p.append(elbow(G[2] - 50, yA, [y2c], G[2], LINE))
    p += [bar(G[2], yA), TB(G[2] + 22, yA - 12, f"pruned · fitness {EVO['fit_aimnet2_g2']:+.1f}".replace("-", "−"), BONE)]
    # gen 1 children
    p += [hl(G[1], G[2], y1a, EMBER), dot(G[1], y1a, EMBER), hollow(G[1], y1b), cross(G[1], y1c)]
    p += [bar(G[2], y1a), TB(G[2] + 22, y1a - 12, f"pruned · fitness {EVO['fit_g1a_g2']:+.1f}".replace("-", "−"), BONE)]
    p.append(cross(G[2], y2c))
    # gen 2 children of g1_a
    p.append(elbow(G[2] - 50, y1a, [y2a, y2b], G[2], LINE))
    p.append(f'<path d="M{G[2] - 50} {y1a}V{y2b}H{G[2]}" stroke="{EMBER}" stroke-width="4" fill="none"/>')
    p += [cross(G[2], y2a), hl(G[2], END, y2b, EMBER, 6), dot(G[2], y2b, EMBER)]
    # gen 3 children of g2_b
    p.append(elbow(G[3] - 50, y2b, [510, 550, 590], G[3], LINE))
    p += [hollow(G[3], 510), hollow(G[3], 550), cross(G[3], 590)]
    p += [dot(END, y2b, EMBER, 16), T(END, y2b - 34, f"fitness {EVO['fit_g2b_g3']:+.1f}", EMBER, "end", 28)]
    p += [T(END, y2b + 50, "survives", BONE, "end", 26, "Geist, sans-serif")]
    svg = (f'<svg aria-label="Family tree from run B seed 1: AIMNet2 breeds children; two fail the gate, one is a rejected clone; '
           f'one passes and replaces its pruned parent; its child passes the gate and survives with fitness +8.1" '
           f'viewBox="0 0 1200 610" style="width:1200px;height:610px;flex:none">{"".join(p)}</svg>')

    def step(n, glyph, head, txt):
        return f'''<div style="display:flex;gap:20px;align-items:flex-start">
<svg viewBox="-16 -16 32 32" style="width:32px;height:32px;flex:none;margin-top:6px">{glyph}</svg>
<div><p style="font-size:30px;font-weight:600;color:{BONE}">{n} · {head}</p>
<p style="font-size:24px;line-height:1.35;color:#C9C3B6">{txt}</p></div></div>'''
    gdot = '<circle r="10" fill="#F26A21"/>'
    gx = '<path d="M-10 -10L10 10M10 -10L-10 10" stroke="#B9420E" stroke-width="5" stroke-linecap="round"/>'
    gfit = '<path d="M-12 0H12M0 -12V12" stroke="#F26A21" stroke-width="5" stroke-linecap="round"/>'
    gbar = f'<path d="M0 -14V14" stroke="{BONE}" stroke-width="6" stroke-linecap="round"/>'
    steps = "\n".join([
        step(1, gdot, "Breed", "Children are fine-tuned on the swarm's DFT labels."),
        step(2, gx, "Gate", "Worse on held-out labels: stillborn. Near-copies (○) are rejected."),
        step(3, gfit, "Fitness", "Marginal contribution to the swarm's answers."),
        step(4, gbar, "Prune", "No contribution: removed, even a parent."),
    ])
    foot = (f"A real lineage: run B, seed 1 — one of {EVO['held_runs']} of {EVO['relE_runs']} pre-registered seed runs "
            f"where evolution held (verdict: no; see backup)")
    body = f'''<!-- source: /home/azureuser/fittest/{B_TREE} ["evolved_s1"] (statuses, born/died, fitness_per_generation); DFT labels per generation: /home/azureuser/fittest/{B_SUM} (evolved_s1 table) -->
{kicker("04", "Next: let them breed (experimental)", EMBER)}
<h2 style="font-family:{SERIF};font-size:96px;font-weight:400;line-height:1.05;letter-spacing:-2px">Let them compete. <i style="color:{EMBER}">What adds nothing is pruned.</i></h2>
<div style="display:flex;gap:64px;align-items:flex-start">
{svg}
<div style="flex:1;display:flex;flex-direction:column;gap:36px;padding-top:72px">
{steps}
</div>
</div>
{footer(foot, MUTE_D)}'''
    notes = ("Talking points: this is one real family from the run, not an illustration. Each generation the swarm buys 50 more DFT "
             "labels and breeds children from them. A child must not be worse than its parent on held-out labels (forces and "
             "reaction energies) or it is stillborn; near-copies are rejected because copies cannot disagree. Survivors are scored "
             "by their marginal contribution to the swarm's served answers, and anything that adds nothing is pruned, including the "
             "AIMNet2 founder and the child that replaced it. Its own child survives with the highest fitness in the swarm. "
             "Breeding is the experimental part: under the pre-registered test it did not beat the static swarm (it held in 3 of 10 "
             "seed runs; backup slide). The result on the next slide is the static swarm.")
    write("compete", sec("compete", INK, BONE, body, notes, pad="128px 128px 150px", extra=";gap:40px"))


# ---------- backup: what didn't work (yet)
def notyet():
    m = lambda t: t.replace("-", "−")
    rows = [("A", "MACE LoRA offspring", [("", [EVO["A_holds"][0], EVO["A_holds"][1]])],
             m(f"{EVO['A_stillborn']} of {EVO['A_children']} children stillborn. Seed 1: forces "
               f"{EVO['A_s1_dF']:+.3f} eV/Å, but reaction energy {EVO['A_s1_dRelE']:+.3f} eV worse.")),
            ("B", "AIMNet2 offspring", [("", [EVO["B_holds"][0], EVO["B_holds"][1]])],
             m(f"Seed 1 beats static ({EVO['B_s1_dF']:+.3f} eV/Å); seed 0 does not ({EVO['B_s0_dF']:+.3f}, n.s.).")),
            ("C", "Niche specialists<br>(3 families)", [(f"{b} labels", list(h)) for b, h in EVO["C_holds"].items()],
             m(f"{EVO['C_stillborn']} of {EVO['C_children']} children stillborn; {EVO['C_worse_F'][0]} of 6 runs worse than "
               f"static on forces (+{EVO['C_worse_F'][1]:.3f} to +{EVO['C_worse_F'][2]:.3f} eV/Å).")),
            ("D", 'Fix, on new seeds<span style="display:block;font-size:24px;font-weight:400;line-height:1.3;color:#2E3631;margin-top:6px">'
                  'pre-registered: protect base models + gentler gate</span>',
             [("seeds 6–11 (CPU)", list(EVO["D_cpu_holds"]))],
             f"{sum(EVO['D_cpu_holds'])} of 6 (needed {EVO['D_needed']}); on GPU also {EVO['D_gpu_held']} of 6, "
             f"two seeds worse than static.", "not confirmed")]
    rows = [r if len(r) == 5 else (*r, "does not hold") for r in rows]

    def dots(h):
        out = []
        for i, v in enumerate(h):
            if v is None:
                out.append(f'<span title="pending" style="width:44px;height:44px;border-radius:50%;border:3px dashed {EMBER_D};'
                           f'display:inline-flex;align-items:center;justify-content:center;font-family:{MONO};font-size:24px;'
                           f'color:{EMBER_D}">?</span>')
            else:
                out.append(f'<span style="width:44px;height:44px;border-radius:50%;'
                           f'{"background:" + EMBER_D if v else "border:3px solid #3C4640"};display:inline-block"></span>')
        return "".join(out)

    def group(lab, h):
        top = (f'<p style="font-family:{MONO};font-size:24px;color:{MUTE_L};height:30px">{lab}</p>' if lab else "")
        return f'<div style="display:flex;flex-direction:column;gap:10px">{top}<div style="display:flex;gap:16px">{dots(h)}</div></div>'

    rowh = []
    for k, name, groups, note, verdict in rows:
        rowh.append(f'''<div style="display:grid;grid-template-columns:72px 380px 520px 1fr;align-items:center;column-gap:32px;border-top:2px solid #CFC7B6;padding:6px 0">
<p style="font-family:{SERIF};font-size:80px;line-height:1;color:{INK}">{k}</p>
<p style="font-size:32px;font-weight:600">{name}</p>
<div style="display:flex;gap:44px">{"".join(group(l, h) for l, h in groups)}</div>
<div style="display:flex;flex-direction:column;gap:8px"><p style="font-family:{MONO};font-size:24px;letter-spacing:2px;text-transform:uppercase;color:{EMBER_D}">{verdict}</p>
<p style="font-size:24px;line-height:1.35;color:#2E3631">{note}</p></div>
</div>''')
    sw = lambda fill: (f'<span style="width:24px;height:24px;border-radius:50%;display:inline-block;'
                       f'{"background:" + EMBER_D if fill else "border:3px solid #3C4640"}"></span>')
    legend = (f'<span style="display:inline-flex;gap:14px;align-items:center;font-family:{MONO};font-size:24px;color:{MUTE_L};white-space:nowrap">'
              f'{sw(True)} held {sw(False)} not · seed 0, seed 1</span>')
    r0, r1 = EVO["relE_range"]
    body = f'''<!-- sources: run A: /home/azureuser/fittest/{A_VERDICT} (per_run.holds), /home/azureuser/fittest/{A_TREE} (children: 18, stillborn: 15, seeds 0+1),
/home/azureuser/fittest/{A_SUM} ("evolved_s1 final − static": F −0.011 *, rel-E +0.056 *);
run B: /home/azureuser/fittest/{B_VERDICT} (per_run.holds), /home/azureuser/fittest/{B_SUM} ("final − static (same budget)" rows: s1 F −0.024 *, s0 F −0.001 n.s.);
run C (pre-registered GPU run, AIMNet2+Orb+MACE specialists): /home/azureuser/fittest/{C_SUM} (verdict.per_run, verdict.per_budget; budgets.*.runs.evolved_*.vs_static.force_rmse: +0.034, +0.019, +0.011 significant);
gate failures run C: /home/azureuser/fittest/results/evolution_c/b*/family_tree.json evolved_* (54 children, 36 stillborn, 35 on rel-E);
run D (pre-registered fix, seeds 6–11): /home/azureuser/fittest/results/evolution_d/cpu/summary.json verdict.per_run (s6 T, s7 T, s8 F, s9 T, s10 F, s11 T), PREREG.md (needs ≥5 of 6);
/home/azureuser/fittest/results/evolution_d/gpu/summary.json (4 of 6 hold; s9 +0.015 *, s10 +0.005 * worse than static on forces);
secondary: rel-E "final − static" rows of both summary_tables.md + {C_SUM} vs_static.rel_e_mae; also /home/azureuser/fittest/{EVO_SUMMARY} "Secondary outcome" -->
{kicker("Backup", "What didn't work (yet)", EMBER_D)}
<h2 style="font-family:{SERIF};font-size:96px;font-weight:400;line-height:1.05;letter-spacing:-2px">Evolution did not beat <i style="color:{EMBER_D}">the static swarm.</i></h2>
<div style="display:flex;justify-content:space-between;align-items:center;gap:32px">
<p style="font-size:26px;line-height:1.4;color:#2E3631">Pre-registered: beat static (paired 95% CI) and own gen 0 on force error, <b>in every seed</b>, ≥3 gens.</p>
{legend}
</div>
<div style="display:flex;flex-direction:column">{"".join(rowh)}</div>
<div style="display:flex;flex-direction:column;gap:10px;border-top:2px solid #CFC7B6;padding-top:16px">
<p style="font-size:28px;line-height:1.35;color:{INK}">Secondary (pre-registered for run C): evolution lowers reaction-energy error vs static in <b>{EVO['relE_lower']} of {EVO['relE_runs']}</b> seed runs ({r0:.3f} to {r1:.3f} eV; fine-tuning without selection gains as much).</p>
</div>
{footer("results/evolution · evolution_aimnet · evolution_c · evolution_d · results/evolution/SUMMARY.md", MUTE_L)}'''.replace("[-", "[−").replace(", -", ", −").replace("(-0.", "(−0.").replace("to -0.", "to −0.")
    notes = ("Talking points: we pre-registered what would count as evolution working, and it has not cleared that bar. "
             "Run A bred MACE children: 15 of 18 were stillborn, mostly on reaction energy, and in seed 1 the child that replaced base MACE "
             "improved forces but made reaction energies worse. Run B bred AIMNet2 children with a reaction-energy loss: it holds in "
             "one seed of two. Run C bred niche specialists from all three families at three DFT budgets: one seed holds at 200 and "
             "at 800, none at 400. The rule needs every seed, so the verdict is no. Most children fail the gate on reaction energy "
             "(36 of 54 in run C, 15 of 18 in run A). Why evolved swarms sometimes end up worse than static is not yet established; "
             "one hypothesis is that selection prunes base models for niche specialists, but no-selection changes several things "
             "at once and lost to evolved in one of the two pre-registered seeds, so we do not claim it. "
             "The pre-registered secondary outcome: the evolved swarm has lower reaction-energy error than static in "
             "9 of 10 seed runs, but children are trained and gated on reaction energy, and fine-tuning without selection gains as much. "
             "Run D pre-registered a fix on six new seeds (protect base models, gentler gate): it needed 5 of 6 and held in 4; "
             "on GPU with three families also 4 of 6, with two seeds worse than static. Not confirmed. The static swarm on slide 05 is the result we stand on.")
    write("notyet", sec("notyet", BONE, INK, body, notes, pad="128px 128px 140px", extra=";gap:20px"))


# ---------- deck.json (order, sections, fonts)
ORDER = ["cover", "abundant", "silent", "disagree", "compete", "wins", "yours", "notyet"]


def deck():
    d = {"v": 4, "createdOnFiles": {"v": 1, "at": "2026-10-31T05:00:00Z"}, "lists": "css", "title": "Fittest",
         "order": ORDER,
         "sections": {
             "s1": {"description": "The problem: ground truth is scarce and models fail silently", "start": "cover"},
             "s2": {"description": "The idea: independent models disagree; let them compete", "start": "disagree"},
             "s3": {"description": "Evidence: same DFT budget", "start": "wins"},
             "s4": {"description": "Use it: the live MCP server", "start": "yours"},
             "s5": {"description": "Backup: what didn't work (yet)", "start": "notyet"}},
         "faces": {
             "instrument-serif": {"family": "Instrument Serif",
                                  "href": "https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap"},
             "geist": {"family": "Geist", "href": "https://fonts.googleapis.com/css2?family=Geist:wght@300..700&display=swap"},
             "geist-mono": {"family": "Geist Mono",
                            "href": "https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500&display=swap"}},
         "designSystems": []}
    (OUT.parent / "deck.json").write_text(json.dumps(d), encoding="utf-8")


def fetch_users():
    """Read the live user count from the MCP server's /stats and store it in cfg.json."""
    import urllib.request
    with urllib.request.urlopen("https://fittest-mcp.westus2.cloudapp.azure.com/stats", timeout=20) as r:
        n = int(json.loads(r.read())["users"])
    CFG["users"] = n
    (HERE / "cfg.json").write_text(json.dumps(CFG, indent=1) + "\n")
    print("users =", n)


if __name__ == "__main__":
    # python gen.py                  -> every slide in ORDER + deck.json (uses cfg.json "users")
    # python gen.py --fetch-users    -> refresh cfg.json from /stats first, then the same
    # python gen.py yours compete    -> only those slides
    args = sys.argv[1:]
    if "--fetch-users" in args:
        args.remove("--fetch-users")
        fetch_users()
    OUT.mkdir(parents=True, exist_ok=True)
    verify_evolution()
    for f in (args or ORDER):
        globals()[f]()
    deck()
    print("ok")
