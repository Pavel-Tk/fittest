"""End-to-end demo of the trust service against the real model agents.

Writes a few held-out Transition1x frames to XYZ, scores them with
``python -m fittest.service score`` (three model-agent processes), and prints the
verdicts next to the true DFT force error (which the service never sees).

    python3 scripts/demo_service.py --snapshot results/trust_snapshot.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fittest.cache import load_frames  # noqa: E402
from fittest.data import ELEMENT_SYMBOLS, assign_leakfree_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames.pkl")
    ap.add_argument("--snapshot", default="results/trust_snapshot.json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--base-port", type=int, default=6400)
    ap.add_argument("--out", default="results/service_demo.json")
    a = ap.parse_args()

    frames, _ = assign_leakfree_splits(load_frames(a.frames), seed=0)
    ev = [f for f in frames if f.split == "eval"]
    rng = np.random.default_rng(0)
    pick = [ev[i] for i in rng.choice(len(ev), size=a.n, replace=False)]
    xyz = "results/service_demo.xyz"
    with open(xyz, "w") as fh:
        for f in pick:
            fh.write(f"{len(f.Z)}\n{f.uid}\n")
            for z, r in zip(f.Z, f.R):
                fh.write(f"{ELEMENT_SYMBOLS[int(z)]} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}\n")
    cmd = [sys.executable, "-m", "fittest.service", "score", xyz, "--snapshot", a.snapshot,
           "--base-port", str(a.base_port)]
    if os.environ.get("FITTEST_ORB_PYTHON"):
        cmd += ["--orb-python", os.environ["FITTEST_ORB_PYTHON"]]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        print(out.stderr[-3000:]); raise SystemExit(out.returncode)
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    report = []
    for f, r in zip(pick, rows):
        served = np.asarray(r["served_forces"])
        err = float(np.sqrt(((served - f.F) ** 2).mean()))
        report.append({"uid": f.uid, "role": f.role, "sigma_F": round(r["sigma_F"], 3),
                       "flagged": r["flagged"], "trust": r["trust"], "recommended": r["recommended_model"],
                       "served_force_rmse_vs_dft": round(err, 3), "wrong@0.25": err > 0.25})
        print(json.dumps(report[-1]))
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
