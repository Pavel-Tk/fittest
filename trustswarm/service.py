"""TrustSwarm trust service: structures in → per-system trust + recommended model out.

The reusable half of the project for other teams. It needs only the model agent
processes (``trustswarm.model_server``) and a calibrator snapshot written by a
TrustSwarm run (``results/trust_snapshot.json``). Nothing here needs labels.

    # HTTP
    python -m trustswarm.service serve --snapshot results/trust_snapshot.json --port 8000
    curl -X POST localhost:8000/trust -H 'content-type: application/json' \
         -d '{"systems": [{"Z": [8,1,1], "R": [[0,0,0],[0.96,0,0],[-0.24,0.93,0]]}]}'

    # CLI
    python -m trustswarm.service score molecules.xyz --snapshot results/trust_snapshot.json

Response per system::

    {"sigma_F": 0.12, "sigma_F_max": 0.40, "flagged": false, "trust": 0.83,
     "recommended_model": "aimnet2" | "consensus", "bucket": [1, 0], "served_forces": [[...]],
     "per_model": {"aimnet2": {"energy": -2081.2, "max_force": 0.31}, ...}}
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trustswarm.signals import force_spread

CONSENSUS = "consensus"  # the ensemble-mean candidate the Calibrator can recommend

# A predictor maps (Z, R, batch) -> {"energy": (S,), "forces": (N,3)}; ModelClient fits.
Predictor = Callable[[np.ndarray, np.ndarray, np.ndarray], dict]


@dataclass
class TrustSnapshot:
    """Calibrator state learned from oracle outcomes during a TrustSwarm run."""

    models: list[str]
    tau: float
    sigma_edges: list[float]
    fbar_edges: list[float]
    p_correct: dict[str, list[list[float]]]
    counts: dict[str, list[list[int]]] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)  # models, subset means, "consensus"
    candidate_members: dict[str, list[str]] = field(default_factory=dict)
    p_global: dict[str, float] = field(default_factory=dict)  # tie-break 1: global posterior
    tiebreak_order: list[str] = field(default_factory=list)  # tie-break 2: Calibrator's order
    meta: dict = field(default_factory=dict)

    def rank_key(self, candidate: str, p_bucket: float) -> tuple[float, float, int]:
        """Same ordering the Calibrator uses: (P(correct | bucket), global P(correct), -tiebreak)."""
        order = self.tiebreak_order or self.candidates
        tb = order.index(candidate) if candidate in order else len(order)
        return (p_bucket, float(self.p_global.get(candidate, 0.0)), -tb)

    def members(self, candidate: str, models: Sequence[str]) -> list[str]:
        """Model agents whose mean prediction a candidate stands for."""
        if candidate in self.candidate_members:
            return list(self.candidate_members[candidate])
        if candidate == CONSENSUS:
            return list(models)
        return [candidate]

    @classmethod
    def load(cls, path: str | Path) -> "TrustSnapshot":
        d = json.loads(Path(path).read_text())
        known = {"models", "tau", "sigma_edges", "fbar_edges", "p_correct", "counts", "candidates",
                 "candidate_members", "p_global", "tiebreak_order"}
        return cls(models=list(d["models"]), tau=float(d["tau"]),
                   sigma_edges=[float(x) for x in d["sigma_edges"]],
                   fbar_edges=[float(x) for x in d["fbar_edges"]],
                   p_correct=d["p_correct"], counts=d.get("counts", {}),
                   candidates=list(d.get("candidates", d["models"])),
                   candidate_members={k: list(v) for k, v in d.get("candidate_members", {}).items()},
                   p_global={k: float(v) for k, v in d.get("p_global", {}).items()},
                   tiebreak_order=list(d.get("tiebreak_order", [])),
                   meta={k: v for k, v in d.items() if k not in known})

    def bucket(self, sigma_F: float, mean_abs_fbar: float) -> tuple[int, int]:
        try:  # identical bucketing to the Calibrator when available
            from trustswarm.agents.calibrator import bucket_of

            return bucket_of(sigma_F, mean_abs_fbar, self.sigma_edges, self.fbar_edges)
        except ImportError:
            return (int(np.searchsorted(self.sigma_edges, sigma_F, side="right")),
                    int(np.searchsorted(self.fbar_edges, mean_abs_fbar, side="right")))


def _pack(systems: Sequence[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z, R, b = [], [], []
    for k, s in enumerate(systems):
        z = np.asarray(s["Z"], dtype=np.int64).reshape(-1)
        r = np.asarray(s["R"], dtype=np.float64).reshape(-1, 3)
        if len(z) != len(r) or len(z) == 0:
            raise ValueError(f"system {k}: Z and R must be non-empty and the same length")
        Z.append(z); R.append(r); b.append(np.full(len(z), k))
    return np.concatenate(Z), np.concatenate(R), np.concatenate(b)


def score(systems: Sequence[dict], predictors: dict[str, Predictor],
          snapshot: TrustSnapshot) -> list[dict]:
    """Run every model agent once on the batch and turn disagreement into a verdict."""
    Z, R, batch = _pack(systems)
    outs = {m: p(Z, R, batch) for m, p in predictors.items()}
    missing = sorted(set(snapshot.models) - set(outs))
    if missing:  # sigma and its bins are only meaningful for the committee the snapshot learned
        raise ValueError(f"model agents missing for this snapshot: {missing}")
    models = list(snapshot.models)
    results = []
    for k in range(len(systems)):
        idx = np.flatnonzero(batch == k)
        F = np.stack([np.asarray(outs[m]["forces"])[idx] for m in models])  # (M, n, 3)
        s_mean, s_max = force_spread(F)
        fbar = float(np.linalg.norm(F.mean(axis=0), axis=-1).mean())
        bkt = snapshot.bucket(float(s_mean), fbar)
        candidates = [c for c in (snapshot.candidates or models) if c in snapshot.p_correct
                      and set(snapshot.members(c, models)) <= set(models)]
        p_ok = {c: float(snapshot.p_correct[c][bkt[0]][bkt[1]]) for c in candidates}
        rec = max(p_ok, key=lambda c: snapshot.rank_key(c, p_ok[c])) if p_ok else CONSENSUS
        energies = {m: float(np.asarray(outs[m]["energy"])[k]) for m in models}
        flagged = bool(s_mean >= snapshot.tau)
        results.append({
            "sigma_F": float(s_mean), "sigma_F_max": float(s_max), "mean_abs_fbar": fbar,
            "flagged": flagged, "trust": round(p_ok.get(rec, float("nan")), 4),
            "recommended_model": rec, "bucket": list(bkt),
            "action": "send to oracle / DFT" if flagged else f"use {rec}",
            # the served answer: the recommended model's prediction, or the ensemble mean
            "served_forces": F[[models.index(m) for m in snapshot.members(rec, models)]].mean(axis=0).tolist(),
            "served_members": snapshot.members(rec, models),
            "per_model": {m: {"energy": energies[m],
                              "max_force": float(np.linalg.norm(F[i], axis=-1).max())}
                          for i, m in enumerate(models)},
        })
    return results


# --------------------------------------------------------------------------- I/O
def read_xyz(path: str | Path) -> list[dict]:
    """Minimal multi-frame XYZ reader (ase is used instead when installed)."""
    try:
        from ase.io import read

        return [{"Z": a.numbers.tolist(), "R": a.positions.tolist(), "name": f"{Path(path).stem}:{i}"}
                for i, a in enumerate(read(str(path), index=":"))]
    except ImportError:
        pass
    from trustswarm.data import ELEMENT_SYMBOLS  # noqa: PLC0415

    SYMBOL_TO_Z = {v: k for k, v in ELEMENT_SYMBOLS.items()}
    lines, out, i = Path(path).read_text().splitlines(), [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i]); rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        out.append({"Z": [SYMBOL_TO_Z[r[0]] for r in rows],
                    "R": [[float(x) for x in r[1:4]] for r in rows], "name": f"frame{len(out)}"})
        i += n + 2
    return out


def connect_models(models: Sequence[str], base_port: int, orb_python: str | None,
                   device: str) -> dict:
    from trustswarm.model_server import spawn

    clients = {}
    for k, m in enumerate(models):
        python = orb_python if (m == "orb" and orb_python) else sys.executable
        clients[m] = spawn(m, base_port + k, python=python, device=device).connect()
    return clients


try:  # request schema at module level so FastAPI can resolve the annotations
    from pydantic import BaseModel

    class System(BaseModel):
        Z: list[int]
        R: list[list[float]]
        name: str | None = None

    class TrustRequest(BaseModel):
        systems: list[System]
except ImportError:  # the service extra is optional; score()/CLI work without it
    TrustRequest = None


def create_app(snapshot: TrustSnapshot, predictors: dict[str, Predictor]):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="TrustSwarm trust service", version="0.1.0")

    @app.get("/health")
    def health():
        return {"ok": True, "models": list(predictors), "tau": snapshot.tau,
                "snapshot": snapshot.meta.get("created")}

    @app.post("/trust")
    def trust(req: TrustRequest):
        try:
            res = score([s.model_dump() for s in req.systems], predictors, snapshot)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        for s, r in zip(req.systems, res):
            r["name"] = s.name
        return {"results": res}

    return app


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="trustswarm.service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "score"):
        p = sub.add_parser(name)
        p.add_argument("--snapshot", required=True)
        p.add_argument("--models", default=None, help="comma list; default: snapshot models")
        p.add_argument("--base-port", type=int, default=6300)
        p.add_argument("--orb-python", default="/content/envs/orb/bin/python")
        p.add_argument("--device", default="cuda")
        if name == "serve":
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--host", default="127.0.0.1")
        else:
            p.add_argument("xyz")
    a = ap.parse_args(argv)
    snap = TrustSnapshot.load(a.snapshot)
    models = a.models.split(",") if a.models else snap.models
    clients = connect_models(models, a.base_port, a.orb_python, a.device)
    predictors = {m: c.predict for m, c in clients.items()}
    try:
        if a.cmd == "score":
            systems = read_xyz(a.xyz)
            for s, r in zip(systems, score(systems, predictors, snap)):
                print(json.dumps({"name": s.get("name"), **r}))
        else:
            import uvicorn

            uvicorn.run(create_app(snap, predictors), host=a.host, port=a.port)
    finally:
        for c in clients.values():
            c.close()


if __name__ == "__main__":
    main()
