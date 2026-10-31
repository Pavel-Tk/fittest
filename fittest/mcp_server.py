"""Fittest MCP server: ask the swarm whether to trust an ML potential on your structures.

Hosted (streamable HTTP, behind Caddy/TLS; model agents are separate warm processes)::

    fittest-mcp --transport http --port 8765 --public-host fittest-mcp.westus2.cloudapp.azure.com \\
        --connect mace_mpa0=6300,aimnet2=6301,orb=6302 --data-dir /var/lib/fittest-mcp

Local (stdio, CPU; spawns whichever model agents are installed; logs only locally)::

    uvx --from git+https://github.com/Pavel-Tk/fittest fittest-mcp              # real models if installed
    uvx --from git+https://github.com/Pavel-Tk/fittest fittest-mcp --fake-models  # demo / tests

Tools: ``check_structures``, ``explain``, ``swarm_status``, ``usage_stats``. Every call is logged
by :mod:`fittest.usage_log` (counts and an anonymous salted-hash client id, never structures).
"""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np

from fittest.service import TrustSnapshot, fake_predictors, score
from fittest.usage_log import CallRecord, TokenBucket, UsageLog

try:  # module level so FastMCP can resolve the tool annotations
    from mcp.server.fastmcp import Context
except ImportError:  # parsing/engine helpers stay importable without the SDK
    Context = Any  # type: ignore[misc,assignment]

#: what `structures` accepts: XYZ text, one {Z|symbols, R} object, or a list mixing both
StructuresArg = Union[str, list[Union[str, dict[str, Any]]], dict[str, Any]]

log = logging.getLogger("fittest.mcp")

REPO_URL = "https://github.com/Pavel-Tk/fittest"
PUBLIC_URL = "https://fittest-mcp.westus2.cloudapp.azure.com/mcp"
ROOT = Path(__file__).resolve().parents[1]

MAX_STRUCTURES = 20
MAX_ATOMS = 100
MAX_TEXT_CHARS = 400_000
REQUEST_TIMEOUT_S = 120.0
BUSY_TIMEOUT_S = 90.0

CAL_ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O"}
T1X_MAX_HEAVY = 7  # Transition1x: GDB7-derived reactions, <= 7 heavy atoms (C/N/O)
T1X_MAX_ATOMS = 26

LEVEL_OF_THEORY = {
    "mace_mpa0": "MACE-MPA-0 medium, PBE/PBE+U (MPtrj + sAlex)",
    "aimnet2": "AIMNet2, wB97M-D3(BJ)/def2-TZVPP",
    "orb": "Orb-v3 conservative, PBE/PBE+U (OMat24)",
}

LOGGING_NOTICE = (
    "Logging: every call to this server is logged (UTC time, tool name, an anonymous client id = "
    "salted hash of client name/version + IP; raw IPs are not stored), structure and atom counts, "
    "verdict counts, latency and ok/error. Structures, coordinates and names are NOT stored. "
    "Aggregate counts are public at /stats.")

SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se "
    "Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy "
    "Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf "
    "Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
SYMBOL_TO_Z = {s.lower(): z for z, s in enumerate(SYMBOLS) if z > 0}


class InputError(ValueError):
    """Bad user input; message is shown to the caller (never logged)."""


# --------------------------------------------------------------------------- parsing
def _z_of(token: Any, where: str) -> int:
    if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
        z = int(token)
    else:
        t = str(token).strip()
        if t.isdigit():
            z = int(t)
        else:
            t = "".join(ch for ch in t if ch.isalpha())  # "C1" / "O_a" style labels
            z = SYMBOL_TO_Z.get(t.lower()) or SYMBOL_TO_Z.get(t[:2].lower()) or SYMBOL_TO_Z.get(t[:1].lower())
            if z is None:
                raise InputError(f"{where}: unknown element {token!r}")
    if not 1 <= z <= 118:
        raise InputError(f"{where}: atomic number {z} outside 1..118")
    return z


def parse_xyz_text(text: str, start_index: int = 0) -> list[dict]:
    """XYZ (single or multi-frame, extended-XYZ comments fine) or bare 'El x y z' lines."""
    if len(text) > MAX_TEXT_CHARS:
        raise InputError(f"XYZ text too long ({len(text)} chars > {MAX_TEXT_CHARS})")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n").split("\n")
    lines = [ln.rstrip() for ln in lines]
    first = next((ln for ln in lines if ln.strip()), "")
    if not first:
        raise InputError("empty XYZ text")
    out: list[dict] = []
    head = first.split()[0]
    if not head.lstrip("+").isdigit():  # bare coordinate block, no count/comment lines
        rows = [ln for ln in lines if ln.strip()]
        out.append(_rows_to_structure(rows, f"structure {start_index}", None))
        return out
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        k = start_index + len(out)
        try:
            n = int(lines[i].split()[0])
        except ValueError:
            raise InputError(f"XYZ line {i + 1}: expected an atom count for frame {len(out)}") from None
        if n < 1:
            raise InputError(f"XYZ line {i + 1}: atom count must be >= 1")
        if n > MAX_ATOMS:
            raise InputError(f"structure {k}: {n} atoms > limit {MAX_ATOMS}")
        comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
        rows = lines[i + 2:i + 2 + n]
        if len(rows) < n or any(not r.strip() for r in rows):
            raise InputError(f"structure {k}: XYZ frame truncated (expected {n} atom lines)")
        name = comment.split("Lattice")[0].strip()[:80] or None
        out.append(_rows_to_structure(rows, f"structure {k}", name))
        i += 2 + n
        if len(out) > MAX_STRUCTURES:
            raise InputError(f"more than {MAX_STRUCTURES} structures in one call")
    return out


def _rows_to_structure(rows: list[str], where: str, name: str | None) -> dict:
    Z, R = [], []
    for j, row in enumerate(rows):
        p = row.split()
        if len(p) < 4:
            raise InputError(f"{where}, atom {j}: expected 'element x y z'")
        Z.append(_z_of(p[0], f"{where}, atom {j}"))
        try:
            R.append([float(p[1]), float(p[2]), float(p[3])])
        except ValueError:
            raise InputError(f"{where}, atom {j}: coordinates must be numbers") from None
    return {"Z": Z, "R": R, "name": name}


def _dict_to_structure(d: dict, k: int) -> dict:
    where = f"structure {k}"
    zs = next((d[key] for key in ("Z", "numbers", "atomic_numbers", "symbols", "elements", "species")
               if key in d), None)
    rs = next((d[key] for key in ("R", "positions", "coords", "coordinates", "xyz") if key in d), None)
    if zs is None or rs is None:
        raise InputError(f"{where}: give 'Z' (atomic numbers) or 'symbols', and 'R' (Angstrom, Nx3)")
    if isinstance(zs, str):
        zs = zs.replace(",", " ").split()
    try:
        R = np.asarray(rs, dtype=float)
    except (TypeError, ValueError):
        raise InputError(f"{where}: 'R' must be an N x 3 list of numbers") from None
    if R.ndim == 1 and R.size % 3 == 0:
        R = R.reshape(-1, 3)
    if R.ndim != 2 or R.shape[1] != 3:
        raise InputError(f"{where}: 'R' must be an N x 3 list of numbers")
    Z = [_z_of(z, f"{where}, atom {j}") for j, z in enumerate(list(zs))]
    if len(Z) != len(R):
        raise InputError(f"{where}: {len(Z)} elements but {len(R)} positions")
    name = d.get("name")
    return {"Z": Z, "R": R.tolist(), "name": None if name is None else str(name)[:80]}


def parse_structures(structures: Any) -> list[dict]:
    """XYZ text, one {Z|symbols, R} dict, or a list mixing both -> [{"Z", "R", "name"}]."""
    if isinstance(structures, str):
        s = structures.strip()
        if s[:1] in "[{":
            try:
                return parse_structures(json.loads(s))
            except json.JSONDecodeError:
                pass
        items: list[Any] = [structures]
    elif isinstance(structures, dict):
        items = [structures]
    elif isinstance(structures, (list, tuple)):
        items = list(structures)
    else:
        raise InputError("structures must be XYZ text, a {Z|symbols, R} object, or a list of them")
    out: list[dict] = []
    for item in items:
        if isinstance(item, str):
            out.extend(parse_xyz_text(item, start_index=len(out)))
        elif isinstance(item, dict):
            out.append(_dict_to_structure(item, len(out)))
        else:
            raise InputError(f"structure {len(out)}: expected XYZ text or a {{Z|symbols, R}} object")
        if len(out) > MAX_STRUCTURES:
            raise InputError(f"more than {MAX_STRUCTURES} structures in one call; split the batch")
    if not out:
        raise InputError("send at least one structure")
    for k, s in enumerate(out):
        _validate(s, k)
    return out


def _validate(s: dict, k: int) -> None:
    n = len(s["Z"])
    if n == 0:
        raise InputError(f"structure {k}: no atoms")
    if n > MAX_ATOMS:
        raise InputError(f"structure {k}: {n} atoms > limit {MAX_ATOMS}")
    R = np.asarray(s["R"], dtype=float)
    if not np.isfinite(R).all():
        raise InputError(f"structure {k}: coordinates must be finite")
    if np.abs(R).max() > 1000:
        raise InputError(f"structure {k}: coordinates beyond 1000 Angstrom; are units Angstrom?")
    if n > 1:
        d = np.linalg.norm(R[:, None] - R[None], axis=-1) + np.eye(n) * 1e9
        if d.min() < 0.3:
            i, j = np.unravel_index(np.argmin(d), d.shape)
            raise InputError(f"structure {k}: atoms {i} and {j} are {d.min():.3f} A apart (overlapping)")


def formula(Z: Sequence[int]) -> str:
    from collections import Counter

    c = Counter(SYMBOLS[int(z)] for z in Z)
    order = [e for e in ("C", "H") if e in c] + sorted(e for e in c if e not in ("C", "H"))
    return "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in order)


def domain_warning(Z: Sequence[int], R: Sequence[Sequence[float]]) -> str | None:
    notes = []
    other = sorted({SYMBOLS[int(z)] for z in Z if int(z) not in CAL_ELEMENTS})
    if other:
        notes.append(f"elements {', '.join(other)} are outside the calibration set (C/H/N/O)")
    heavy = sum(1 for z in Z if int(z) != 1)
    if heavy > T1X_MAX_HEAVY or len(Z) > T1X_MAX_ATOMS:
        notes.append(f"size ({len(Z)} atoms, {heavy} heavy) is beyond the Transition1x range "
                     f"(<= {T1X_MAX_HEAVY} heavy atoms)")
    R = np.asarray(R, dtype=float)
    if len(Z) > 1:
        d = np.linalg.norm(R[:, None] - R[None], axis=-1) + np.eye(len(Z)) * 1e9
        if d.min(axis=1).max() > 4.0:
            notes.append("at least one atom is > 4 A from every other atom (fragments / isolated atoms)")
        if d.min() < 0.6:
            notes.append(f"shortest interatomic distance {d.min():.2f} A is unusually short")
    if not notes:
        return None
    return ("; ".join(notes) + ". Calibration covers gas-phase C/H/N/O reaction chemistry "
            "(Transition1x); here the trust score is an extrapolation.")


# --------------------------------------------------------------------------- engine
def subset_snapshot(snap: TrustSnapshot, models: Sequence[str]) -> TrustSnapshot:
    """The snapshot restricted to the model agents that are actually running."""
    models = [m for m in snap.models if m in set(models)]
    if models == list(snap.models):
        return snap
    keep = [c for c in (snap.candidates or snap.models)
            if set(snap.members(c, snap.models)) <= set(models)]
    members = {c: [m for m in snap.members(c, snap.models)] for c in keep}
    return TrustSnapshot(models=models, tau=snap.tau, sigma_edges=snap.sigma_edges,
                         fbar_edges=snap.fbar_edges,
                         p_correct={c: v for c, v in snap.p_correct.items() if c in keep},
                         counts={c: v for c, v in snap.counts.items() if c in keep},
                         candidates=keep, candidate_members=members,
                         p_global={c: v for c, v in snap.p_global.items() if c in keep},
                         tiebreak_order=[c for c in snap.tiebreak_order if c in keep],
                         risk_iso={c: v for c, v in snap.risk_iso.items() if c in keep},
                         risk_feature=snap.risk_feature, meta=dict(snap.meta))


class RemoteModel:
    """A warm model-agent process reached over its socket; reconnects if it restarted."""

    def __init__(self, model: str, port: int):
        self.model, self.port = model, port
        self._client = None
        self._lock = threading.Lock()

    def connect(self, timeout: float = 5.0) -> "RemoteModel":
        from fittest.model_server import ModelClient

        self._client = ModelClient(self.model, self.port).connect(timeout=timeout)
        return self

    def predict(self, Z, R, batch):
        with self._lock:
            for attempt in (0, 1):
                try:
                    if self._client is None:
                        self.connect()
                    return self._client.predict(Z, R, batch)
                except (EOFError, OSError, ConnectionError, TimeoutError):
                    self._client = None
                    if attempt:
                        raise

    def close(self) -> None:
        if self._client is not None:
            self._client.close(shutdown=False)


class Engine:
    """Holds the Calibrator snapshot and the model-agent predictors; one evaluation at a time."""

    def __init__(self, snapshot: TrustSnapshot, mode: str, snapshot_path: str = ""):
        self.full = snapshot
        self.mode = mode  # "hosted" | "local" | "fake"
        self.snapshot_path = snapshot_path
        self.predictors: dict[str, Any] = {}
        self.model_status: dict[str, str] = {m: "not started" for m in snapshot.models}
        self._lock = threading.Lock()
        self.closers: list[Any] = []

    def add_model(self, name: str, predictor) -> None:
        self.predictors[name] = predictor
        self.model_status[name] = "active"

    @property
    def active(self) -> list[str]:
        return [m for m in self.full.models if m in self.predictors]

    def require_models(self) -> list[str]:
        active = self.active
        if not active:
            states = ", ".join(f"{m}: {s}" for m, s in self.model_status.items())
            if all(s == "not installed" for s in self.model_status.values()):
                raise InputError(f"no model agents installed ({states}). Install the [models] extra "
                                 "(MACE + AIMNet2) and/or set FITTEST_ORB_PYTHON, restart with --fake-models "
                                 f"for a demo, or use the hosted server {PUBLIC_URL}")
            raise InputError(f"no model agent is ready yet ({states}); retry in a minute")
        return active

    def evaluate(self, systems: list[dict]) -> tuple[list[dict], dict, TrustSnapshot]:
        active = self.require_models()
        snap = subset_snapshot(self.full, active)
        if not self._lock.acquire(timeout=BUSY_TIMEOUT_S):
            raise TimeoutError("server busy")
        try:
            from fittest.service import _pack

            Z, R, b = _pack(systems)
            outs = {}
            for m in active:
                try:
                    outs[m] = self.predictors[m](Z, R, b)
                except Exception as exc:  # a dead model agent must not take the whole call down
                    self.model_status[m] = f"error: {type(exc).__name__}"
                    log.warning("model %s failed: %s", m, exc)
            if not outs:
                raise RuntimeError("all model agents failed")
            if set(outs) != set(active):
                snap = subset_snapshot(self.full, list(outs))
            fixed = {m: (lambda *_a, o=o: o) for m, o in outs.items()}
            return score(systems, fixed, snap), outs, snap
        finally:
            self._lock.release()

    def close(self) -> None:
        for c in self.closers:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass


def _served_energy(res: dict) -> float:
    members = res["served_members"]
    return float(np.mean([res["per_model"][m]["energy"] for m in members]))


def summarize(sys_in: dict, res: dict, snap: TrustSnapshot, k: int, include_forces: bool) -> dict:
    F = np.asarray(res["served_forces"])
    fmag = np.linalg.norm(F, axis=-1)
    verdict = "send_to_dft" if res["flagged"] else "trust"
    out = {
        "index": k, "name": sys_in.get("name"), "formula": formula(sys_in["Z"]), "n_atoms": len(sys_in["Z"]),
        "verdict": verdict,
        "trust_score": res["trust"],
        "recommended": res["recommended_model"],
        "served_members": res["served_members"],
        "served_energy_eV": round(_served_energy(res), 6),
        "max_force_eV_per_A": round(float(fmag.max()), 5),
        "mean_force_eV_per_A": round(float(fmag.mean()), 5),
        "sigma_F": round(res["sigma_F"], 5), "sigma_F_max_atom": round(res["sigma_F_max"], 5),
        "tau": round(snap.tau, 5),
        "per_model": {m: {"energy_eV": round(v["energy"], 6), "max_force_eV_per_A": round(v["max_force"], 5)}
                      for m, v in res["per_model"].items()},
        "domain_warning": domain_warning(sys_in["Z"], sys_in["R"]),
    }
    if include_forces:
        out["served_forces_eV_per_A"] = np.round(F, 6).tolist()
    return out


def explain_one(sys_in: dict, res: dict, outs: dict, snap: TrustSnapshot) -> dict:
    models = list(snap.models)
    F = np.stack([np.asarray(outs[m]["forces"]) for m in models])  # (M, n, 3) — single system
    Fbar = F.mean(axis=0)
    dev = {m: float(np.sqrt(((F[i] - Fbar) ** 2).sum(-1).mean())) for i, m in enumerate(models)}
    pair = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pair[f"{models[i]} vs {models[j]}"] = round(float(np.sqrt(((F[i] - F[j]) ** 2).sum(-1).mean())), 5)
    per_atom = np.sqrt(((F - Fbar) ** 2).sum(-1).mean(0))  # (n,)
    worst = np.argsort(-per_atom)[:3]
    bkt = res["bucket"]
    se, fe = snap.sigma_edges, snap.fbar_edges

    def rng(edges, i):
        lo = "-inf" if i == 0 else f"{edges[i - 1]:.3f}"
        hi = "inf" if i >= len(edges) else f"{edges[i]:.3f}"
        return f"[{lo}, {hi})"

    p_ok = {c: round(float(snap.p_correct[c][bkt[0]][bkt[1]]), 4) for c in snap.candidates
            if c in snap.p_correct}
    n_bucket = None
    if snap.counts:
        c0 = next(iter(snap.counts.values()))
        n_bucket = int(c0[bkt[0]][bkt[1]])
    s, tau = res["sigma_F"], snap.tau
    wt = snap.meta.get("wrong_thresholds") or {}
    verdict = "send_to_dft" if res["flagged"] else "trust"
    most = max(dev, key=dev.get) if dev else None
    reason = (f"sigma_F = {s:.3f} eV/A {'>=' if res['flagged'] else '<'} tau = {tau:.3f}: "
              + ("the committee disagrees more than the Calibrator's escalation threshold, so this "
                 "structure goes to DFT." if res["flagged"] else
                 f"the committee agrees well enough; the Calibrator serves {res['recommended_model']} "
                 f"(highest P(correct) in this bucket)."))
    return {
        "formula": formula(sys_in["Z"]), "verdict": verdict, "reason": reason,
        "sigma_F": round(s, 5), "tau": round(tau, 5), "sigma_over_tau": round(s / tau, 3),
        "trust_score": res["trust"],
        "trust_score_meaning": ("1 - P(served answer wrong | " + snap.risk_feature + "), isotonic map learned "
                                "from DFT oracle labels; 'wrong' = force RMSE > "
                                f"{wt.get('force_rmse', 0.25)} eV/A or relative energy error > "
                                f"{wt.get('rel_e', 0.25)}"),
        "bucket": {"index": bkt, "sigma_F_range": rng(se, bkt[0]), "mean_abs_force_range": rng(fe, bkt[1]),
                   "oracle_labels_in_bucket": n_bucket},
        "p_correct_in_bucket": dict(sorted(p_ok.items(), key=lambda kv: -kv[1])),
        "recommended": res["recommended_model"], "served_members": res["served_members"],
        "member_deviation_from_mean_eV_per_A": {m: round(v, 5) for m, v in dev.items()},
        "most_divergent_member": most,
        "pairwise_force_rms_eV_per_A": pair,
        "most_uncertain_atoms": [{"atom": int(i), "element": SYMBOLS[int(sys_in["Z"][i])],
                                  "sigma_eV_per_A": round(float(per_atom[i]), 5)} for i in worst],
        "domain_warning": domain_warning(sys_in["Z"], sys_in["R"]),
    }


# --------------------------------------------------------------------------- swarm status
def family_tree_path() -> Path:
    return Path(os.environ.get("FITTEST_FAMILY_TREE", ROOT / "results" / "evolution" / "family_tree.json"))


def family_tree_summary(max_chars: int = 6000) -> dict:
    p = family_tree_path()
    gh = f"{REPO_URL}/blob/main/results/evolution/family_tree.json"
    if not p.exists():
        return {"available": False, "note": "no family tree published yet (evolution run in progress)",
                "github": gh}
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "note": f"family tree unreadable ({type(exc).__name__})", "github": gh}
    text = json.dumps(data)
    out: dict[str, Any] = {"available": True, "github": gh,
                           "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))}
    if len(text) <= max_chars:
        out["tree"] = data
    elif isinstance(data, dict):
        out["summary"] = {k: (v if not isinstance(v, (list, dict)) or len(json.dumps(v)) < 800
                              else f"<{type(v).__name__} of {len(v)} items>") for k, v in data.items()}
    else:
        out["summary"] = f"<{type(data).__name__} of {len(data)} items; see github>"
    return out


def swarm_status_payload(engine: Engine) -> dict:
    snap = engine.full
    meta = snap.meta
    rep = meta.get("reputation", {})
    n_labels = None
    if snap.counts:
        c0 = next(iter(snap.counts.values()))
        n_labels = int(np.asarray(c0).sum())
    return {
        "mode": engine.mode,
        "fake_models": engine.mode == "fake",
        "population": {
            "answerers": [{"agent": m, "role": "answerer (model agent: predicts energy + forces)",
                           "model": LEVEL_OF_THEORY.get(m, m), "status": engine.model_status.get(m),
                           "reputation": rep.get(m)} for m in snap.models],
            "critics": [{"agent": "skeptic", "role": "critic (audits the Calibrator's trust claims "
                         "against held-out oracle labels)", "reputation": rep.get("skeptic")}],
            "coordinators": [
                {"agent": "calibrator", "role": "learns P(correct | sigma_F, |F| bucket) per candidate from "
                 "DFT oracle labels; picks the served model or blend; sets tau"},
                {"agent": "oracle", "role": "spends the DFT label budget on escalated structures"},
                {"agent": "curator", "role": "repairs the weakest model where none can be trusted (LoRA)"}],
        },
        "ledger": {"reputation": rep, "oracle_labels_used_for_calibration": n_labels,
                   "budget": meta.get("budget"), "global_p_correct": snap.p_global,
                   "candidates": snap.candidates},
        "calibration": {
            "domain": "gas-phase C/H/N/O reaction chemistry: Transition1x (reactants, products, NEB paths, "
                      "transition states), <= 7 heavy atoms; reference DFT wB97x/6-31G(d)",
            "tau_sigma_F_eV_per_A": snap.tau, "sigma_edges": snap.sigma_edges, "fbar_edges": snap.fbar_edges,
            "risk_feature": snap.risk_feature, "risk_feature_cv_auroc": meta.get("risk_feature_cv_auroc"),
            "wrong_thresholds": meta.get("wrong_thresholds"), "snapshot_created": meta.get("created"),
            "outside_domain": "trust scores extrapolate; check_structures adds a domain_warning",
        },
        "active_models": engine.active,
        "committee_complete": engine.active == list(snap.models),
        "family_tree": family_tree_summary(),
        "repo": REPO_URL,
    }


# --------------------------------------------------------------------------- MCP app
_TOOL_DESCRIPTIONS = {
    "check_structures": (
        "Ask the Fittest swarm whether a machine-learned interatomic potential can be trusted on your "
        "molecular structures, or whether they need DFT. Three model agents (MACE-MPA-0, AIMNet2, Orb-v3) "
        "predict forces; their disagreement sigma_F is turned into a calibrated verdict.\n\n"
        "Input `structures`: XYZ text (single or multi-frame; bare 'El x y z' lines also work) and/or a "
        "list of objects {\"Z\": [8,1,1] or \"symbols\": [\"O\",\"H\",\"H\"], \"R\": [[x,y,z], ...]} in "
        f"Angstrom. Limits: <= {MAX_STRUCTURES} structures per call, <= {MAX_ATOMS} atoms each, "
        "non-periodic molecules.\n\n"
        "Returns per structure: verdict (trust / send_to_dft), trust_score (0-1), recommended model or blend, "
        "served energy (eV) and max/mean force magnitude (eV/A), sigma_F vs tau, per-model summary, and a "
        "domain_warning when outside the calibration domain (C/H/N/O, Transition1x sizes). Set "
        "include_forces=true for full force arrays.\n\n" + LOGGING_NOTICE),
    "explain": (
        "Explain why Fittest trusted or escalated ONE structure (XYZ text or {Z|symbols, R}): sigma_F vs the "
        "escalation threshold tau, the Calibrator bucket and P(correct) of every candidate in it, which model "
        "agents disagree most, and the most uncertain atoms.\n\n" + LOGGING_NOTICE),
    "swarm_status": (
        "Show the Fittest population: model agents (answerers) and the Skeptic (critic) with reputations from "
        "the ledger, the Calibrator's calibration domain and threshold, which models are live, and the "
        "evolution family tree if published.\n\n" + LOGGING_NOTICE),
    "usage_stats": (
        "Aggregate usage of this server: distinct users (anonymous client ids, internal test clients "
        "excluded), total calls, calls per tool, first/last call time. No per-user data.\n\n" + LOGGING_NOTICE),
}

INSTRUCTIONS = (
    "Fittest decides which ML interatomic potential to trust on a molecular structure, and when to "
    "escalate to DFT. Use check_structures for verdicts, explain for the reasoning on one structure, "
    "swarm_status for the agent population. Calibrated on C/H/N/O reaction chemistry (Transition1x). "
    + LOGGING_NOTICE)


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "internal")


class Caller:
    """Who is calling, derived from the MCP context (HTTP request / session client info)."""

    def __init__(self, ctx, ulog: UsageLog, transport: str, trusted_proxies: set[str]):
        name = version = ip = None
        internal = _truthy(os.environ.get("FITTEST_INTERNAL"))
        req = None
        try:
            params = ctx.session.client_params
            if params is not None and params.clientInfo is not None:
                name, version = params.clientInfo.name, params.clientInfo.version
        except Exception:  # noqa: BLE001
            pass
        try:
            req = ctx.request_context.request
        except Exception:  # noqa: BLE001
            req = None
        if req is not None and getattr(req, "headers", None) is not None:
            peer = req.client.host if req.client else None
            xff = req.headers.get("x-forwarded-for")
            ip = xff.split(",")[-1].strip() if (xff and peer in trusted_proxies) else peer
            internal = internal or _truthy(req.headers.get("x-fittest-internal"))
            if name is None:
                name = (req.headers.get("user-agent") or "?")[:60]
        self.transport = transport
        self.client_id = ulog.client_id(name, version, ip)
        self.ip_key = ulog.client_id("ip", "", ip)
        if internal:
            ulog.mark_internal(self.client_id, "marked internal by header/env")
        self.internal = internal or ulog.is_internal_client(self.client_id)


def build_server(engine: Engine, ulog: UsageLog, transport: str = "stdio", host: str = "127.0.0.1",
                 port: int = 8765, public_host: str | None = None,
                 limiter: TokenBucket | None = None, ip_limiter: TokenBucket | None = None):
    import anyio
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse

    security = None
    if public_host:  # public server behind a TLS proxy: no local secrets to protect from DNS rebinding
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    mcp = FastMCP("fittest", instructions=INSTRUCTIONS, website_url=REPO_URL, host=host, port=port,
                  transport_security=security, max_request_body_size=2 * 1024 * 1024)
    limiter = limiter or TokenBucket(capacity=20, rate=0.5)
    ip_limiter = ip_limiter or TokenBucket(capacity=60, rate=1.0)
    trusted = {"127.0.0.1", "::1"}

    async def run_logged(tool: str, ctx, fn):
        t0 = time.perf_counter()
        caller = Caller(ctx, ulog, transport, trusted)
        rec = CallRecord(tool=tool, client_id=caller.client_id, transport=transport, internal=caller.internal)
        try:
            ok, wait = limiter.allow(caller.client_id)
            ok2, wait2 = ip_limiter.allow(caller.ip_key) if ok else (True, 0.0)
            if not (ok and ok2):
                raise RateLimited(f"rate limit: retry in {max(wait, wait2):.0f} s")
            with anyio.fail_after(REQUEST_TIMEOUT_S):
                payload = await anyio.to_thread.run_sync(lambda: fn(rec), abandon_on_cancel=True)
            rec.ok = True
            return payload
        except InputError as exc:
            rec.ok, rec.error = False, "input_error"
            raise ToolError(str(exc)) from None
        except RateLimited as exc:
            rec.ok, rec.error = False, "rate_limited"
            raise ToolError(str(exc)) from None
        except (TimeoutError, anyio.get_cancelled_exc_class()) as exc:
            rec.ok, rec.error = False, "timeout"
            if isinstance(exc, TimeoutError):
                raise ToolError("the swarm is busy or the request timed out; retry shortly") from None
            raise
        except Exception as exc:  # noqa: BLE001
            rec.ok, rec.error = False, type(exc).__name__
            log.exception("tool %s failed", tool)
            raise ToolError(f"internal error ({type(exc).__name__}); the model agents may be restarting") from None
        finally:
            rec.latency_ms = (time.perf_counter() - t0) * 1000
            try:
                ulog.record(rec)
            except Exception:  # noqa: BLE001
                log.exception("usage log write failed")

    def header(snap: TrustSnapshot) -> dict:
        h = {"models_active": list(snap.models), "committee_complete": list(snap.models) == list(engine.full.models),
             "mode": engine.mode}
        if engine.mode == "fake":
            h["warning"] = "FAKE MODELS: stand-in predictors for demos/tests; verdicts are meaningless."
        elif not h["committee_complete"]:
            h["warning"] = (f"partial committee ({', '.join(snap.models)}): the Calibrator learned tau and the "
                            f"trust map on {', '.join(engine.full.models)}; verdicts are approximate.")
        return h

    @mcp.tool(description=_TOOL_DESCRIPTIONS["check_structures"])
    async def check_structures(structures: StructuresArg, include_forces: bool = False,
                               ctx: Context = None) -> dict:
        def work(rec: CallRecord):
            systems = parse_structures(structures)
            rec.n_structures, rec.n_atoms = len(systems), sum(len(s["Z"]) for s in systems)
            results, _outs, snap = engine.evaluate(systems)
            rows = [summarize(s, r, snap, k, include_forces) for k, (s, r) in enumerate(zip(systems, results))]
            rec.n_trust = sum(r["verdict"] == "trust" for r in rows)
            rec.n_dft = len(rows) - rec.n_trust
            return {**header(snap), "n_structures": len(rows),
                    "summary": {"trust": rec.n_trust, "send_to_dft": rec.n_dft},
                    "results": rows,
                    "notes": "Energies are each model's absolute total energy on its own reference; compare "
                             "differences within one model. Forces in eV/A.",
                    "logging": LOGGING_NOTICE}
        return await run_logged("check_structures", ctx, work)

    @mcp.tool(description=_TOOL_DESCRIPTIONS["explain"])
    async def explain(structure: StructuresArg, ctx: Context = None) -> dict:
        def work(rec: CallRecord):
            systems = parse_structures(structure)
            if len(systems) != 1:
                raise InputError(f"explain takes exactly one structure (got {len(systems)}); "
                                 "use check_structures for batches")
            rec.n_structures, rec.n_atoms = 1, len(systems[0]["Z"])
            results, outs, snap = engine.evaluate(systems)
            ex = explain_one(systems[0], results[0], outs, snap)
            rec.n_trust = int(ex["verdict"] == "trust")
            rec.n_dft = 1 - rec.n_trust
            return {**header(snap), **ex}
        return await run_logged("explain", ctx, work)

    @mcp.tool(description=_TOOL_DESCRIPTIONS["swarm_status"])
    async def swarm_status(ctx: Context = None) -> dict:
        return await run_logged("swarm_status", ctx, lambda rec: swarm_status_payload(engine))

    @mcp.tool(description=_TOOL_DESCRIPTIONS["usage_stats"])
    async def usage_stats(ctx: Context = None) -> dict:
        def work(rec):
            st = ulog.stats()
            st["scope"] = ("this hosted server" if transport != "stdio"
                           else "this local install only (local mode never reports anywhere)")
            return st
        return await run_logged("usage_stats", ctx, work)

    @mcp.custom_route("/stats", methods=["GET"])
    async def stats_route(request: Request):
        st = ulog.stats()
        st["active_models"] = engine.active
        return JSONResponse(st, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"})

    @mcp.custom_route("/health", methods=["GET"])
    async def health_route(request: Request):
        return JSONResponse({"ok": bool(engine.active), "mode": engine.mode, "active_models": engine.active,
                             "model_status": engine.model_status})

    @mcp.custom_route("/", methods=["GET"])
    async def landing(request: Request):
        url = f"https://{public_host}/mcp" if public_host else f"http://{host}:{port}/mcp"
        return HTMLResponse(landing_html(url, engine))

    return mcp


class RateLimited(Exception):
    pass


def landing_html(url: str, engine: Engine) -> str:
    models = ", ".join(LEVEL_OF_THEORY.get(m, m) for m in engine.full.models)
    live = ", ".join(engine.active) or "warming up"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Fittest MCP</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mute:#5f5e5a;--card:#fff;--line:#e4e1d8;--acc:#1f6f50}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151614;--fg:#ecebe6;--mute:#a9a79f;--card:#1e1f1c;--line:#34352f;--acc:#6fcf9f}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:760px;margin:0 auto}} h1{{margin:0 0 4px;font-size:30px}} h2{{font-size:19px;margin-top:28px}}
.sub{{color:var(--mute);margin:0 0 18px}} code,pre{{font:13.5px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
pre{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;white-space:pre-wrap;word-break:break-all}}
a{{color:var(--acc)}} .stat{{display:inline-block;border:1px solid var(--line);background:var(--card);border-radius:8px;padding:8px 14px;margin:4px 8px 4px 0}}
.stat b{{font-size:22px;display:block}} li{{margin:4px 0}}
</style></head><body><main>
<h1>Fittest MCP</h1>
<p class="sub">Survival of the fittest, in materials science. Ask a swarm of ML interatomic potentials whether
to trust them on your molecule, or send it to DFT.</p>
<div><span class="stat"><b id="u">-</b>users</span><span class="stat"><b id="c">-</b>calls</span>
<span class="stat"><b>{live}</b>live models</span></div>
<h2>What it does</h2>
<p>Model agents ({models}) predict forces on your structure. Their disagreement &sigma;<sub>F</sub> is mapped by a
Calibrator, learned from DFT oracle labels on Transition1x reactions, to a verdict (<code>trust</code> /
<code>send_to_dft</code>), a trust score and the model or blend to use. Calibrated for gas-phase C/H/N/O
chemistry; elsewhere the score is an extrapolation and the response says so.</p>
<p>Tools: <code>check_structures</code>, <code>explain</code>, <code>swarm_status</code>, <code>usage_stats</code>.
Limits: {MAX_STRUCTURES} structures per call, {MAX_ATOMS} atoms each, rate-limited per client.</p>
<h2>Connect</h2>
<p>Claude Code</p><pre>claude mcp add --transport http fittest {url}</pre>
<p>Claude Desktop / claude.ai: Settings &rarr; Connectors &rarr; Add custom connector &rarr; <code>{url}</code></p>
<p>Cursor (<code>~/.cursor/mcp.json</code>)</p><pre>{{"mcpServers": {{"fittest": {{"url": "{url}"}}}}}}</pre>
<p>Run it locally instead (stdio, CPU; uses the models you have installed, logs only on your machine)</p>
<pre>claude mcp add fittest -- uvx --from git+{REPO_URL} fittest-mcp
uvx --from git+{REPO_URL} fittest-mcp --fake-models   # demo without models</pre>
<p>Then ask: <i>"Use fittest to check whether I can trust an ML potential on this water molecule: O 0 0 0, H 0.96 0 0,
H -0.24 0.93 0."</i></p>
<h2>What is logged</h2>
<p>{LOGGING_NOTICE.replace("Logging: ", "")} Live counts: <a href="/stats">/stats</a>.</p>
<h2>Source</h2><p><a href="{REPO_URL}">{REPO_URL}</a> (MIT)</p>
</main><script>
fetch('/stats').then(r=>r.json()).then(s=>{{document.getElementById('u').textContent=s.users;
document.getElementById('c').textContent=s.external_calls}}).catch(()=>{{}});
</script></body></html>"""


# --------------------------------------------------------------------------- startup
def default_snapshot_path() -> str:
    env = os.environ.get("FITTEST_SNAPSHOT")
    if env:
        return env
    repo = ROOT / "results" / "trust_snapshot.json"
    if repo.exists():
        return str(repo)
    return str(Path(__file__).resolve().parent / "snapshots" / "trust_snapshot.json")


def default_data_dir() -> str:
    return os.environ.get("FITTEST_MCP_DATA", str(Path.home() / ".fittest-mcp"))


def installed_models(orb_python: str | None) -> list[str]:
    """Model agents this machine can run: MACE + AIMNet2 via nvalchemi in this env, Orb in its own env."""
    have = []
    if importlib.util.find_spec("nvalchemi") and importlib.util.find_spec("mace"):
        have.append("mace_mpa0")
    if importlib.util.find_spec("aimnet"):
        have.append("aimnet2")
    if (orb_python and Path(orb_python).exists()) or importlib.util.find_spec("orb_models"):
        have.append("orb")
    return have


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


WATER = {"Z": [8, 1, 1], "R": [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]}


def _warm(engine: Engine, name: str, pred) -> None:
    from fittest.service import _pack

    Z, R, b = _pack([WATER])
    pred(Z, R, b)
    engine.add_model(name, pred)
    log.info("model agent %s warm", name)


def start_hosted_models(engine: Engine, ports: dict[str, int]) -> None:
    """Connect to systemd-managed model servers; keep retrying the ones not up yet."""
    def loop():
        while True:
            for m, port in ports.items():
                if m in engine.predictors and not engine.model_status.get(m, "").startswith("error"):
                    continue
                try:
                    rm = RemoteModel(m, port).connect(timeout=3.0)
                    engine.model_status[m] = "warming"
                    engine.predictors.pop(m, None)
                    _warm(engine, m, rm.predict)
                    engine.closers.append(rm)
                except Exception as exc:  # noqa: BLE001
                    engine.model_status[m] = f"waiting for model server on port {port} ({type(exc).__name__})"
            time.sleep(10)
    threading.Thread(target=loop, daemon=True, name="model-connector").start()


def start_local_models(engine: Engine, models: list[str], device: str, orb_python: str | None,
                       log_dir: str) -> None:
    from fittest.model_server import spawn

    def boot(m: str):
        try:
            engine.model_status[m] = "loading"
            python = orb_python if (m == "orb" and orb_python) else sys.executable
            client = spawn(m, _free_port(), python=python, device=device, log_dir=log_dir).connect(timeout=1800)
            engine.closers.append(client)
            _warm(engine, m, client.predict)
        except Exception as exc:  # noqa: BLE001
            engine.model_status[m] = f"failed: {type(exc).__name__}: {str(exc)[:200]}"
            log.error("model agent %s failed to start: %s", m, exc)

    for m in models:
        threading.Thread(target=boot, args=(m,), daemon=True, name=f"boot-{m}").start()


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fittest-mcp", description="Fittest MCP server (trust verdicts for MLIPs).")
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--public-host", default=None, help="public hostname when served behind a TLS proxy")
    ap.add_argument("--snapshot", default=None, help="Calibrator snapshot (default: packaged trust_snapshot.json)")
    ap.add_argument("--fake-models", action="store_true",
                    help="CPU stand-in predictors (no torch needed); verdicts are meaningless")
    ap.add_argument("--models", default=None, help="comma list of model agents to run (default: all installed)")
    ap.add_argument("--connect", default=None,
                    help="use running model servers instead of spawning: mace_mpa0=6300,aimnet2=6301,orb=6302")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--orb-python", default=os.environ.get("FITTEST_ORB_PYTHON"))
    ap.add_argument("--data-dir", default=None, help="usage log + salt (default: ~/.fittest-mcp)")
    ap.add_argument("--salt-file", default=os.environ.get("FITTEST_SALT_FILE"))
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=a.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    snap_path = a.snapshot or default_snapshot_path()
    snap = TrustSnapshot.load(snap_path)
    data_dir = Path(a.data_dir or default_data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    ulog = UsageLog(data_dir / "usage.sqlite", salt_path=a.salt_file or data_dir / "salt")
    wanted = a.models.split(",") if a.models else list(snap.models)

    if a.fake_models:
        engine = Engine(snap, "fake", snap_path)
        for m, p in fake_predictors(wanted).items():
            engine.add_model(m, p)
    elif a.connect:
        engine = Engine(snap, "hosted", snap_path)
        ports = {k: int(v) for k, v in (kv.split("=") for kv in a.connect.split(","))}
        start_hosted_models(engine, ports)
    else:
        engine = Engine(snap, "local", snap_path)
        have = [m for m in installed_models(a.orb_python) if m in wanted]
        missing = [m for m in snap.models if m not in have]
        for m in missing:
            engine.model_status[m] = "not installed"
        if not have:
            print("fittest-mcp: no model agents installed (need nvalchemi-toolkit[mace,aimnet] and/or orb-models "
                  "via FITTEST_ORB_PYTHON). Serving anyway; tools will report this. Use --fake-models for a demo.",
                  file=sys.stderr)
        else:
            print(f"fittest-mcp: starting model agents {have}" + (f" (not installed: {missing})" if missing else ""),
                  file=sys.stderr)
        start_local_models(engine, have, a.device, a.orb_python, str(data_dir / "logs"))
    atexit.register(engine.close)
    # Import everything the tools need on the main thread now. On Windows, a worker thread that
    # imports sklearn/joblib blocks while the stdio transport sits in a blocking stdin read.
    try:
        import fittest.agents.calibrator  # noqa: F401
    except ImportError:
        pass
    if engine.mode == "fake":
        engine.evaluate([WATER])

    server = build_server(engine, ulog, transport=a.transport, host=a.host, port=a.port,
                          public_host=a.public_host)
    if a.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http")


if __name__ == "__main__":
    main()
