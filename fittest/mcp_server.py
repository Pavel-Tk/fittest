"""Fittest MCP server: ask the swarm whether to trust an ML potential on your structures.

Hosted (streamable HTTP, behind Caddy/TLS; model agents are separate warm processes)::

    fittest-mcp --transport http --port 8765 --public-host fittest-mcp.westus2.cloudapp.azure.com \\
        --connect mace_mpa0=7300,aimnet2=7301,orb=7302 --data-dir /var/lib/fittest-mcp

Local (stdio; spawns whichever model agents are installed in *this* environment; logs only locally)::

    fittest-mcp                  # from an env with the [models] extra (MACE + AIMNet2), + FITTEST_ORB_PYTHON
    uvx --from git+https://github.com/Pavel-Tk/fittest fittest-mcp --fake-models  # demo / tests only

Tools: ``check_structures``, ``explain``, ``swarm_status``, ``usage_stats``. Every call is logged
by :mod:`fittest.usage_log` (counts and an anonymous salted-hash client id, never structures).

Verdict rule (one rule, used everywhere): a structure is ``trust`` only if all model agents can
handle its elements, their forces do not all vanish, σ_F < τ, and the trust score >= 0.5;
otherwise ``send_to_dft`` with the reasons listed. The trust score is the Calibrator's Beta
posterior P(served answer correct | bucket) for the served candidate (prior Beta(2, 1), so never
exactly 1), reported with its 90 % credible interval and the number of DFT labels behind it.
τ is fixed: the 70th percentile of round-0 σ_F; it never moved during the run (results/RESULTS.md).
"""

from __future__ import annotations

import argparse
import atexit
import contextvars
import importlib.util
import json
import logging
import os
import re
import socket
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np

from fittest import __version__
from fittest.service import TrustSnapshot, _pack, fake_predictors, score
from fittest.usage_log import USERS_DEFINITION, CallRecord, TokenBucket, UsageLog

try:  # module level so FastMCP can resolve the tool annotations
    from mcp.server.fastmcp import FastMCP as _FastMCP
except ImportError:  # parsing/engine helpers stay importable without the SDK
    _FastMCP = object  # type: ignore[misc,assignment]

#: what `structures` accepts: XYZ text, one {Z|symbols, R} object, or a list mixing both
StructuresArg = Union[str, list[Union[str, dict[str, Any]]], dict[str, Any]]

log = logging.getLogger("fittest.mcp")

REPO_URL = "https://github.com/Pavel-Tk/fittest"
BRANCH = "master"
PUBLIC_URL = "https://fittest-mcp.westus2.cloudapp.azure.com/mcp"
ROOT = Path(__file__).resolve().parents[1]
TOOL_NAMES = ("check_structures", "explain", "swarm_status", "usage_stats")

MAX_STRUCTURES = 20
MAX_ATOMS = 100
MAX_TEXT_CHARS = 400_000
REQUEST_TIMEOUT_S = 120.0
BUSY_TIMEOUT_S = 90.0
TRUST_MIN_SCORE = 0.5
FORCE_VANISH_EV_A = 0.05  # every model's max |F| below this: forces carry no disagreement signal

CAL_ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O"}
T1X_MAX_HEAVY = 7  # Transition1x: GDB7-derived reactions, <= 7 heavy atoms (C/N/O)
T1X_MAX_ATOMS = 26

#: elements each model agent returns finite energies/forces for (probed on the hosted agents,
#: X-H dimers for Z = 1..118; AIMNet2 is the binding constraint for the full committee)
MODEL_ELEMENTS = {
    "mace_mpa0": frozenset(range(1, 95)),
    "aimnet2": frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}),
    "orb": frozenset(range(1, 118)),
}
#: model whose energy is reported as the batch-comparable ``reference_energy_eV`` (first present)
REFERENCE_ENERGY_ORDER = ("aimnet2", "mace_mpa0", "orb")

LEVEL_OF_THEORY = {
    "mace_mpa0": "MACE-MPA-0 medium, PBE/PBE+U (MPtrj + sAlex)",
    "aimnet2": "AIMNet2, wB97M-D3(BJ)/def2-TZVPP",
    "orb": "Orb-v3 conservative, PBE/PBE+U (OMat24)",
}

TAU_MEANING = ("tau is a fixed escalation threshold: the 70th percentile of sigma_F on the round-0 pool. "
               "The Calibrator's threshold rule never moved it during the run (see results/RESULTS.md).")
SCORE_MEANING = ("trust_score = the Calibrator's Beta posterior mean of P(served answer correct) for the served "
                 "candidate in this (sigma_F, |F|) bucket, from DFT oracle labels on Transition1x; prior Beta(2, 1), "
                 "so it is never exactly 1. 'Correct' = force RMSE <= 0.25 eV/A and relative energy error <= 0.25 eV. "
                 "Its calibration has only been spot-checked (12 held-out frames), not measured. "
                 "labels_behind_score is the number of DFT labels in that bucket; few labels = wide interval.")
VERDICT_RULE = ("trust only if: every model agent supports all elements, the forces do not all vanish, "
                "sigma_F < tau, and trust_score >= 0.5. Otherwise send_to_dft, with verdict_reasons.")

LOGGING_NOTICE = (
    "Logging: every call to this server is logged (UTC time, tool name, an anonymous client id = "
    "salted hash of client name + IP), structure and atom counts, verdict counts, latency and "
    "ok/error category. Raw IPs are not stored by the app or written to the access log; since "
    "2026-10-31 19:11 UTC Caddy's error/unmapped-host logs drop them too (before that, some Caddy "
    "error and bare-IP entries in the system journal contained client IPs, kept until journal "
    "rotation). Structures, coordinates, element lists and names are NOT stored. Aggregate counts "
    "are public at /stats.")

SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se "
    "Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy "
    "Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf "
    "Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
SYMBOL_TO_Z = {s: z for z, s in enumerate(SYMBOLS) if z > 0}

#: errors that mean the link to a model agent is broken (vs the agent rejecting one input)
CONN_ERRORS = (EOFError, OSError, ConnectionError)


class InputError(ValueError):
    """Bad user input; message is shown to the caller (never logged)."""


class ModelsNotReady(RuntimeError):
    """No (or too few) model agents are connected."""


class Busy(RuntimeError):
    """Another evaluation held the swarm for too long."""


class RateLimited(RuntimeError):
    pass


# --------------------------------------------------------------------------- parsing
def _z_of(token: Any, where: str) -> int:
    """Exact element symbols (canonical case, or all-lowercase), atomic numbers, or PDB-style
    labels that are a symbol plus trailing digits ("C1", "H12"). Anything else is an error:
    labels like "CA", "HE1" or "Hello" are ambiguous or not elements."""
    if isinstance(token, (int, np.integer)) and not isinstance(token, bool):
        z = int(token)
    else:
        t = str(token).strip()
        if t.isascii() and t.isdigit():
            z = int(t)
        else:
            base = t.rstrip("0123456789") if t.isascii() else t
            canon = base[:1].upper() + base[1:].lower()
            ok_case = base == canon or len(base) == 1 or base == base.lower()
            if not (base and canon in SYMBOL_TO_Z and ok_case and base.isascii() and base.isalpha()):
                hint = ""
                if base.isascii() and base.isalpha() and len(base) == 2 and canon in SYMBOL_TO_Z:
                    hint = f" (write {canon} for {canon}, or {base[0]} if you meant that element)"
                raise InputError(f"{where}: {token!r} is not an element symbol{hint}; use symbols like "
                                 "C, H, Cl or atomic numbers")
            z = SYMBOL_TO_Z[canon]
    if not 1 <= z <= 118:
        raise InputError(f"{where}: atomic number {z} outside 1..118")
    return z


_PERIODIC = re.compile(r"Lattice\s*=|pbc\s*=\s*\"?\s*[Tt]")


def _err(k: int, msg: str, name: str | None = None) -> dict:
    return {"index": k, "name": name, "error": msg}


def _expand_bare_rows(lines: list[str]) -> list[str]:
    """Bare coordinates: one atom per line, or atoms separated by ';' or ',' (e.g. the prompt
    'O 0 0 0, H 0.96 0 0, H -0.24 0.93 0'). Commas between coordinates also work."""
    rows: list[str] = []
    for ln in lines:
        for chunk in ln.split(";"):
            if not chunk.strip():
                continue
            parts = [p for p in chunk.split(",") if p.strip()]
            if len(parts) > 1 and all(len(p.split()) == 4 for p in parts):
                rows.extend(parts)
            else:
                rows.append(chunk.replace(",", " "))
    return rows


def parse_xyz_text(text: str, start_index: int = 0) -> list[dict]:
    """XYZ (single or multi-frame, extended-XYZ comments fine) or bare 'El x y z' atoms.
    Returns one entry per structure: a structure dict, or ``{"index", "error"}``."""
    if len(text) > MAX_TEXT_CHARS:
        raise InputError(f"XYZ text too long ({len(text)} chars > {MAX_TEXT_CHARS})")
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").strip("\n").split("\n")]
    first = next((ln for ln in lines if ln.strip()), "")
    if not first:
        raise InputError("empty XYZ text")
    out: list[dict] = []
    head = first.split()[0]
    if not (head.isascii() and head.lstrip("+-").isdigit() and len(first.split()) == 1):
        rows = _expand_bare_rows([ln for ln in lines if ln.strip()])
        out.append(_try_rows(rows, start_index, None))
        return out
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        k = start_index + len(out)
        tok = lines[i].split()[0]
        if not (tok.isascii() and tok.isdigit()):
            out.append(_err(k, f"XYZ line {i + 1}: expected an atom count (a positive integer); parsing "
                               "stopped here"))
            break
        n = int(tok)
        if n < 1:
            out.append(_err(k, f"XYZ line {i + 1}: atom count must be >= 1; parsing stopped here"))
            break
        comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
        rows = lines[i + 2:i + 2 + n]
        name = comment.split("Lattice")[0].strip()[:80] or None
        if n > MAX_ATOMS:
            out.append(_err(k, f"structure {k}: {n} atoms > limit {MAX_ATOMS}", name))
        elif len(rows) < n or any(not r.strip() for r in rows):
            out.append(_err(k, f"structure {k}: XYZ frame truncated (expected {n} atom lines); parsing "
                               "stopped here", name))
            break
        elif _PERIODIC.search(comment):
            out.append(_err(k, f"structure {k}: periodic structure (Lattice=/pbc in the comment line); "
                               "Fittest is calibrated for isolated molecules only", name))
        else:
            out.append(_try_rows(rows, k, name))
        i += 2 + n
        if len(out) > MAX_STRUCTURES:
            raise InputError(f"more than {MAX_STRUCTURES} structures in one call; split the batch")
    return out


def _try_rows(rows: list[str], k: int, name: str | None) -> dict:
    try:
        s = _rows_to_structure(rows, f"structure {k}", name)
        _validate(s, k)
        return s
    except InputError as exc:
        return _err(k, str(exc), name)


def _rows_to_structure(rows: list[str], where: str, name: str | None) -> dict:
    Z, R = [], []
    for j, row in enumerate(rows):
        p = row.replace(",", " ").split()
        if len(p) < 4:
            raise InputError(f"{where}, atom {j}: expected 'element x y z'")
        Z.append(_z_of(p[0], f"{where}, atom {j}"))
        try:
            R.append([float(p[1]), float(p[2]), float(p[3])])
        except ValueError:
            raise InputError(f"{where}, atom {j}: coordinates must be numbers") from None
    if len(Z) > MAX_ATOMS:
        raise InputError(f"{where}: {len(Z)} atoms > limit {MAX_ATOMS}")
    return {"Z": Z, "R": R, "name": name}


def _dict_to_structure(d: dict, k: int) -> dict:
    where = f"structure {k}"
    name = d.get("name")
    name = None if name is None else str(name)[:80]
    pbc, cell = d.get("pbc"), d.get("cell", d.get("lattice"))
    if (pbc is not None and np.any(np.asarray(pbc, dtype=bool))) or \
            (cell is not None and np.size(cell) and np.any(np.asarray(cell, dtype=float) != 0)):
        raise InputError(f"{where}: periodic structure (pbc/cell given); Fittest is calibrated for isolated "
                         "molecules only")
    zs = next((d[key] for key in ("Z", "numbers", "atomic_numbers", "symbols", "elements", "species")
               if key in d), None)
    rs = next((d[key] for key in ("R", "positions", "coords", "coordinates", "xyz") if key in d), None)
    if zs is None or rs is None:
        raise InputError(f"{where}: give 'Z' (atomic numbers) or 'symbols', and 'R' (Angstrom, Nx3)")
    if isinstance(zs, str):
        zs = zs.replace(",", " ").split()
    if isinstance(rs, str):
        rs = rs.replace(",", " ").split()
    try:
        R = np.asarray(rs, dtype=float)
    except (TypeError, ValueError):
        raise InputError(f"{where}: 'R' must be an N x 3 list of numbers") from None
    if R.ndim == 1 and R.size % 3 == 0:
        R = R.reshape(-1, 3)
    if R.ndim != 2 or R.shape[1] != 3:
        raise InputError(f"{where}: 'R' must be an N x 3 list of numbers")
    if len(R) > MAX_ATOMS:
        raise InputError(f"{where}: {len(R)} atoms > limit {MAX_ATOMS}")
    Z = [_z_of(z, f"{where}, atom {j}") for j, z in enumerate(list(zs))]
    if len(Z) != len(R):
        raise InputError(f"{where}: {len(Z)} elements but {len(R)} positions")
    return {"Z": Z, "R": R.tolist(), "name": name}


def parse_structures(structures: Any) -> list[dict]:
    """XYZ text, one {Z|symbols, R} object, or a list mixing both. One bad structure gives one
    ``{"index", "error"}`` entry; only call-level problems (type, size, count) raise InputError."""
    if isinstance(structures, str):
        s = structures.strip()
        if s[:1] in "[{":
            try:
                return parse_structures(json.loads(s))
            except json.JSONDecodeError:
                pass
            except RecursionError:
                raise InputError("structures JSON is nested too deeply") from None
        items: list[Any] = [structures]
    elif isinstance(structures, dict):
        items = [structures]
    elif isinstance(structures, (list, tuple)):
        items = list(structures)
    else:
        raise InputError("structures must be XYZ text, a {Z|symbols, R} object, or a list of them")
    if len(items) > MAX_STRUCTURES:
        raise InputError(f"more than {MAX_STRUCTURES} structures in one call; split the batch")
    out: list[dict] = []
    for item in items:
        k = len(out)
        if isinstance(item, str):
            out.extend(parse_xyz_text(item, start_index=k))
        elif isinstance(item, dict):
            try:
                s = _dict_to_structure(item, k)
                _validate(s, k)
                out.append(s)
            except InputError as exc:
                out.append(_err(k, str(exc), None))
        else:
            out.append(_err(k, f"structure {k}: expected XYZ text or a {{Z|symbols, R}} object"))
        if len(out) > MAX_STRUCTURES:
            raise InputError(f"more than {MAX_STRUCTURES} structures in one call; split the batch")
    if not out:
        raise InputError("send at least one structure")
    return out


def _validate(s: dict, k: int) -> None:
    n = len(s["Z"])
    if n == 0:
        raise InputError(f"structure {k}: no atoms")
    if n == 1:
        raise InputError(f"structure {k}: a single atom has zero force by symmetry for every model, so "
                         "disagreement carries no information; send at least 2 atoms")
    if n > MAX_ATOMS:
        raise InputError(f"structure {k}: {n} atoms > limit {MAX_ATOMS}")
    R = np.asarray(s["R"], dtype=float)
    if not np.isfinite(R).all():
        raise InputError(f"structure {k}: coordinates must be finite")
    if np.abs(R).max() > 1000:
        raise InputError(f"structure {k}: coordinates beyond 1000 Angstrom; are units Angstrom?")
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


def unsupported_elements(Z: Sequence[int], models: Sequence[str]) -> dict[str, list[str]]:
    zs = {int(z) for z in Z}
    out = {}
    for m in models:
        bad = sorted(zs - MODEL_ELEMENTS.get(m, frozenset(range(1, 119))))
        if bad:
            out[m] = [SYMBOLS[z] for z in bad]
    return out


# --------------------------------------------------------------------------- scoring
def subset_snapshot(snap: TrustSnapshot, models: Sequence[str]) -> TrustSnapshot:
    """The snapshot restricted to the model agents that are actually running."""
    models = [m for m in snap.models if m in set(models)]
    if models == list(snap.models):
        return snap
    keep = [c for c in (snap.candidates or snap.models)
            if set(snap.members(c, snap.models)) <= set(models)]
    members = {c: list(snap.members(c, snap.models)) for c in keep}
    return TrustSnapshot(models=models, tau=snap.tau, sigma_edges=snap.sigma_edges,
                         fbar_edges=snap.fbar_edges,
                         p_correct={c: v for c, v in snap.p_correct.items() if c in keep},
                         counts={c: v for c, v in snap.counts.items() if c in keep},
                         candidates=keep, candidate_members=members,
                         p_global={c: v for c, v in snap.p_global.items() if c in keep},
                         tiebreak_order=[c for c in snap.tiebreak_order if c in keep],
                         risk_iso={c: v for c, v in snap.risk_iso.items() if c in keep},
                         risk_feature=snap.risk_feature, meta=dict(snap.meta))


def _prior() -> tuple[float, float]:
    try:
        from fittest.config import CORRECTNESS_PRIOR

        return float(CORRECTNESS_PRIOR[0]), float(CORRECTNESS_PRIOR[1])
    except Exception:  # noqa: BLE001
        return 2.0, 1.0


def posterior(snap: TrustSnapshot, cand: str, bucket: Sequence[int]) -> dict:
    """Beta posterior of P(correct) for a candidate in a bucket, as the Calibrator keeps it."""
    a0, b0 = _prior()
    p = float(snap.p_correct[cand][bucket[0]][bucket[1]])
    counts = snap.counts.get(cand) or (next(iter(snap.counts.values())) if snap.counts else None)
    n = int(counts[bucket[0]][bucket[1]]) if counts is not None else 0
    k = min(max(p * (n + a0 + b0) - a0, 0.0), float(n))
    a, b = a0 + k, b0 + n - k
    try:
        from scipy.stats import beta as _beta

        lo, hi = (float(x) for x in _beta.ppf([0.05, 0.95], a, b))
    except Exception:  # noqa: BLE001
        lo = hi = float("nan")
    return {"mean": a / (a + b), "ci90": [round(lo, 3), round(hi, 3)], "n_labels": n}


def decide(res: dict, snap: TrustSnapshot, forces: dict[str, np.ndarray]) -> dict:
    """The one verdict rule: see VERDICT_RULE."""
    post = posterior(snap, res["recommended_model"], res["bucket"])
    reasons = []
    if all(float(np.linalg.norm(F, axis=-1).max()) < FORCE_VANISH_EV_A for F in forces.values()):
        reasons.append(f"forces vanish for every model (max |F| < {FORCE_VANISH_EV_A} eV/A: symmetry or "
                       "exact equilibrium), so their disagreement carries no information")
    if res["sigma_F"] >= snap.tau:
        reasons.append(f"sigma_F = {res['sigma_F']:.3f} >= tau = {snap.tau:.3f}")
    if post["mean"] < TRUST_MIN_SCORE:
        reasons.append(f"trust_score = {post['mean']:.2f} < {TRUST_MIN_SCORE} "
                       f"({post['n_labels']} DFT labels in this bucket)")
    return {"verdict": "send_to_dft" if reasons else "trust", "verdict_reasons": reasons,
            "trust_score": round(post["mean"], 4), "trust_score_90ci": post["ci90"],
            "labels_behind_score": post["n_labels"]}


# --------------------------------------------------------------------------- engine
class RemoteModel:
    """A warm model-agent process reached over its socket. Exactly one link at a time: the old
    one is always closed before reconnecting (the server serves one connection at a time)."""

    def __init__(self, model: str, port: int):
        self.model, self.port = model, port
        self._client = None
        self._lock = threading.Lock()

    def _drop(self) -> None:
        if self._client is not None:
            try:
                self._client.close(shutdown=False)
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def _ensure(self, timeout: float) -> None:
        if self._client is None:
            from fittest.model_server import ModelClient

            self._client = ModelClient(self.model, self.port).connect(timeout=timeout)

    def reconnect(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._drop()
            self._ensure(timeout)

    def predict(self, Z, R, batch):
        with self._lock:
            self._ensure(5.0)
            try:
                return self._client.predict(Z, R, batch)
            except CONN_ERRORS:
                self._drop()
                raise

    def close(self) -> None:
        with self._lock:
            self._drop()


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

    def drop_model(self, name: str, why: str) -> None:
        self.predictors.pop(name, None)
        self.model_status[name] = why

    @property
    def active(self) -> list[str]:
        return [m for m in self.full.models if m in self.predictors]

    @property
    def healthy(self) -> bool:
        return self.active == list(self.full.models)

    def require_models(self) -> list[str]:
        active = self.active
        if len(active) < 2:
            states = ", ".join(f"{m}: {s}" for m, s in self.model_status.items())
            if all(s == "not installed" for s in self.model_status.values()):
                raise ModelsNotReady(f"no model agents installed ({states}). Install the [models] extra "
                                     "(MACE + AIMNet2) and/or set FITTEST_ORB_PYTHON, restart with --fake-models "
                                     f"for a demo, or use the hosted server {PUBLIC_URL}")
            raise ModelsNotReady(f"fewer than 2 model agents are ready ({states}); retry in a minute")
        return active

    def _run_model(self, m: str, systems: list[dict], idxs: list[int]) -> tuple[dict, set]:
        """Per-structure (E, F) for model m; a model-reported failure only fails that structure."""
        pred = self.predictors[m]

        def call(sub):
            Z, R, b = _pack([systems[k] for k in sub])
            o = pred(Z, R, b)
            E = np.asarray(o["energy"], dtype=float).reshape(-1)
            F = np.asarray(o["forces"], dtype=float).reshape(-1, 3)
            if len(E) != len(sub) or len(F) != len(Z):
                raise ValueError("model returned arrays of the wrong shape")
            res = {}
            for j, k in enumerate(sub):
                Fk = F[b == j]
                res[k] = (float(E[j]), Fk) if np.isfinite(E[j]) and np.isfinite(Fk).all() else None
            return res

        try:
            got = call(idxs)
        except CONN_ERRORS:
            raise
        except Exception:  # noqa: BLE001  model rejected the batch: find the culprit(s)
            got = {}
            for k in idxs:
                try:
                    got.update(call([k]))
                except CONN_ERRORS:
                    raise
                except Exception:  # noqa: BLE001
                    got[k] = None
        ok = {k: v for k, v in got.items() if v is not None}
        return ok, {k for k in idxs if k not in ok}

    def evaluate(self, systems: list[dict]) -> tuple[dict[int, dict], TrustSnapshot]:
        """Per input index: {"kind": "ok", "res", "forces", "energies"} or {"kind": "rejected", ...}."""
        self.require_models()
        out: dict[int, dict] = {}
        todo = []
        for k, s in enumerate(systems):
            bad = unsupported_elements(s["Z"], self.full.models)
            if bad:
                out[k] = {"kind": "rejected", "reason": "unsupported_elements",
                          "detail": "; ".join(f"{m} does not support {', '.join(e)}" for m, e in bad.items())
                          + ". Without the full committee there is no calibrated verdict."}
            else:
                todo.append(k)
        if not todo:
            return out, subset_snapshot(self.full, self.active)
        if not self._lock.acquire(timeout=BUSY_TIMEOUT_S):
            raise Busy("server busy")
        try:
            per_model, failed = {}, {}
            for m in self.require_models():
                try:
                    per_model[m], failed[m] = self._run_model(m, systems, todo)
                except CONN_ERRORS as exc:  # the link broke: this model sits out, the connector reconnects
                    self.drop_model(m, f"reconnecting ({type(exc).__name__})")
                    log.warning("model %s link lost: %s", m, type(exc).__name__)
            used = [m for m in self.full.models if m in per_model]
            if len(used) < 2:
                raise ModelsNotReady("fewer than 2 model agents answered; retry in a minute")
        finally:
            self._lock.release()
        snap = subset_snapshot(self.full, used)
        good = []
        for k in todo:
            bad = [m for m in used if k in failed[m]]
            if bad:
                out[k] = {"kind": "rejected", "reason": "model_failed",
                          "detail": f"{', '.join(bad)} could not evaluate this structure; no calibrated verdict"}
            else:
                good.append(k)
        if good:
            fixed = {m: (lambda *_a, m=m: {"energy": np.array([per_model[m][k][0] for k in good]),
                                           "forces": np.concatenate([per_model[m][k][1] for k in good])})
                     for m in used}
            for k, r in zip(good, score([systems[k] for k in good], fixed, snap)):
                out[k] = {"kind": "ok", "res": r, "forces": {m: per_model[m][k][1] for m in used},
                          "energies": {m: per_model[m][k][0] for m in used}}
        return out, snap

    def close(self) -> None:
        for c in self.closers:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass


def summarize(sys_in: dict, ev: dict, snap: TrustSnapshot, k: int, include_forces: bool) -> dict:
    base = {"index": k, "name": sys_in.get("name"), "formula": formula(sys_in["Z"]), "n_atoms": len(sys_in["Z"])}
    if ev["kind"] != "ok":
        return {**base, "verdict": "send_to_dft", "verdict_reasons": [ev["reason"]], "trust_score": None,
                "detail": ev["detail"], "domain_warning": domain_warning(sys_in["Z"], sys_in["R"])}
    res = ev["res"]
    F = np.asarray(res["served_forces"])
    fmag = np.linalg.norm(F, axis=-1)
    ref_model = next(m for m in REFERENCE_ENERGY_ORDER + tuple(snap.models) if m in ev["energies"])
    out = {
        **base,
        **decide(res, snap, ev["forces"]),
        "recommended": res["recommended_model"],
        "served_members": res["served_members"],
        "served_energy_eV": round(float(np.mean([ev["energies"][m] for m in res["served_members"]])), 6),
        "reference_energy_eV": round(ev["energies"][ref_model], 6),
        "reference_energy_model": ref_model,
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


def explain_one(sys_in: dict, ev: dict, snap: TrustSnapshot) -> dict:
    if ev["kind"] != "ok":
        return {"formula": formula(sys_in["Z"]), "verdict": "send_to_dft", "verdict_reasons": [ev["reason"]],
                "reason": ev["detail"], "trust_score": None, "verdict_rule": VERDICT_RULE,
                "domain_warning": domain_warning(sys_in["Z"], sys_in["R"])}
    res = ev["res"]
    models = list(snap.models)
    F = np.stack([np.asarray(ev["forces"][m]) for m in models])  # (M, n, 3)
    Fbar = F.mean(axis=0)
    dev = {m: float(np.sqrt(((F[i] - Fbar) ** 2).sum(-1).mean())) for i, m in enumerate(models)}
    pair = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pair[f"{models[i]} vs {models[j]}"] = round(float(np.sqrt(((F[i] - F[j]) ** 2).sum(-1).mean())), 5)
    per_atom = np.sqrt(((F - Fbar) ** 2).sum(-1).mean(0))
    worst = np.argsort(-per_atom)[:3]
    bkt = res["bucket"]
    se, fe = snap.sigma_edges, snap.fbar_edges

    def rng(edges, i):
        lo = "-inf" if i == 0 else f"{edges[i - 1]:.3f}"
        hi = "inf" if i >= len(edges) else f"{edges[i]:.3f}"
        return f"[{lo}, {hi})"

    cands = {c: posterior(snap, c, bkt) for c in snap.candidates if c in snap.p_correct}
    d = decide(res, snap, ev["forces"])
    s, tau = res["sigma_F"], snap.tau
    if d["verdict"] == "trust":
        reason = (f"sigma_F = {s:.3f} eV/A < tau = {tau:.3f} and the served candidate "
                  f"{res['recommended_model']} has P(correct) = {d['trust_score']:.2f} "
                  f"(90% CI {d['trust_score_90ci'][0]:.2f}-{d['trust_score_90ci'][1]:.2f}, "
                  f"{d['labels_behind_score']} DFT labels) in this bucket: trust.")
    else:
        reason = "send to DFT because " + "; ".join(d["verdict_reasons"]) + "."
    return {
        "formula": formula(sys_in["Z"]), **d, "reason": reason, "verdict_rule": VERDICT_RULE,
        "sigma_F": round(s, 5), "tau": round(tau, 5), "sigma_over_tau": round(s / tau, 3),
        "tau_meaning": TAU_MEANING, "trust_score_meaning": SCORE_MEANING,
        "bucket": {"index": bkt, "sigma_F_range": rng(se, bkt[0]), "mean_abs_force_range": rng(fe, bkt[1]),
                   "oracle_labels_in_bucket": d["labels_behind_score"]},
        "p_correct_in_bucket": {c: {"posterior_mean": round(v["mean"], 4), "ci90": v["ci90"]}
                                for c, v in sorted(cands.items(), key=lambda kv: -kv[1]["mean"])},
        "recommended": res["recommended_model"], "served_members": res["served_members"],
        "member_deviation_from_mean_eV_per_A": {m: round(v, 5) for m, v in dev.items()},
        "most_divergent_member": max(dev, key=dev.get),
        "pairwise_force_rms_eV_per_A": pair,
        "most_uncertain_atoms": [{"atom": int(i), "element": SYMBOLS[int(sys_in["Z"][i])],
                                  "sigma_eV_per_A": round(float(per_atom[i]), 5)} for i in worst],
        "domain_warning": domain_warning(sys_in["Z"], sys_in["R"]),
    }


# --------------------------------------------------------------------------- swarm status
def results_dir() -> Path:
    return Path(os.environ.get("FITTEST_RESULTS_DIR", ROOT / "results"))


def family_tree_paths() -> list[Path]:
    """Evolution family trees in run order: A (evolution/), B (evolution_aimnet/), C (evolution_c/,
    one tree per budget sub-run, e.g. evolution_c/b800/family_tree.json)."""
    r = results_dir()
    paths = [r / "evolution" / "family_tree.json", r / "evolution_aimnet" / "family_tree.json",
             r / "evolution_c" / "family_tree.json"]
    if (r / "evolution_c").is_dir():
        paths += sorted(p for p in (r / "evolution_c").glob("*family_tree*.json") if p not in paths)
        paths += sorted(p for p in (r / "evolution_c").glob("*/family_tree.json") if p not in paths)
    if os.environ.get("FITTEST_FAMILY_TREE"):
        paths.insert(0, Path(os.environ["FITTEST_FAMILY_TREE"]))
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _tree_summary(p: Path, max_chars: int) -> dict:
    try:
        rel = p.resolve().relative_to(results_dir().resolve().parent).as_posix()
    except ValueError:
        rel = p.name
    gh = f"{REPO_URL}/blob/{BRANCH}/{rel}"
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": rel, "available": False, "note": f"unreadable ({type(exc).__name__})", "github": gh}
    out: dict[str, Any] = {"path": rel, "available": True, "github": gh,
                           "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))}
    text = json.dumps(data)
    if len(text) <= max_chars:
        out["tree"] = data
    elif isinstance(data, dict):
        out["summary"] = {k: (v if not isinstance(v, (list, dict)) or len(json.dumps(v)) < 800
                              else f"<{type(v).__name__} of {len(v)} items>") for k, v in data.items()}
    else:
        out["summary"] = f"<{type(data).__name__} of {len(data)} items; see github>"
    return out


def family_tree_summary(max_chars: int = 6000) -> dict:
    trees = [_tree_summary(p, max_chars) for p in family_tree_paths() if p.exists()]
    if not trees:
        return {"available": False, "note": "no family tree published yet (evolution run in progress)",
                "github": f"{REPO_URL}/tree/{BRANCH}/results"}
    out = {"available": any(t["available"] for t in trees), "trees": trees}
    first = next((t for t in trees if t["available"]), trees[0])
    out.update({k: v for k, v in first.items() if k in ("tree", "summary", "github")})
    return out


def swarm_status_payload(engine: Engine) -> dict:
    snap = engine.full
    meta = snap.meta
    rep = meta.get("reputation", {})
    n_labels = None
    if snap.counts:
        c0 = next(iter(snap.counts.values()))
        n_labels = int(np.asarray(c0).sum())
    committee_elements = sorted(frozenset.intersection(*(MODEL_ELEMENTS[m] for m in snap.models
                                                         if m in MODEL_ELEMENTS)))
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
                {"agent": "calibrator", "role": "keeps a Beta ledger of P(correct | sigma_F, |F| bucket) per "
                 "candidate from DFT oracle labels and picks the served model or blend"},
                {"agent": "oracle", "role": "spends the DFT label budget on escalated structures"},
                {"agent": "curator", "role": "repairs the weakest model where none can be trusted (LoRA)"}],
        },
        "ledger": {"reputation": rep, "oracle_labels_used_for_calibration": n_labels,
                   "budget": meta.get("budget"), "global_p_correct": snap.p_global,
                   "candidates": snap.candidates},
        "calibration": {
            "domain": "gas-phase C/H/N/O reaction chemistry: Transition1x (reactants, products, NEB paths, "
                      "transition states), <= 7 heavy atoms; reference DFT wB97x/6-31G(d)",
            "tau_sigma_F_eV_per_A": snap.tau, "tau_meaning": TAU_MEANING,
            "trust_score_meaning": SCORE_MEANING, "verdict_rule": VERDICT_RULE,
            "sigma_edges": snap.sigma_edges, "fbar_edges": snap.fbar_edges,
            "wrong_thresholds": meta.get("wrong_thresholds"), "snapshot_created": meta.get("created"),
            "committee_elements": [SYMBOLS[z] for z in committee_elements],
            "outside_domain": "trust scores extrapolate; check_structures adds a domain_warning. Elements "
                              "outside committee_elements get send_to_dft (no calibrated verdict).",
        },
        "active_models": engine.active,
        "committee_complete": engine.healthy,
        "family_tree": family_tree_summary(),
        "repo": REPO_URL,
        "server_version": __version__,
    }


# --------------------------------------------------------------------------- MCP app
_TOOL_DESCRIPTIONS = {
    "check_structures": (
        "Ask the Fittest swarm whether a machine-learned interatomic potential can be trusted on your "
        "molecular structures, or whether they need DFT. Three model agents (MACE-MPA-0, AIMNet2, Orb-v3) "
        "predict forces; their disagreement sigma_F and the Calibrator's DFT-label ledger give the verdict.\n\n"
        "Input `structures`: XYZ text (single or multi-frame; bare 'El x y z' atoms on separate lines or "
        "separated by commas/semicolons also work) and/or a list of objects {\"Z\": [8,1,1] or \"symbols\": "
        "[\"O\",\"H\",\"H\"], \"R\": [[x,y,z], ...]} in Angstrom. Exact element symbols only. Limits: "
        f"<= {MAX_STRUCTURES} structures per call, 2..{MAX_ATOMS} atoms each, isolated molecules (no "
        "periodic cells). Elements the full committee supports: H B C N O F Si P S Cl As Se Br I.\n\n"
        "Returns per structure: verdict (trust / send_to_dft) with verdict_reasons, trust_score (Beta "
        "posterior P(correct), never exactly 1) with its 90% interval and labels_behind_score, the recommended "
        "model or blend, served and reference energy (eV; compare reference_energy_eV across a batch), "
        "max/mean force (eV/A), sigma_F vs tau, per-model summary, domain_warning outside C/H/N/O or "
        "Transition1x sizes. A bad structure gets its own error entry; the rest are still scored. "
        f"Rule: {VERDICT_RULE} {TAU_MEANING}\n\n" + LOGGING_NOTICE),
    "explain": (
        "Explain why Fittest trusted or escalated ONE structure (XYZ text or {Z|symbols, R}): the verdict "
        "rule applied, sigma_F vs tau, the Calibrator bucket with its label count, the posterior P(correct) "
        "of every candidate in it, which model agents disagree most, and the most uncertain atoms.\n\n"
        + LOGGING_NOTICE),
    "swarm_status": (
        "Show the Fittest population: model agents (answerers) and the Skeptic (critic) with reputations from "
        "the ledger, the calibration domain, tau and the verdict rule, which models are live, and the "
        "evolution family trees if published.\n\n" + LOGGING_NOTICE),
    "usage_stats": (
        "Aggregate usage of this server: distinct users (anonymous clients that got at least one structure "
        "verdict; internal test clients excluded), total calls, calls per tool, first/last call time. No "
        "per-user data.\n\n" + LOGGING_NOTICE),
}

INSTRUCTIONS = (
    "Fittest decides which ML interatomic potential to trust on a molecular structure, and when to "
    "escalate to DFT. Use check_structures for verdicts, explain for the reasoning on one structure, "
    "swarm_status for the agent population. Calibrated on C/H/N/O reaction chemistry (Transition1x). "
    + LOGGING_NOTICE)

#: the usage record of the tool call in flight (filled by the tools, written by the dispatcher)
_REC: contextvars.ContextVar[CallRecord | None] = contextvars.ContextVar("fittest_rec", default=None)


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "internal")


class Caller:
    """Who is calling: client name from the MCP session + the peer IP (uvicorn resolves the
    proxy's X-Forwarded-For only in hosted mode, where Caddy overwrites it)."""

    def __init__(self, ctx, ulog: UsageLog):
        name = ip = None
        internal = _truthy(os.environ.get("FITTEST_INTERNAL"))
        try:
            params = ctx.session.client_params
            if params is not None and params.clientInfo is not None:
                name = params.clientInfo.name
        except Exception:  # noqa: BLE001
            pass
        try:
            req = ctx.request_context.request
        except Exception:  # noqa: BLE001
            req = None
        if req is not None and getattr(req, "headers", None) is not None:
            ip = req.client.host if req.client else None
            internal = internal or _truthy(req.headers.get("x-fittest-internal"))
            if name is None:
                name = (req.headers.get("user-agent") or "?").split("/")[0][:60]
        self.client_id = ulog.client_id(name, ip)
        self.ip_key = ulog.client_id("ip", ip)
        if internal:
            ulog.mark_internal(self.client_id, "marked internal by header/env")
        self.internal = internal or ulog.is_internal_client(self.client_id)


def _classify(exc: BaseException, name: str) -> tuple[str, str]:
    """(log category, message for the caller). Never puts an exception message in the log."""
    try:
        from pydantic import ValidationError
    except ImportError:  # pragma: no cover
        ValidationError = ()  # type: ignore[assignment]
    cause = exc.__cause__ or exc
    if name not in TOOL_NAMES:
        return "unknown_tool", f"unknown tool; available: {', '.join(TOOL_NAMES)}"
    if isinstance(cause, InputError):
        return "input_error", str(cause)
    if isinstance(cause, ValidationError) or isinstance(cause, RecursionError):
        msg = "invalid arguments" if isinstance(cause, RecursionError) else \
            "invalid arguments: " + "; ".join(f"{'.'.join(map(str, e.get('loc', ())))}: {e.get('msg')}"
                                              for e in cause.errors()[:3])
        return "invalid_arguments", msg
    if isinstance(cause, ModelsNotReady):
        return "not_ready", str(cause)
    if isinstance(cause, Busy):
        return "busy", "the swarm is busy; retry shortly"
    if isinstance(cause, RateLimited):
        return "rate_limited", str(cause)
    if isinstance(cause, TimeoutError):
        return "timeout", "the request timed out; retry shortly"
    return type(cause).__name__[:40], f"internal error ({type(cause).__name__}); the model agents may be restarting"


class LoggedFastMCP(_FastMCP):  # type: ignore[misc,valid-type]
    """FastMCP whose every tools/call (unknown tools and schema errors included) goes through one
    dispatcher that rate-limits, times out and writes exactly one usage-log row."""

    dispatcher = None

    async def call_tool(self, name, arguments):  # noqa: D401
        return await self.dispatcher(name, arguments, super().call_tool)


def build_server(engine: Engine, ulog: UsageLog, transport: str = "stdio", host: str = "127.0.0.1",
                 port: int = 8765, public_host: str | None = None,
                 limiter: TokenBucket | None = None, ip_limiter: TokenBucket | None = None,
                 log_level: str = "WARNING"):
    import anyio
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse

    security = None
    if public_host:  # public server behind a TLS proxy: no local secrets to protect from DNS rebinding
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    mcp = LoggedFastMCP("fittest", instructions=INSTRUCTIONS, website_url=REPO_URL, host=host, port=port,
                        transport_security=security, max_request_body_size=2 * 1024 * 1024,
                        log_level=log_level.upper())
    mcp._mcp_server.version = __version__  # serverInfo.version = fittest's, not the SDK's
    # the SDK logs raw protocol errors (which can quote user input): keep it quiet
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    limiter = limiter or TokenBucket(capacity=20, rate=0.5)
    ip_limiter = ip_limiter or TokenBucket(capacity=60, rate=1.0)

    async def dispatch(name, arguments, call):
        t0 = time.perf_counter()
        caller = Caller(mcp.get_context(), ulog)
        tool = name if name in TOOL_NAMES else "unknown"
        rec = CallRecord(tool=tool, client_id=caller.client_id, transport=transport, internal=caller.internal)
        token = _REC.set(rec)
        try:
            ok, wait = limiter.allow(caller.client_id)
            ok2, wait2 = ip_limiter.allow(caller.ip_key) if ok else (True, 0.0)
            if not (ok and ok2):
                raise ToolError("rate limited") from RateLimited(f"rate limit: retry in {max(wait, wait2):.0f} s")
            with anyio.fail_after(REQUEST_TIMEOUT_S):
                out = await call(name, arguments)
            rec.ok = True
            return out
        except Exception as exc:  # noqa: BLE001
            rec.ok = False
            rec.error, msg = _classify(exc, name)
            if rec.error not in ("input_error", "invalid_arguments", "unknown_tool", "rate_limited", "not_ready"):
                log.warning("tool %s failed: %s", tool, rec.error)
            raise ToolError(msg) from None
        finally:
            rec.latency_ms = (time.perf_counter() - t0) * 1000
            _REC.reset(token)
            try:
                ulog.record(rec)
            except Exception:  # noqa: BLE001
                log.warning("usage log write failed")

    mcp.dispatcher = dispatch

    def header(snap: TrustSnapshot) -> dict:
        h = {"models_active": list(snap.models), "committee_complete": list(snap.models) == list(engine.full.models),
             "mode": engine.mode, "server_version": __version__}
        if engine.mode == "fake":
            h["warning"] = "FAKE MODELS: stand-in predictors for demos/tests; verdicts are meaningless."
        elif not h["committee_complete"]:
            h["warning"] = (f"partial committee ({', '.join(snap.models)}): the Calibrator's buckets and ledger "
                            f"were built on {', '.join(engine.full.models)}; verdicts are approximate.")
        return h

    async def in_thread(fn):
        rec = _REC.get()
        return await anyio.to_thread.run_sync(lambda: fn(rec), abandon_on_cancel=True)

    def run_batch(structures, rec: CallRecord):
        items = parse_structures(structures)
        valid = [(k, s) for k, s in enumerate(items) if "error" not in s]
        rec.n_structures = len(items)
        rec.n_atoms = sum(len(s["Z"]) for _, s in valid)
        rec.n_failed = len(items) - len(valid)
        evs, snap = engine.evaluate([s for _, s in valid]) if valid else ({}, subset_snapshot(engine.full,
                                                                                               engine.active))
        return items, valid, evs, snap

    @mcp.tool(description=_TOOL_DESCRIPTIONS["check_structures"])
    async def check_structures(structures: StructuresArg, include_forces: bool = False) -> dict:
        def work(rec: CallRecord):
            items, valid, evs, snap = run_batch(structures, rec)
            rows: list[dict] = [dict(s) if "error" in s else None for s in items]
            for j, (k, s) in enumerate(valid):
                rows[k] = summarize(s, evs[j], snap, k, include_forces)
            rec.n_trust = sum(r.get("verdict") == "trust" for r in rows)
            rec.n_dft = sum(r.get("verdict") == "send_to_dft" for r in rows)
            served = {tuple(r["served_members"]) for r in rows if r.get("served_members")}
            out = {**header(snap), "n_structures": len(rows),
                   "summary": {"trust": rec.n_trust, "send_to_dft": rec.n_dft, "errors": rec.n_failed},
                   "results": rows, "verdict_rule": VERDICT_RULE, "tau_meaning": TAU_MEANING,
                   "notes": "Forces in eV/A. served_energy_eV is the served model's (or blend's) total energy "
                            "on its own reference; reference_energy_eV uses one model for every structure "
                            "and is the field to compare across structures.",
                   "logging": LOGGING_NOTICE}
            if len(served) > 1:
                out["energy_warning"] = ("the served model differs between structures in this batch, so "
                                         "served_energy_eV values are NOT comparable with each other; use "
                                         "reference_energy_eV")
            return out
        return await in_thread(work)

    @mcp.tool(description=_TOOL_DESCRIPTIONS["explain"])
    async def explain(structure: StructuresArg) -> dict:
        def work(rec: CallRecord):
            items = parse_structures(structure)
            if len(items) != 1:
                raise InputError(f"explain takes exactly one structure (got {len(items)}); "
                                 "use check_structures for batches")
            if "error" in items[0]:
                raise InputError(items[0]["error"])
            items, valid, evs, snap = run_batch(structure, rec)
            ex = explain_one(valid[0][1], evs[0], snap)
            rec.n_trust = int(ex["verdict"] == "trust")
            rec.n_dft = 1 - rec.n_trust
            return {**header(snap), **ex}
        return await in_thread(work)

    @mcp.tool(description=_TOOL_DESCRIPTIONS["swarm_status"])
    async def swarm_status() -> dict:
        return await in_thread(lambda rec: swarm_status_payload(engine))

    @mcp.tool(description=_TOOL_DESCRIPTIONS["usage_stats"])
    async def usage_stats() -> dict:
        def work(rec):
            st = ulog.stats()
            st["scope"] = ("this hosted server" if transport != "stdio"
                           else "this local install only (local mode never reports anywhere)")
            return st
        return await in_thread(work)

    @mcp.custom_route("/stats", methods=["GET"])
    async def stats_route(request: Request):
        st = ulog.stats()
        st["active_models"] = engine.active
        st["server_version"] = __version__
        return JSONResponse(st, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"})

    @mcp.custom_route("/health", methods=["GET"])
    async def health_route(request: Request):
        ok = engine.healthy
        return JSONResponse({"ok": ok, "status": "ok" if ok else ("degraded" if engine.active else "down"),
                             "mode": engine.mode, "active_models": engine.active,
                             "model_status": engine.model_status, "server_version": __version__},
                            status_code=200 if ok else 503)

    @mcp.custom_route("/", methods=["GET"])
    async def landing(request: Request):
        url = f"https://{public_host}/mcp" if public_host else f"http://{host}:{port}/mcp"
        return HTMLResponse(landing_html(url, engine))

    return mcp


EXAMPLE_PROMPT = ("Use fittest to check whether I can trust an ML potential on this water molecule: "
                  "O 0 0 0, H 0.96 0 0, H -0.24 0.93 0")


def landing_html(url: str, engine: Engine) -> str:
    import html as _html

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
.stat b{{font-size:22px;display:block}} li{{margin:4px 0}} small{{color:var(--mute)}}
</style></head><body><main>
<h1>Fittest MCP</h1>
<p class="sub">Survival of the fittest, in materials science. Ask a swarm of ML interatomic potentials whether
to trust them on your molecule, or send it to DFT.</p>
<div><span class="stat"><b id="u">-</b>users</span><span class="stat"><b id="c">-</b>calls</span>
<span class="stat"><b>{live}</b>live models</span></div>
<p><small>Users: {_html.escape(USERS_DEFINITION)}</small></p>
<h2>What it does</h2>
<p>Model agents ({models}) predict forces on your structure. A structure is <code>trust</code> only if every
model supports its elements, the forces don't all vanish, their disagreement &sigma;<sub>F</sub> is below the
fixed threshold &tau; (70th percentile of round-0 &sigma;<sub>F</sub>), and the Calibrator's Beta-posterior
P(correct) for the served model, learned from DFT labels on Transition1x reactions, is at least 0.5. Otherwise
<code>send_to_dft</code>, with the reasons. Each score comes with its 90% interval and the number of DFT labels
behind it. Calibrated for gas-phase C/H/N/O chemistry; elsewhere the score is an extrapolation and the response
says so.</p>
<p>Tools: <code>check_structures</code>, <code>explain</code>, <code>swarm_status</code>, <code>usage_stats</code>.
Limits: {MAX_STRUCTURES} structures per call, 2&ndash;{MAX_ATOMS} atoms each, molecules only, rate-limited per
client.</p>
<h2>Connect</h2>
<p>Claude Code</p><pre>claude mcp add --transport http fittest {url}</pre>
<p>Claude Desktop / claude.ai: Settings &rarr; Connectors &rarr; Add custom connector &rarr; <code>{url}</code></p>
<p>Cursor (<code>~/.cursor/mcp.json</code>)</p><pre>{{"mcpServers": {{"fittest": {{"url": "{url}"}}}}}}</pre>
<p>Local demo (stdio, stand-in models: verdicts are meaningless, for trying the tools offline)</p>
<pre>claude mcp add fittest-demo -- uvx --from git+{REPO_URL} fittest-mcp --fake-models</pre>
<p>Local with real models: install <code>"fittest[models] @ git+{REPO_URL}"</code> (CPU torch) into a fresh
environment, optionally Orb-v3 in a second environment named by <code>FITTEST_ORB_PYTHON</code>, and point your
client at that environment's <code>fittest-mcp</code>; exact commands in
<a href="{REPO_URL}#12-local-mcp-server">README &sect;1.2</a>. This install is documented but not tested end to
end; the hosted server is the tested path. A plain <code>uvx</code> install has no models.</p>
<p>Then ask: <i>"{_html.escape(EXAMPLE_PROMPT)}."</i></p>
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
    Z, R, b = _pack([WATER])
    pred(Z, R, b)
    engine.add_model(name, pred)
    log.info("model agent %s warm", name)


def start_hosted_models(engine: Engine, ports: dict[str, int], period: float = 10.0) -> threading.Thread:
    """Keep one link per systemd-managed model server; (re)connect any model that is not active.
    A model rejecting an input never changes its status (see Engine._run_model)."""
    remotes = {m: RemoteModel(m, port) for m, port in ports.items()}
    engine.closers.extend(remotes.values())

    def loop():
        while True:
            for m, rm in remotes.items():
                if engine.model_status.get(m) == "active":
                    continue
                try:
                    rm.reconnect(timeout=3.0)
                    _warm(engine, m, rm.predict)
                except Exception as exc:  # noqa: BLE001
                    rm.close()
                    engine.model_status[m] = f"waiting for model server on port {rm.port} ({type(exc).__name__})"
            time.sleep(period)
    th = threading.Thread(target=loop, daemon=True, name="model-connector")
    th.start()
    return th


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
            engine.model_status[m] = f"failed: {type(exc).__name__}"
            log.error("model agent %s failed to start: %s", m, type(exc).__name__)

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
                    help="use running model servers instead of spawning: mace_mpa0=7300,aimnet2=7301,orb=7302")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--orb-python", default=os.environ.get("FITTEST_ORB_PYTHON"))
    ap.add_argument("--data-dir", default=None, help="usage log + salt (default: ~/.fittest-mcp)")
    ap.add_argument("--salt-file", default=os.environ.get("FITTEST_SALT_FILE"))
    ap.add_argument("--log-level", default="WARNING")
    ap.add_argument("--version", action="version", version=f"fittest-mcp {__version__}")
    a = ap.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=a.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    snap_path = a.snapshot or default_snapshot_path()
    snap = TrustSnapshot.load(snap_path)
    data_dir = Path(a.data_dir or default_data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    ulog = UsageLog(data_dir / "usage.sqlite", salt_path=a.salt_file or data_dir / "salt")
    wanted = a.models.split(",") if a.models else list(snap.models)
    have: list[str] = []

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
            print("fittest-mcp: no model agents installed in this environment (need nvalchemi-toolkit[mace,aimnet] "
                  "and/or orb-models via FITTEST_ORB_PYTHON). Serving anyway; tools will report this. Use "
                  "--fake-models for a demo.", file=sys.stderr)
        else:
            print(f"fittest-mcp: starting model agents {have}" + (f" (not installed: {missing})" if missing else ""),
                  file=sys.stderr)
    atexit.register(engine.close)
    # Import everything the tools need on the main thread now. On Windows, a worker thread that
    # imports sklearn/joblib/scipy blocks while the stdio transport sits in a blocking stdin read.
    for mod in ("fittest.agents.calibrator", "scipy.stats", "pydantic"):
        try:
            importlib.import_module(mod)
        except ImportError:
            pass
    if engine.mode == "fake":
        engine.evaluate([WATER])
        posterior(snap, snap.candidates[0], (0, 0))

    server = build_server(engine, ulog, transport=a.transport, host=a.host, port=a.port,
                          public_host=a.public_host, log_level=a.log_level)
    if engine.mode == "local" and have:  # spawn only once the server is built (no orphans on errors)
        start_local_models(engine, have, a.device, a.orb_python, str(data_dir / "logs"))
    if a.transport == "stdio":
        server.run("stdio")
    else:
        import uvicorn

        # access_log=False: uvicorn's access log would record client IPs, which we never store.
        # proxy_headers only behind our own proxy (Caddy overwrites X-Forwarded-For); otherwise a
        # client could forge the header and mint new client ids.
        uvicorn.run(server.streamable_http_app(), host=a.host, port=a.port, access_log=False,
                    log_level=a.log_level.lower(), proxy_headers=bool(a.public_host),
                    forwarded_allow_ips="127.0.0.1" if a.public_host else "", timeout_keep_alive=30)


if __name__ == "__main__":
    main()
