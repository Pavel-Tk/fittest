"""DisagreementHook: cross-model force disagreement inside an ALCHEMI Toolkit dynamics run.

A drop-in Toolkit hook (``nvalchemi.hooks`` protocol: ``stage``, ``frequency``,
``__call__(ctx, stage)``, plus ``_runs_on_stage`` for multi-stage dispatch) for any
``FusedStage`` / ``FIRE2`` / ``NEB`` run:

* at **admission** (pristine dataset geometry, before force priming) and/or **every
  ``every`` steps** it sends the live systems to every committee member (one round trip
  per model, all members concurrently on threads), computes per-system
  ``sigma_F = mean_i sqrt(mean_m ||F_m,i - F̄_i||²)`` (identical to
  :func:`trustswarm.signals.force_spread`, DESIGN.md §3), and records it;
* systems with ``sigma_F >= tau`` are **flagged**; with ``graduate=True`` their status is
  set to the workflow's ``exit_status`` so the next ``refill_check`` evicts them and the
  ``SizeAwareSampler`` refills the slot (DESIGN.md §6). With ``graduate=False`` (NEB) it
  only observes;
* ``ON_GRADUATE`` is observed to record when every other system left the batch.

The Toolkit has no graduation-reason tag, so the hook keeps its own ledger:
``hook.records[key]`` (one dict per system) and ``hook.flagged`` (ordered list of keys),
where ``key = id_map(system_id)`` (default: the sampler's ``system_id``; graph index for
batches built outside a sampler, e.g. NEB bands).

The numerical core (:class:`DisagreementMonitor`) is toolkit-agnostic and works on flat
numpy arrays, so it is unit-testable with fake models and reusable outside dynamics.

Committee members ("clients") may be any mix of
* objects with ``predict(Z, R, batch, charge) -> {"energy": (S,), "forces": (N,3)}``
  (``trustswarm.model_server.ModelClient``),
* objects/callables returning ``(E (S,), F (N,3))`` (e.g. ``BatchedMACE.predict``),
given as a list (names from ``.model``/``.name``) or a ``{name: client}`` dict.
"""

from __future__ import annotations

import concurrent.futures
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

Array = np.ndarray
ClientLike = Any


# --------------------------------------------------------------------------- sigma
def per_system_force_spread(forces: Array, batch: Array, n_systems: int | None = None
                            ) -> tuple[Array, Array]:
    """Vectorised :func:`trustswarm.signals.force_spread` over many systems (numpy).

    ``forces`` (M, N, 3) for the concatenated atoms, ``batch`` (N,) system index.
    Returns ``(sigma_mean (S,), sigma_max (S,))`` in eV/Å.
    """
    f = np.asarray(forces, dtype=np.float64)
    if f.ndim != 3 or f.shape[-1] != 3 or f.shape[0] < 2:
        raise ValueError(f"forces must be (M>=2, N, 3), got {f.shape}")
    b = np.asarray(batch, dtype=np.int64)
    S = int(b.max()) + 1 if n_systems is None else int(n_systems)
    dev = f - f.mean(axis=0, keepdims=True)
    s = np.sqrt(np.mean(np.sum(dev * dev, axis=-1), axis=0))          # (N,) per-atom spread
    counts = np.bincount(b, minlength=S).astype(np.float64)
    mean = np.bincount(b, weights=s, minlength=S) / np.maximum(counts, 1.0)
    mx = np.full(S, -np.inf)
    np.maximum.at(mx, b, s)
    mean[counts == 0] = np.nan
    mx[counts == 0] = np.nan
    return mean, mx


def per_system_force_spread_torch(forces, batch_idx, n_systems: int):
    """Torch twin of :func:`per_system_force_spread` (``forces`` [M,V,3], ``batch_idx`` [V])."""
    import torch

    f = forces.to(torch.float64)
    dev = f - f.mean(dim=0, keepdim=True)
    s = dev.pow(2).sum(-1).mean(0).sqrt()                                # [V]
    idx = batch_idx.long()
    counts = torch.zeros(n_systems, dtype=f.dtype, device=f.device).index_add_(
        0, idx, torch.ones_like(s))
    mean = torch.zeros(n_systems, dtype=f.dtype, device=f.device).index_add_(0, idx, s) / counts.clamp_min(1)
    mx = torch.full((n_systems,), -math.inf, dtype=f.dtype, device=f.device).scatter_reduce(
        0, idx, s, reduce="amax", include_self=True)
    return mean, mx


# --------------------------------------------------------------------------- clients
def client_name(c: ClientLike, i: int) -> str:
    for attr in ("model", "name"):
        v = getattr(c, attr, None)
        if isinstance(v, str):
            return v
    return getattr(c, "__name__", f"m{i}")


def call_client(c: ClientLike, Z: Array, R: Array, batch: Array, charge: Array) -> tuple[Array, Array]:
    """One committee prediction, normalised to ``(E (S,) float64, F (N,3) float64)``."""
    fn = c.predict if hasattr(c, "predict") else c
    try:
        out = fn(Z, R, batch, charge)
    except TypeError:
        out = fn(Z, R, batch)
    if isinstance(out, Mapping):
        E, F = out["energy"], out["forces"]
    else:
        E, F = out
    E = np.asarray(E, dtype=np.float64).reshape(-1)
    F = np.asarray(F, dtype=np.float64).reshape(-1, 3)
    if F.shape[0] != len(Z):
        raise ValueError(f"forces have {F.shape[0]} rows for {len(Z)} atoms")
    return E, F


# --------------------------------------------------------------------------- core
class DisagreementMonitor:
    """Toolkit-agnostic committee gather + sigma + flag ledger.

    Parameters
    ----------
    clients
        Committee members (list or ``{name: client}``); at least two must answer.
    tau
        Flag threshold on ``sigma_F`` [eV/Å], a float or a zero-arg callable (read at every
        check, e.g. the Calibrator's current ``threshold`` from the Blackboard).
    on_flag
        Called once per newly flagged system with its record.
    max_failures
        A member that raises this many times in a row is dropped from the committee
        (reported in ``failures``); the check continues with the rest while >= 2 remain.
    keep_forces
        ``"flag"`` keeps per-model forces + geometry for the admission check
        (``rec["at_admission"]``) and the flagging check (``rec["at_flag"]``, what the oracle
        queue needs); ``"last"`` also keeps them for the latest check of every system;
        ``"none"`` keeps energies and sigma only.
    """

    def __init__(self, clients: Sequence[ClientLike] | Mapping[str, ClientLike],
                 tau: float | Callable[[], float], *, on_flag: Callable[[dict], None] | None = None,
                 max_failures: int = 2, keep_forces: str = "flag", timeout: float | None = 600.0):
        if isinstance(clients, Mapping):
            self.clients = dict(clients)
        else:
            self.clients = {client_name(c, i): c for i, c in enumerate(clients)}
        if len(self.clients) < 2:
            raise ValueError("a disagreement committee needs at least two models")
        if keep_forces not in ("flag", "last", "none"):
            raise ValueError("keep_forces must be 'flag', 'last' or 'none'")
        self._tau = tau
        self.on_flag = on_flag
        self.max_failures = int(max_failures)
        self.keep_forces = keep_forces
        self.timeout = timeout
        self.records: dict[Any, dict] = {}
        self.flagged: list[Any] = []
        self.failures: dict[str, list[str]] = {}
        self.dead: set[str] = set()
        self._consecutive: dict[str, int] = {n: 0 for n in self.clients}
        self.n_checks = 0
        self.n_round_trips = 0
        self.gather_seconds = 0.0
        self._pool = ThreadPoolExecutor(max_workers=len(self.clients), thread_name_prefix="committee")
        self._lock = threading.Lock()

    # ---- threshold
    def tau(self) -> float:
        return float(self._tau() if callable(self._tau) else self._tau)

    @property
    def live_models(self) -> list[str]:
        return [n for n in self.clients if n not in self.dead]

    # ---- gather
    def gather(self, Z: Array, R: Array, batch: Array, charge: Array | None = None
               ) -> dict[str, tuple[Array, Array]]:
        """Concurrent round trip to every live member; failed members are logged and skipped."""
        S = int(np.max(batch)) + 1 if len(batch) else 0
        q = np.zeros(S) if charge is None else np.asarray(charge, dtype=np.float64)
        names = self.live_models
        t0 = time.perf_counter()
        futs = {n: self._pool.submit(call_client, self.clients[n], Z, R, batch, q) for n in names}
        out: dict[str, tuple[Array, Array]] = {}
        for n, fu in futs.items():
            try:
                E, F = fu.result(timeout=self.timeout)
                if E.shape[0] != S or not (np.isfinite(E).all() and np.isfinite(F).all()):
                    raise ValueError(f"bad output: {E.shape[0]} energies for {S} systems or non-finite values")
                out[n] = (E, F)
                self._consecutive[n] = 0
            except Exception as exc:  # report, keep going with the rest of the committee
                msg = f"{type(exc).__name__}: {exc}"
                self.failures.setdefault(n, []).append(msg)
                self._consecutive[n] += 1
                if isinstance(exc, concurrent.futures.TimeoutError):
                    # The worker may still be blocked in recv() on this client's connection; a
                    # later request could then read this stale reply. Retire the member.
                    self.dead.add(n)
                if self._consecutive[n] >= self.max_failures:
                    self.dead.add(n)
        self.gather_seconds += time.perf_counter() - t0
        self.n_round_trips += len(names)
        return out

    # ---- one check over S systems
    def check(self, keys: Sequence[Any], Z: Array, R: Array, batch: Array, *, step: int,
              phase: str, charge: Array | None = None, local_steps: Sequence[int] | None = None
              ) -> tuple[Array, Array]:
        """Score S systems; returns ``(sigma (S,), newly_flagged (S,) bool)``.

        ``keys[s]`` identifies system ``s`` in the ledger. Systems already flagged are
        scored again (recorded) but not re-flagged.
        """
        S = len(keys)
        preds = self.gather(Z, R, batch, charge)
        tau = self.tau()
        self.n_checks += 1
        if len(preds) < 2:
            for k in keys:
                rec = self._rec(k)
                rec["checks"].append({"step": step, "phase": phase, "sigma": None, "tau": tau,
                                      "models": sorted(preds), "note": "fewer than 2 models answered"})
            return np.full(S, np.nan), np.zeros(S, dtype=bool)
        names = sorted(preds)
        Fm = np.stack([preds[n][1] for n in names])                       # (M, N, 3)
        sig, sig_max = per_system_force_spread(Fm, batch, S)
        newly = np.zeros(S, dtype=bool)
        ptr = np.concatenate([[0], np.cumsum(np.bincount(batch, minlength=S))])
        for s, k in enumerate(keys):
            rec = self._rec(k)
            a, e = int(ptr[s]), int(ptr[s + 1])
            entry = {"step": int(step), "phase": phase, "sigma": float(sig[s]),
                     "sigma_max": float(sig_max[s]), "tau": tau, "models": names,
                     "energy": {n: float(preds[n][0][s]) for n in names}}
            if local_steps is not None:
                entry["local_step"] = int(local_steps[s])
            rec["checks"].append(entry)
            rec["sigma_last"] = float(sig[s])
            if phase == "admission" and rec.get("sigma_admission") is None:
                rec["sigma_admission"] = float(sig[s])
            snap = {"Z": Z[a:e].copy(), "R": R[a:e].copy(),
                    "forces": {n: preds[n][1][a:e].copy() for n in names}}
            if self.keep_forces == "last":
                rec["last"] = snap
            if phase == "admission" and self.keep_forces != "none" and "at_admission" not in rec:
                rec["at_admission"] = snap
            if not rec["flagged"] and sig[s] >= tau:
                rec.update(flagged=True, flag_step=int(step), flag_phase=phase,
                           sigma_at_flag=float(sig[s]), tau_at_flag=tau)
                if local_steps is not None:
                    rec["flag_local_step"] = int(local_steps[s])
                if self.keep_forces != "none":
                    rec["at_flag"] = snap
                newly[s] = True
                self.flagged.append(k)
                if self.on_flag is not None:
                    self.on_flag(rec)
        return sig, newly

    def _rec(self, k: Any) -> dict:
        rec = self.records.get(k)
        if rec is None:
            rec = self.records[k] = {"key": k, "checks": [], "flagged": False,
                                     "sigma_admission": None, "sigma_last": None}
        return rec

    def summary(self) -> dict:
        return {"n_systems": len(self.records), "n_flagged": len(self.flagged),
                "n_checks": self.n_checks, "n_round_trips": self.n_round_trips,
                "gather_seconds": self.gather_seconds, "models": list(self.clients),
                "dead_models": sorted(self.dead),
                "failures": {k: v[-3:] for k, v in self.failures.items()}}

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------- toolkit hook
_NBR_KEYS = ("neighbor_list", "neighbor_list_shifts", "neighbor_matrix", "num_neighbors",
             "neighbor_matrix_shifts")


class DisagreementHook(DisagreementMonitor):
    """ALCHEMI Toolkit hook around :class:`DisagreementMonitor`.

    Parameters
    ----------
    clients, tau, on_flag, max_failures, keep_forces
        See :class:`DisagreementMonitor`.
    every
        Re-check live systems every ``every`` engine steps (``AFTER_STEP``; the Toolkit
        registry gates on ``step_count % every == 0``).
    stage
        ``"admission"``, ``"step"`` or ``"both"`` (default): when to check.
    graduate
        ``True``: flagged systems get ``status = exit_status`` (evicted at the next
        ``refill_check``; the sampler refills the slot). ``False``: observe only (NEB).
    id_map
        Maps a sampler ``system_id`` to the ledger key (e.g. dataset index -> frame uid).
        For batches built outside a sampler (``system_id`` = -1 or absent) the key is the
        graph index within the batch.

    Register on the outer ``FusedStage`` (so ``ctx.workflow.exit_status`` is the fused
    exit code) via ``FusedStage(..., hooks=[hook])``, on a single dynamics, or through
    ``NEB(..., extra_hooks=[hook])``.
    """

    def __init__(self, clients, tau, every: int = 1, stage: str = "both", *,
                 on_flag: Callable[[dict], None] | None = None, graduate: bool = True,
                 id_map: Callable[[int], Any] | None = None, exit_status: int | None = None,
                 max_failures: int = 2, keep_forces: str = "flag", timeout: float | None = 600.0):
        super().__init__(clients, tau, on_flag=on_flag, max_failures=max_failures,
                         keep_forces=keep_forces, timeout=timeout)
        from nvalchemi.dynamics.base import DynamicsStage

        if stage not in ("admission", "step", "both"):
            raise ValueError("stage must be 'admission', 'step' or 'both'")
        if int(every) < 1:
            raise ValueError("every must be >= 1")
        self._S = DynamicsStage
        self.frequency = int(every)
        self.check_admission = stage in ("admission", "both")
        self.check_steps = stage in ("step", "both")
        self.stage = DynamicsStage.AFTER_STEP if self.check_steps else DynamicsStage.ON_ADMISSION
        self.graduate = bool(graduate)
        self.id_map = id_map
        self.exit_status = exit_status
        self.admitted_step: dict[Any, int] = {}
        self.n_graduated_by_hook = 0
        self.events: list[dict] = []   # admission/graduation events (refill bookkeeping)

    # Toolkit multi-stage dispatch
    def _runs_on_stage(self, stage) -> bool:
        S = self._S
        return ((stage == S.ON_ADMISSION and self.check_admission)
                or (stage == S.AFTER_STEP and self.check_steps)
                or stage == S.ON_ADMISSION or stage == S.ON_GRADUATE)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # ---- helpers
    def _keys(self, batch, gidx: Array) -> list:
        sid = getattr(batch, "system_id", None)
        if sid is None:
            return [int(g) for g in gidx]
        sid = sid.view(-1).detach().cpu().numpy()
        keys = []
        for g in gidx:
            s = int(sid[g])
            if s < 0:
                keys.append(int(g))
            else:
                keys.append(self.id_map(s) if self.id_map is not None else s)
        return keys

    def _exit_status(self, ctx) -> int:
        if self.exit_status is not None:
            return int(self.exit_status)
        return int(getattr(ctx.workflow, "exit_status", 1))

    def _active(self, ctx):
        from nvalchemi.dynamics.base import BaseDynamics

        m = BaseDynamics.active_graph_mask(ctx.batch, self._exit_status(ctx))
        import torch

        if m is None:
            return torch.ones(ctx.batch.num_graphs, dtype=torch.bool, device=ctx.batch.positions.device)
        return m.view(-1)[: ctx.batch.num_graphs]

    def __call__(self, ctx, stage) -> None:
        import torch

        with torch.no_grad():
            if stage == self._S.ON_GRADUATE:
                self._on_graduate(ctx)
                return
            batch = ctx.batch
            step = int(getattr(ctx, "step_count", 0) or 0)
            active = self._active(ctx)
            gidx = torch.nonzero(active, as_tuple=True)[0].cpu().numpy()
            if gidx.size == 0:
                return
            keys = self._keys(batch, gidx)
            if stage == self._S.ON_ADMISSION:
                new = [i for i, k in enumerate(keys) if k not in self.admitted_step]
                for i in new:
                    self.admitted_step[keys[i]] = step
                    self._rec(keys[i])["admitted_step"] = step
                if new:
                    self.events.append({"event": "admission", "step": step, "n": len(new)})
                if not self.check_admission:
                    return
                # pristine geometry: score each system once, at its first admission
                gidx, keys = gidx[new], [keys[i] for i in new]
                if len(keys) == 0:
                    return
                phase = "admission"
            else:
                if not self.check_steps:
                    return
                phase = "step"
            self._check_graphs(ctx, gidx, keys, step, phase)

    def _check_graphs(self, ctx, gidx: Array, keys: list, step: int, phase: str) -> None:
        import torch
        from trustswarm.models.toolkit_io import arrays_from_batch

        batch = ctx.batch
        Z, R, b, q = arrays_from_batch(batch, gidx)
        local = [step - self.admitted_step.get(k, step) for k in keys]
        _, newly = self.check(keys, Z, R, b, step=step, phase=phase, charge=q, local_steps=local)
        if not newly.any():
            return
        for k in np.asarray(keys, dtype=object)[newly]:
            self.records[k]["graduated_by_hook"] = self.graduate
        if self.graduate:
            g = torch.as_tensor(gidx[newly], dtype=torch.long, device=batch.status.device)
            status = batch.status.view(-1)
            status[g] = self._exit_status(ctx)
            self.n_graduated_by_hook += int(newly.sum())
            for k in np.asarray(keys, dtype=object)[newly]:
                rec = self.records[k]
                rec["graduated_step"] = step
                rec["n_steps"] = step - self.admitted_step.get(k, step)
                rec["reason"] = "disagreement"

    def _on_graduate(self, ctx) -> None:
        import torch

        m = getattr(ctx, "graduated_mask", None)
        if m is None or not bool(m.any()):
            return
        b = ctx.batch
        step = int(getattr(ctx, "step_count", 0) or 0)
        gidx = torch.nonzero(m.view(-1)[: b.num_graphs], as_tuple=True)[0].cpu().numpy()
        keys = self._keys(b, gidx)
        fmax = None
        if getattr(b, "forces", None) is not None:
            fn = b.forces.detach().norm(dim=-1)
            fmax_t = torch.zeros(b.num_graphs, dtype=fn.dtype, device=fn.device).scatter_reduce(
                0, b.batch_idx.long(), fn, reduce="amax", include_self=False)
            fmax = fmax_t.cpu().numpy()
        for g, k in zip(gidx, keys):
            rec = self._rec(k)
            if rec.get("reason") == "disagreement":
                continue
            rec["graduated_step"] = step
            rec["n_steps"] = step - self.admitted_step.get(k, step)
            rec["fmax_at_graduation"] = None if fmax is None else float(fmax[g])
            rec["reason"] = "dynamics"   # converged or step budget; caller decides via fmax
        self.events.append({"event": "graduate", "step": step, "n": len(keys)})
