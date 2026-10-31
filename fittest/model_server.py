"""One process per model agent.

Each ModelAgent's weights live in their own OS process (and, for Orb, their own
Python environment, because orb-models pins a different toolkit-ops version).
The DisagreementHook and the scoring scripts talk to them over a local socket
with a tiny request/response protocol: send flat arrays for a batch of systems,
get back per-system energies and per-atom forces.

    python -m fittest.model_server --model orb --port 6003 --device cuda

Protocol (pickled dicts over ``multiprocessing.connection``; numpy only):
    request  {"cmd": "predict", "Z": int64 (N,), "R": float64 (N,3),
              "batch": int64 (N,), "charge": float64 (S,)}
    response {"energy": float64 (S,), "forces": float64 (N,3), "seconds": float}
    request  {"cmd": "info"} -> {"model": str, "level_of_theory": str, ...}
    request  {"cmd": "shutdown"}
"""

from __future__ import annotations

import argparse
import os
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener

import numpy as np

# The protocol is pickle over a socket, so the authkey is what keeps other local users out.
# There is no fixed default: without FITTEST_AUTHKEY each process draws a random key, and
# ``spawn`` hands a fresh random key to every server it starts (and to its ModelClient).
AUTHKEY = (os.environ.get("FITTEST_AUTHKEY") or secrets.token_hex(16)).encode()
CONNECT_HANDSHAKE_TIMEOUT = 15.0
PREDICT_TIMEOUT = float(os.environ.get("FITTEST_PREDICT_TIMEOUT", "300"))


class ModelError(RuntimeError):
    """The model agent answered, but rejected or failed on this input (the connection is fine)."""


# --------------------------------------------------------------------------- backends
class Backend:
    name: str = "base"
    level_of_theory: str = ""

    def predict(self, Z: np.ndarray, R: np.ndarray, batch: np.ndarray,
                charge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def _split(batch: np.ndarray) -> list[np.ndarray]:
    n_sys = int(batch.max()) + 1 if batch.size else 0
    return [np.flatnonzero(batch == s) for s in range(n_sys)]


class OrbBackend(Backend):
    level_of_theory = "PBE/PBE+U (OMat24)"

    def __init__(self, device: str, checkpoint: str = "orb_v3_conservative_inf_omat"):
        from orb_models.forcefield import pretrained

        self.name = "orb"
        self.device = device
        loaded = getattr(pretrained, checkpoint)(device=device, precision="float32-highest", compile=False)
        # orb-models >= 0.6 returns (model, adapter); 0.5.x returns the model only.
        self.model, self.adapter = loaded if isinstance(loaded, tuple) else (loaded, None)

    def predict(self, Z, R, batch, charge):
        from ase import Atoms

        idx = _split(batch)
        atoms = [Atoms(numbers=Z[i], positions=R[i]) for i in idx]
        if self.adapter is not None:
            g = self.adapter.from_ase_atoms_list(atoms, device=self.device)
        else:
            from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
            from orb_models.forcefield.base import batch_graphs

            g = batch_graphs([ase_atoms_to_atom_graphs(a, self.model.system_config, device=self.device)
                              for a in atoms])
        out = self.model.predict(g, split=False)
        # Conservative checkpoints return -dE/dR as "grad_forces"; prefer it over any direct head.
        forces_key = next(k for k in ("grad_forces", "forces") if k in out)
        E = out["energy"].detach().double().cpu().numpy().reshape(-1)
        F_flat = out[forces_key].detach().double().cpu().numpy()
        F = np.empty_like(R)
        F[np.concatenate(idx)] = F_flat
        return E, F


class AIMNet2Backend(Backend):
    level_of_theory = "wB97M-D3(BJ)/def2-TZVPP"

    def __init__(self, device: str, checkpoint: str = "aimnet2"):
        from aimnet.calculators import AIMNet2Calculator

        self.name = checkpoint
        self.device = device
        self.calc = AIMNet2Calculator(checkpoint, device=device)

    def predict(self, Z, R, batch, charge):
        import torch

        out = self.calc({
            "coord": torch.as_tensor(R, dtype=torch.float32, device=self.device),
            "numbers": torch.as_tensor(Z, dtype=torch.long, device=self.device),
            "charge": torch.as_tensor(charge, dtype=torch.float32, device=self.device),
            "mol_idx": torch.as_tensor(batch, dtype=torch.long, device=self.device),
        }, forces=True)
        E = out["energy"].detach().double().cpu().numpy().reshape(-1)
        F = out["forces"].detach().double().cpu().numpy().reshape(-1, 3)
        return E, F


class MACEBackend(Backend):
    level_of_theory = "PBE/PBE+U (MPtrj + sAlex)"

    def __init__(self, device: str, checkpoint: str = "medium-mpa-0",
                 model_path: str | None = None):
        from fittest.models.mace_batched import BatchedMACE

        self.name = "mace_mpa0" if model_path is None else os.path.basename(model_path)
        self.impl = BatchedMACE(checkpoint=checkpoint, model_path=model_path, device=device)

    def predict(self, Z, R, batch, charge):
        return self.impl.predict(Z, R, batch)


def make_backend(model: str, device: str, model_path: str | None = None) -> Backend:
    if model == "orb":
        return OrbBackend(device)
    if model == "aimnet2":
        return AIMNet2Backend(device, "aimnet2")
    if model.startswith("aimnet2_m"):  # shipped ensemble members aimnet2_m0..m3
        return AIMNet2Backend(device, f"aimnet2-wb97m-d3_{model[-1]}")
    if model.startswith("mace"):
        return MACEBackend(device, model_path=model_path)
    raise ValueError(f"unknown model {model!r}")


# --------------------------------------------------------------------------- server
def serve(model: str, port: int, device: str, model_path: str | None = None) -> None:
    backend = make_backend(model, device, model_path)
    if not os.environ.get("FITTEST_AUTHKEY"):
        print("[model_server] FITTEST_AUTHKEY not set: using a random key (only spawn() callers can connect)",
              flush=True)
    with Listener(("127.0.0.1", port), authkey=AUTHKEY) as listener:
        print(f"[model_server] {model} ready on {port}", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    cmd = msg.get("cmd")
                    if cmd == "shutdown":
                        conn.send({"ok": True})
                        return
                    if cmd == "info":
                        conn.send({"model": model, "level_of_theory": backend.level_of_theory,
                                   "pid": os.getpid(), "python": sys.executable})
                        continue
                    t0 = time.perf_counter()
                    try:
                        E, F = backend.predict(np.asarray(msg["Z"]), np.asarray(msg["R"], dtype=np.float64),
                                               np.asarray(msg["batch"]), np.asarray(msg["charge"]))
                        conn.send({"energy": E, "forces": F, "seconds": time.perf_counter() - t0})
                    except Exception as exc:  # report, never kill the server
                        conn.send({"error": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------- client
@dataclass
class ModelClient:
    """Handle to a model agent's process. ``predict`` is one round trip."""

    model: str
    port: int
    proc: subprocess.Popen | None = None
    log_path: str | None = None
    authkey: bytes | None = None  # default: this process's AUTHKEY
    predict_timeout: float = PREDICT_TIMEOUT

    def __post_init__(self):
        self._conn = None
        if self.authkey is None:
            self.authkey = AUTHKEY

    def _open(self, handshake_timeout: float):
        """``Client()`` with a bound on the auth handshake (it has none: a server busy with another
        connection would leave us blocked forever). An abandoned late connection closes itself."""
        box: dict = {}
        abandoned = threading.Event()

        def work():
            try:
                c = Client(("127.0.0.1", self.port), authkey=self.authkey)
            except BaseException as exc:  # noqa: BLE001
                box["exc"] = exc
                return
            if abandoned.is_set():
                c.close()
            else:
                box["conn"] = c

        th = threading.Thread(target=work, daemon=True, name=f"connect-{self.model}")
        th.start()
        th.join(handshake_timeout)
        if th.is_alive():
            abandoned.set()
            if "conn" in box:  # finished in the race window
                box["conn"].close()
            raise TimeoutError(f"{self.model}: handshake on port {self.port} timed out")
        if "exc" in box:
            raise box["exc"]
        return box["conn"]

    def connect(self, timeout: float = 600.0) -> "ModelClient":
        t0 = time.time()
        while True:
            try:
                self._conn = self._open(min(CONNECT_HANDSHAKE_TIMEOUT, max(1.0, timeout)))
            except TimeoutError:
                self._conn = None
                raise
            except (ConnectionRefusedError, OSError):
                self._conn = None
                if self.proc is not None and self.proc.poll() is not None:
                    tail = ""
                    if self.log_path and os.path.exists(self.log_path):
                        with open(self.log_path, errors="replace") as fh:
                            tail = "".join(fh.readlines()[-8:])
                    raise RuntimeError(f"{self.model} server exited with {self.proc.returncode}"
                                       f" (log {self.log_path}):\n{tail}")
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"{self.model} server did not come up on {self.port}")
                time.sleep(1.0)
                continue
            # Never trust whoever happens to hold the port: a stale server from a crashed run
            # would silently answer for the wrong model.
            info = self.info()
            if info.get("model") != self.model or (self.proc is not None and info.get("pid") != self.proc.pid):
                self._conn.close()
                self._conn = None
                raise RuntimeError(f"port {self.port} is held by {info.get('model')!r} (pid {info.get('pid')}), "
                                   f"not the {self.model!r} server we started; stop stale servers first")
            return self

    def predict(self, Z, R, batch, charge=None) -> dict:
        n_sys = int(np.max(batch)) + 1
        charge = np.zeros(n_sys) if charge is None else charge
        self._conn.send({"cmd": "predict", "Z": np.asarray(Z, dtype=np.int64),
                         "R": np.asarray(R, dtype=np.float64),
                         "batch": np.asarray(batch, dtype=np.int64), "charge": np.asarray(charge, dtype=np.float64)})
        if not self._conn.poll(self.predict_timeout):
            # the reply may still arrive later and would desynchronize the stream: drop the link
            self._conn.close()
            self._conn = None
            raise TimeoutError(f"{self.model}: no reply within {self.predict_timeout:.0f} s")
        out = self._conn.recv()
        if "error" in out:
            raise ModelError(f"{self.model}: {out['error']}")
        return out

    def info(self) -> dict:
        self._conn.send({"cmd": "info"})
        return self._conn.recv()

    def close(self, shutdown: bool = True) -> None:
        if self._conn is not None:
            if shutdown:
                try:
                    self._conn.send({"cmd": "shutdown"})
                    self._conn.recv()
                except (EOFError, OSError):
                    pass
            self._conn.close()
        if self.proc is not None:
            self.proc.wait(timeout=60)


def spawn(model: str, port: int, python: str = sys.executable, device: str = "cuda",
          model_path: str | None = None, log_dir: str = "logs") -> ModelClient:
    """Start a model agent in its own process (optionally another interpreter/env).

    Each spawned server gets its own random authkey, known only to it and the returned client.
    """
    import socket

    with socket.socket() as probe:  # fail fast if a stale server still holds the port
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use; stop the stale model server first")
    os.makedirs(log_dir, exist_ok=True)
    cmd = [python, "-m", "fittest.model_server", "--model", model, "--port", str(port), "--device", device]
    if model_path:
        cmd += ["--model-path", model_path]
    log_path = os.path.join(log_dir, f"server_{model}.log")
    log = open(log_path, "w")
    key = secrets.token_hex(16)
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
               FITTEST_AUTHKEY=key)
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
    return ModelClient(model, port, proc, log_path, authkey=key.encode())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-path", default=None)
    a = ap.parse_args()
    serve(a.model, a.port, a.device, a.model_path)
