"""Batched MACE inference through the ALCHEMI Toolkit ``MACEWrapper``.

One code path for the foundation model and for TrustSwarm's LoRA-repaired model:

* ``BatchedMACE()``                      -> ``MACEWrapper.from_checkpoint("medium-mpa-0")``
* ``BatchedMACE(model_path=<dir with trustswarm.json>)`` -> TrustSwarm fine-tune checkpoint
  (``trustswarm.mace_lora`` format written by ``scripts/finetune.py``: merged LoRA weights +
  refitted E0 in ``model.pt``), loaded with ``trustswarm.mace_lora.load_mace_lora`` into a
  ``MACEWrapper`` (cuEq conversion optional). This is the primary fine-tune path.
* ``BatchedMACE(model_path=<dir>)``      -> bare ALCHEMI PEFT checkpoint written by
  ``FineTuningStrategy.save_checkpoint(dir, save_trainable_state_only=True)``
  (``scripts/finetune.py``). The base model is rebuilt from the checkpoint's own
  ``model_specs["main"]`` (the exact ``from_checkpoint`` kwargs used for training: cuEq,
  dtype, re-fitted E0s), falling back to ``checkpoint`` + ``<dir>/atomic_energies.json``;
  adapters are then loaded and merged with ``nvalchemi.training.peft.load_peft_checkpoint_into_model``.
* ``BatchedMACE(model_path=<file.pt>)``  -> a full pickled MACE module
  (``MACEWrapper.export_model(path)``), or a plain ``state_dict`` of the base architecture
  (``export_model(path, as_state_dict=True)``), loaded into ``checkpoint``.

Neighbor lists are built per call with ``nvalchemi.neighbors.compute_neighbors`` using the
wrapper's own ``NeighborConfig`` (COO, full list, cutoff = MACE ``r_max``). Molecules are
non-periodic (no cell, no pbc); the Toolkit treats that as open boundaries.

``predict`` follows the model-server protocol: flat numpy arrays in, float64 eV / eV/Å out.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from trustswarm.models.toolkit_io import batch_from_arrays, split_index

DEFAULT_CHECKPOINT = "medium-mpa-0"


def _cueq_available() -> bool:
    import importlib.util

    return (importlib.util.find_spec("cuequivariance_torch") is not None
            and importlib.util.find_spec("cuequivariance_ops_torch") is not None)


def _freeze(wrapper: torch.nn.Module) -> torch.nn.Module:
    for p in wrapper.parameters():
        p.requires_grad_(False)
    wrapper.eval()
    return wrapper


def load_mace_wrapper(checkpoint: str = DEFAULT_CHECKPOINT, model_path: str | None = None,
                      device: str | torch.device = "cuda", dtype: torch.dtype = torch.float32,
                      enable_cueq: bool | None = None):
    """Return an inference-ready (frozen, eval) Toolkit ``MACEWrapper`` plus a provenance dict."""
    from nvalchemi.models.mace import MACEWrapper

    device = torch.device(device)
    if enable_cueq is None:
        enable_cueq = device.type == "cuda" and _cueq_available()
    info: dict = {"checkpoint": checkpoint, "model_path": model_path, "dtype": str(dtype),
                  "enable_cueq": bool(enable_cueq)}

    if model_path is None:
        w = MACEWrapper.from_checkpoint(checkpoint, device=device, dtype=dtype,
                                        enable_cueq=enable_cueq, compile_model=False)
        info["format"] = "foundation"
        return _freeze(w), info

    p = Path(model_path)
    if p.is_dir() and (p / "trustswarm.json").exists():
        # TrustSwarm fine-tune checkpoint (trustswarm.mace_lora format, scripts/finetune.py):
        # merged LoRA + refitted E0 in <dir>/model.pt; optional cuEq conversion on load.
        from trustswarm.mace_lora import load_mace_lora, read_manifest

        w = load_mace_lora(p, device=device, dtype=dtype, enable_cueq=enable_cueq)
        info.update(format="trustswarm.mace_lora", manifest_base=read_manifest(p).get("base_checkpoint"))
        return _freeze(w.to(device)), info
    if p.is_dir():
        # Bare ALCHEMI PEFT checkpoint directory (FineTuningStrategy.save_checkpoint).
        from nvalchemi.training.peft import load_peft_checkpoint_into_model

        base = None
        meta_files = sorted(p.glob("**/*.json"))
        spec = None
        for mf in meta_files:
            try:
                meta = json.loads(mf.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(meta, dict) and isinstance(meta.get("model_specs"), dict) \
                    and "main" in meta["model_specs"]:
                spec = meta["model_specs"]["main"]
                break
        if spec is not None:
            try:
                from nvalchemi.specs import create_model_spec_from_json

                base = create_model_spec_from_json(dict(spec)).build(device=device)
                info["base_from"] = "checkpoint model_specs"
            except Exception as exc:  # fall back to explicit reconstruction below
                info["base_spec_error"] = f"{type(exc).__name__}: {exc}"
        if base is None:
            e0_path = p / "atomic_energies.json"
            # Adapters trained on the e3nn layout (scripts/finetune.py) -> rebuild without cuEq.
            base = MACEWrapper.from_checkpoint(
                checkpoint, device=device, dtype=dtype, enable_cueq=False,
                atomic_energies_path=e0_path if e0_path.exists() else None)
            info["base_from"] = f"{checkpoint} (+{e0_path.name})" if e0_path.exists() else checkpoint
        w = load_peft_checkpoint_into_model(base, p, map_location=device, merge=True)
        info["format"] = "peft_checkpoint_dir"
        return _freeze(w.to(device)), info

    obj = torch.load(p, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        if isinstance(obj, MACEWrapper):
            w = obj
        else:
            w = MACEWrapper(obj.to(dtype=dtype))
        info["format"] = "pickled_module"
        return _freeze(w.to(device)), info
    if isinstance(obj, dict):
        state = obj.get("state_dict", obj)
        w = MACEWrapper.from_checkpoint(checkpoint, device=device, dtype=dtype, enable_cueq=False)
        missing, unexpected = w.model.load_state_dict(state, strict=False)
        if unexpected or missing:
            # state dict of the wrapper (keys prefixed "model.") rather than the inner MACE
            w.load_state_dict(state, strict=True)
        info["format"] = "state_dict"
        return _freeze(w), info
    raise TypeError(f"unrecognised MACE checkpoint at {model_path}: {type(obj).__name__}")


class BatchedMACE:
    """Toolkit MACE for many small molecules per call.

    ``predict(Z (N,), R (N,3), batch (N,)) -> (E (S,) float64 eV, F (N,3) float64 eV/Å)``.
    Systems are processed in chunks of at most ``max_atoms`` atoms to bound GPU memory.
    ``self.model`` is the underlying ``MACEWrapper`` (a Toolkit ``BaseModelMixin``) so the
    same object can drive FIRE2/NEB/FusedStage runs.
    """

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, model_path: str | None = None,
                 device: str = "cuda", dtype: torch.dtype = torch.float32,
                 enable_cueq: bool | None = None, max_atoms: int = 20000):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_atoms = int(max_atoms)
        t0 = time.perf_counter()
        self.model, self.info = load_mace_wrapper(checkpoint, model_path, self.device, dtype, enable_cueq)
        self.model.model_config.active_outputs = {"energy", "forces"}
        self.info["load_seconds"] = time.perf_counter() - t0
        self.neighbor_config = self.model.model_config.neighbor_config

    @property
    def cutoff(self) -> float:
        return float(self.neighbor_config.cutoff)

    def _forward(self, b) -> tuple[torch.Tensor, torch.Tensor]:
        from nvalchemi.neighbors import compute_neighbors

        compute_neighbors(b, config=self.neighbor_config)
        with torch.enable_grad():
            out = self.model(b)
        return out["energy"].detach().reshape(-1), out["forces"].detach()

    def predict(self, Z: np.ndarray, R: np.ndarray, batch: np.ndarray,
                charge: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        Z = np.asarray(Z, dtype=np.int64)
        R = np.asarray(R, dtype=np.float64).reshape(-1, 3)
        batch = np.asarray(batch, dtype=np.int64)
        ptr = split_index(batch)
        S = len(ptr) - 1
        E = np.empty(S, dtype=np.float64)
        F = np.empty((len(Z), 3), dtype=np.float64)
        s0 = 0
        while s0 < S:
            # Greedy chunk of whole systems within the atom budget (at least one system).
            s1 = s0 + 1
            while s1 < S and ptr[s1 + 1] - ptr[s0] <= self.max_atoms:
                s1 += 1
            a, e = int(ptr[s0]), int(ptr[s1])
            b = batch_from_arrays(Z[a:e], R[a:e], batch[a:e] - s0, device=self.device)
            Ec, Fc = self._forward(b)
            E[s0:s1] = Ec.double().cpu().numpy()
            F[a:e] = Fc.double().cpu().numpy()
            s0 = s1
        return E, F

    __call__ = predict
