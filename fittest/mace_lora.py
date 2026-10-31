"""On-disk format of a Fittest MACE LoRA fine-tune, and its loader (DESIGN.md §4 Curator).

Written by ``scripts/finetune.py``; read by ``fittest.models.mace_batched.BatchedMACE``
(``model_path=<checkpoint dir>``) and by anything else that speaks plain mace-torch.

Layout of one checkpoint directory (format ``fittest.mace_lora/1``)::

    <ckpt>/
      model.pt                 merged model: a plain mace-torch ``ScaleShiftMACE`` nn.Module in
                               e3nn layout, float32, saved with ``torch.save(module)``. The LoRA
                               adapters are folded into the base weights
                               (``merge_lora_into_model``) and the T1x-refitted per-element
                               reference energies (E0) are baked into
                               ``atomic_energies_fn``. Outputs are total energies on the
                               ωB97x/6-31G(d) scale [eV] and forces [eV/Å].
      fittest.json          {"format", "base_checkpoint", "model_file", "layout": "e3nn",
                                "dtype", "atomic_energies" {Z: E0 eV}, training config, ...}
      atomic_energies.json     {"Z": E0 eV} (the ``atomic_energies_path`` format of
                               ``MACEWrapper.from_checkpoint``)
      manifest.json, strategy.json, models/, optimizers/, ...
                               the ALCHEMI PEFT checkpoint itself
                               (``FineTuningStrategy.save_checkpoint(<ckpt>,
                               save_trainable_state_only=True)``), so ``<ckpt>`` is also a PEFT
                               checkpoint root: ``load_peft_checkpoint_into_model(
                               MACEWrapper.from_checkpoint(base_checkpoint, enable_cueq=False,
                               atomic_energies_path="<ckpt>/atomic_energies.json"), "<ckpt>")``.
                               The adapters are e3nn-layout: rebuild the base with
                               ``enable_cueq=False``.
      acceptance_report.json   ALCHEMI acceptance report + Fittest hold-out metrics.

Loading ``model.pt`` needs only torch + mace-torch::

    module = torch.load("<ckpt>/model.pt", map_location="cuda", weights_only=False)

or, as an ALCHEMI model (frozen, eval mode, optional cuEquivariance conversion)::

    from fittest.mace_lora import load_mace_lora
    wrapper = load_mace_lora("<ckpt>", device="cuda", enable_cueq=True)

Do **not** pass ``model.pt`` to ``MACEWrapper.from_checkpoint``: in nvalchemi 0.3.0-rc that
routes local paths through ``mace...download_mace_mp_checkpoint``, which tries to download the
path as a URL and caches by sanitised basename, so two checkpoints both named ``model.pt``
would silently collide in ``~/.cache/mace``. :func:`load_mace_lora` replicates the relevant
``from_checkpoint`` steps (dtype cast, optional cuEq conversion, ``MACEWrapper(model)``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FORMAT = "fittest.mace_lora/1"
MODEL_FILE = "model.pt"
MANIFEST_FILE = "fittest.json"  # manifest.json belongs to the PEFT checkpoint
E0_FILE = "atomic_energies.json"
REPORT_FILE = "acceptance_report.json"


def resolve_model_file(model_path: str | Path) -> Path:
    """``<ckpt>`` or ``<ckpt>/model.pt`` -> path of the merged model file."""
    p = Path(model_path)
    f = p / MODEL_FILE if p.is_dir() else p
    if not f.is_file():
        raise FileNotFoundError(f"no merged MACE model at {f}")
    return f


def read_manifest(model_path: str | Path) -> dict[str, Any]:
    p = Path(model_path)
    d = p if p.is_dir() else p.parent
    manifest = json.loads((d / MANIFEST_FILE).read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{d / MANIFEST_FILE}: format {manifest.get('format')!r} != {FORMAT!r}")
    return manifest


def load_mace_module(model_path: str | Path, device: Any = "cpu", dtype: Any = None):
    """The merged mace-torch nn.Module (e3nn layout), weights frozen, eval mode."""
    import torch

    module = torch.load(resolve_model_file(model_path), map_location=torch.device(device),
                        weights_only=False)  # our own artefact: a pickled nn.Module
    if dtype is not None:
        module = module.to(dtype=dtype)
    for prm in module.parameters():
        prm.requires_grad_(False)
    return module.eval()


def load_mace_lora(model_path: str | Path, device: Any = "cuda", dtype: Any = None,
                   enable_cueq: bool = False):
    """Load a fine-tuned checkpoint as a frozen ``nvalchemi`` ``MACEWrapper``.

    ``dtype`` defaults to float32 (the training dtype). ``enable_cueq=True`` converts the e3nn
    layout to cuEquivariance kernels (CUDA only), exactly as ``MACEWrapper.from_checkpoint``
    does for named checkpoints.
    """
    import torch
    from nvalchemi.models.mace import MACEWrapper

    device = torch.device(device)
    module = load_mace_module(model_path, device=device, dtype=dtype or torch.float32)
    if enable_cueq:
        if device.type != "cuda":
            raise ValueError("cuEquivariance conversion requires a CUDA device")
        from mace.cli.convert_e3nn_cueq import run as convert_e3nn_cueq

        with torch.cuda.device(device):
            module = convert_e3nn_cueq(module, return_model=True, device="cuda")
        module = module.to(device)
    return MACEWrapper(module, train=False)


def write_checkpoint(ckpt_dir: str | Path, merged_module, manifest: dict[str, Any]) -> Path:
    """Save the merged module + manifest. Returns the model file path."""
    import torch

    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(merged_module, d / MODEL_FILE)
    (d / MANIFEST_FILE).write_text(json.dumps({"format": FORMAT, "model_file": MODEL_FILE,
                                               **manifest}, indent=2, sort_keys=True))
    if "atomic_energies" in manifest:
        (d / E0_FILE).write_text(json.dumps(manifest["atomic_energies"], indent=2, sort_keys=True))
    return d / MODEL_FILE
