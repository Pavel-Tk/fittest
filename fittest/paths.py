"""Machine-specific locations, resolved from environment variables (no hard-coded VM paths).

Every script takes these as CLI defaults, so another team only sets env vars once:

=========================  ==========================================================  ===================
variable                   meaning                                                     default
=========================  ==========================================================  ===================
``FITTEST_T1X``         Transition1x HDF5 file (figshare 36035789, 6.6 GB)          ``data/Transition1x.h5``
``FITTEST_CORE_PYTHON`` interpreter with torch + nvalchemi + mace + aimnet           ``sys.executable``
``FITTEST_ORB_PYTHON``  interpreter of the separate Orb env (orb-models)            core python
``FITTEST_NEB_DIR``     ``batched_neb_endpoints`` dir of the nvalchemi-toolkit repo  found via ``nvalchemi``
``FITTEST_LOG_DIR``     model-server logs                                           ``logs``
=========================  ==========================================================  ===================

Our Colab L4 values: ``FITTEST_T1X=/content/data/Transition1x.h5``,
``FITTEST_ORB_PYTHON=/content/envs/orb/bin/python``, ``FITTEST_LOG_DIR=/content/logs``
(see ``scripts/setup_gpu.sh`` and ``docs/REPRODUCE.md``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def t1x_path() -> str:
    return os.environ.get("FITTEST_T1X", str(ROOT / "data" / "Transition1x.h5"))


def core_python() -> str:
    return os.environ.get("FITTEST_CORE_PYTHON", sys.executable)


def orb_python() -> str:
    return os.environ.get("FITTEST_ORB_PYTHON", core_python())


def log_dir() -> str:
    return os.environ.get("FITTEST_LOG_DIR", "logs")


def model_python(model: str) -> str:
    """Interpreter that hosts a model server (Orb lives in its own env)."""
    return orb_python() if model == "orb" else core_python()


def neb_dir() -> str:
    """``examples/advanced/batched_neb_endpoints`` of the nvalchemi-toolkit checkout.

    ``FITTEST_NEB_DIR`` wins; otherwise the directory is located next to an editable
    (source) install of ``nvalchemi``. The examples are not part of a wheel install, so with a
    wheel you must clone the toolkit repo and set the variable.
    """
    env = os.environ.get("FITTEST_NEB_DIR")
    if env:
        return env
    spec = importlib.util.find_spec("nvalchemi")
    if spec is not None and spec.origin:
        cand = Path(spec.origin).resolve().parents[1] / "examples" / "advanced" / "batched_neb_endpoints"
        if cand.is_dir():
            return str(cand)
    return str(ROOT.parent / "nvalchemi-toolkit" / "examples" / "advanced" / "batched_neb_endpoints")
