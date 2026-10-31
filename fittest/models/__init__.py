"""Model loaders built on the NVIDIA ALCHEMI Toolkit wrappers.

Import submodules explicitly (``from fittest.models.mace_batched import BatchedMACE``);
this package does not import torch or nvalchemi at import time, so CPU-only code that only
needs ``fittest.models.toolkit_io`` type hints stays importable without a GPU stack.
"""

__all__ = ["mace_batched", "toolkit_io"]
