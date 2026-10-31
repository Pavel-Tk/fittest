"""Full AIMNet2 calculator (incl. its D3/Coulomb terms) exposed as a Toolkit model.

Same pattern as ``AIMNet2rxnWrapper`` in ``nvalchemi-toolkit/examples/advanced/11_batched_neb.py``
but for any checkpoint name the ``aimnet`` registry resolves (default ``"aimnet2"``, the
wB97M-D3 model that the ``aimnet2`` model server uses), so a dynamics driver and the
committee member are the same potential. The calculator builds its own neighbor list, so
``neighbor_config`` is ``None`` and no neighbor hook is needed.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig


class AIMNet2CalculatorWrapper(nn.Module, BaseModelMixin):
    def __init__(self, checkpoint: str = "aimnet2", device: str | torch.device = "cuda"):
        super().__init__()
        from aimnet.calculators import AIMNet2Calculator

        self.checkpoint = checkpoint
        self.calculator = AIMNet2Calculator(checkpoint, device=str(device))
        self.model = self.calculator.model
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            required_inputs=frozenset({"charge"}),
            supports_pbc=False,
            needs_pbc=False,
            active_outputs={"energy", "forces"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs: Any):
        raise NotImplementedError("AIMNet2CalculatorWrapper does not expose embeddings")

    def forward(self, data: AtomicData | Batch, **kwargs: Any):
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        charge = getattr(data, "charge", None)
        if charge is None:
            charge = torch.zeros(data.num_graphs, 1, device=data.positions.device)
        out = self.calculator({
            "coord": data.positions.detach().to(torch.float32),
            "numbers": data.atomic_numbers.long(),
            "charge": charge.reshape(-1).to(torch.float32),
            "mol_idx": data.batch_idx.long(),
        }, forces=True)
        energy = out["energy"].reshape(-1, 1).to(data.positions.dtype)
        return OrderedDict(energy=energy.detach(), forces=out["forces"].detach().to(data.positions.dtype))
