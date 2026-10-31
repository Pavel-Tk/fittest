"""Conversions between Fittest's flat numpy layout and ALCHEMI Toolkit ``AtomicData``/``Batch``.

Flat layout (the model-server protocol, ``fittest.model_server``):
``Z`` int (N,), ``R`` float (N, 3) Å, ``batch`` int (N,) system index 0..S-1 (contiguous,
non-decreasing), optional ``charge`` float (S,).

Molecules are non-periodic: ``cell``/``pbc`` are left unset, which the Toolkit treats as
open boundaries (MACE fills an identity cell with zero shifts, ``compute_neighbors`` drops
the cell for non-periodic systems).
"""

from __future__ import annotations

import numpy as np
import torch

from nvalchemi.data import AtomicData, Batch


def atomic_data(Z: np.ndarray, R: np.ndarray, *, charge: float = 0.0,
                device: str | torch.device = "cpu", with_dynamics_fields: bool = False) -> AtomicData:
    """One non-periodic molecule as ``AtomicData`` (float32 positions, explicit total charge).

    ``with_dynamics_fields`` adds zero ``energy``/``forces``/``velocities`` so the object can
    enter a FIRE2/FusedStage batch directly (FIRE2 requires velocities on the input).
    """
    n = len(Z)
    kw = dict(
        atomic_numbers=torch.as_tensor(np.asarray(Z), dtype=torch.long),
        positions=torch.as_tensor(np.asarray(R, dtype=np.float32), dtype=torch.float32),
        charge=torch.tensor([[float(charge)]], dtype=torch.float32),
    )
    if with_dynamics_fields:
        kw["energy"] = torch.zeros(1, 1)
        kw["forces"] = torch.zeros(n, 3)
    d = AtomicData(**kw)
    if with_dynamics_fields:
        d.add_node_property("velocities", torch.zeros(n, 3))
    return d.to(device)


def split_index(batch: np.ndarray) -> np.ndarray:
    """Atom offsets ``ptr`` (S+1,) for a contiguous, non-decreasing system index."""
    b = np.asarray(batch, dtype=np.int64)
    if b.size == 0:
        return np.zeros(1, dtype=np.int64)
    if np.any(np.diff(b) < 0) or b[0] != 0 or np.any(np.diff(b) > 1):
        raise ValueError("batch index must be contiguous and non-decreasing, starting at 0")
    counts = np.bincount(b)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def batch_from_arrays(Z: np.ndarray, R: np.ndarray, batch: np.ndarray,
                      charge: np.ndarray | None = None,
                      device: str | torch.device = "cpu") -> Batch:
    """Flat arrays -> Toolkit ``Batch`` (graph order preserved)."""
    ptr = split_index(batch)
    S = len(ptr) - 1
    q = np.zeros(S) if charge is None else np.asarray(charge, dtype=float).reshape(-1)
    datas = [atomic_data(Z[a:b], R[a:b], charge=float(q[s]))
             for s, (a, b) in enumerate(zip(ptr[:-1], ptr[1:]))]
    return Batch.from_data_list(datas, device=device)


def arrays_from_batch(batch: Batch, graph_idx: torch.Tensor | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Toolkit ``Batch`` (optionally a subset of graphs) -> flat ``(Z, R, batch, charge)`` numpy.

    ``graph_idx`` selects graphs (in the given order); the returned ``batch`` index is
    renumbered 0..len(graph_idx)-1 in that order.
    """
    ptr = batch.batch_ptr.detach().cpu().numpy()
    Z_all = batch.atomic_numbers.detach().cpu().numpy().astype(np.int64)
    R_all = batch.positions.detach().cpu().numpy().astype(np.float64)
    q_all = getattr(batch, "charge", None)
    q_all = (np.zeros(batch.num_graphs) if q_all is None
             else q_all.detach().cpu().numpy().reshape(-1).astype(np.float64))
    gs = (np.arange(batch.num_graphs) if graph_idx is None
          else np.asarray(graph_idx.detach().cpu().numpy() if torch.is_tensor(graph_idx) else graph_idx,
                          dtype=np.int64))
    Z, R, b = [], [], []
    for k, g in enumerate(gs):
        a, e = int(ptr[g]), int(ptr[g + 1])
        Z.append(Z_all[a:e]); R.append(R_all[a:e]); b.append(np.full(e - a, k, dtype=np.int64))
    if not Z:
        return (np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0))
    return np.concatenate(Z), np.concatenate(R), np.concatenate(b), q_all[gs]
