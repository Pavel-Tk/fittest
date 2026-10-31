"""Served-answer candidates: every non-empty subset mean of the model agents' predictions.

Shared by the Oracle (reports each candidate's error), the Calibrator (keeps a ledger per
candidate) and the trust service (rebuilds a candidate from its members). Standard library
only, so any process can import it.

Naming: a singleton is the model's own name; the full set is ``"consensus"``; any other
subset is ``"mean(a+b)"`` with members sorted.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

CONSENSUS = "consensus"


def candidate_members(models: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Ordered ``{name: members}``: singletons (model order), then larger subsets, then consensus."""
    models = tuple(models)
    if len(set(models)) != len(models) or CONSENSUS in models:
        raise ValueError(f"model names must be unique and not {CONSENSUS!r}: {models}")
    out: dict[str, tuple[str, ...]] = {m: (m,) for m in models}
    for size in range(2, len(models)):
        for combo in combinations(sorted(models), size):
            out[f"mean({'+'.join(combo)})"] = combo
    if len(models) > 1:
        out[CONSENSUS] = tuple(sorted(models))
    return out
