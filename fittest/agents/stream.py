"""The structure stream: a passive source that releases *label-free* frames on schedule.

It is not a decision maker — it is the CPU stand-in for "new systems arrive in the
ALCHEMI batch". Pool reactions arrive in chunks (one per triage round); the held-out
``eval`` frames arrive once, in ``phase="serve"``, after triage, so the collective's served
answer can be scored on data nobody ever had labels for.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from fittest.agents.base import Agent, to_list
from fittest.blackboard import SYSTEM, Blackboard, Event
from fittest.data import Frame

TRIAGE = "triage"
SERVE = "serve"


def structure_payload(frame: Frame, phase: str) -> dict[str, Any]:
    """Public, label-free description of one structure (shared by ``system`` and ``plant``)."""
    if frame.has_labels:
        raise ValueError(f"refusing to publish labelled frame {frame.uid}")
    return {"uid": frame.uid, "reaction": frame.reaction, "formula": frame.formula,
            "role": frame.role, "phase": phase, "Z": [int(z) for z in frame.Z],
            "R": to_list(frame.R, 5)}


class PoolStream(Agent):
    subscriptions: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, frames: Sequence[Frame], arrival_round: Mapping[str, int],
                 serve_frames: Sequence[Frame] = (), serve_round: int | None = None,
                 name: str = "stream") -> None:
        super().__init__(name)
        if any(f.has_labels for f in [*frames, *serve_frames]):
            raise ValueError("PoolStream must only ever hold label-free frames")
        self._schedule: dict[int, list[tuple[Frame, str]]] = defaultdict(list)
        for f in frames:
            self._schedule[int(arrival_round[f.uid])].append((f, TRIAGE))
        if serve_frames:
            if serve_round is None:
                raise ValueError("serve_round is required when serve_frames are given")
            self._schedule[serve_round].extend((f, SERVE) for f in serve_frames)
        self._released: set[int] = set()

    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        r = bb.round
        if r in self._released:
            return
        self._released.add(r)
        for frame, phase in sorted(self._schedule.get(r, []), key=lambda fp: fp[0].uid):
            self.post(bb, SYSTEM, structure_payload(frame, phase))
