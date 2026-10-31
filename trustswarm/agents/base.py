"""Agent base class: subscribe to event kinds, react to new events, post new events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from trustswarm.blackboard import Blackboard, Event


class Agent(ABC):
    """An agent sees *only* the event kinds in ``subscriptions`` and only events it has not
    seen before. All other knowledge is private state set at construction."""

    subscriptions: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, name: str) -> None:
        self.name = name
        self._cursor = 0

    def step(self, bb: Blackboard) -> None:
        events = bb.read(kind=self.subscriptions, since=self._cursor)
        self._cursor = bb.next_id
        self.handle(bb, events)

    @abstractmethod
    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        """React to new events (possibly none: agents may also act on the clock)."""

    def post(self, bb: Blackboard, kind: str, payload: dict[str, Any]) -> int:
        return bb.post(kind, self.name, payload)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


def to_list(a: np.ndarray, decimals: int = 6) -> list[Any]:
    """Arrays → rounded nested lists for JSON payloads."""
    return np.round(np.asarray(a, dtype=float), decimals).tolist()
