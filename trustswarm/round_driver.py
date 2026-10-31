"""Round driver: advances the clock. It makes no decisions (DESIGN.md §4).

Within a round, every agent is stepped repeatedly, in a fixed order, until no agent posts
anything (quiescence); then the clock ticks. Which events exist, and what they say, is
entirely up to the agents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from trustswarm.agents.base import Agent
from trustswarm.blackboard import Blackboard


class QuiescenceError(RuntimeError):
    """Agents kept posting for ``max_substeps`` passes within one round."""


@dataclass
class RoundDriver:
    blackboard: Blackboard
    agents: Sequence[Agent]
    max_substeps: int = 100

    def run_round(self) -> int:
        """Step all agents until quiescent; returns the number of passes taken."""
        for passes in range(1, self.max_substeps + 1):
            before = self.blackboard.next_id
            for agent in self.agents:
                agent.step(self.blackboard)
            if self.blackboard.next_id == before:
                return passes
        raise QuiescenceError(f"no quiescence after {self.max_substeps} passes "
                              f"in round {self.blackboard.round}")

    def run(self, n_rounds: int) -> None:
        for _ in range(n_rounds):
            self.run_round()
            self.blackboard.tick()
