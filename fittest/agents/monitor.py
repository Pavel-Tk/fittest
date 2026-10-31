"""DisagreementMonitor: CPU twin of the ALCHEMI ``DisagreementHook`` (fittest.hook).

Same contract as the hook (DESIGN.md §3/§6): gather every model's forces on a system,
compute σ_F with :func:`fittest.signals.force_spread`, post ``flag`` if σ ≥ τ else
``pass``. It reads only ``prediction`` and ``threshold`` — it never sees labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from fittest.agents.base import Agent, batch_payload, iter_rows
from fittest.blackboard import FLAG, PASS, PREDICTION, THRESHOLD, Blackboard, Event
from fittest.signals import energy_spread, force_spread, mean_abs_force, model_deviations


@dataclass
class Measurement:
    phase: str
    sigma_F: float
    sigma_F_max: float
    sigma_E: float | None
    mean_abs_fbar: float
    deviation: dict[str, float]
    versions: dict[str, int]

    def payload(self, uid: str, tau: float) -> dict[str, Any]:
        return {"uid": uid, "phase": self.phase, "sigma_F": self.sigma_F,
                "sigma_F_max": self.sigma_F_max, "sigma_E": self.sigma_E,
                "mean_abs_fbar": self.mean_abs_fbar, "deviation": self.deviation,
                "versions": self.versions, "tau": tau}

    def pass_row(self, uid: str) -> dict[str, Any]:
        return {"uid": uid, "phase": self.phase, "sigma_F": self.sigma_F}


def measure(models: Sequence[str], preds: dict[str, dict[str, Any]]) -> Measurement:
    """σ statistics for one system from the latest prediction payload of each model."""
    forces = np.stack([np.asarray(preds[m]["forces"], dtype=float) for m in models])
    mean_f, max_f = force_spread(forces)
    rel = [preds[m]["rel_energy"] for m in models]
    return Measurement(
        phase=preds[models[0]]["phase"], sigma_F=mean_f, sigma_F_max=max_f,
        sigma_E=energy_spread(rel) if all(r is not None for r in rel) else None,
        mean_abs_fbar=mean_abs_force(forces.mean(axis=0)),
        deviation={m: float(d) for m, d in zip(models, model_deviations(forces))},
        versions={m: int(preds[m]["version"]) for m in models},
    )


PREDICTION_KEYS = ("uid", "phase", "energy", "rel_energy", "forces")


class DisagreementMonitor(Agent):
    subscriptions: ClassVar[frozenset[str]] = frozenset({PREDICTION, THRESHOLD})

    def __init__(self, models: Sequence[str], name: str = "hook") -> None:
        super().__init__(name)
        if len(models) < 2:
            raise ValueError("disagreement needs at least two models")
        self._models = tuple(models)
        self._latest: dict[str, dict[str, dict[str, Any]]] = {}
        self._measured: dict[str, Measurement] = {}
        self._status: dict[str, str | None] = {}
        self._tau: float | None = None

    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        tau_changed = False
        for e in events:
            if e.kind == PREDICTION:
                model = e.payload["model"]
                if model not in self._models:
                    continue
                for _, row in iter_rows(e.payload, PREDICTION_KEYS):
                    uid = row["uid"]
                    self._latest.setdefault(uid, {})[model] = {**row, "version": e.payload["version"]}
                    if len(self._latest[uid]) == len(self._models):
                        self._measured[uid] = measure(self._models, self._latest[uid])
                        self._status[uid] = None  # (re-)measured: decide again
                continue
            if e.kind == THRESHOLD:
                tau_changed = tau_changed or e.payload["tau"] != self._tau
                self._tau = float(e.payload["tau"])
        if self._tau is None:
            return  # no threshold yet: hold measurements until the Calibrator posts one
        passes = []
        for uid in sorted(self._measured):
            m, status = self._measured[uid], self._status[uid]
            flagged = m.sigma_F >= self._tau
            if flagged and (status is None or (status == PASS and tau_changed)):
                self._status[uid] = FLAG
                self.post(bb, FLAG, m.payload(uid, self._tau))
            elif not flagged and status is None:
                self._status[uid] = PASS
                passes.append(m.pass_row(uid))
        if passes:  # passes are uneventful: one batch per step
            self.post(bb, PASS, batch_payload(passes, tau=self._tau))
