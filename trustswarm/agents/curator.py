"""Curator: after triage, repairs the worst model on the oracle labels (DESIGN.md §4).

Fine-tuning itself is GPU work done offline (``scripts/finetune.py``); the agent takes an
injectable ``finetune_fn(model_name, labelled_frames) -> FinetuneReport``:

* :class:`NullFinetune` — never ships (fine-tuning disabled; the arm reports it).
* :class:`ReplayFinetune` — replays a predictions file produced offline by the GPU script,
  so the full ablation can be rerun on CPU. Its acceptance report is read from that file's
  ``meta["acceptance_report"]``; it ships only if the report's verdict is ``"pass"``, and
  every mismatch (wrong base model, missing file, different training set) is reported.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

import numpy as np

from trustswarm.agents.base import Agent
from trustswarm.agents.stream import TRIAGE
from trustswarm.blackboard import (FINETUNE_REPORT, MODEL_UPDATE, ORACLE_RESULT, SYSTEM,
                                   Blackboard, Event)
from trustswarm.cache import load_predictions
from trustswarm.data import POOL, Frame


@dataclass(frozen=True)
class FinetuneReport:
    shipped: bool
    report: dict[str, Any] = field(default_factory=dict)
    new_predictions_path: str | None = None


class FinetuneFn(Protocol):
    def __call__(self, model_name: str, labelled_frames: Sequence[Frame]) -> FinetuneReport: ...


def uid_digest(uids: Sequence[str]) -> str:
    """Order-independent fingerprint of a training set (shared with the GPU script)."""
    return hashlib.sha256("\n".join(sorted(uids)).encode()).hexdigest()[:16]


class NullFinetune:
    def __call__(self, model_name: str, labelled_frames: Sequence[Frame]) -> FinetuneReport:
        return FinetuneReport(False, {"mode": "null", "reason": "fine-tuning disabled (NullFinetune)",
                                      "n_labelled": len(labelled_frames)})


class ReplayFinetune:
    """Replay offline fine-tunes. ``replay`` maps base model -> predictions file (.npz).

    Provenance is enforced, not assumed. A replay ships only if its ``meta`` has
    * ``base_model`` equal to the model the Curator chose,
    * a non-empty ``train_uids`` list that is a subset of the labels the Curator handed over
      (otherwise it was trained on another run's labels: "fine-tune not re-run"),
    * no train uid among the held-out uids (eval / skeptic_ref / planted; see
      :meth:`bind_held_out`), and
    * ``acceptance_report.verdict == "pass"``.
    Every refusal is reported with its reason.
    """

    def __init__(self, replay: Mapping[str, str | Path],
                 held_out_uids: Iterable[str] = ()) -> None:
        self._replay = {m: Path(p) for m, p in replay.items()}
        self._held_out = frozenset(held_out_uids)

    def bind_held_out(self, uids: Iterable[str]) -> None:
        """Called by the ablation with every uid that must never be trained on."""
        self._held_out = frozenset(uids)

    def __call__(self, model_name: str, labelled_frames: Sequence[Frame]) -> FinetuneReport:
        uids = [f.uid for f in labelled_frames]
        base: dict[str, Any] = {"mode": "replay", "n_labelled": len(uids),
                                "train_digest_requested": uid_digest(uids)}
        path = self._replay.get(model_name)
        if path is None or not path.exists():
            return FinetuneReport(False, {**base, "reason": f"no offline fine-tune available for "
                                                            f"{model_name!r} ({path})"})
        meta = load_predictions(path).meta
        acceptance = dict(meta.get("acceptance_report", {}))
        report = {**base, "predictions_path": str(path), "acceptance_report": acceptance,
                  "weights": meta.get("weights", path.stem)}

        def refuse(reason: str) -> FinetuneReport:
            return FinetuneReport(False, {**report, "reason": reason})

        if meta.get("base_model") != model_name:
            return refuse(f"replay file was fine-tuned from {meta.get('base_model')!r}, "
                          f"not {model_name!r}")
        trained = set(meta.get("train_uids") or [])
        if not trained:
            return refuse("replay file has no train_uids: provenance unknown")
        report["n_train"] = len(trained)
        report["train_digest_replayed"] = uid_digest(sorted(trained))
        leaked = sorted(trained & self._held_out)
        if leaked:
            return refuse(f"{len(leaked)} train uids are held-out frames (e.g. {leaked[0]})")
        foreign = trained - set(uids)
        if foreign:
            return refuse(f"replay was trained on a different label set ({len(foreign)} of "
                          f"{len(trained)} train uids are not this run's oracle labels): "
                          f"fine-tune not re-run for this budget/configuration")
        if acceptance.get("verdict") != "pass":
            return refuse(f"acceptance verdict {acceptance.get('verdict')!r} != 'pass'")
        return FinetuneReport(True, report, str(path))


class Curator(Agent):
    subscriptions: ClassVar[frozenset[str]] = frozenset({ORACLE_RESULT, SYSTEM})

    def __init__(self, models: Sequence[str], finetune_fn: FinetuneFn, repair_round: int,
                 name: str = "curator") -> None:
        super().__init__(name)
        self._models = tuple(models)
        self._finetune = finetune_fn
        self._repair_round = repair_round
        self._geometry: dict[str, dict[str, Any]] = {}
        self._labels: dict[str, dict[str, Any]] = {}
        self._done = False
        self.report: FinetuneReport | None = None

    def handle(self, bb: Blackboard, events: list[Event]) -> None:
        for e in events:
            if e.kind == SYSTEM and e.payload["phase"] == TRIAGE:
                self._geometry[e.payload["uid"]] = e.payload
            elif e.kind == ORACLE_RESULT:
                self._labels[e.payload["uid"]] = e.payload
        if self._done or bb.round < self._repair_round:
            return
        self._done = True
        self._repair(bb)

    def ledger_force_rmse(self) -> dict[str, float]:
        """Mean oracle-revealed force RMSE per model on labelled *pool* systems."""
        errs: dict[str, list[float]] = defaultdict(list)
        for uid, p in self._labels.items():
            if uid in self._geometry:
                for m in self._models:
                    if m in p["errors"]:
                        errs[m].append(p["errors"][m]["force_rmse"])
        return {m: float(np.mean(v)) for m, v in sorted(errs.items())}

    def _repair(self, bb: Blackboard) -> None:
        rmse = self.ledger_force_rmse()
        frames = [Frame(uid=u, reaction=g["reaction"], formula=g["formula"], split=POOL,
                        role=g["role"], Z=np.asarray(g["Z"], dtype=np.int64),
                        R=np.asarray(g["R"], dtype=float), E=float(self._labels[u]["E"]),
                        F=np.asarray(self._labels[u]["F"], dtype=float))
                  for u, g in sorted(self._geometry.items()) if u in self._labels]
        if not rmse:
            self.report = FinetuneReport(False, {"reason": "no oracle labels on pool systems"})
            self.post(bb, FINETUNE_REPORT, {"model": None, "shipped": False,
                                            "report": self.report.report, "ledger_force_rmse": rmse})
            return
        worst = max(rmse, key=lambda m: (rmse[m], m))
        self.report = self._finetune(worst, frames)
        self.post(bb, FINETUNE_REPORT, {"model": worst, "n_labels": len(frames),
                                        "ledger_force_rmse": rmse, "shipped": self.report.shipped,
                                        "report": self.report.report,
                                        "new_predictions_path": self.report.new_predictions_path})
        if self.report.shipped and self.report.new_predictions_path:
            self.post(bb, MODEL_UPDATE, {
                "model": worst, "predictions_path": self.report.new_predictions_path,
                "weights": self.report.report.get("weights",
                                                  Path(self.report.new_predictions_path).stem)})
