"""Append-only event log through which agents communicate (DESIGN.md §4 "Blackboard").

Nothing else is shared between agents: every fact an agent acts on that it did not hold
privately from the start reached it as an :class:`Event` here. The optional JSONL file is
a deliverable — it is the audit trail of who knew what, when.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

# Event kinds (DESIGN.md §4 table, plus the stream/clock kinds needed on CPU).
SYSTEM = "system"              # a label-free structure enters the stream (stream source)
PLANT = "plant"                # a label-free planted structure enters the stream (Skeptic)
PREDICTION = "prediction"      # one model's energy/forces on one structure (ModelAgent)
FLAG = "flag"                  # σ ≥ τ (DisagreementMonitor / ALCHEMI hook)
PASS = "pass"                  # σ < τ
THRESHOLD = "threshold"        # τ for the current round (Calibrator)
VERDICT = "verdict"            # trust + recommended model (Calibrator)
VETO = "veto"                  # Skeptic disputes a trusted verdict
BID = "bid"                    # request for an oracle call (ModelAgent, Skeptic)
ORACLE_RESULT = "oracle_result"  # DFT labels + per-model errors (Oracle) — the ONLY label channel
REPUTATION = "reputation"      # bidder reputations (Calibrator)
FINETUNE_REPORT = "finetune_report"  # Curator
MODEL_UPDATE = "model_update"  # Curator: a model's weights were replaced
TICK = "tick"                  # clock advanced (round driver)

ARRIVAL_KINDS: frozenset[str] = frozenset({SYSTEM, PLANT})
HEAVY_KEYS: frozenset[str] = frozenset({"forces", "F", "R"})
"""Array-valued payload keys kept on in-memory events (the data plane) but summarised as
``{"omitted": "array", "shape": [...]}`` in the JSONL audit log, so the log stays small."""
LOG_SIGNIFICANT_DIGITS = 5
MEASUREMENT_KINDS: frozenset[str] = frozenset({FLAG, PASS})


@dataclass(frozen=True)
class Event:
    id: int
    round: int
    kind: str
    author: str
    payload: dict[str, Any]
    t: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), allow_nan=False)

    def to_log_json(self, heavy_keys: frozenset[str] = HEAVY_KEYS) -> str:
        """Compact audit-log form: arrays summarised, floats rounded."""
        record = asdict(self)
        record["payload"] = _compact(self.payload, heavy_keys)
        record["t"] = round(self.t, 3)
        return json.dumps(record, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def _shape(value: Any) -> list[int]:
    shape = []
    while isinstance(value, list):
        shape.append(len(value))
        value = value[0] if value else None
    return shape


def _compact(value: Any, heavy_keys: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {k: ({"omitted": "array", "shape": _shape(v)} if k in heavy_keys and isinstance(v, list)
                    else _compact(v, heavy_keys)) for k, v in value.items()}
    if isinstance(value, list):
        return [_compact(v, heavy_keys) for v in value]
    if isinstance(value, float):
        return float(f"{value:.{LOG_SIGNIFICANT_DIGITS}g}")
    return value


class Blackboard:
    """In-memory append-only log with an optional JSONL audit mirror.

    ``post`` validates that the payload is JSON-serialisable (no numpy arrays, no NaN). The
    JSONL file is the human-readable audit trail: identical events, except that array-valued
    keys (:data:`HEAVY_KEYS`: forces, labels, coordinates) are replaced by their shape and
    floats are rounded (:meth:`Event.to_log_json`).
    """

    def __init__(self, log_path: str | Path | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self._events: list[Event] = []
        self._round = 0
        self._clock = clock
        self._log_path = Path(log_path) if log_path is not None else None
        self._fh: IO[str] | None = None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._log_path.open("w", encoding="utf-8")

    # --- clock -------------------------------------------------------------------------
    @property
    def round(self) -> int:
        return self._round

    def tick(self) -> int:
        """Advance the clock one round; returns the new round."""
        self._round += 1
        self.post(TICK, "clock", {"round": self._round})
        return self._round

    # --- write -------------------------------------------------------------------------
    def post(self, kind: str, author: str, payload: dict[str, Any]) -> int:
        line = json.dumps(payload, allow_nan=False)  # raises on non-JSON payloads
        event = Event(id=len(self._events), round=self._round, kind=kind, author=author,
                      payload=json.loads(line), t=self._clock())
        self._events.append(event)
        if self._fh is not None:
            self._fh.write(event.to_log_json() + "\n")
            self._fh.flush()
        return event.id

    # --- read --------------------------------------------------------------------------
    @property
    def next_id(self) -> int:
        return len(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def read(self, kind: str | Iterable[str] | None = None, since: int = 0,
             round: int | None = None) -> list[Event]:
        """Events with ``id >= since``, optionally filtered by kind(s) and round, in id order."""
        kinds = None if kind is None else ({kind} if isinstance(kind, str) else set(kind))
        return [e for e in self._events[max(since, 0):]
                if (kinds is None or e.kind in kinds) and (round is None or e.round == round)]

    # --- lifecycle ---------------------------------------------------------------------
    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> Blackboard:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_jsonl(path: str | Path) -> list[Event]:
    """Read a blackboard audit log back (arrays appear as shape summaries)."""
    with Path(path).open(encoding="utf-8") as fh:
        return [Event(**json.loads(line)) for line in fh if line.strip()]
