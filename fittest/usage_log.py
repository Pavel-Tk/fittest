"""Usage log for the Fittest MCP server: one SQLite row per tool call, no structures.

What is stored per call (and nothing else):

* UTC time, tool name, transport (``http`` / ``stdio``)
* an anonymous client id: ``HMAC-SHA256(secret salt, client name | remote IP)``, truncated to
  16 hex chars (the client *version* is deliberately left out, so upgrades or a forged version
  string do not mint new ids). The salt is a server-side secret file (never in the repo), so the
  id cannot be reversed to an IP. Raw IPs are never written.
* whether the call is internal (our own tests), structure count, total atom count,
  verdict counts (trust / send_to_dft), per-structure failures, latency, ok/error and a short
  error *category* (never an exception message, so no user input can leak into the log).

"Users" (see :data:`USERS_DEFINITION`) = distinct non-internal client ids that got at least one
structure verdict from ``check_structures`` or ``explain``. Status/stats calls alone do not count.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc       TEXT    NOT NULL,
    tool         TEXT    NOT NULL,
    client_id    TEXT    NOT NULL,
    transport    TEXT    NOT NULL,
    internal     INTEGER NOT NULL DEFAULT 0,
    n_structures INTEGER NOT NULL DEFAULT 0,
    n_atoms      INTEGER NOT NULL DEFAULT 0,
    n_trust      INTEGER NOT NULL DEFAULT 0,
    n_dft        INTEGER NOT NULL DEFAULT 0,
    latency_ms   REAL    NOT NULL DEFAULT 0,
    ok           INTEGER NOT NULL,
    error        TEXT,
    n_failed     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS calls_client ON calls (client_id);
CREATE TABLE IF NOT EXISTS internal_clients (
    client_id TEXT PRIMARY KEY,
    note      TEXT,
    added_utc TEXT NOT NULL
);
"""

#: columns of ``calls`` — tests assert nothing else is ever stored.
CALL_COLUMNS = ("id", "ts_utc", "tool", "client_id", "transport", "internal", "n_structures",
                "n_atoms", "n_trust", "n_dft", "latency_ms", "ok", "error", "n_failed")
VERDICT_TOOLS = ("check_structures", "explain")
USERS_DEFINITION = ("distinct anonymous clients (salted hash of client name + IP) that received at least one "
                    "structure verdict from check_structures or explain, excluding internal test clients. "
                    "People sharing one public IP and client (same office, venue Wi-Fi, or claude.ai "
                    "connectors that egress from shared IPs) merge into one user; one person on several "
                    "networks or clients counts several times.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_or_create_salt(path: str | Path) -> bytes:
    """Read the secret salt, creating a random one (mode 600) on first use."""
    p = Path(path)
    if p.exists():
        data = p.read_bytes().strip()
        if data:
            return data
    p.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32).encode()
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(salt)
    return salt


@dataclass
class CallRecord:
    tool: str
    client_id: str
    transport: str = "stdio"
    internal: bool = False
    n_structures: int = 0
    n_atoms: int = 0
    n_trust: int = 0
    n_dft: int = 0
    n_failed: int = 0
    latency_ms: float = 0.0
    ok: bool = True
    error: str | None = None


class UsageLog:
    """Thread-safe SQLite usage log."""

    def __init__(self, db_path: str | Path, salt: bytes | None = None,
                 salt_path: str | Path | None = None):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        if salt is None:
            salt = load_or_create_salt(salt_path or Path(self.db_path).with_name("salt"))
        self._salt = salt
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
        if self.db_path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(calls)")}
        if "n_failed" not in cols:  # migrate logs written before per-structure failures existed
            self._db.execute("ALTER TABLE calls ADD COLUMN n_failed INTEGER NOT NULL DEFAULT 0")
        self._db.commit()

    # ------------------------------------------------------------------ identity
    def client_id(self, client_name: str | None, remote_ip: str | None) -> str:
        name = (client_name or "?").strip().lower()[:64]
        msg = f"{name}|{remote_ip or 'local'}".encode()
        return hmac.new(self._salt, msg, hashlib.sha256).hexdigest()[:16]

    def mark_internal(self, client_id: str, note: str = "") -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO internal_clients VALUES (?, ?, ?)",
                             (client_id, note[:200], utc_now()))
            self._db.commit()

    def is_internal_client(self, client_id: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM internal_clients WHERE client_id = ?",
                                   (client_id,)).fetchone()
        return row is not None

    # ------------------------------------------------------------------ writing
    def record(self, rec: CallRecord) -> None:
        err = None if rec.error is None else str(rec.error)[:64]
        with self._lock:
            self._db.execute(
                "INSERT INTO calls (ts_utc, tool, client_id, transport, internal, n_structures, n_atoms,"
                " n_trust, n_dft, n_failed, latency_ms, ok, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (utc_now(), rec.tool[:64], rec.client_id[:64], rec.transport[:16], int(bool(rec.internal)),
                 int(rec.n_structures), int(rec.n_atoms), int(rec.n_trust), int(rec.n_dft),
                 int(rec.n_failed), round(float(rec.latency_ms), 2), int(bool(rec.ok)), err))
            self._db.commit()

    # ------------------------------------------------------------------ reading
    def _q(self, sql: str, args: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def stats(self) -> dict:
        ext = ("internal = 0 AND client_id NOT IN (SELECT client_id FROM internal_clients)")
        tools = ", ".join(f"'{t}'" for t in VERDICT_TOOLS)
        users = self._q(f"SELECT COUNT(DISTINCT client_id) FROM calls WHERE ok = 1 AND {ext}"
                        f" AND tool IN ({tools}) AND n_trust + n_dft >= 1")[0][0]
        total, ok_calls = self._q("SELECT COUNT(*), COALESCE(SUM(ok), 0) FROM calls")[0]
        ext_calls = self._q(f"SELECT COUNT(*) FROM calls WHERE {ext}")[0][0]
        internal_ids = self._q(f"SELECT COUNT(DISTINCT client_id) FROM calls WHERE NOT ({ext})")[0][0]
        per_tool = {t: {"calls": n, "external_calls": e}
                    for t, n, e in self._q(f"SELECT tool, COUNT(*), SUM(CASE WHEN {ext} THEN 1 ELSE 0 END)"
                                           " FROM calls GROUP BY tool ORDER BY tool")}
        first, last = self._q("SELECT MIN(ts_utc), MAX(ts_utc) FROM calls")[0]
        n_struct, n_atoms, n_trust, n_dft = self._q(
            f"SELECT COALESCE(SUM(n_structures),0), COALESCE(SUM(n_atoms),0), COALESCE(SUM(n_trust),0),"
            f" COALESCE(SUM(n_dft),0) FROM calls WHERE ok = 1 AND {ext}")[0]
        return {
            "users": int(users),
            "users_definition": USERS_DEFINITION,
            "total_calls": int(total), "successful_calls": int(ok_calls),
            "external_calls": int(ext_calls), "internal_calls": int(total - ext_calls),
            "internal_client_ids": int(internal_ids),
            "calls_per_tool": per_tool,
            "external_structures_checked": int(n_struct), "external_atoms_checked": int(n_atoms),
            "external_verdicts": {"trust": int(n_trust), "send_to_dft": int(n_dft)},
            "first_call_utc": first, "last_call_utc": last,
            "generated_utc": utc_now(),
        }

    def rows(self) -> list[dict]:
        """All call rows (for tests and audits)."""
        with self._lock:
            cur = self._db.execute("SELECT * FROM calls ORDER BY id")
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._db.close()


class TokenBucket:
    """Per-key token bucket: ``capacity`` burst, ``rate`` tokens/second refill."""

    def __init__(self, capacity: float = 20.0, rate: float = 0.5, max_keys: int = 50_000):
        self.capacity, self.rate, self.max_keys = float(capacity), float(rate), max_keys
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        """Take ``cost`` tokens; returns (allowed, seconds until enough tokens)."""
        now = time.monotonic()
        with self._lock:
            if len(self._state) > self.max_keys:
                self._state.clear()
            tokens, t = self._state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - t) * self.rate)
            if tokens >= cost:
                self._state[key] = (tokens - cost, now)
                return True, 0.0
            self._state[key] = (tokens, now)
            return False, (cost - tokens) / self.rate if self.rate > 0 else float("inf")
