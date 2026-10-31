"""Usage log: anonymous ids, user counting, no structure data, rate limiting."""

import sqlite3

import pytest

from fittest.usage_log import CALL_COLUMNS, CallRecord, TokenBucket, UsageLog, load_or_create_salt


@pytest.fixture
def ulog(tmp_path):
    lg = UsageLog(tmp_path / "usage.sqlite", salt=b"test-salt")
    yield lg
    lg.close()


def test_client_id_is_salted_hash_without_ip(tmp_path):
    a = UsageLog(tmp_path / "a.sqlite", salt=b"salt-a")
    b = UsageLog(tmp_path / "b.sqlite", salt=b"salt-b")
    ida = a.client_id("claude-code", "2.0", "203.0.113.7")
    assert ida == a.client_id("claude-code", "2.0", "203.0.113.7")  # stable
    assert ida != b.client_id("claude-code", "2.0", "203.0.113.7")  # depends on the secret salt
    assert ida != a.client_id("claude-code", "2.0", "203.0.113.8")  # depends on IP
    assert ida != a.client_id("cursor", "2.0", "203.0.113.7")       # depends on client info
    assert len(ida) == 16 and "203" not in ida
    a.record(CallRecord(tool="swarm_status", client_id=ida))
    raw = (tmp_path / "a.sqlite").read_bytes()
    assert b"203.0.113.7" not in raw and b"claude-code" not in raw


def test_salt_file_created_once(tmp_path):
    p = tmp_path / "sub" / "salt"
    s1 = load_or_create_salt(p)
    assert p.exists() and len(s1) == 64
    assert load_or_create_salt(p) == s1


def test_users_exclude_internal_failed_and_registered(ulog):
    ulog.record(CallRecord(tool="check_structures", client_id="u1", n_structures=2, n_atoms=6, n_trust=1, n_dft=1))
    ulog.record(CallRecord(tool="check_structures", client_id="u1"))
    ulog.record(CallRecord(tool="explain", client_id="u2"))
    ulog.record(CallRecord(tool="check_structures", client_id="u3", ok=False, error="input_error"))  # failed only
    ulog.record(CallRecord(tool="check_structures", client_id="t1", internal=True))  # flagged internal
    ulog.mark_internal("t2", "laptop test")
    ulog.record(CallRecord(tool="swarm_status", client_id="t2"))  # registered internal client
    st = ulog.stats()
    assert st["users"] == 2
    assert st["total_calls"] == 6 and st["successful_calls"] == 5
    assert st["internal_calls"] == 2 and st["external_calls"] == 4
    assert st["internal_client_ids"] == 2
    assert st["calls_per_tool"]["check_structures"] == {"calls": 4, "external_calls": 3}
    assert st["external_verdicts"] == {"trust": 1, "send_to_dft": 1}
    assert st["external_structures_checked"] == 2 and st["first_call_utc"] <= st["last_call_utc"]


def test_schema_has_only_counts(ulog):
    cols = [r[1] for r in ulog._db.execute("PRAGMA table_info(calls)").fetchall()]
    assert tuple(cols) == CALL_COLUMNS
    for forbidden in ("structure", "xyz", "positions", "coords", "ip", "name"):
        assert not any(forbidden == c or c.startswith(forbidden + "_") for c in cols)


def test_error_is_truncated(ulog):
    ulog.record(CallRecord(tool="x", client_id="c", ok=False, error="E" * 500))
    assert len(ulog.rows()[0]["error"]) == 64


def test_empty_stats(tmp_path):
    lg = UsageLog(tmp_path / "e.sqlite", salt=b"s")
    st = lg.stats()
    assert st["users"] == 0 and st["total_calls"] == 0 and st["first_call_utc"] is None


def test_wal_db_readable_by_other_connection(tmp_path, ulog):
    ulog.record(CallRecord(tool="usage_stats", client_id="c"))
    con = sqlite3.connect(ulog.db_path)
    assert con.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 1


def test_token_bucket():
    tb = TokenBucket(capacity=3, rate=0.0)
    assert all(tb.allow("k")[0] for _ in range(3))
    ok, wait = tb.allow("k")
    assert not ok and wait == float("inf")
    assert tb.allow("other")[0]  # per key
    tb2 = TokenBucket(capacity=1, rate=1000.0)
    assert tb2.allow("k")[0]
    import time

    time.sleep(0.05)  # Windows monotonic clock ticks ~16 ms
    assert tb2.allow("k")[0]
