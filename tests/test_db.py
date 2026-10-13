"""Tests for the unified database (session index)."""

import os
import sqlite3

import pytest

from cross_session_mcp.db import SCHEMA_VERSION, CrossSessionDB
from cross_session_mcp.models import SessionSummary


class FakeReader:
    def __init__(self, name, sessions, fail=False):
        self.name, self._sessions, self._fail = name, sessions, fail

    def can_read(self):
        return True

    def list_sessions(self, workspace_filter=None, limit=50):
        if self._fail:
            raise RuntimeError("corrupt store")
        return self._sessions[:limit]


def summary(sid, harness="claude", ws="/p/app", updated="2026-09-01T10:00:00.000000+00:00"):
    return SessionSummary(session_id=sid, harness=harness, workspace_path=ws, title=sid, updated_at=updated)


@pytest.fixture
def db(tmp_path):
    return CrossSessionDB(db_path=tmp_path / "t.db")


def test_empty(db):
    assert db.query_sessions() == []
    assert db.seconds_since_last_sync() == float("inf")


def test_sync_isolates_failing_reader(db):
    readers = [FakeReader("agy", [], fail=True), FakeReader("claude", [summary("a"), summary("b")])]
    assert db.sync_from_readers(readers=readers) == 2
    assert {s.session_id for s in db.query_sessions()} == {"a", "b"}
    assert db.seconds_since_last_sync() < 5


def test_workspace_filter_prefix_and_escape(db):
    db.sync_from_readers(
        readers=[
            FakeReader(
                "claude",
                [
                    summary("a", ws="/p/app"),
                    summary("b", ws="/p/app/sub"),
                    summary("c", ws="/p/app2"),
                    summary("d", ws="/p/x_y"),
                    summary("e", ws="/p/xzy"),
                ],
            )
        ]
    )
    assert {s.session_id for s in db.query_sessions("/p/app")} == {"a", "b"}
    assert {s.session_id for s in db.query_sessions("/p/app/")} == {"a", "b"}
    assert {s.session_id for s in db.query_sessions("x_y")} == {"d"}  # '_' is not a wildcard
    assert {s.session_id for s in db.query_sessions("file:///p/app")} == {"a", "b"}


def test_ordering_across_harnesses(db):
    db.sync_from_readers(
        readers=[
            FakeReader("agy", [summary("old", "agy", updated="2026-09-01T09:00:00.000000+00:00")]),
            FakeReader("claude", [summary("new", "claude", updated="2026-09-01T11:00:00.000000+00:00")]),
        ]
    )
    assert [s.session_id for s in db.query_sessions()] == ["new", "old"]


def test_find_sessions_prefix(db):
    db.sync_from_readers(readers=[FakeReader("claude", [summary("4cbb572e-aaaa"), summary("4cbb9999")])])
    assert [s.session_id for s in db.find_sessions("4cbb572e")] == ["4cbb572e-aaaa"]
    assert len(db.find_sessions("4cbb")) == 2
    assert db.find_sessions("4cbb572e", harness="agy") == []


def test_migrates_v1_schema(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE handoffs (session_id TEXT, harness TEXT, workspace_path TEXT, markdown_handoff TEXT,"
        " model_used TEXT, generated_at TEXT, PRIMARY KEY (session_id, harness))"
    )
    conn.commit()
    conn.close()
    CrossSessionDB(db_path=path)
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "handoffs" not in names  # the old cache is gone: the handoff lives in session_digests
    if os.name != "nt":
        assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_sync_purges_excluded_workspaces(db, monkeypatch):
    monkeypatch.setenv("CROSS_SESSION_EXCLUDE_WORKSPACES", "")
    db.sync_from_readers(readers=[FakeReader("claude", [summary("a", ws="/noise/x"), summary("b")])])
    monkeypatch.setenv("CROSS_SESSION_EXCLUDE_WORKSPACES", "/noise")
    db.sync_from_readers(readers=[])
    assert [s.session_id for s in db.query_sessions()] == ["b"]
