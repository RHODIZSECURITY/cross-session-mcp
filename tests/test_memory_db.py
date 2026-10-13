"""Tests for the persistent digest and memory tables (migration, CRUD and optimistic concurrency)."""

import os
import sqlite3
import stat

import pytest

from cross_session_mcp.db import SCHEMA_VERSION, ConcurrentUpdateError, CrossSessionDB
from cross_session_mcp.models import SessionSummary


def entry(eid="D1", **kw):
    base = dict(
        id=eid,
        kind=eid[0],
        text="text " + eid,
        why="",
        scope=None,
        evidence=["claude:s1"],
        status="active",
        pinned=False,
        text_hash="h",
        first_seen="2026-09-01",
        last_seen="2026-09-01",
        updated_at="2026-09-01",
    )
    base.update(kw)
    return base


def save(db, base, ws="/p/app", entries=None, md="# M"):
    return db.save_memory_version(
        ws,
        base,
        md,
        [{"key": "claude:s1", "harness": "claude", "session_id": "s1"}],
        ["state"],
        "gemini",
        entries if entries is not None else [entry()],
    )


def make_v3_db(path):
    """DB with the real v3 schema: session_digests without format_version and the `handoffs` cache."""
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE cross_sessions (session_id TEXT NOT NULL, harness TEXT NOT NULL, workspace_path TEXT, title TEXT,
            created_at TEXT, updated_at TEXT, turn_count INTEGER DEFAULT 0, last_prompt TEXT,
            status TEXT DEFAULT 'active', PRIMARY KEY (session_id, harness));
        CREATE TABLE handoffs (session_id TEXT NOT NULL, harness TEXT NOT NULL, focus_area TEXT NOT NULL DEFAULT '',
            workspace_path TEXT, markdown_handoff TEXT NOT NULL, model_used TEXT NOT NULL, generated_at TEXT NOT NULL,
            source_updated_at TEXT, PRIMARY KEY (session_id, harness, focus_area));
        CREATE TABLE session_digests (session_id TEXT NOT NULL, harness TEXT NOT NULL, workspace_path TEXT,
            source_updated_at TEXT, relevant INTEGER NOT NULL DEFAULT 1, digest_json TEXT NOT NULL,
            handoff_md TEXT NOT NULL, model_used TEXT NOT NULL, generated_at TEXT NOT NULL,
            PRIMARY KEY (session_id, harness));
        INSERT INTO cross_sessions (session_id, harness, workspace_path) VALUES ('old', 'claude', '/x');
        INSERT INTO handoffs VALUES ('old', 'claude', '', '/x', 'md', 'm', '2026', '2026');
        INSERT INTO session_digests VALUES ('s1', 'claude', '/p/app', '2026', 1, '{}', '### 1. Request', 'm', '2026');
        PRAGMA user_version = 3;
    """)
    conn.commit()
    conn.close()


def test_migration_v3_to_v4_invalidates_old_digests_and_keeps_memory(tmp_path):
    path = tmp_path / "old.db"
    make_v3_db(path)
    CrossSessionDB(path)  # creates the memory tables missing from the simulated DB
    save(CrossSessionDB(path), 0)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()

    db = CrossSessionDB(path)
    assert db.schema_version() == SCHEMA_VERSION == 4
    assert db.get_digest("s1", "claude") is None  # legacy format invalidated
    assert db.digest_format_counts() == {"current": 0, "legacy": 0}
    assert db.get_memory("/p/app")["version"] == 1 and len(db.get_entries("/p/app")) == 1  # memory intact
    assert db.query_sessions() == []  # index rebuilt
    names = {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "handoffs" not in names and "maintenance_locks" in names


def test_old_format_digest_is_ignored_and_new_one_is_saved_as_current(tmp_path):
    db = CrossSessionDB(tmp_path / "t.db")
    conn = sqlite3.connect(db.db_path)
    conn.execute(
        "INSERT INTO session_digests (session_id, harness, workspace_path, source_updated_at, relevant, "
        "digest_json, handoff_md, model_used, generated_at, format_version) "
        "VALUES ('s1','claude','/p','2026',1,'{}','old','m','2026',1)"
    )
    conn.commit()
    conn.close()
    assert db.get_digest("s1", "claude") is None
    assert db.digest_format_counts() == {"current": 0, "legacy": 1}
    db.save_digest("s1", "claude", "/p", "2026", {"relevant": True}, "### new", "m")
    assert db.get_digest("s1", "claude")["handoff_md"] == "### new"
    assert db.digest_format_counts() == {"current": 1, "legacy": 0}


def test_maintenance_lock_is_exclusive_expires_and_is_released_only_by_owner(tmp_path):
    db = CrossSessionDB(tmp_path / "t.db")
    assert db.acquire_maintenance_lock("/p", "a") is True
    assert db.acquire_maintenance_lock("/p", "b") is False  # another instance
    assert db.acquire_maintenance_lock("/q", "b") is True  # another workspace
    db.release_maintenance_lock("/p", "b")  # not the owner: does not release
    assert db.acquire_maintenance_lock("/p", "b") is False
    assert {lk["workspace_path"] for lk in db.active_maintenance_locks()} == {"/p", "/q"}
    assert db.acquire_maintenance_lock("/p", "b", ttl=0) is True  # expired lock: reclaimed
    db.release_maintenance_lock("/p", "b")
    assert db.acquire_maintenance_lock("/p", "c") is True


def test_tables_exist_and_file_is_private(tmp_path):
    db = CrossSessionDB(tmp_path / "t.db")
    names = {r[0] for r in sqlite3.connect(db.db_path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"session_digests", "project_memory", "memory_entries", "memory_versions", "maintenance_locks"} <= names
    if os.name != "nt":
        assert stat.S_IMODE(db.db_path.stat().st_mode) == 0o600


def test_digest_cache_valid_and_stale(tmp_path):
    db = CrossSessionDB(tmp_path / "t.db")
    db.save_digest(
        "s1", "claude", "/p", "2026-09-01T10:00:00+00:00", {"relevant": False, "summary_line": "a"}, "# H", "m"
    )
    assert db.get_digest("s1", "claude", "2026-09-01T10:00:00+00:00")["handoff_md"] == "# H"  # current
    assert db.get_digest("s1", "claude", "2026-09-01T09:00:00+00:00") is not None  # session not newer
    assert db.get_digest("s1", "claude", "2026-09-02T00:00:00+00:00") is None  # stale
    assert db.get_digest("s1", "codex") is None
    db.save_digest(
        "s1", "claude", "/p", "2026-09-02T00:00:00+00:00", {"relevant": True, "summary_line": "b"}, "# H2", "m"
    )  # replaces
    assert db.get_digest("s1", "claude", "2026-09-02T00:00:00+00:00")["digest"]["summary_line"] == "b"


def test_window_sessions_exact_workspace(tmp_path):
    db = CrossSessionDB(tmp_path / "t.db")

    class R:
        name = "claude"

        def can_read(self):
            return True

        def list_sessions(self, limit=50, workspace_filter=None):
            return [
                SessionSummary("a", "claude", "/p/app", "t", updated_at="2026-09-01T10:00:00+00:00"),
                SessionSummary("b", "claude", "/p/app/sub", "t", updated_at="2026-09-02T10:00:00+00:00"),
            ]

    db.sync_from_readers(readers=[R()])
    assert [s.session_id for s in db.window_sessions("/p/app", 8)] == ["a"]


def test_optimistic_concurrency_second_write_fails(tmp_path):
    db = CrossSessionDB(tmp_path / "t.db")
    assert save(db, 0) == 1
    assert save(db, 1) == 2
    with pytest.raises(ConcurrentUpdateError):
        save(db, 1)  # same base version as before
    with pytest.raises(ConcurrentUpdateError):
        save(db, 0)
    assert db.get_memory("/p/app")["version"] == 2


def test_versions_retention_and_entries_replaced(tmp_path):
    db = CrossSessionDB(tmp_path / "t.db")
    for i in range(25):
        save(db, i, entries=[entry("D1", status="active" if i % 2 else "archived")])
    versions = db.list_memory_versions("/p/app")
    assert len(versions) == 20 and versions[0]["version"] == 25 and versions[-1]["version"] == 6
    assert db.count_memory_projects() == 1
    assert len(db.get_entries("/p/app")) == 1
