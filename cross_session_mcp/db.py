"""Unified SQLite database: session index and memory bank.

Several harnesses may launch their own server instance on the same file at once,
hence WAL, busy_timeout and one connection per operation.
"""

import json
import os
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cross_session_mcp.models import SessionSummary
from cross_session_mcp.readers.base import BaseSessionReader
from cross_session_mcp.utils import (
    is_absolute_workspace,
    is_excluded_workspace,
    logger,
    normalize_workspace_filter,
    now_utc_iso,
)

SCHEMA_VERSION = 4
# Digest format version (8-section Extended Handoff = 2). Older digests are invalidated.
DIGEST_FORMAT_VERSION = 2
MAINTENANCE_LOCK_TTL = 600  # seconds: an older lock is considered abandoned
MEMORY_VERSIONS_RETAINED = 20


class ConcurrentUpdateError(Exception):
    """The base memory version changed during consolidation (failed optimistic write)."""


_DROP_REBUILDABLE = """
    DROP TABLE IF EXISTS handoffs;
    DROP TABLE IF EXISTS cross_sessions;
"""

_REBUILDABLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS cross_sessions (
        session_id TEXT NOT NULL,
        harness TEXT NOT NULL,
        workspace_path TEXT,
        title TEXT,
        created_at TEXT,
        updated_at TEXT,
        turn_count INTEGER DEFAULT 0,
        last_prompt TEXT,
        status TEXT DEFAULT 'active',
        PRIMARY KEY (session_id, harness)
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_updated ON cross_sessions(updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON cross_sessions(workspace_path);

    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );
"""

_PERSISTENT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS session_digests (
        session_id TEXT NOT NULL,
        harness TEXT NOT NULL,
        workspace_path TEXT,
        source_updated_at TEXT,
        relevant INTEGER NOT NULL DEFAULT 1,
        digest_json TEXT NOT NULL,
        handoff_md TEXT NOT NULL,
        model_used TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        format_version INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (session_id, harness)
    );

    CREATE TABLE IF NOT EXISTS maintenance_locks (
        workspace_path TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        acquired_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS project_memory (
        workspace_path TEXT PRIMARY KEY,
        version INTEGER NOT NULL,
        rendered_md TEXT NOT NULL,
        window_json TEXT NOT NULL,
        state_json TEXT NOT NULL DEFAULT '[]',
        model_used TEXT,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS memory_entries (
        workspace_path TEXT NOT NULL,
        entry_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        text TEXT NOT NULL,
        why TEXT,
        scope TEXT,
        evidence_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'active',
        pinned INTEGER NOT NULL DEFAULT 0,
        text_hash TEXT,
        first_seen TEXT,
        last_seen TEXT,
        updated_at TEXT,
        PRIMARY KEY (workspace_path, entry_id)
    );

    CREATE TABLE IF NOT EXISTS memory_versions (
        workspace_path TEXT NOT NULL,
        version INTEGER NOT NULL,
        rendered_md TEXT NOT NULL,
        window_json TEXT NOT NULL,
        model_used TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (workspace_path, version)
    );
"""


def get_default_db_path() -> Path:
    """Returns the database path (CROSS_SESSION_DB or ~/.ai/cross_sessions.db)."""
    env_path = os.environ.get("CROSS_SESSION_DB")
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".ai" / "cross_sessions.db"


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class CrossSessionDB:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else get_default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ connection
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version < SCHEMA_VERSION:
                # Only the index is rebuildable (plus the old `handoffs` cache, which is dropped). The persistent tables
                # (digests and memory) are NEVER dropped: they only allow CREATE IF NOT EXISTS / ADD COLUMN.
                conn.executescript(_DROP_REBUILDABLE)
            conn.executescript(_REBUILDABLE_SCHEMA)
            conn.executescript(_PERSISTENT_SCHEMA)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(session_digests)")}
            if "format_version" not in columns:  # DB v3: the column is added, the table is never recreated
                conn.execute("ALTER TABLE session_digests ADD COLUMN format_version INTEGER NOT NULL DEFAULT 1")
            # Digests in older formats (10-section compact handoff): invalidated
            conn.execute("DELETE FROM session_digests WHERE format_version < ?", (DIGEST_FORMAT_VERSION,))
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        finally:
            conn.close()
        try:
            os.chmod(self.db_path, 0o600)  # the cache holds summaries of private conversations
        except OSError:
            pass

    # ------------------------------------------------------------------ sync
    def sync_from_readers(
        self,
        limit_per_harness: int = 50,
        readers: Iterable[BaseSessionReader] | None = None,
    ) -> int:
        """Indexes recent sessions from every harness. A broken reader does not block the rest."""
        if readers is None:
            from cross_session_mcp.readers import get_all_readers

            readers = get_all_readers()

        collected: list[SessionSummary] = []
        for reader in readers:
            try:
                if reader.can_read():
                    collected.extend(reader.list_sessions(limit=limit_per_harness))
            except Exception as exc:
                logger.warning("Failed to sync '%s': %s", reader.name, exc)

        conn = self._connect()
        try:
            conn.executemany(
                """
                INSERT INTO cross_sessions (
                    session_id, harness, workspace_path, title,
                    created_at, updated_at, turn_count, last_prompt, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, harness) DO UPDATE SET
                    workspace_path=excluded.workspace_path,
                    title=excluded.title,
                    created_at=COALESCE(excluded.created_at, cross_sessions.created_at),
                    updated_at=excluded.updated_at,
                    turn_count=excluded.turn_count,
                    last_prompt=excluded.last_prompt,
                    status=excluded.status
                """,
                [
                    (
                        s.session_id,
                        s.harness,
                        s.workspace_path,
                        s.title,
                        s.created_at,
                        s.updated_at,
                        s.turn_count,
                        s.last_prompt,
                        s.status,
                    )
                    for s in collected
                ],
            )
            # Purge sessions of excluded workspaces left over from earlier syncs
            for (ws,) in conn.execute("SELECT DISTINCT workspace_path FROM cross_sessions").fetchall():
                if ws and is_excluded_workspace(ws):
                    conn.execute("DELETE FROM cross_sessions WHERE workspace_path = ?", (ws,))
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('last_sync', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(time.time()),),
            )
            conn.commit()
        finally:
            conn.close()
        return len(collected)

    def seconds_since_last_sync(self) -> float:
        conn = self._connect()
        try:
            row = conn.execute("SELECT value FROM meta WHERE key = 'last_sync'").fetchone()
        finally:
            conn.close()
        return time.time() - float(row[0]) if row else float("inf")

    # ------------------------------------------------------------------ consultas
    @staticmethod
    def _row_to_summary(r: Any) -> SessionSummary:
        return SessionSummary(
            session_id=r[0],
            harness=r[1],
            workspace_path=r[2] or "",
            title=r[3] or "",
            created_at=r[4],
            updated_at=r[5],
            turn_count=r[6] or 0,
            last_prompt=r[7] or "",
            status=r[8] or "active",
        )

    _SELECT = """
        SELECT session_id, harness, workspace_path, title, created_at, updated_at,
               turn_count, last_prompt, status
        FROM cross_sessions
    """

    def query_sessions(
        self,
        workspace_filter: str | None = None,
        harness_filter: str | None = None,
        limit: int = 25,
    ) -> list[SessionSummary]:
        """Queries sessions. An absolute path filters by that directory and its subdirectories;
        relative text filters by substring."""
        conditions: list[str] = []
        params: list[Any] = []

        ws = normalize_workspace_filter(workspace_filter)
        if ws:
            if is_absolute_workspace(ws):
                ws_fwd = ws.replace("\\", "/").rstrip("/")
                ws_back = ws.replace("/", "\\").rstrip("\\")
                conditions.append(
                    "(workspace_path = ? OR workspace_path = ? OR workspace_path LIKE ? ESCAPE '\\' OR workspace_path LIKE ? ESCAPE '\\')"
                )
                params.extend([ws_fwd, ws_back, _like_escape(ws_fwd) + "/%", _like_escape(ws_back) + "\\%"])
            else:
                conditions.append("workspace_path LIKE ? ESCAPE '\\'")
                params.append(f"%{_like_escape(ws)}%")

        if harness_filter:
            conditions.append("harness = ?")
            params.append(harness_filter)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(max(1, int(limit)))

        conn = self._connect()
        try:
            rows = conn.execute(f"{self._SELECT} {where} ORDER BY updated_at DESC LIMIT ?", params).fetchall()
        finally:
            conn.close()
        return [self._row_to_summary(r) for r in rows]

    def find_sessions(self, session_id: str, harness: str | None = None) -> list[SessionSummary]:
        """Looks up by exact ID or, failing that, by prefix (e.g. '4cbb572e')."""
        conn = self._connect()
        try:
            extra, extra_params = ("AND harness = ?", [harness]) if harness else ("", [])
            rows = conn.execute(f"{self._SELECT} WHERE session_id = ? {extra}", [session_id] + extra_params).fetchall()
            if not rows:
                rows = conn.execute(
                    f"{self._SELECT} WHERE session_id LIKE ? ESCAPE '\\' {extra} ORDER BY updated_at DESC LIMIT 10",
                    [_like_escape(session_id) + "%"] + extra_params,
                ).fetchall()
        finally:
            conn.close()
        return [self._row_to_summary(r) for r in rows]

    # ------------------------------------------------------------------ digests
    def get_digest(self, session_id: str, harness: str, source_updated_at: str | None = None) -> dict[str, Any] | None:
        """Saved digest in the current format, or None if missing or the session moved on since it was
        generated. Without `source_updated_at` it returns the saved one even if stale."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT workspace_path, source_updated_at, relevant, digest_json, handoff_md, model_used, generated_at "
                "FROM session_digests WHERE session_id = ? AND harness = ? AND format_version >= ?",
                (session_id, harness, DIGEST_FORMAT_VERSION),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        cached_source = row[1]
        if source_updated_at and (not cached_source or source_updated_at > cached_source):
            return None
        return {
            "session_id": session_id,
            "harness": harness,
            "workspace_path": row[0] or "",
            "source_updated_at": cached_source,
            "relevant": bool(row[2]),
            "digest": json.loads(row[3]),
            "handoff_md": row[4],
            "model_used": row[5],
            "generated_at": row[6],
        }

    def save_digest(
        self,
        session_id: str,
        harness: str,
        workspace_path: str,
        source_updated_at: str | None,
        digest: dict[str, Any],
        handoff_md: str,
        model_used: str,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO session_digests (session_id, harness, workspace_path, source_updated_at,
                                             relevant, digest_json, handoff_md, model_used, generated_at,
                                             format_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, harness) DO UPDATE SET
                    workspace_path=excluded.workspace_path,
                    source_updated_at=excluded.source_updated_at,
                    relevant=excluded.relevant,
                    digest_json=excluded.digest_json,
                    handoff_md=excluded.handoff_md,
                    model_used=excluded.model_used,
                    generated_at=excluded.generated_at,
                    format_version=excluded.format_version
                """,
                (
                    session_id,
                    harness,
                    workspace_path,
                    source_updated_at,
                    1 if digest.get("relevant") else 0,
                    json.dumps(digest, ensure_ascii=False),
                    handoff_md,
                    model_used,
                    now_utc_iso(),
                    DIGEST_FORMAT_VERSION,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_digests(self) -> int:
        """Deletes saved Extended Handoffs (regenerated on demand; consolidated memory is kept)."""
        conn = self._connect()
        try:
            count = conn.execute("DELETE FROM session_digests").rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def digest_format_counts(self) -> dict[str, int]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT format_version >= ?, COUNT(*) FROM session_digests GROUP BY 1", (DIGEST_FORMAT_VERSION,)
            ).fetchall()
        finally:
            conn.close()
        counts = {"current": 0, "legacy": 0}
        for current, n in rows:
            counts["current" if current else "legacy"] += n
        return counts

    def search_digests(self, workspace_path: str | None = None) -> list[dict[str, Any]]:
        """Current-format digests (of one exact workspace or all of them) for topic search."""
        sql = (
            "SELECT session_id, harness, workspace_path, source_updated_at, digest_json, handoff_md "
            "FROM session_digests WHERE format_version >= ?"
        )
        params: list[Any] = [DIGEST_FORMAT_VERSION]
        if workspace_path:
            sql += " AND workspace_path = ?"
            params.append(workspace_path)
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [
            {
                "session_id": r[0],
                "harness": r[1],
                "workspace_path": r[2] or "",
                "updated_at": r[3] or "",
                "digest": json.loads(r[4]),
                "handoff_md": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ maintenance lock
    def acquire_maintenance_lock(self, workspace_path: str, owner: str, ttl: float = MAINTENANCE_LOCK_TTL) -> bool:
        """Cross-process lock per workspace. A lock older than `ttl` seconds can be reclaimed."""
        now = time.time()
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT INTO maintenance_locks (workspace_path, owner, acquired_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(workspace_path) DO UPDATE SET owner = excluded.owner, acquired_at = excluded.acquired_at "
                    "WHERE maintenance_locks.acquired_at < ?",
                    (workspace_path, owner, now, now - ttl),
                )
                acquired = conn.execute("SELECT changes()").fetchone()[0] > 0
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return acquired

    def release_maintenance_lock(self, workspace_path: str, owner: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM maintenance_locks WHERE workspace_path = ? AND owner = ?", (workspace_path, owner)
            )
            conn.commit()
        finally:
            conn.close()

    def active_maintenance_locks(self, ttl: float = MAINTENANCE_LOCK_TTL) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT workspace_path, owner, acquired_at FROM maintenance_locks WHERE acquired_at >= ?",
                (time.time() - ttl,),
            ).fetchall()
        finally:
            conn.close()
        return [{"workspace_path": r[0], "owner": r[1], "acquired_at": r[2]} for r in rows]

    # ------------------------------------------------------------------ project memory
    def window_sessions(self, workspace_path: str, limit: int) -> list[SessionSummary]:
        """Last `limit` sessions whose workspace_path is EXACTLY `workspace_path` (no subdirectories)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                f"{self._SELECT} WHERE workspace_path = ? ORDER BY updated_at DESC LIMIT ?",
                (workspace_path, max(1, int(limit))),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_summary(r) for r in rows]

    def get_memory(self, workspace_path: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT version, rendered_md, window_json, state_json, model_used, updated_at "
                "FROM project_memory WHERE workspace_path = ?",
                (workspace_path,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {
            "workspace_path": workspace_path,
            "version": row[0],
            "rendered_md": row[1],
            "window": json.loads(row[2]),
            "state": json.loads(row[3] or "[]"),
            "model_used": row[4] or "",
            "updated_at": row[5],
        }

    def get_entries(self, workspace_path: str | None) -> list[dict[str, Any]]:
        """Entries of one exact workspace, or of all with `workspace_path=None` (they include 'workspace_path')."""
        sql = (
            "SELECT entry_id, kind, text, why, scope, evidence_json, status, pinned, text_hash, "
            "first_seen, last_seen, updated_at, workspace_path FROM memory_entries"
        )
        params: list[Any] = []
        if workspace_path is not None:
            sql += " WHERE workspace_path = ?"
            params.append(workspace_path)
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        entries = []
        for r in rows:
            e = {
                "id": r[0],
                "kind": r[1],
                "text": r[2],
                "why": r[3] or "",
                "scope": r[4],
                "evidence": json.loads(r[5] or "[]"),
                "status": r[6],
                "pinned": bool(r[7]),
                "text_hash": r[8],
                "first_seen": r[9],
                "last_seen": r[10],
                "updated_at": r[11],
            }
            if workspace_path is None:
                e["workspace_path"] = r[12]
            entries.append(e)
        return entries

    def save_memory_version(
        self,
        workspace_path: str,
        base_version: int,
        rendered_md: str,
        window: list[dict[str, Any]],
        state: list[str],
        model_used: str,
        entries: list[dict[str, Any]],
        note: str | None = None,
    ) -> int:
        """Saves a new version with optimistic control: fails with ConcurrentUpdateError if the
        current version is no longer `base_version` (0 = no memory yet)."""
        now = now_utc_iso()
        new_version = base_version + 1
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT version FROM project_memory WHERE workspace_path = ?", (workspace_path,)
                ).fetchone()
                window_json = json.dumps(window, ensure_ascii=False)
                if row is None:
                    if base_version != 0:
                        raise ConcurrentUpdateError("The base memory no longer exists.")
                    conn.execute(
                        "INSERT INTO project_memory (workspace_path, version, rendered_md, window_json, "
                        "state_json, model_used, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            workspace_path,
                            new_version,
                            rendered_md,
                            window_json,
                            json.dumps(state, ensure_ascii=False),
                            model_used,
                            now,
                        ),
                    )
                else:
                    cur = conn.execute(
                        "UPDATE project_memory SET version = ?, rendered_md = ?, window_json = ?, state_json = ?, "
                        "model_used = ?, updated_at = ? WHERE workspace_path = ? AND version = ?",
                        (
                            new_version,
                            rendered_md,
                            window_json,
                            json.dumps(state, ensure_ascii=False),
                            model_used,
                            now,
                            workspace_path,
                            base_version,
                        ),
                    )
                    if cur.rowcount == 0:
                        raise ConcurrentUpdateError(
                            f"Memory went from version {base_version} to {row[0]} during the update."
                        )
                conn.execute("DELETE FROM memory_entries WHERE workspace_path = ?", (workspace_path,))
                conn.executemany(
                    "INSERT INTO memory_entries (workspace_path, entry_id, kind, text, why, scope, evidence_json, "
                    "status, pinned, text_hash, first_seen, last_seen, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            workspace_path,
                            e["id"],
                            e["kind"],
                            e["text"],
                            e.get("why") or "",
                            e.get("scope"),
                            json.dumps(e.get("evidence") or []),
                            e["status"],
                            1 if e.get("pinned") else 0,
                            e.get("text_hash"),
                            e.get("first_seen"),
                            e.get("last_seen"),
                            e.get("updated_at"),
                        )
                        for e in entries
                    ],
                )
                conn.execute(
                    "INSERT INTO memory_versions (workspace_path, version, rendered_md, window_json, model_used, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (workspace_path, new_version, rendered_md, window_json, note or model_used, now),
                )
                conn.execute(
                    "DELETE FROM memory_versions WHERE workspace_path = ? AND version <= ?",
                    (workspace_path, new_version - MEMORY_VERSIONS_RETAINED),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return new_version

    def list_memory_versions(self, workspace_path: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT version, model_used, created_at, window_json FROM memory_versions "
                "WHERE workspace_path = ? ORDER BY version DESC",
                (workspace_path,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {"version": r[0], "model_used": r[1] or "", "created_at": r[2], "window": json.loads(r[3])} for r in rows
        ]

    def count_memory_projects(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM project_memory").fetchone()[0]
        finally:
            conn.close()

    def schema_version(self) -> int:
        conn = self._connect()
        try:
            return conn.execute("PRAGMA user_version").fetchone()[0]
        finally:
            conn.close()
