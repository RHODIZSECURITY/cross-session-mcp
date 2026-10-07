"""Session reader for OpenCode (~/.local/share/opencode/opencode.db)."""

import json
from pathlib import Path
from typing import Any

from cross_session_mcp.models import SessionDetail, SessionSummary, SessionTurn
from cross_session_mcp.readers.base import TEXT_TURN_LIMIT, TOOL_TURN_LIMIT, BaseSessionReader
from cross_session_mcp.utils import env_path, is_excluded_workspace, logger, to_utc_iso, truncate, workspace_matches


def _loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


class OpenCodeSessionReader(BaseSessionReader):
    name = "opencode"

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or env_path(
            "CROSS_SESSION_OPENCODE_DB", Path.home() / ".local" / "share" / "opencode" / "opencode.db"
        )

    def can_read(self) -> bool:
        return self.db_path.exists()

    # ------------------------------------------------------------------ index
    def _query(self, where: str = "", params: tuple = (), limit: int = 50) -> list[dict[str, Any]]:
        try:
            conn = self._connect_ro(self.db_path)
            try:
                cols = self._columns(conn, "session")
                wanted = ["id", "directory", "title", "time_created", "time_updated", "time_archived", "parent_id"]
                select = ", ".join(c for c in wanted if c in cols)
                cur = conn.execute(
                    f"SELECT {select} FROM session {where} ORDER BY time_updated DESC LIMIT ?",
                    params + (limit,),
                )
                names = [d[0] for d in cur.description]
                rows = [dict(zip(names, row, strict=True)) for row in cur.fetchall()]
                self._attach_last_prompts(conn, rows)
                return rows
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("opencode: error reading %s: %s", self.db_path, exc)
            return []

    @staticmethod
    def _attach_last_prompts(conn, rows: list[dict[str, Any]]) -> None:
        """Adds the last user text to each session with a single query."""
        ids = [r["id"] for r in rows]
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        sql = f"""
            SELECT m.session_id, p.data
            FROM message m JOIN part p ON p.message_id = m.id
            WHERE m.session_id IN ({marks})
              AND json_extract(m.data, '$.role') = 'user'
              AND json_extract(p.data, '$.type') = 'text'
            ORDER BY m.time_created ASC, p.time_created ASC
        """
        last: dict[str, str] = {}
        try:
            for sid, pdata in conn.execute(sql, ids):
                text = _loads(pdata).get("text", "")
                if text.strip():
                    last[sid] = text
        except Exception as exc:  # json_extract is missing in very old SQLite
            logger.debug("opencode: no last_prompt (%s)", exc)
        for r in rows:
            r["last_prompt"] = last.get(r["id"], "")

    def _row_to_summary(self, row: dict[str, Any]) -> SessionSummary:
        sid = str(row.get("id"))
        return SessionSummary(
            session_id=sid,
            harness=self.name,
            workspace_path=row.get("directory") or "",
            title=row.get("title") or f"OpenCode session ({sid[:8]})",
            created_at=to_utc_iso(row.get("time_created")),
            updated_at=to_utc_iso(row.get("time_updated")),
            last_prompt=(row.get("last_prompt") or "")[:300],
            status="archived" if row.get("time_archived") else "active",
        )

    def list_sessions(
        self,
        workspace_filter: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        if not self.can_read():
            return []
        # Sessions with parent_id are subagents: left out of the main listing
        fetch = limit * 4 if workspace_filter else limit
        rows = self._query("WHERE parent_id IS NULL", limit=fetch)
        sessions = [self._row_to_summary(r) for r in rows]
        return [
            s for s in sessions
            if not is_excluded_workspace(s.workspace_path) and workspace_matches(workspace_filter, s.workspace_path)
        ][:limit]

    # ------------------------------------------------------------------ transcript
    def _part_to_text(self, part: dict[str, Any]) -> tuple[str, str]:
        """Returns (forced_role, text). Empty forced_role = the message role."""
        ptype = part.get("type")
        if ptype == "text" and not part.get("synthetic"):
            return "", part.get("text", "")
        if ptype == "tool":
            state = part.get("state") or {}
            call = self._tool_call_text(part.get("tool", "tool"), state.get("input"))
            output = state.get("output") or state.get("error") or ""
            if output:
                call += "\n" + truncate(str(output), TOOL_TURN_LIMIT)
            return "tool", call
        if ptype == "patch":
            files = part.get("files") or []
            return "tool", f"→ patch: {', '.join(map(str, files))[:400]}"
        return "", ""  # reasoning, step-start, step-finish, file...

    def get_session_detail(
        self,
        session_id: str,
        max_turns: int | None = 25,
    ) -> SessionDetail | None:
        if not self.can_read():
            return None
        rows = self._query("WHERE id = ?", (session_id,), limit=1)
        if not rows:
            return None
        summary = self._row_to_summary(rows[0])

        turns: list[SessionTurn] = []
        try:
            conn = self._connect_ro(self.db_path)
            try:
                cur = conn.execute(
                    """
                    SELECT m.id, m.data, m.time_created, p.data
                    FROM message m
                    LEFT JOIN part p ON m.id = p.message_id
                    WHERE m.session_id = ?
                    ORDER BY m.time_created ASC, p.time_created ASC
                    """,
                    (session_id,),
                )
                current_id = None
                role = "user"
                ts = None
                texts: list[str] = []

                def flush() -> None:
                    text = "\n".join(t for t in texts if t.strip()).strip()
                    if text:
                        turns.append(SessionTurn(role=role, content=truncate(text, TEXT_TURN_LIMIT), timestamp=ts))
                    texts.clear()

                for mid, m_data, m_time, p_data in cur:
                    if mid != current_id:
                        flush()
                        current_id = mid
                        role = _loads(m_data).get("role", "user")
                        ts = to_utc_iso(m_time)
                    forced_role, text = self._part_to_text(_loads(p_data))
                    if not text:
                        continue
                    if forced_role == "tool":
                        flush()  # keeps the text -> tool order
                        turns.append(SessionTurn(role="tool", content=text, timestamp=ts))
                    else:
                        texts.append(text)
                flush()
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("opencode: error reading session %s: %s", session_id, exc)

        summary.turn_count = sum(1 for t in turns if t.role in ("user", "assistant"))
        user_turns = [t for t in turns if t.role == "user"]
        if user_turns:
            summary.last_prompt = user_turns[-1].content[:300]
        return SessionDetail(summary=summary, turns=self._tail(turns, max_turns))
