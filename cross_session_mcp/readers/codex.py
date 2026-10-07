"""Session reader for the OpenAI Codex CLI.

The index lives in `~/.codex/state_5.sqlite` (table `threads`) and each thread points to its
JSONL transcript through `rollout_path` (usually `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`).
"""

import json
from pathlib import Path
from typing import Any

from cross_session_mcp.models import SessionDetail, SessionSummary, SessionTurn
from cross_session_mcp.readers.base import TEXT_TURN_LIMIT, TOOL_TURN_LIMIT, BaseSessionReader
from cross_session_mcp.utils import (
    env_path,
    is_excluded_workspace,
    is_safe_session_id,
    logger,
    to_utc_iso,
    truncate,
    workspace_matches,
)

# "user" messages Codex injects on its own that the user never typed
_INJECTED_PREFIXES = ("<environment_context>", "<user_instructions>", "# AGENTS.md", "<permissions", "<INSTRUCTIONS>")


class CodexSessionReader(BaseSessionReader):
    name = "codex"

    def __init__(self, codex_home: Path | None = None, db_path: Path | None = None):
        self.codex_home = codex_home or env_path("CROSS_SESSION_CODEX_HOME", Path.home() / ".codex")
        self.db_path = db_path or (self.codex_home / "state_5.sqlite")

    def can_read(self) -> bool:
        return self.db_path.exists()

    # ------------------------------------------------------------------ index
    def _query(self, where: str = "", params: tuple = (), limit: int = 50) -> list[dict[str, Any]]:
        try:
            conn = self._connect_ro(self.db_path)
            try:
                cols = self._columns(conn, "threads")
                wanted = [
                    "id", "title", "name", "cwd", "created_at", "updated_at", "created_at_ms",
                    "updated_at_ms", "preview", "first_user_message", "rollout_path", "archived",
                ]
                select = ", ".join(c for c in wanted if c in cols)
                order = "updated_at_ms" if "updated_at_ms" in cols else "updated_at"
                cur = conn.execute(
                    f"SELECT {select} FROM threads {where} ORDER BY {order} DESC LIMIT ?",
                    params + (limit,),
                )
                names = [d[0] for d in cur.description]
                return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("codex: error reading %s: %s", self.db_path, exc)
            return []

    def _row_to_summary(self, row: dict[str, Any]) -> SessionSummary:
        tid = str(row.get("id"))
        first = row.get("first_user_message") or ""
        title = row.get("name") or row.get("title") or truncate(first, 80) or f"Codex session ({tid[:8]})"
        return SessionSummary(
            session_id=tid,
            harness=self.name,
            workspace_path=row.get("cwd") or "",
            title=title,
            created_at=to_utc_iso(row.get("created_at_ms") or row.get("created_at")),
            updated_at=to_utc_iso(row.get("updated_at_ms") or row.get("updated_at")),
            last_prompt=(row.get("preview") or first)[:300],
            status="archived" if row.get("archived") else "active",
        )

    def list_sessions(
        self,
        workspace_filter: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        if not self.can_read():
            return []
        fetch = limit * 4 if workspace_filter else limit
        sessions = [self._row_to_summary(r) for r in self._query(limit=fetch)]
        return [
            s for s in sessions
            if not is_excluded_workspace(s.workspace_path) and workspace_matches(workspace_filter, s.workspace_path)
        ][:limit]

    # ------------------------------------------------------------------ transcript
    def _find_rollout(self, session_id: str, row: dict[str, Any] | None) -> Path | None:
        if row and row.get("rollout_path"):
            path = Path(row["rollout_path"]).expanduser()
            if path.exists():
                return path
        sessions_dir = self.codex_home / "sessions"
        if sessions_dir.is_dir():
            return next(sessions_dir.rglob(f"rollout-*{session_id}*.jsonl"), None)
        return None

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for item in content or []:
            if isinstance(item, dict) and item.get("type") in ("input_text", "output_text", "text"):
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _output_text(output: Any) -> str:
        if isinstance(output, str):
            try:
                decoded = json.loads(output)
                if isinstance(decoded, dict) and "output" in decoded:
                    return str(decoded["output"])
            except ValueError:
                pass
            return output
        if isinstance(output, dict):
            return str(output.get("output") or output.get("content") or output)
        return str(output or "")

    def _item_to_turns(self, item: dict[str, Any], ts: str | None) -> list[SessionTurn]:
        itype = item.get("type")
        if itype == "message":
            role = item.get("role")
            text = self._message_text(item.get("content")).strip()
            if role not in ("user", "assistant") or not text:
                return []
            if role == "user" and text.startswith(_INJECTED_PREFIXES):
                return []
            return [SessionTurn(role=role, content=truncate(text, TEXT_TURN_LIMIT), timestamp=ts)]
        if itype in ("function_call", "custom_tool_call"):
            args = item.get("arguments") if "arguments" in item else item.get("input")
            return [SessionTurn(role="tool", content=self._tool_call_text(item.get("name", "tool"), args), timestamp=ts)]
        if itype == "local_shell_call":
            action = item.get("action") or {}
            return [SessionTurn(role="tool", content=self._tool_call_text("shell", action.get("command", action)), timestamp=ts)]
        if itype in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
            return [SessionTurn(role="tool", content=truncate(self._output_text(item.get("output")), TOOL_TURN_LIMIT), timestamp=ts)]
        return []  # 'reasoning' and other internal events

    def _parse_rollout(self, path: Path) -> tuple[list[SessionTurn], dict[str, Any]]:
        turns: list[SessionTurn] = []
        meta: dict[str, Any] = {}
        for rec in self._iter_jsonl(path):
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            ts = to_utc_iso(rec.get("timestamp"))
            payload = rec.get("payload")
            if rtype == "session_meta" and isinstance(payload, dict):
                meta = payload
            elif rtype == "response_item" and isinstance(payload, dict):
                turns.extend(self._item_to_turns(payload, ts))
            elif rtype in ("message", "function_call", "function_call_output", "local_shell_call"):
                turns.extend(self._item_to_turns(rec, ts))  # legacy format without envelope
        return turns, meta

    def get_session_detail(
        self,
        session_id: str,
        max_turns: int | None = 25,
    ) -> SessionDetail | None:
        if not is_safe_session_id(session_id):
            return None
        rows = self._query("WHERE id = ?", (session_id,), limit=1) if self.can_read() else []
        row = rows[0] if rows else None
        rollout = self._find_rollout(session_id, row)
        if row is None and rollout is None:
            return None

        turns: list[SessionTurn] = []
        meta: dict[str, Any] = {}
        if rollout is not None:
            turns, meta = self._parse_rollout(rollout)

        if row is not None:
            summary = self._row_to_summary(row)
        else:
            summary = SessionSummary(
                session_id=session_id,
                harness=self.name,
                workspace_path=meta.get("cwd", ""),
                title=f"Codex session ({session_id[:8]})",
                created_at=to_utc_iso(meta.get("timestamp")),
            )
        if not turns and summary.last_prompt:
            turns.append(SessionTurn(role="user", content=summary.last_prompt))
        summary.turn_count = sum(1 for t in turns if t.role in ("user", "assistant"))
        return SessionDetail(summary=summary, turns=self._tail(turns, max_turns))
