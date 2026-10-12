"""Session reader for the Cursor agent/chat (Composer).

Location: `<config>/Cursor/User/globalStorage/state.vscdb`, where `<config>` is `~/.config` on Linux
and `~/Library/Application Support` on macOS. Table `cursorDiskKV`:
- `composerData:<composerId>` → metadata (name, createdAt, lastUpdatedAt) and the message
  order in `fullConversationHeadersOnly` (current format) or the full messages in
  `conversation` (legacy format).
- `bubbleId:<composerId>:<bubbleId>` → each message (type 1 = user, 2 = assistant),
  with `text` and, for a tool, `toolFormerData` (name, rawArgs, result).
Each conversation's workspace comes from `workspaceStorage/<hash>/state.vscdb`
(ItemTable `composer.composerData` → allComposers) + `workspace.json`.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

from cross_session_mcp.models import SessionDetail, SessionSummary, SessionTurn
from cross_session_mcp.readers.base import TEXT_TURN_LIMIT, TOOL_TURN_LIMIT, BaseSessionReader
from cross_session_mcp.utils import (
    config_home,
    env_path,
    file_uri_to_path,
    is_excluded_workspace,
    is_safe_session_id,
    logger,
    to_utc_iso,
    truncate,
    workspace_matches,
)

USER_BUBBLE = 1
ASSISTANT_BUBBLE = 2


def _loads(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        return json.loads(raw)
    except ValueError:
        return None


class CursorSessionReader(BaseSessionReader):
    name = "cursor"

    def __init__(self, user_dir: Path | None = None):
        self.user_dir = user_dir or env_path("CROSS_SESSION_CURSOR_DIR", config_home() / "Cursor" / "User")
        self.db_path = self.user_dir / "globalStorage" / "state.vscdb"

    def can_read(self) -> bool:
        return self.db_path.exists()

    # ------------------------------------------------------------------ workspaces
    def _workspace_map(self) -> dict[str, str]:
        """composerId -> workspace path, built from each workspaceStorage."""
        mapping: dict[str, str] = {}
        root = self.user_dir / "workspaceStorage"
        if not root.is_dir():
            return mapping
        for ws_dir in root.iterdir():
            db = ws_dir / "state.vscdb"
            meta = ws_dir / "workspace.json"
            if not db.exists() or not meta.exists():
                continue
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
                folder = file_uri_to_path(info.get("folder") or info.get("workspace") or "")
                conn = self._connect_ro(db)
                try:
                    row = conn.execute("SELECT value FROM ItemTable WHERE key = 'composer.composerData'").fetchone()
                finally:
                    conn.close()
            except (OSError, ValueError, sqlite3.Error):
                continue
            data = _loads(row[0]) if row else None
            for composer in (data or {}).get("allComposers") or []:
                if isinstance(composer, dict) and composer.get("composerId"):
                    mapping[composer["composerId"]] = folder
        return mapping

    # ------------------------------------------------------------------ datos
    def _composers(self, conn: sqlite3.Connection, composer_id: str | None = None) -> list[dict[str, Any]]:
        if composer_id:
            rows = conn.execute("SELECT value FROM cursorDiskKV WHERE key = ?", (f"composerData:{composer_id}",)).fetchall()
        else:
            rows = conn.execute("SELECT value FROM cursorDiskKV WHERE key LIKE 'composerData:%'").fetchall()
        return [d for d in (_loads(r[0]) for r in rows) if isinstance(d, dict) and d.get("composerId")]

    @staticmethod
    def _headers(composer: dict[str, Any]) -> list[dict[str, Any]]:
        headers = composer.get("fullConversationHeadersOnly")
        if headers:
            return [h for h in headers if isinstance(h, dict)]
        return [b for b in composer.get("conversation") or [] if isinstance(b, dict)]

    def _bubbles(self, conn: sqlite3.Connection, composer: dict[str, Any]) -> list[dict[str, Any]]:
        cid = composer["composerId"]
        inline = {b.get("bubbleId"): b for b in composer.get("conversation") or [] if isinstance(b, dict)}
        bubbles: list[dict[str, Any]] = []
        for header in self._headers(composer):
            bid = header.get("bubbleId")
            bubble = inline.get(bid)
            if bubble is None and bid:
                row = conn.execute("SELECT value FROM cursorDiskKV WHERE key = ?", (f"bubbleId:{cid}:{bid}",)).fetchone()
                bubble = _loads(row[0]) if row else None
            if isinstance(bubble, dict):
                bubble.setdefault("type", header.get("type"))
                bubbles.append(bubble)
        return bubbles

    def _bubble_to_turns(self, bubble: dict[str, Any]) -> list[SessionTurn]:
        ts = to_utc_iso(bubble.get("createdAt") or (bubble.get("timingInfo") or {}).get("clientStartTime"))
        text = (bubble.get("text") or "").strip()
        turns: list[SessionTurn] = []
        if bubble.get("type") == USER_BUBBLE:
            if text:
                turns.append(SessionTurn(role="user", content=truncate(text, TEXT_TURN_LIMIT), timestamp=ts))
            return turns
        if text:
            turns.append(SessionTurn(role="assistant", content=truncate(text, TEXT_TURN_LIMIT), timestamp=ts))
        tool = bubble.get("toolFormerData")
        if isinstance(tool, dict) and (tool.get("name") or tool.get("tool")):
            content = self._tool_call_text(str(tool.get("name") or tool.get("tool")), tool.get("rawArgs") or tool.get("params") or "")
            result = tool.get("result")
            if result:
                content += "\n" + truncate(result if isinstance(result, str) else json.dumps(result, ensure_ascii=False), TOOL_TURN_LIMIT)
            turns.append(SessionTurn(role="tool", content=content, timestamp=ts))
        return turns

    def _summary(self, composer: dict[str, Any], workspace: str, turns: list[SessionTurn] | None = None) -> SessionSummary:
        cid = composer["composerId"]
        headers = self._headers(composer)
        user_turns = [t for t in turns or [] if t.role == "user"]
        return SessionSummary(
            session_id=cid,
            harness=self.name,
            workspace_path=workspace,
            title=composer.get("name") or (truncate(user_turns[0].content.splitlines()[0], 80) if user_turns else "")
            or f"Cursor session ({cid[:8]})",
            created_at=to_utc_iso(composer.get("createdAt")),
            updated_at=to_utc_iso(composer.get("lastUpdatedAt") or composer.get("createdAt")),
            turn_count=len(headers) if turns is None else sum(1 for t in turns if t.role in ("user", "assistant")),
            last_prompt=user_turns[-1].content[:300] if user_turns else "",
            status="archived" if composer.get("isArchived") else "active",
        )

    # ------------------------------------------------------------------ API
    def list_sessions(
        self,
        workspace_filter: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        if not self.can_read():
            return []
        workspaces = self._workspace_map()
        try:
            conn = self._connect_ro(self.db_path)
            try:
                composers = [c for c in self._composers(conn) if self._headers(c)]
                composers.sort(key=lambda c: c.get("lastUpdatedAt") or c.get("createdAt") or 0, reverse=True)
                sessions: list[SessionSummary] = []
                for composer in composers:
                    workspace = workspaces.get(composer["composerId"], "")
                    if is_excluded_workspace(workspace) or not workspace_matches(workspace_filter, workspace):
                        continue
                    # Only user messages are read, for the title and the last prompt
                    user_bubbles = [b for b in self._bubbles(conn, composer) if b.get("type") == USER_BUBBLE]
                    user_turns = [t for b in user_bubbles for t in self._bubble_to_turns(b)]
                    summary = self._summary(composer, workspace, None)
                    if user_turns:
                        summary.last_prompt = user_turns[-1].content[:300]
                        if not composer.get("name"):
                            summary.title = truncate(user_turns[0].content.splitlines()[0], 80)
                    sessions.append(summary)
                    if len(sessions) >= limit:
                        break
                return sessions
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("cursor: error reading %s: %s", self.db_path, exc)
            return []

    def get_session_detail(
        self,
        session_id: str,
        max_turns: int | None = 25,
    ) -> SessionDetail | None:
        if not self.can_read() or not is_safe_session_id(session_id):
            return None
        try:
            conn = self._connect_ro(self.db_path)
            try:
                composers = self._composers(conn, session_id)
                if not composers:
                    return None
                composer = composers[0]
                turns = [t for b in self._bubbles(conn, composer) for t in self._bubble_to_turns(b)]
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("cursor: error reading session %s: %s", session_id, exc)
            return None
        workspace = self._workspace_map().get(session_id, "")
        return SessionDetail(summary=self._summary(composer, workspace, turns), turns=self._tail(turns, max_turns))
