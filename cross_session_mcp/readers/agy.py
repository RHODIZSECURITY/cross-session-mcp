"""Session reader for Google Antigravity (the `agy` CLI and the IDE).

Each installation (~/.gemini/antigravity-cli and ~/.gemini/antigravity) has its own
`conversation_summaries.db` index and transcripts in
`brain/<id>/.system_generated/logs/transcript[_full].jsonl`. Both are read.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from cross_session_mcp.models import SessionDetail, SessionSummary, SessionTurn
from cross_session_mcp.readers.base import TEXT_TURN_LIMIT, TOOL_TURN_LIMIT, BaseSessionReader
from cross_session_mcp.utils import (
    file_uri_to_path,
    is_excluded_workspace,
    is_safe_session_id,
    logger,
    to_utc_iso,
    truncate,
    workspace_matches,
)

_USER_REQUEST_RE = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.S)
_ADDITIONAL_META_RE = re.compile(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", re.S)


def default_agy_dirs() -> list[Path]:
    env = os.environ.get("CROSS_SESSION_AGY_DIRS")
    if env:
        return [Path(p).expanduser() for p in env.split(os.pathsep) if p.strip()]
    gemini = Path.home() / ".gemini"
    return [gemini / "antigravity-cli", gemini / "antigravity"]


def parse_workspace_uris(raw: Any) -> str:
    """'["file:///a/b"]' | 'file:///a/b,file:///c' -> '/a/b' (first workspace)."""
    if not raw:
        return ""
    uris: list[str] = []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            uris = [str(u) for u in parsed] if isinstance(parsed, list) else [str(parsed)]
        except ValueError:
            uris = [u.strip() for u in raw.split(",")]
    elif isinstance(raw, list):
        uris = [str(u) for u in raw]
    uris = [u for u in uris if u]
    return file_uri_to_path(uris[0]) if uris else ""


class AgySessionReader(BaseSessionReader):
    name = "agy"

    def __init__(self, base_dirs: list[Path] | None = None, base_dir: Path | None = None):
        if base_dir is not None:
            base_dirs = [base_dir]
        self.base_dirs = [d for d in (base_dirs or default_agy_dirs())]

    def _stores(self) -> list[tuple[Path, Path]]:
        return [
            (d, d / "conversation_summaries.db") for d in self.base_dirs if (d / "conversation_summaries.db").exists()
        ]

    def can_read(self) -> bool:
        return bool(self._stores())

    # ------------------------------------------------------------------ index
    def _query(self, db_path: Path, where: str = "", params: tuple = (), limit: int = 50) -> list[dict[str, Any]]:
        try:
            conn = self._connect_ro(db_path)
            try:
                cols = self._columns(conn, "conversation_summaries")
                wanted = [
                    "conversation_id",
                    "title",
                    "preview",
                    "step_count",
                    "last_modified_time",
                    "workspace_uris",
                    "status",
                ]
                optional = [c for c in ("parent_conversation_id", "killed") if c in cols]
                select = ", ".join(c for c in wanted + optional if c in cols)
                sql = f"SELECT {select} FROM conversation_summaries {where} ORDER BY last_modified_time DESC LIMIT ?"
                cur = conn.execute(sql, params + (limit,))
                names = [d[0] for d in cur.description]
                return [dict(zip(names, row, strict=True)) for row in cur.fetchall()]
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("agy: error reading %s: %s", db_path, exc)
            return []

    def _row_to_summary(self, row: dict[str, Any]) -> SessionSummary:
        cid = str(row.get("conversation_id"))
        status = str(row.get("status") or "active")
        status = status.replace("CASCADE_RUN_STATUS_", "").lower() or "active"
        return SessionSummary(
            session_id=cid,
            harness=self.name,
            workspace_path=parse_workspace_uris(row.get("workspace_uris")),
            title=row.get("title") or f"Antigravity session ({cid[:8]})",
            updated_at=to_utc_iso(row.get("last_modified_time")),
            turn_count=int(row.get("step_count") or 0),
            last_prompt=(row.get("preview") or "")[:300],
            status=status,
        )

    def list_sessions(
        self,
        workspace_filter: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        sessions: list[SessionSummary] = []
        for _, db_path in self._stores():
            # The query is oversized because the workspace filter runs in Python
            fetch = limit * 4 if workspace_filter else limit
            for row in self._query(db_path, limit=fetch):
                if row.get("parent_conversation_id"):
                    continue  # subagents: queried through their parent conversation
                summary = self._row_to_summary(row)
                if not is_excluded_workspace(summary.workspace_path) and workspace_matches(
                    workspace_filter, summary.workspace_path
                ):
                    sessions.append(summary)
        sessions.sort(key=lambda s: s.updated_at or "", reverse=True)
        return sessions[:limit]

    # ------------------------------------------------------------------ transcript
    @staticmethod
    def _clean_user_input(content: str) -> str:
        match = _USER_REQUEST_RE.search(content)
        if match:
            return match.group(1)
        return _ADDITIONAL_META_RE.sub("", content).strip()

    def _step_to_turns(self, step: dict[str, Any]) -> list[SessionTurn]:
        stype = step.get("type", "")
        content = step.get("content") or ""
        ts = to_utc_iso(step.get("created_at"))
        turns: list[SessionTurn] = []
        if stype == "USER_INPUT":
            text = self._clean_user_input(content)
            if text:
                turns.append(SessionTurn(role="user", content=truncate(text, TEXT_TURN_LIMIT), timestamp=ts))
        elif stype in ("PLANNER_RESPONSE", "MODEL_RESPONSE"):
            if content.strip():
                turns.append(SessionTurn(role="assistant", content=truncate(content, TEXT_TURN_LIMIT), timestamp=ts))
            for call in step.get("tool_calls") or []:
                if isinstance(call, dict):
                    turns.append(
                        SessionTurn(
                            role="tool",
                            content=self._tool_call_text(call.get("name", "tool"), call.get("args")),
                            timestamp=ts,
                        )
                    )
        elif stype in ("GENERIC", "TOOL_EXECUTION"):
            if content.strip():
                turns.append(SessionTurn(role="tool", content=truncate(content, TOOL_TURN_LIMIT), timestamp=ts))
        elif stype == "ERROR_MESSAGE":
            error = step.get("error") or content
            if error:
                turns.append(SessionTurn(role="system", content="ERROR: " + truncate(str(error), 500), timestamp=ts))
        # SYSTEM_MESSAGE and CHECKPOINT are internal harness noise
        return turns

    def _find_transcript(self, session_id: str) -> Path | None:
        for base in self.base_dirs:
            brain = base / "brain" / session_id
            for candidate in (
                brain / ".system_generated" / "logs" / "transcript_full.jsonl",
                brain / ".system_generated" / "logs" / "transcript.jsonl",
                brain / "transcript.jsonl",
            ):
                if candidate.exists():
                    return candidate
        return None

    def get_session_detail(
        self,
        session_id: str,
        max_turns: int | None = 25,
    ) -> SessionDetail | None:
        if not is_safe_session_id(session_id):
            return None

        summary: SessionSummary | None = None
        for _, db_path in self._stores():
            rows = self._query(db_path, "WHERE conversation_id = ?", (session_id,), limit=1)
            if rows:
                summary = self._row_to_summary(rows[0])
                break

        transcript = self._find_transcript(session_id)
        if summary is None and transcript is None:
            return None
        if summary is None:
            summary = SessionSummary(
                session_id=session_id,
                harness=self.name,
                workspace_path="",
                title=f"Antigravity session ({session_id[:8]})",
            )

        turns: list[SessionTurn] = []
        if transcript is not None:
            for step in self._iter_jsonl(transcript):
                if isinstance(step, dict):
                    turns.extend(self._step_to_turns(step))
        return SessionDetail(summary=summary, turns=self._tail(turns, max_turns))
