"""Session reader for Claude Code (~/.claude/projects/<project>/<session>.jsonl)."""

import datetime
import functools
from pathlib import Path
from typing import Any

from cross_session_mcp.models import SessionDetail, SessionSummary, SessionTurn
from cross_session_mcp.readers.base import TEXT_TURN_LIMIT, TOOL_TURN_LIMIT, BaseSessionReader
from cross_session_mcp.utils import (
    env_path,
    is_excluded_workspace,
    is_safe_session_id,
    to_utc_iso,
    truncate,
    workspace_matches,
)


@functools.lru_cache(maxsize=512)
def decode_project_dir(name: str) -> str:
    """Rebuilds the path of a Claude project directory ('-home-user-my-app').

    Claude replaces '/' and '.' with '-', so decoding is ambiguous ('my-app'
    versus 'my/app'). It walks the disk picking the longest segment that exists;
    if nothing matches it returns the naive conversion.
    """
    tokens = name.lstrip("-").split("-")
    current = Path("/")
    i = 0
    while i < len(tokens):
        for j in range(len(tokens), i, -1):
            chunk = tokens[i:j]
            candidates = ["-".join(chunk), ".".join(chunk)]
            if chunk and chunk[0] == "" and len(chunk) > 1:  # '--claude' -> '.claude'
                candidates.append("." + "-".join(chunk[1:]))
            match = next((c for c in candidates if c and (current / c).exists()), None)
            if match:
                current = current / match
                i = j
                break
        else:
            return "/" + name.lstrip("-").replace("-", "/")
    return str(current)


# Per-file metadata cache (path -> (mtime_ns, size, meta)). The server is a long-running
# process: this way each re-sync only rereads the sessions that changed.
_META_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}


class ClaudeSessionReader(BaseSessionReader):
    name = "claude"

    def __init__(self, projects_dir: Path | None = None):
        self.projects_dir = projects_dir or env_path(
            "CROSS_SESSION_CLAUDE_DIR", Path.home() / ".claude" / "projects"
        )

    def can_read(self) -> bool:
        return self.projects_dir.is_dir()

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _content_items(message: Any) -> list[Any]:
        if isinstance(message, str):
            return [{"type": "text", "text": message}]
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return [{"type": "text", "text": content}]
            if isinstance(content, list):
                return content
        return []

    @staticmethod
    def _tool_result_text(item: dict[str, Any]) -> str:
        content = item.get("content")
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
            )
        return str(content or "")

    def _record_to_turns(self, rec: dict[str, Any]) -> list[SessionTurn]:
        rtype = rec.get("type")
        if rtype not in ("user", "assistant") or rec.get("isMeta") or rec.get("isSidechain"):
            return []
        ts = to_utc_iso(rec.get("timestamp"))
        turns: list[SessionTurn] = []
        texts: list[str] = []
        for item in self._content_items(rec.get("message")):
            if isinstance(item, str):
                texts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "text" and item.get("text", "").strip():
                texts.append(item["text"])
            elif itype == "tool_use":
                turns.append(SessionTurn(
                    role="tool",
                    content=self._tool_call_text(item.get("name", "tool"), item.get("input")),
                    timestamp=ts,
                ))
            elif itype == "tool_result":
                turns.append(SessionTurn(
                    role="tool",
                    content=truncate(self._tool_result_text(item), TOOL_TURN_LIMIT),
                    timestamp=ts,
                ))
            # 'thinking' and 'image' are skipped on purpose
        if texts:
            turns.insert(0, SessionTurn(role=rtype, content=truncate("\n".join(texts), TEXT_TURN_LIMIT), timestamp=ts))
        return turns

    def _scan(self, path: Path, collect_turns: bool) -> tuple[dict[str, Any], list[SessionTurn]]:
        meta: dict[str, Any] = {"title": "", "ai_title": "", "cwd": "", "created_at": None, "last_prompt": "", "turn_count": 0}
        turns: list[SessionTurn] = []
        for rec in self._iter_jsonl(path):
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            if rtype == "custom-title" and rec.get("customTitle"):
                meta["title"] = rec["customTitle"]
            elif rtype == "ai-title" and rec.get("aiTitle"):
                meta["ai_title"] = rec["aiTitle"]
            elif rtype == "summary" and rec.get("summary") and not meta["ai_title"]:
                meta["ai_title"] = rec["summary"]
            if not meta["cwd"] and rec.get("cwd"):
                meta["cwd"] = rec["cwd"]
            if meta["created_at"] is None and rec.get("timestamp"):
                meta["created_at"] = to_utc_iso(rec["timestamp"])

            rec_turns = self._record_to_turns(rec)
            for t in rec_turns:
                if t.role in ("user", "assistant"):
                    meta["turn_count"] += 1
                if t.role == "user":
                    meta["last_prompt"] = t.content[:300]
            if collect_turns:
                turns.extend(rec_turns)
        return meta, turns

    def _build_summary(self, path: Path, meta: dict[str, Any]) -> SessionSummary:
        session_id = path.stem
        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc)
        return SessionSummary(
            session_id=session_id,
            harness=self.name,
            workspace_path=meta["cwd"] or decode_project_dir(path.parent.name),
            title=meta["title"] or meta["ai_title"] or f"Claude session ({session_id[:8]})",
            created_at=meta["created_at"],
            updated_at=to_utc_iso(mtime),
            turn_count=meta["turn_count"],
            last_prompt=meta["last_prompt"],
            status="active",
        )

    def _cached_meta(self, path: Path) -> dict[str, Any]:
        st = path.stat()
        key = str(path)
        cached = _META_CACHE.get(key)
        if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            return cached[2]
        meta, _ = self._scan(path, collect_turns=False)
        _META_CACHE[key] = (st.st_mtime_ns, st.st_size, meta)
        return meta

    # ------------------------------------------------------------------ API
    def _session_files(self) -> list[Path]:
        files: list[Path] = []
        try:
            for pdir in self.projects_dir.iterdir():
                if pdir.is_dir():
                    files.extend(pdir.glob("*.jsonl"))
        except OSError:
            return []
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return files

    def list_sessions(
        self,
        workspace_filter: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        if not self.can_read():
            return []
        sessions: list[SessionSummary] = []
        for sfile in self._session_files():
            if is_excluded_workspace(decode_project_dir(sfile.parent.name)):
                continue
            meta = self._cached_meta(sfile)
            if meta["turn_count"] == 0:
                continue  # empty or metadata-only sessions
            summary = self._build_summary(sfile, meta)
            if is_excluded_workspace(summary.workspace_path) or not workspace_matches(workspace_filter, summary.workspace_path):
                continue
            sessions.append(summary)
            if len(sessions) >= limit:
                break
        return sessions

    def get_session_detail(
        self,
        session_id: str,
        max_turns: int | None = 25,
    ) -> SessionDetail | None:
        if not self.can_read() or not is_safe_session_id(session_id):
            return None
        target = next(self.projects_dir.glob(f"*/{session_id}.jsonl"), None)
        if target is None:
            return None
        meta, turns = self._scan(target, collect_turns=True)
        return SessionDetail(summary=self._build_summary(target, meta), turns=self._tail(turns, max_turns))
