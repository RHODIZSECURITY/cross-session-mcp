"""Session reader for GitHub Copilot Chat in VS Code.

Location: `<config>/Code/User/workspaceStorage/<hash>/chatSessions/<id>.jsonl|.json`, where `<config>`
is `~/.config` on Linux and `~/Library/Application Support` on macOS
(+ `globalStorage/emptyWindowChatSessions/` for windows without a folder). The workspace
of each `<hash>` is in `workspaceStorage/<hash>/workspace.json`.

`.jsonl` format (VS Code ≥ 1.10x): an operation log over the session state.
- kind 0: full initial state (`v`)
- kind 1: sets `v` at path `k`
- kind 2: appends the items of `v` to the list at `k`; if it carries `i`, truncates it to `i` first
- kind 3: deletes the key at path `k`
`.json` format (older versions): the full state in a single object.
"""

import datetime
import json
import os
from pathlib import Path
from typing import Any

from cross_session_mcp.models import SessionDetail, SessionSummary, SessionTurn
from cross_session_mcp.readers.base import TEXT_TURN_LIMIT, BaseSessionReader
from cross_session_mcp.utils import (
    config_home,
    file_uri_to_path,
    is_excluded_workspace,
    is_safe_session_id,
    logger,
    to_utc_iso,
    truncate,
    workspace_matches,
)


def default_vscode_user_dirs() -> list[Path]:
    env = os.environ.get("CROSS_SESSION_VSCODE_DIRS")
    if env:
        return [Path(p).expanduser() for p in env.split(os.pathsep) if p.strip()]
    config = config_home()
    return [config / name / "User" for name in ("Code", "Code - Insiders", "VSCodium")]


def replay_operations(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuilds the final state of a session from the operation log."""
    state: dict[str, Any] = {}
    for op in lines:
        kind = op.get("kind")
        if kind == 0:
            state = op.get("v") or {}
            continue
        path = op.get("k") or []
        if not path:
            continue
        try:
            parent: Any = state
            for key in path[:-1]:
                parent = parent[key]
            last = path[-1]
            if kind == 1:
                parent[last] = op.get("v")
            elif kind == 2:
                target = parent[last] if (isinstance(parent, dict) and last in parent) or (
                    isinstance(parent, list) and isinstance(last, int) and last < len(parent)
                ) else None
                if not isinstance(target, list):
                    target = []
                    parent[last] = target
                if op.get("i") is not None:
                    del target[int(op["i"]):]
                target.extend(op.get("v") or [])
            elif kind == 3:
                if isinstance(parent, dict):
                    parent.pop(last, None)
                elif isinstance(parent, list) and isinstance(last, int) and last < len(parent):
                    parent.pop(last)
        except (KeyError, IndexError, TypeError, ValueError):
            continue  # operation on a path that does not exist: ignored
    return state


class CopilotSessionReader(BaseSessionReader):
    name = "copilot"

    def __init__(self, user_dirs: list[Path] | None = None):
        self.user_dirs = user_dirs or default_vscode_user_dirs()

    def can_read(self) -> bool:
        return any((d / "workspaceStorage").is_dir() or (d / "globalStorage" / "emptyWindowChatSessions").is_dir()
                   for d in self.user_dirs)

    # ------------------------------------------------------------------ files
    @staticmethod
    def _workspace_of(storage_dir: Path) -> str:
        meta = storage_dir / "workspace.json"
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return file_uri_to_path(data.get("folder") or data.get("workspace") or "")

    def _session_files(self) -> list[tuple[Path, str]]:
        files: list[tuple[Path, str]] = []
        for user_dir in self.user_dirs:
            ws_root = user_dir / "workspaceStorage"
            if ws_root.is_dir():
                for chat_dir in ws_root.glob("*/chatSessions"):
                    workspace = self._workspace_of(chat_dir.parent)
                    files.extend((f, workspace) for f in chat_dir.iterdir() if f.suffix in (".json", ".jsonl"))
            empty = user_dir / "globalStorage" / "emptyWindowChatSessions"
            if empty.is_dir():
                files.extend((f, "") for f in empty.iterdir() if f.suffix in (".json", ".jsonl"))
        # If both formats exist for the same ID, .jsonl (newer) wins
        by_id: dict[str, tuple[Path, str]] = {}
        for f, ws in files:
            if f.stem not in by_id or f.suffix == ".jsonl":
                by_id[f.stem] = (f, ws)
        result = list(by_id.values())
        result.sort(key=lambda item: item[0].stat().st_mtime, reverse=True)
        return result

    def _load_state(self, path: Path) -> dict[str, Any]:
        if path.suffix == ".jsonl":
            return replay_operations([op for op in self._iter_jsonl(path) if isinstance(op, dict)])
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as exc:
            logger.warning("copilot: could not read %s: %s", path, exc)
            return {}

    # ------------------------------------------------------------------ conversion
    @staticmethod
    def _request_text(request: dict[str, Any]) -> str:
        message = request.get("message") or {}
        if isinstance(message, str):
            return message
        return message.get("text") or ""

    @staticmethod
    def _value_text(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("value") or "")
        return str(value or "")

    def _request_to_turns(self, request: dict[str, Any]) -> list[SessionTurn]:
        ts = to_utc_iso(request.get("timestamp"))
        turns: list[SessionTurn] = []
        text = self._request_text(request).strip()
        if text:
            turns.append(SessionTurn(role="user", content=truncate(text, TEXT_TURN_LIMIT), timestamp=ts))

        markdown: list[str] = []

        def flush() -> None:
            joined = "".join(markdown).strip()
            if joined:
                turns.append(SessionTurn(role="assistant", content=truncate(joined, TEXT_TURN_LIMIT), timestamp=ts))
            markdown.clear()

        for part in request.get("response") or []:
            if not isinstance(part, dict):
                continue
            kind = part.get("kind")
            if kind in (None, "markdownContent", "markdownVuln"):
                markdown.append(self._value_text(part.get("content", part.get("value"))))
            elif kind == "inlineReference":
                ref = part.get("inlineReference") or {}
                markdown.append(f"`{ref.get('name') or Path(str(ref.get('path') or ref.get('fsPath') or '')).name}`")
            elif kind == "toolInvocationSerialized":
                flush()
                message = self._value_text(part.get("pastTenseMessage") or part.get("invocationMessage"))
                turns.append(SessionTurn(role="tool", content=self._tool_call_text(part.get("toolId", "tool"), message), timestamp=ts))
            elif kind == "textEditGroup":
                flush()
                uri = part.get("uri") or {}
                target = (uri.get("fsPath") or uri.get("path") or "") if isinstance(uri, dict) else str(uri)
                turns.append(SessionTurn(role="tool", content=f"→ edit: {target}", timestamp=ts))
            # thinking, undoStop, codeblockUri, progress... are skipped
        flush()
        return turns

    @staticmethod
    def _resolve_multi_root(workspace: str, requests: list[dict[str, Any]]) -> str:
        """A multi-root workspace points to a .code-workspace file: its first folder is used.

        If the file no longer exists (untitled workspaces), the folder is inferred from the
        `baseUri` VS Code stores in Markdown responses.
        """
        if not workspace.endswith((".code-workspace", "workspace.json")):
            return workspace
        ws_file = Path(workspace)
        try:
            folders = json.loads(ws_file.read_text(encoding="utf-8")).get("folders") or []
            first = next((f.get("path") or file_uri_to_path(f.get("uri", "")) for f in folders if isinstance(f, dict)), "")
            if first:
                return str((ws_file.parent / first).resolve()) if not first.startswith("/") else first
        except (OSError, ValueError, AttributeError):
            pass
        for request in requests:
            for part in request.get("response") or []:
                base = part.get("baseUri") if isinstance(part, dict) else None
                if isinstance(base, dict) and base.get("scheme") == "file" and base.get("path"):
                    return base["path"].rstrip("/")
        return workspace

    def _build(self, path: Path, workspace: str, state: dict[str, Any]) -> tuple[SessionSummary, list[SessionTurn]]:
        requests = [r for r in state.get("requests") or [] if isinstance(r, dict)]
        workspace = self._resolve_multi_root(workspace, requests)
        turns: list[SessionTurn] = []
        for request in requests:
            turns.extend(self._request_to_turns(request))
        user_texts = [self._request_text(r).strip() for r in requests if self._request_text(r).strip()]
        session_id = str(state.get("sessionId") or path.stem)
        title = state.get("customTitle") or (truncate(user_texts[0].splitlines()[0], 80) if user_texts else "")
        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc)
        last_ts = max((r.get("timestamp") or 0 for r in requests), default=0)
        summary = SessionSummary(
            session_id=session_id,
            harness=self.name,
            workspace_path=workspace,
            title=title or f"Copilot session ({session_id[:8]})",
            created_at=to_utc_iso(state.get("creationDate")),
            updated_at=to_utc_iso(last_ts) if last_ts else to_utc_iso(mtime),
            turn_count=sum(1 for t in turns if t.role in ("user", "assistant")),
            last_prompt=user_texts[-1][:300] if user_texts else "",
        )
        return summary, turns

    # ------------------------------------------------------------------ API
    def list_sessions(
        self,
        workspace_filter: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        sessions: list[SessionSummary] = []
        for path, workspace in self._session_files():
            summary, _ = self._build(path, workspace, self._load_state(path))
            if summary.turn_count == 0:
                continue  # sessions opened but never used
            if is_excluded_workspace(summary.workspace_path) or not workspace_matches(workspace_filter, summary.workspace_path):
                continue
            sessions.append(summary)
            if len(sessions) >= limit:
                break
        sessions.sort(key=lambda s: s.updated_at or "", reverse=True)
        return sessions

    def get_session_detail(
        self,
        session_id: str,
        max_turns: int | None = 25,
    ) -> SessionDetail | None:
        if not is_safe_session_id(session_id):
            return None
        for path, workspace in self._session_files():
            if path.stem == session_id:
                summary, turns = self._build(path, workspace, self._load_state(path))
                return SessionDetail(summary=summary, turns=self._tail(turns, max_turns))
        return None
