"""MCP server (JSON-RPC 2.0 over stdio) for cross-session-mcp.

Built on the standard library only, so any harness (Claude Code, Codex, OpenCode,
Antigravity, Gemini CLI...) can launch it without dependencies.

Transport rules it follows:
- stdout carries ONLY newline-delimited JSON-RPC messages. The original file
  descriptor 1 is reserved for the protocol and everything else in the process
  (including libraries and subprocesses that print) is redirected to stderr.
- Notifications (no 'id') never get a response.
- Tool execution failures are returned as a result with 'isError': true; JSON-RPC
  errors are reserved for protocol failures.
"""

import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TextIO

from cross_session_mcp import __version__, i18n
from cross_session_mcp.agent import GeminiHandoffAgent
from cross_session_mcp.db import CrossSessionDB
from cross_session_mcp.models import HARNESSES, SessionSummary
from cross_session_mcp.service import CrossSessionService, SessionError, select_turns
from cross_session_mcp.utils import logger, redact, setup_logging, truncate

SUPPORTED_PROTOCOL_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

SERVER_INSTRUCTIONS = (
    "Memory bank of Claude Code, agy, OpenCode, Codex, Cursor and Copilot sessions. Order: "
    "1) get_project_memory; 2) inject_session_context (back=1 last session, up to 10); "
    "3) search_project_memory by topic; 4) get_session_turns only for a specific detail. "
    "Secrets and PII are hidden."
)

_WS = {"type": "string", "description": "Exact absolute project path."}
_SID = {"type": "string", "description": "ID or prefix; if missing, workspace_path+back is used."}
_HARNESS = {"type": "string", "enum": list(HARNESSES)}
_BOOL = {"type": "boolean"}
_HANDOFF_SCHEMA = {
    "type": "object",
    "properties": {
        "workspace_path": _WS,
        "back": {"type": "integer", "minimum": 1, "maximum": 10, "description": "1 = last session."},
        "session_id": _SID,
        "harness": _HARNESS,
        "focus_area": {"type": "string", "description": "Focus; separate call, not saved."},
        "force_refresh": _BOOL,
    },
    "additionalProperties": False,
}
_RO = {"readOnlyHint": True, "openWorldHint": False}
_RO_LLM = {"readOnlyHint": True, "openWorldHint": True}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str], annotations: dict[str, bool]):
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "inputSchema": schema, "annotations": annotations}


TOOLS_METADATA: list[dict[str, Any]] = [
    _tool("get_project_memory", "Project MEMORY.md, no LLM. Use it first.", {"workspace_path": _WS},
          ["workspace_path"], _RO),
    _tool("inject_session_context", "Extended Handoff ready to resume a previous session.",
          _HANDOFF_SCHEMA["properties"], [], _RO_LLM),
    _tool("generate_session_handoff", "Extended Handoff of a session (no resume wrapper).",
          _HANDOFF_SCHEMA["properties"], [], _RO_LLM),
    _tool("search_project_memory", "Searches a topic in active and archived memory and handoffs.", {
        "topic": {"type": "string"},
        "workspace_path": _WS,
        "all_projects": _BOOL,
        "synthesize": {"type": "boolean", "description": "Patterns via LLM; not saved."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
    }, ["topic", "workspace_path"], _RO_LLM),
    _tool("update_project_memory", "Forces memory consolidation (LLM).", {
        "workspace_path": _WS,
        "sessions": {"type": "integer", "minimum": 1, "maximum": 20},
        "force_refresh": _BOOL,
        "export_path": {"type": "string", "description": "Also writes the file (600)."},
    }, ["workspace_path"], {"readOnlyHint": False, "openWorldHint": True}),
    _tool("suggest_agent_rules", "Memory rules for AGENTS.md or CLAUDE.md, no LLM.", {
        "workspace_path": _WS,
        "target": {"type": "string", "enum": ["agents", "claude", "both"]},
    }, ["workspace_path"], _RO),
    _tool("list_cross_sessions", "Recent sessions from every harness.", {
        "workspace_path": {"type": "string", "description": "Path (subdirectories included) or substring."},
        "harness": _HARNESS,
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        "refresh": _BOOL,
    }, [], _RO),
    _tool("get_session_turns", "Verbatim turns; only for a specific detail.", {
        "session_id": {"type": "string"},
        "harness": _HARNESS,
        "query": {"type": "string", "description": "Only turns with this text (±1)."},
        "max_turns": {"type": "integer", "minimum": 1, "maximum": 50},
        "include_tools": _BOOL,
    }, ["session_id"], _RO),
]
_TOOL_NAMES = {t["name"] for t in TOOLS_METADATA}

MAX_OUTPUT_CHARS = 40_000
_NARROWING_HINTS = {
    "get_session_turns": "use query or lower max_turns",
    "list_cross_sessions": "lower limit or filter by workspace_path/harness",
    "search_project_memory": "narrow topic or lower limit",
}


def max_output_chars() -> int:
    try:
        return max(2_000, int(os.environ.get("CROSS_SESSION_MAX_OUTPUT_CHARS", MAX_OUTPUT_CHARS)))
    except ValueError:
        return MAX_OUTPUT_CHARS


def cap_output(text: str, tool_name: str, limit: int | None = None) -> str:
    """Central cap: no result exceeds `limit` characters; when cutting, it explains how to narrow down."""
    limit = limit or max_output_chars()
    if len(text) <= limit:
        return text
    hint = _NARROWING_HINTS.get(tool_name, "narrow the request")
    notice = f"\n\n> ✂️ Result truncated: {len(text) - limit:,} characters omitted. To narrow it down, {hint}."
    cut = text.rfind("\n", 0, limit - len(notice))
    cut = cut if cut > limit // 2 else limit - len(notice)
    return text[:cut] + notice


class InvalidParams(Exception):
    """Arguments that do not match the tool schema (JSON-RPC error -32602)."""


def _arg_str(args: dict[str, Any], key: str, required: bool = False) -> str | None:
    value = args.get(key)
    if value is None or value == "":
        if required:
            raise InvalidParams(f"Missing required parameter '{key}'.")
        return None
    if not isinstance(value, str):
        raise InvalidParams(f"Parameter '{key}' must be a string.")
    return value.strip()


def _arg_int(args: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool):
        raise InvalidParams(f"Parameter '{key}' must be an integer.")
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise InvalidParams(f"Parameter '{key}' must be an integer.") from None
    if not lo <= value <= hi:
        raise InvalidParams(f"Parameter '{key}' must be between {lo} and {hi}.")
    return value


def _arg_bool(args: dict[str, Any], key: str, default: bool = False) -> bool:
    value = args.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


class MCPServer:
    def __init__(
        self,
        db: CrossSessionDB | None = None,
        agent: GeminiHandoffAgent | None = None,
        service: CrossSessionService | None = None,
    ):
        # Nothing expensive here: the 'initialize' handshake must answer at once.
        self.service = service or CrossSessionService(db=db, agent=agent)
        # The harness launches this process when the session opens: anything created later is the current session
        self.service.process_started_at = time.time()
        self.service.background = True
        self._out: TextIO = sys.stdout
        self._write_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mcp-tool")

    # ------------------------------------------------------------------ tools
    def handle_tool_call(self, name: str, args: dict[str, Any]) -> str:
        if name not in _TOOL_NAMES:
            raise InvalidParams(f"Unknown tool: {name}")
        if not isinstance(args, dict):
            raise InvalidParams("'arguments' must be an object.")
        return getattr(self, f"_tool_{name}")(args)

    def _tool_list_cross_sessions(self, args: dict[str, Any]) -> str:
        harness = _arg_str(args, "harness")
        sessions = self.service.list_sessions(
            workspace=_arg_str(args, "workspace_path"),
            harness=harness,
            limit=_arg_int(args, "limit", 15, 1, 50),
            refresh=_arg_bool(args, "refresh"),
        )
        if not sessions:
            return "No sessions match the given filters."
        lines = [f"### 📋 Recent Sessions ({len(sessions)} found)\n"]
        for s in sessions:
            lines.append(self._format_session(s))
        return "\n".join(lines)

    @staticmethod
    def _format_session(s: SessionSummary) -> str:
        when = (s.updated_at or "unknown")[:16].replace("T", " ")
        prompt = truncate(redact(s.last_prompt).replace("\n", " "), 160)
        text = (
            f"- **[{s.harness.upper()}]** `{s.session_id}`: **{redact(s.title)}**\n"
            f"  *Path:* `{s.workspace_path}` | *Updated:* {when} UTC | *Turns:* {s.turn_count} | *Status:* {s.status}\n"
        )
        if prompt:
            text += f"  *Last user message:* {prompt}\n"
        return text

    def _tool_get_session_turns(self, args: dict[str, Any]) -> str:
        include_tools = _arg_bool(args, "include_tools")
        max_turns = _arg_int(args, "max_turns", 10, 1, 50)
        query = _arg_str(args, "query")
        detail = self.service.get_detail(_arg_str(args, "session_id", required=True), _arg_str(args, "harness"),
                                         max_turns=None)
        turns, omitted = select_turns(detail.turns, query, include_tools, max_turns)
        s = detail.summary
        if query and not turns:
            return f"No turn of `{s.session_id}` contains “{redact(query)}”."
        lines = [
            f"### 💬 {redact(s.title)} ({s.harness}, `{s.session_id}`)",
            f"*Showing:* {len(turns)} turns{f' with “{redact(query)}” (±1)' if query else ''} · *omitted:* {omitted}\n",
        ]
        icons = {"user": "👤", "assistant": "🤖", "tool": "🔧", "system": "⚙️"}
        for t in turns:
            lines.append(f"**{icons.get(t.role, '•')} {t.role.upper()}:**\n{truncate(redact(t.content), 2000)}\n")
        return "\n".join(lines)

    def _handoff(self, args: dict[str, Any]):
        session_id = _arg_str(args, "session_id")
        workspace = _arg_str(args, "workspace_path")
        if not session_id and not workspace:
            raise InvalidParams("Provide 'workspace_path' (and optionally 'back') or 'session_id'.")
        return self.service.get_handoff(
            session_id,
            _arg_str(args, "harness"),
            focus_area=_arg_str(args, "focus_area"),
            force_refresh=_arg_bool(args, "force_refresh"),
            workspace=workspace,
            back=_arg_int(args, "back", 1, 1, 10),
        )

    def _tool_generate_session_handoff(self, args: dict[str, Any]) -> str:
        h = self._handoff(args)
        origin = "local bank" if h.from_cache else h.model_used
        return f"{h.markdown_handoff}\n\n---\n*Source: {origin} · generated {h.generated_at[:16].replace('T', ' ')} UTC*"

    def _tool_inject_session_context(self, args: dict[str, Any]) -> str:
        h = self._handoff(args)
        return f"""# {i18n.t('resume_title')} ({h.harness.upper()} · `{h.session_id}`)
{i18n.t('resume_workdir')}: `{h.workspace_path}`

{h.markdown_handoff}

---
{i18n.t('resume_footer')}
"""

    def _tool_search_project_memory(self, args: dict[str, Any]) -> str:
        return self.service.search_project_memory(
            _arg_str(args, "topic", required=True),
            _arg_str(args, "workspace_path", required=True),
            all_projects=_arg_bool(args, "all_projects"),
            limit=_arg_int(args, "limit", 20, 1, 50),
            synthesize=_arg_bool(args, "synthesize"),
        )

    def _tool_update_project_memory(self, args: dict[str, Any]) -> str:
        result = self.service.update_project_memory(
            _arg_str(args, "workspace_path", required=True),
            sessions=_arg_int(args, "sessions", 0, 0, 20) or None,
            force_refresh=_arg_bool(args, "force_refresh"),
            export_path=_arg_str(args, "export_path"),
        )
        return f"{result.markdown}\n---\n*{result.summary()}*"

    def _tool_get_project_memory(self, args: dict[str, Any]) -> str:
        return self.service.get_project_memory(_arg_str(args, "workspace_path", required=True))

    def _tool_suggest_agent_rules(self, args: dict[str, Any]) -> str:
        target = _arg_str(args, "target") or "both"
        if target not in ("agents", "claude", "both"):
            raise InvalidParams("Parameter 'target' must be 'agents', 'claude' or 'both'.")
        return self.service.suggest_agent_rules(_arg_str(args, "workspace_path", required=True), target)

    # ------------------------------------------------------------------ JSON-RPC
    def _send(self, message: dict[str, Any]) -> None:
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._write_lock:
            self._out.write(data)
            self._out.flush()

    def _error(self, req_id: Any, code: int, message: str) -> None:
        self._send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

    def _result(self, req_id: Any, result: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _schedule_maintenance(self, args: Any) -> None:
        """Any call with workspace_path checks (without waiting) whether that project's bank is up to date."""
        workspace = args.get("workspace_path") if isinstance(args, dict) else None
        if not isinstance(workspace, str) or not workspace.strip().startswith("/"):
            return
        try:
            self.service.maybe_schedule_maintenance(workspace)
        except Exception:
            logger.exception("Could not schedule maintenance for %s", workspace)

    def _run_tool(self, req_id: Any, name: str, args: dict[str, Any]) -> None:
        try:
            text = cap_output(self.handle_tool_call(name, args), name)
            self._schedule_maintenance(args)
            self._result(req_id, {"content": [{"type": "text", "text": text}], "isError": False})
        except InvalidParams as exc:
            self._error(req_id, -32602, str(exc))
        except SessionError as exc:
            self._result(req_id, {"content": [{"type": "text", "text": f"Error: {exc}"}], "isError": True})
        except Exception as exc:
            logger.exception("Unexpected failure in tool %s", name)
            msg = redact(f"{type(exc).__name__}: {exc}")
            self._result(req_id, {"content": [{"type": "text", "text": f"Internal error: {msg}"}], "isError": True})

    def handle_message(self, message: Any) -> None:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            req_id = message.get("id") if isinstance(message, dict) else None
            if req_id is not None or not isinstance(message, dict):
                self._error(req_id, -32600, "Invalid JSON-RPC request.")
            return

        req_id = message.get("id")
        method = message["method"]
        params = message.get("params") or {}
        is_notification = "id" not in message

        if is_notification:
            return  # notifications/initialized, notifications/cancelled, etc.

        if method == "initialize":
            requested = params.get("protocolVersion") if isinstance(params, dict) else None
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
            self._result(req_id, {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "cross-session-mcp", "title": "Cross-Session Handoff", "version": __version__},
                "instructions": SERVER_INSTRUCTIONS,
            })
        elif method == "ping":
            self._result(req_id, {})
        elif method == "tools/list":
            self._result(req_id, {"tools": TOOLS_METADATA})
        elif method == "tools/call":
            if not isinstance(params, dict) or not isinstance(params.get("name"), str):
                self._error(req_id, -32602, "tools/call requires 'name'.")
                return
            # Tools may be slow (Gemini call): they run in a thread so they do not
            # block ping or other requests.
            self._executor.submit(self._run_tool, req_id, params["name"], params.get("arguments") or {})
        elif method in ("resources/list", "resources/templates/list", "prompts/list"):
            key = {"resources/list": "resources", "resources/templates/list": "resourceTemplates", "prompts/list": "prompts"}[method]
            self._result(req_id, {key: []})
        else:
            self._error(req_id, -32601, f"Unsupported method: {method}")

    def run(self, stdin: TextIO, stdout: TextIO) -> None:
        self._out = stdout
        try:
            for line in stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    self._error(None, -32700, "JSON parse error.")
                    continue
                if isinstance(message, list):  # batches (protocol 2025-03-26)
                    for item in message:
                        self.handle_message(item)
                else:
                    self.handle_message(message)
        finally:
            self._executor.shutdown(wait=True)

    def run_stdio(self) -> None:
        """Starts on stdio, shielding stdout from writes outside the protocol."""
        protocol_fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # fd 1 -> stderr (subprocesses, C, print)
        sys.stdout = sys.stderr
        protocol_out = io.TextIOWrapper(os.fdopen(protocol_fd, "wb"), encoding="utf-8", newline="\n")
        stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        self.run(stdin, protocol_out)


def main() -> None:
    setup_logging()
    MCPServer().run_stdio()


if __name__ == "__main__":
    main()
