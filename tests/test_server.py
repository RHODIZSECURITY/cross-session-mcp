"""Tests for the MCP protocol: in process and as a real stdio subprocess."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

from cross_session_mcp.server import LATEST_PROTOCOL_VERSION, TOOLS_METADATA, MCPServer

ROOT = Path(__file__).resolve().parent.parent


def rpc(id_, method, params=None):
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def call(id_, name, args):
    return rpc(id_, "tools/call", {"name": name, "arguments": args})


def run_server(messages, stores_env=None):
    server = MCPServer()
    stdin = io.StringIO("".join((m if isinstance(m, str) else json.dumps(m)) + "\n" for m in messages))
    stdout = io.StringIO()
    server.run(stdin, stdout)
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return {r.get("id"): r for r in responses}, responses


def text_of(resp):
    return resp["result"]["content"][0]["text"]


def test_tools_metadata_schema():
    from cross_session_mcp.models import HARNESSES
    by_name = {t["name"]: t for t in TOOLS_METADATA}
    assert list(by_name) == ["get_project_memory", "inject_session_context", "generate_session_handoff",
                             "search_project_memory", "update_project_memory", "suggest_agent_rules",
                             "list_cross_sessions", "get_session_turns"]
    assert by_name["list_cross_sessions"]["inputSchema"]["properties"]["harness"]["enum"] == list(HARNESSES)
    assert by_name["update_project_memory"]["annotations"] == {"readOnlyHint": False, "openWorldHint": True}
    for name in ("get_project_memory", "suggest_agent_rules", "search_project_memory", "get_session_turns"):
        assert by_name[name]["annotations"]["readOnlyHint"] is True
    assert by_name["suggest_agent_rules"]["inputSchema"]["properties"]["target"]["enum"] == ["agents", "claude", "both"]
    props = by_name["update_project_memory"]["inputSchema"]["properties"]
    assert (props["sessions"]["minimum"], props["sessions"]["maximum"]) == (1, 20)
    back = by_name["inject_session_context"]["inputSchema"]["properties"]["back"]
    assert (back["minimum"], back["maximum"]) == (1, 10)
    assert "required" not in by_name["inject_session_context"]["inputSchema"]  # session_id is no longer required
    assert by_name["get_session_turns"]["inputSchema"]["properties"]["max_turns"]["maximum"] == 50
    assert by_name["list_cross_sessions"]["inputSchema"]["properties"]["limit"]["maximum"] == 50
    for t in TOOLS_METADATA:
        assert t["inputSchema"]["type"] == "object"
        assert "harness" not in t["inputSchema"].get("required", [])


def test_tool_definitions_budget():
    """Fixed cost per session: definitions + instructions in compact JSON (what travels) ≤ 4,800 characters."""
    from cross_session_mcp.server import SERVER_INSTRUCTIONS
    size = len(json.dumps(TOOLS_METADATA, ensure_ascii=False, separators=(",", ":"))) + len(SERVER_INSTRUCTIONS)
    assert size <= 4800, size


def test_cap_output_trims_with_narrowing_hint(monkeypatch):
    from cross_session_mcp.server import cap_output
    text = "\n".join("x" * 99 for _ in range(1200))  # 120,000 characters
    out = cap_output(text, "get_session_turns")
    assert len(out) <= 40_000 and "use query or lower max_turns" in out and "characters omitted" in out
    assert cap_output("short", "get_session_turns") == "short"
    monkeypatch.setenv("CROSS_SESSION_MAX_OUTPUT_CHARS", "5000")
    assert len(cap_output(text, "other")) <= 5000


def test_handshake_and_protocol_errors(stores):
    by_id, responses = run_server([
        rpc(1, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        rpc(2, "initialize", {"protocolVersion": "1999-01-01"}),
        rpc(3, "ping"),
        rpc(4, "tools/list"),
        rpc(5, "weird/method"),
        call(6, "does_not_exist", {}),
        rpc(7, "tools/call", {}),
        "this is not json",
    ])
    assert by_id[1]["result"]["protocolVersion"] == "2024-11-05"
    assert "instructions" in by_id[1]["result"]
    assert by_id[2]["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION
    assert by_id[3]["result"] == {}
    assert len(by_id[4]["result"]["tools"]) == 8
    assert by_id[5]["error"]["code"] == -32601
    assert by_id[6]["error"]["code"] == -32602
    assert by_id[7]["error"]["code"] == -32602
    assert by_id[None]["error"]["code"] == -32700
    assert len(responses) == 8  # 9 messages; the notification gets no response


def test_tools_end_to_end(stores):
    by_id, _ = run_server([
        call(1, "list_cross_sessions", {"workspace_path": "/home/test/projects/my-app"}),
        call(2, "list_cross_sessions", {"harness": "codex"}),
        call(3, "get_session_turns", {"session_id": "agy-sess"}),  # prefix, no harness
        call(4, "get_session_turns", {"session_id": "claude-sess-1", "include_tools": False}),
        call(5, "generate_session_handoff", {"session_id": "claude-sess-1"}),
        call(6, "inject_session_context", {"session_id": "oc-sess-1", "harness": "opencode"}),
        call(7, "get_session_turns", {"session_id": "missing"}),
        call(8, "get_session_turns", {"session_id": "../../etc/passwd"}),
        call(9, "list_cross_sessions", {"harness": "vscode"}),
        call(10, "list_cross_sessions", {"limit": "lots"}),
        call(11, "get_session_turns", {"session_id": "cursor-sess-1"}),
        call(12, "generate_session_handoff", {"session_id": "copilot-sess", "harness": "copilot"}),
    ])
    listing = text_of(by_id[1])
    for sid in ("claude-sess-1", "agy-sess-1", "oc-sess-1", "codex-sess-1", "cursor-sess-1", "copilot-sess-1"):
        assert sid in listing
    assert "agy-child" not in listing and "oc-child" not in listing
    assert "codex-sess-1" in text_of(by_id[2]) and "claude-sess-1" not in text_of(by_id[2])

    assert "Audit the repo" in text_of(by_id[3])
    turns4 = text_of(by_id[4])
    assert "TOOL" not in turns4 and "Fix the login" in turns4
    assert "AIza" not in turns4  # secret hidden

    handoff = text_of(by_id[5])
    assert by_id[5]["result"]["isError"] is False
    assert "Extended Handoff" in handoff and "AIza" not in handoff
    assert "SESSION RESUME" in text_of(by_id[6]) and "### 7. Immediate plan" in text_of(by_id[6])
    assert "Migrate the project to Vite" in text_of(by_id[11])
    assert "charges and refunds" in text_of(by_id[12])

    for i in (7, 8, 9):
        assert by_id[i]["result"]["isError"] is True, i
    assert by_id[10]["error"]["code"] == -32602


def test_stdio_subprocess_keeps_stdout_clean(stores, tmp_path):
    """The real server over stdio: everything on stdout must be valid JSON-RPC,
    even when a library prints to stdout."""
    sitecustom = tmp_path / "noisy"
    sitecustom.mkdir()
    (sitecustom / "sitecustomize.py").write_text("import atexit\nprint('noise at startup')\natexit.register(lambda: print('noise at exit'))\n")
    env = dict(os.environ, PYTHONPATH=f"{sitecustom}{os.pathsep}{ROOT}")
    payload = "\n".join(json.dumps(m) for m in [
        rpc(1, "initialize", {"protocolVersion": LATEST_PROTOCOL_VERSION, "capabilities": {}}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        call(2, "list_cross_sessions", {}),
    ]) + "\n"
    proc = subprocess.run(
        [sys.executable, "-c", "print('import noise'); from cross_session_mcp.server import main; main()"],
        input=payload.encode(), capture_output=True, env=env, timeout=60,
    )
    lines = [line for line in proc.stdout.decode().splitlines() if line.strip()]
    # 'noise at startup' and 'import' come out before stdout is shielded; the rest must be JSON
    json_lines = [line for line in lines if line.startswith("{")]
    assert all(json.loads(line)["jsonrpc"] == "2.0" for line in json_lines)
    assert "noise at exit" not in proc.stdout.decode()
    ids = {json.loads(line)["id"] for line in json_lines}
    assert ids == {1, 2}
    assert b"noise at exit" in proc.stderr


def test_launcher_script(stores):
    proc = subprocess.run(
        [str(ROOT / "bin" / "cross-session-mcp-server")],
        input=(json.dumps(rpc(1, "initialize", {"protocolVersion": LATEST_PROTOCOL_VERSION})) + "\n").encode(),
        capture_output=True, timeout=60, cwd="/",
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.decode().splitlines()[0])["result"]["serverInfo"]["name"] == "cross-session-mcp"


def test_project_memory_tools(stores):
    ws = "/home/test/projects/my-app"
    by_id, _ = run_server([
        call(1, "get_project_memory", {"workspace_path": ws}),
        call(2, "update_project_memory", {"workspace_path": ws}),  # no Gemini key -> isError
        call(3, "suggest_agent_rules", {"workspace_path": ws}),
        call(4, "suggest_agent_rules", {"workspace_path": ws, "target": "other"}),
        call(5, "update_project_memory", {}),
        call(6, "update_project_memory", {"workspace_path": "relative"}),
        call(7, "update_project_memory", {"workspace_path": ws, "sessions": "x"}),
    ])
    assert by_id[1]["result"]["isError"] is False and "No memory saved" in text_of(by_id[1])
    assert by_id[2]["result"]["isError"] is True and "key" in text_of(by_id[2])
    assert by_id[3]["result"]["isError"] is False and "update_project_memory" in text_of(by_id[3])
    assert by_id[4]["error"]["code"] == -32602 and by_id[5]["error"]["code"] == -32602
    assert by_id[6]["result"]["isError"] is True
    assert by_id[7]["error"]["code"] == -32602




def test_get_session_turns_defaults_query_and_limits(stores):
    by_id, _ = run_server([
        call(1, "get_session_turns", {"session_id": "oc-sess-1"}),
        call(2, "get_session_turns", {"session_id": "oc-sess-1", "include_tools": True}),
        call(3, "get_session_turns", {"session_id": "oc-sess-1", "query": "REFACTOR"}),
        call(4, "get_session_turns", {"session_id": "oc-sess-1", "query": "nowhere to be found"}),
        call(5, "get_session_turns", {"session_id": "oc-sess-1", "max_turns": 500}),
        call(6, "list_cross_sessions", {"limit": 200}),
        call(7, "get_session_turns", {"session_id": "oc-sess-1", "max_turns": 1}),
    ])
    assert "TOOL" not in text_of(by_id[1]) and "3 passed" not in text_of(by_id[1])  # no outputs by default
    assert "3 passed" in text_of(by_id[2])
    assert "Refactor utils.py" in text_of(by_id[3]) and "(±1)" in text_of(by_id[3])
    assert "No turn" in text_of(by_id[4])
    assert by_id[5]["error"]["code"] == -32602 and "50" in by_id[5]["error"]["message"]
    assert by_id[6]["error"]["code"] == -32602 and "50" in by_id[6]["error"]["message"]
    assert "omitted:* 2" in text_of(by_id[7])


def test_handoff_by_position_and_parameter_errors(stores):
    from cross_session_mcp.service import CrossSessionService
    ws = "/home/test/projects/my-app"
    ordered = [s.session_id for s in CrossSessionService().list_sessions(workspace=ws, limit=50)
               if s.workspace_path == ws]
    by_id, _ = run_server([
        call(1, "inject_session_context", {"workspace_path": ws}),
        call(2, "generate_session_handoff", {"workspace_path": ws, "back": 2}),
        call(3, "generate_session_handoff", {}),
        call(4, "generate_session_handoff", {"workspace_path": ws, "back": 11}),
        call(5, "generate_session_handoff", {"workspace_path": ws, "back": 10}),
    ])
    assert len(ordered) >= 2
    assert f"`{ordered[0]}`" in text_of(by_id[1]) and "# Extended Handoff —" in text_of(by_id[1])
    assert ordered[1][:8] in text_of(by_id[2]) and ordered[0][:8] not in text_of(by_id[2])
    assert by_id[3]["error"]["code"] == -32602 and by_id[4]["error"]["code"] == -32602
    assert by_id[5]["result"]["isError"] is True and "previous session(s)" in text_of(by_id[5])


def test_search_project_memory_tool(stores):
    ws = "/home/test/projects/my-app"
    by_id, _ = run_server([
        call(1, "search_project_memory", {"topic": "docker", "workspace_path": ws}),
        call(2, "search_project_memory", {"workspace_path": ws}),
        call(3, "search_project_memory", {"topic": "docker", "workspace_path": ws, "limit": 99}),
    ])
    assert by_id[1]["result"]["isError"] is False and "No matches" in text_of(by_id[1])
    assert by_id[2]["error"]["code"] == -32602 and by_id[3]["error"]["code"] == -32602


def test_instructions_order_memory_handoff_search_turns():
    from cross_session_mcp.server import SERVER_INSTRUCTIONS as text
    order = [text.index(n) for n in ("get_project_memory", "inject_session_context", "search_project_memory",
                                     "get_session_turns")]
    assert order == sorted(order)
