"""Fixtures with synthetic stores for every harness (the real HOME is never touched)."""

import json
import sqlite3
from pathlib import Path

import pytest

FAKE_KEY = "AIzaSyD" + "x" * 32
WORKSPACE = "/home/test/projects/my-app"


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


def make_claude_store(root: Path) -> Path:
    projects = root / "claude" / "projects"
    _write_jsonl(projects / "-home-test-projects-my-app" / "claude-sess-1.jsonl", [
        {"type": "user", "cwd": WORKSPACE, "timestamp": "2026-09-01T10:00:00Z",
         "message": {"role": "user", "content": f"Fix the login. My key is {FAKE_KEY}"}},
        {"type": "user", "isMeta": True, "timestamp": "2026-09-01T10:00:01Z",
         "message": {"role": "user", "content": "<internal meta>"}},
        {"type": "assistant", "timestamp": "2026-09-01T10:00:02Z",
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "pondering"},
             {"type": "text", "text": "I will check auth.py"},
             {"type": "tool_use", "name": "Read", "input": {"file_path": "auth.py"}},
         ]}},
        {"type": "user", "timestamp": "2026-09-01T10:00:03Z",
         "message": {"role": "user", "content": [{"type": "tool_result", "content": "def login(): ..."}]}},
        {"type": "ai-title", "aiTitle": "Fix login"},
    ])
    # Empty session (metadata only): must not be listed
    _write_jsonl(projects / "-home-test-other" / "claude-empty.jsonl", [{"type": "ai-title", "aiTitle": "empty"}])
    return projects


def make_agy_store(root: Path) -> Path:
    base = root / "agy"
    base.mkdir(parents=True)
    conn = sqlite3.connect(base / "conversation_summaries.db")
    conn.execute(
        "CREATE TABLE conversation_summaries (conversation_id text, title text, preview text, step_count integer,"
        " last_modified_time datetime, workspace_uris text, status text, parent_conversation_id text DEFAULT '')"
    )
    conn.execute(
        "INSERT INTO conversation_summaries VALUES (?,?,?,?,?,?,?,?)",
        ("agy-sess-1", "Audit", "Audit the repo", 5, "2026-09-02 12:00:00.123456789+00:00",
         json.dumps(["file:///home/test/projects/my-app"]), "CASCADE_RUN_STATUS_IDLE", ""),
    )
    conn.execute(
        "INSERT INTO conversation_summaries VALUES (?,?,?,?,?,?,?,?)",
        ("agy-child", "Subagent", "", 1, "2026-09-02 12:30:00+00:00", "[]", "", "agy-sess-1"),
    )
    conn.commit()
    conn.close()
    _write_jsonl(base / "brain" / "agy-sess-1" / ".system_generated" / "logs" / "transcript.jsonl", [
        {"type": "USER_INPUT", "created_at": "2026-09-02T11:00:00Z",
         "content": "<USER_REQUEST>\nAudit the repo\n</USER_REQUEST>\n<ADDITIONAL_METADATA>time</ADDITIONAL_METADATA>"},
        {"type": "PLANNER_RESPONSE", "created_at": "2026-09-02T11:00:01Z", "content": "",
         "tool_calls": [{"name": "run_command", "args": {"CommandLine": "ls -la"}}]},
        {"type": "GENERIC", "created_at": "2026-09-02T11:00:02Z", "content": "total 8"},
        {"type": "SYSTEM_MESSAGE", "created_at": "2026-09-02T11:00:03Z", "content": "noise"},
        {"type": "PLANNER_RESPONSE", "created_at": "2026-09-02T11:00:04Z", "content": "Audit complete"},
    ])
    return base


def make_opencode_store(root: Path) -> Path:
    db = root / "opencode.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE session (id text PRIMARY KEY, project_id text, parent_id text, directory text, title text,
                              time_created integer, time_updated integer, time_archived integer);
        CREATE TABLE message (id text PRIMARY KEY, session_id text, time_created integer, time_updated integer, data text);
        CREATE TABLE part (id text PRIMARY KEY, message_id text, session_id text, time_created integer,
                           time_updated integer, data text);
    """)
    conn.execute("INSERT INTO session VALUES ('oc-sess-1','p',NULL,?, 'Refactor', 1790000000000, 1790000500000, NULL)", (WORKSPACE,))
    conn.execute("INSERT INTO session VALUES ('oc-child','p','oc-sess-1',?, 'Sub', 1790000000000, 1790000600000, NULL)", (WORKSPACE,))
    conn.execute("INSERT INTO message VALUES ('m1','oc-sess-1',1790000001000,0,?)", (json.dumps({"role": "user"}),))
    conn.execute("INSERT INTO message VALUES ('m2','oc-sess-1',1790000002000,0,?)", (json.dumps({"role": "assistant"}),))
    parts = [
        ("p1", "m1", 1, {"type": "text", "text": "Refactor utils.py"}),
        ("p2", "m2", 2, {"type": "reasoning", "text": "thinking it over"}),
        ("p3", "m2", 3, {"type": "text", "text": "Starting"}),
        ("p4", "m2", 4, {"type": "tool", "tool": "bash", "state": {"input": {"command": "pytest"}, "output": "3 passed"}}),
        ("p5", "m2", 5, {"type": "text", "text": "Done"}),
    ]
    for pid, mid, t, data in parts:
        conn.execute("INSERT INTO part VALUES (?,?,?,?,0,?)", (pid, mid, "oc-sess-1", t, json.dumps(data)))
    conn.commit()
    conn.close()
    return db


def make_codex_store(root: Path) -> Path:
    home = root / "codex"
    rollout = home / "sessions" / "2026" / "09" / "03" / "rollout-2026-09-03-codex-sess-1.jsonl"
    _write_jsonl(rollout, [
        {"timestamp": "2026-09-03T09:00:00Z", "type": "session_meta", "payload": {"id": "codex-sess-1", "cwd": WORKSPACE}},
        {"timestamp": "2026-09-03T09:00:01Z", "type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]}},
        {"timestamp": "2026-09-03T09:00:02Z", "type": "response_item", "payload": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Add tests"}]}},
        {"timestamp": "2026-09-03T09:00:03Z", "type": "response_item", "payload": {"type": "reasoning", "summary": []}},
        {"timestamp": "2026-09-03T09:00:04Z", "type": "response_item", "payload": {
            "type": "function_call", "name": "shell", "arguments": "{\"command\":[\"pytest\"]}", "call_id": "c1"}},
        {"timestamp": "2026-09-03T09:00:05Z", "type": "response_item", "payload": {
            "type": "function_call_output", "call_id": "c1", "output": "{\"output\":\"5 passed\"}"}},
        {"timestamp": "2026-09-03T09:00:06Z", "type": "response_item", "payload": {
            "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Tests added"}]}},
    ])
    conn = sqlite3.connect(home / "state_5.sqlite")
    conn.execute(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT, created_at INTEGER, updated_at INTEGER, cwd TEXT,"
        " title TEXT, first_user_message TEXT DEFAULT '', preview TEXT DEFAULT '', archived INTEGER DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO threads VALUES ('codex-sess-1', ?, 1790200000, 1790200300, ?, 'Tests', 'Add tests', '', 0)",
        (str(rollout), WORKSPACE),
    )
    conn.commit()
    conn.close()
    return home


def make_cursor_store(root: Path) -> Path:
    user = root / "cursor" / "User"
    (user / "globalStorage").mkdir(parents=True)
    conn = sqlite3.connect(user / "globalStorage" / "state.vscdb")
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
    kv = {
        # Current format: headers + separate bubbles
        "composerData:cursor-sess-1": {
            "composerId": "cursor-sess-1", "name": "Migrate to Vite", "createdAt": 1790300000000,
            "lastUpdatedAt": 1790300900000,
            "fullConversationHeadersOnly": [{"bubbleId": "b1", "type": 1}, {"bubbleId": "b2", "type": 2},
                                            {"bubbleId": "b3", "type": 2}],
        },
        "bubbleId:cursor-sess-1:b1": {"type": 1, "text": "Migrate the project to Vite", "createdAt": "2026-09-04T10:00:00Z"},
        "bubbleId:cursor-sess-1:b2": {"type": 2, "text": "", "toolFormerData": {
            "name": "run_terminal_cmd", "rawArgs": "{\"command\": \"npm i vite\"}", "result": "added 1 package"}},
        "bubbleId:cursor-sess-1:b3": {"type": 2, "text": "Migration complete"},
        # Legacy format: embedded conversation
        "composerData:cursor-old": {
            "composerId": "cursor-old", "createdAt": 1780000000000,
            "conversation": [{"bubbleId": "x1", "type": 1, "text": "Old question"},
                             {"bubbleId": "x2", "type": 2, "text": "Old answer"}],
        },
        # Empty conversation: must not be listed
        "composerData:cursor-empty": {"composerId": "cursor-empty", "fullConversationHeadersOnly": [], "createdAt": 1790300000000},
    }
    for k, v in kv.items():
        conn.execute("INSERT INTO cursorDiskKV VALUES (?, ?)", (k, json.dumps(v)))
    conn.commit()
    conn.close()

    ws = user / "workspaceStorage" / "abc123"
    ws.mkdir(parents=True)
    (ws / "workspace.json").write_text(json.dumps({"folder": "file:///home/test/projects/my-app"}))
    conn = sqlite3.connect(ws / "state.vscdb")
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
    conn.execute("INSERT INTO ItemTable VALUES ('composer.composerData', ?)",
                 (json.dumps({"allComposers": [{"composerId": "cursor-sess-1"}]}),))
    conn.commit()
    conn.close()
    return user


def make_copilot_store(root: Path) -> Path:
    user = root / "vscode" / "User"
    ws = user / "workspaceStorage" / "hash1"
    (ws / "chatSessions").mkdir(parents=True)
    (ws / "workspace.json").write_text(json.dumps({"folder": "file:///home/test/projects/my-app"}))
    request = {
        "requestId": "r1", "timestamp": 1790400000000,
        "message": {"text": "Explain the payments module"},
        "response": [
            {"kind": "thinking", "value": "thinking it over"},
            {"kind": "toolInvocationSerialized", "toolId": "copilot_readFile",
             "pastTenseMessage": {"value": "Read payments.py"}},
            {"value": "The module ", "baseUri": {"scheme": "file", "path": "/home/test/projects/my-app/"}},
        ],
    }
    ops = [
        {"kind": 0, "v": {"version": 3, "sessionId": "copilot-sess-1", "creationDate": 1790399000000, "requests": []}},
        {"kind": 2, "k": ["requests"], "v": [request]},
        {"kind": 2, "k": ["requests", 0, "response"], "v": [{"value": "handles charges."}]},
        {"kind": 2, "k": ["requests", 0, "response"], "i": 3, "v": [{"value": "handles charges and refunds."}]},
        {"kind": 1, "k": ["customTitle"], "v": "Payments module"},
    ]
    _write_jsonl(ws / "chatSessions" / "copilot-sess-1.jsonl", ops)
    # Session without requests: must not be listed
    _write_jsonl(ws / "chatSessions" / "copilot-empty.jsonl", [{"kind": 0, "v": {"sessionId": "copilot-empty", "requests": []}}])
    # Legacy .json format in a window without a folder
    empty = user / "globalStorage" / "emptyWindowChatSessions"
    empty.mkdir(parents=True)
    (empty / "copilot-old.json").write_text(json.dumps({
        "sessionId": "copilot-old", "creationDate": 1780000000000,
        "requests": [{"timestamp": 1780000001000, "message": {"text": "Hello"}, "response": [{"value": "Hello, how can I help?"}]}],
    }))
    return user


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """Creates the synthetic stores and points the readers at them through environment variables."""
    claude = make_claude_store(tmp_path)
    agy = make_agy_store(tmp_path)
    opencode = make_opencode_store(tmp_path)
    codex = make_codex_store(tmp_path)
    cursor = make_cursor_store(tmp_path)
    copilot = make_copilot_store(tmp_path)
    monkeypatch.setenv("CROSS_SESSION_CLAUDE_DIR", str(claude))
    monkeypatch.setenv("CROSS_SESSION_AGY_DIRS", str(agy))
    monkeypatch.setenv("CROSS_SESSION_OPENCODE_DB", str(opencode))
    monkeypatch.setenv("CROSS_SESSION_CODEX_HOME", str(codex))
    monkeypatch.setenv("CROSS_SESSION_CURSOR_DIR", str(cursor))
    monkeypatch.setenv("CROSS_SESSION_VSCODE_DIRS", str(copilot))
    monkeypatch.setenv("CROSS_SESSION_DB", str(tmp_path / "index.db"))
    monkeypatch.setenv("CROSS_SESSION_KEY_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(tmp_path / "missing"))
    monkeypatch.delenv("CROSS_SESSION_GOOGLE_PROJECT", raising=False)
    monkeypatch.setenv("CROSS_SESSION_DISABLE_SDK", "1")
    monkeypatch.delenv("CROSS_SESSION_LANGUAGE", raising=False)
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return {"claude": claude, "agy": agy, "opencode": opencode, "codex": codex,
            "cursor": cursor, "copilot": copilot, "root": tmp_path}
