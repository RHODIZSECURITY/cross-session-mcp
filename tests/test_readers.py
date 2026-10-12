"""Tests for the session readers against synthetic stores."""

import json
import shutil
import sys

import pytest

from cross_session_mcp.readers import get_all_readers, get_reader_by_name
from cross_session_mcp.readers.agy import AgySessionReader, parse_workspace_uris
from cross_session_mcp.readers.claude import ClaudeSessionReader, decode_project_dir
from cross_session_mcp.readers.codex import CodexSessionReader
from cross_session_mcp.readers.copilot import CopilotSessionReader, replay_operations
from cross_session_mcp.readers.cursor import CursorSessionReader
from cross_session_mcp.readers.opencode import OpenCodeSessionReader
from tests.conftest import WORKSPACE, make_copilot_store, make_cursor_store

_STORE_VARS = ("CROSS_SESSION_CLAUDE_DIR", "CROSS_SESSION_AGY_DIRS", "CROSS_SESSION_OPENCODE_DB",
               "CROSS_SESSION_CODEX_HOME", "CROSS_SESSION_CURSOR_DIR", "CROSS_SESSION_VSCODE_DIRS")


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """An empty HOME with none of the store overrides set."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for var in _STORE_VARS:
        monkeypatch.delenv(var, raising=False)
    return home


def test_registry():
    assert {r.name for r in get_all_readers()} == {"claude", "agy", "opencode", "codex", "cursor", "copilot"}
    assert isinstance(get_reader_by_name("claude"), ClaudeSessionReader)
    with pytest.raises(ValueError):
        get_reader_by_name("unknown")


def test_missing_stores_are_safe(tmp_path):
    readers = [
        ClaudeSessionReader(tmp_path / "x"),
        AgySessionReader(base_dirs=[tmp_path / "y"]),
        OpenCodeSessionReader(tmp_path / "z.db"),
        CodexSessionReader(codex_home=tmp_path / "w"),
        CursorSessionReader(tmp_path / "v"),
        CopilotSessionReader([tmp_path / "u"]),
    ]
    for r in readers:
        assert r.can_read() is False
        assert r.list_sessions() == []
        assert r.get_session_detail("abc") is None


def test_claude(stores):
    r = ClaudeSessionReader()
    sessions = r.list_sessions()
    assert [s.session_id for s in sessions] == ["claude-sess-1"]  # the empty one is skipped
    s = sessions[0]
    assert s.title == "Fix login"
    assert s.workspace_path == WORKSPACE
    assert s.updated_at.endswith("+00:00")

    d = r.get_session_detail("claude-sess-1", max_turns=None)
    roles = [t.role for t in d.turns]
    assert roles == ["user", "assistant", "tool", "tool"]  # no isMeta and no thinking
    assert "Read" in d.turns[2].content and "auth.py" in d.turns[2].content
    assert d.turns[3].content == "def login(): ..."
    assert r.get_session_detail("../../etc/passwd") is None
    assert r.list_sessions(workspace_filter="/home/test/other") == []


def test_decode_project_dir(tmp_path):
    (tmp_path / "my-app" / "sub").mkdir(parents=True)
    encoded = str(tmp_path).replace("/", "-") + "-my-app-sub"
    assert decode_project_dir(encoded) == str(tmp_path / "my-app" / "sub")


def test_agy(stores):
    r = AgySessionReader()
    sessions = r.list_sessions()
    assert [s.session_id for s in sessions] == ["agy-sess-1"]  # subagent left out
    s = sessions[0]
    assert s.workspace_path == WORKSPACE
    assert s.updated_at == "2026-09-02T12:00:00.123456+00:00"
    assert s.status == "idle"

    d = r.get_session_detail("agy-sess-1", max_turns=None)
    assert [t.role for t in d.turns] == ["user", "tool", "tool", "assistant"]
    assert d.turns[0].content == "Audit the repo"
    assert "run_command" in d.turns[1].content


@pytest.mark.parametrize("raw,expected", [
    ('["file:///a/b%20c"]', "/a/b c"),
    ("file:///a/b,file:///c", "/a/b"),
    ("", ""),
    ("[]", ""),
])
def test_parse_workspace_uris(raw, expected):
    assert parse_workspace_uris(raw) == expected


def test_opencode(stores):
    r = OpenCodeSessionReader()
    sessions = r.list_sessions()
    assert [s.session_id for s in sessions] == ["oc-sess-1"]
    assert sessions[0].last_prompt == "Refactor utils.py"
    assert sessions[0].updated_at.startswith("2026-")

    d = r.get_session_detail("oc-sess-1", max_turns=None)
    assert [t.role for t in d.turns] == ["user", "assistant", "tool", "assistant"]
    assert "pytest" in d.turns[2].content and "3 passed" in d.turns[2].content


def test_codex(stores):
    r = CodexSessionReader()
    sessions = r.list_sessions()
    assert len(sessions) == 1
    s = sessions[0]
    assert s.workspace_path == WORKSPACE
    assert s.updated_at.startswith("2026-")  # epoch -> ISO UTC

    d = r.get_session_detail("codex-sess-1", max_turns=None)
    assert [t.role for t in d.turns] == ["user", "tool", "tool", "assistant"]  # no environment_context
    assert d.turns[0].content == "Add tests"
    assert d.turns[2].content == "5 passed"


def test_max_turns_tail(stores):
    d = OpenCodeSessionReader().get_session_detail("oc-sess-1", max_turns=2)
    assert [t.content for t in d.turns][-1] == "Done"
    assert len(d.turns) == 2


def test_excluded_workspaces(stores, monkeypatch):
    monkeypatch.setenv("CROSS_SESSION_EXCLUDE_WORKSPACES", "/home/test/projects")
    for reader in get_all_readers():
        # Only sessions without a workspace remain (empty Cursor/VS Code windows)
        assert all(s.workspace_path == "" for s in reader.list_sessions()), reader.name
    monkeypatch.setenv("CROSS_SESSION_EXCLUDE_WORKSPACES", "")
    assert all(reader.list_sessions() for reader in get_all_readers())


def test_cursor(stores):
    r = CursorSessionReader()
    sessions = r.list_sessions()
    assert [s.session_id for s in sessions] == ["cursor-sess-1", "cursor-old"]  # empty one left out, by date
    s = sessions[0]
    assert s.title == "Migrate to Vite"
    assert s.workspace_path == WORKSPACE
    assert s.last_prompt == "Migrate the project to Vite"
    assert s.updated_at.startswith("2026-")
    assert sessions[1].title == "Old question" and sessions[1].workspace_path == ""

    d = r.get_session_detail("cursor-sess-1", max_turns=None)
    assert [t.role for t in d.turns] == ["user", "tool", "assistant"]
    assert "run_terminal_cmd" in d.turns[1].content and "added 1 package" in d.turns[1].content
    assert [t.role for t in r.get_session_detail("cursor-old", max_turns=None).turns] == ["user", "assistant"]
    assert r.get_session_detail("missing") is None
    assert r.list_sessions(workspace_filter="/other") == []


def test_copilot(stores):
    r = CopilotSessionReader()
    sessions = r.list_sessions()
    assert [s.session_id for s in sessions] == ["copilot-sess-1", "copilot-old"]
    s = sessions[0]
    assert s.title == "Payments module"
    assert s.workspace_path == WORKSPACE
    assert s.last_prompt == "Explain the payments module"
    assert sessions[1].title == "Hello" and sessions[1].workspace_path == ""

    d = r.get_session_detail("copilot-sess-1", max_turns=None)
    assert [t.role for t in d.turns] == ["user", "tool", "assistant"]
    assert "copilot_readFile" in d.turns[1].content
    assert d.turns[2].content == "The module handles charges and refunds."  # the op with 'i' replaces the tail


def test_copilot_multi_root_workspace_resolution(tmp_path):
    user = tmp_path / "User"
    ws = user / "workspaceStorage" / "h"
    (ws / "chatSessions").mkdir(parents=True)
    ws_file = tmp_path / "proj.code-workspace"
    ws_file.write_text(json.dumps({"folders": [{"path": "repo"}]}))
    (ws / "workspace.json").write_text(json.dumps({"workspace": ws_file.as_uri()}))
    (ws / "chatSessions" / "s.json").write_text(json.dumps(
        {"sessionId": "s", "requests": [{"message": {"text": "hello"}, "response": []}]}))
    assert CopilotSessionReader([user]).list_sessions()[0].workspace_path == str(tmp_path / "repo")


def test_replay_operations():
    state = replay_operations([
        {"kind": 0, "v": {"a": {"list": [1, 2, 3]}}},
        {"kind": 2, "k": ["a", "list"], "v": [4]},
        {"kind": 2, "k": ["a", "list"], "i": 1, "v": [9]},
        {"kind": 1, "k": ["a", "x"], "v": "y"},
        {"kind": 3, "k": ["a", "x"]},
        {"kind": 1, "k": ["does", "not-exist"], "v": 1},  # ignored without breaking
        {"kind": 2, "k": ["new"], "v": [1]},
    ])
    assert state == {"a": {"list": [1, 9]}, "new": [1]}


@pytest.mark.parametrize("platform, config", [("darwin", "Library/Application Support"), ("linux", ".config")])
def test_cursor_and_copilot_default_paths(fake_home, tmp_path, monkeypatch, platform, config):
    monkeypatch.setattr(sys, "platform", platform)
    shutil.move(make_cursor_store(tmp_path / "src"), fake_home / config / "Cursor" / "User")
    shutil.move(make_copilot_store(tmp_path / "src"), fake_home / config / "Code" / "User")

    cursor = CursorSessionReader().list_sessions()
    assert [s.session_id for s in cursor] == ["cursor-sess-1", "cursor-old"]
    assert cursor[0].workspace_path == WORKSPACE
    assert [s.session_id for s in CopilotSessionReader().list_sessions()] == ["copilot-sess-1", "copilot-old"]


def test_store_overrides_win_on_macos(fake_home, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    cursor_user = make_cursor_store(tmp_path / "custom")
    copilot_user = make_copilot_store(tmp_path / "custom")
    shutil.copytree(cursor_user, fake_home / "Library" / "Application Support" / "Cursor" / "User")
    monkeypatch.setenv("CROSS_SESSION_CURSOR_DIR", str(cursor_user))
    monkeypatch.setenv("CROSS_SESSION_VSCODE_DIRS", str(copilot_user))
    assert CursorSessionReader().user_dir == cursor_user
    assert len(CopilotSessionReader().list_sessions()) == 2


def test_empty_home_lists_nothing(fake_home):
    for r in get_all_readers():
        assert r.can_read() is False
        assert r.list_sessions() == []
