"""Tests for the session readers against synthetic stores."""

import pytest

from cross_session_mcp.readers import get_all_readers, get_reader_by_name
from cross_session_mcp.readers.agy import AgySessionReader, parse_workspace_uris
from cross_session_mcp.readers.claude import ClaudeSessionReader, decode_project_dir
from cross_session_mcp.readers.codex import CodexSessionReader
from cross_session_mcp.readers.opencode import OpenCodeSessionReader
from tests.conftest import WORKSPACE


def test_registry():
    assert {r.name for r in get_all_readers()} == {"claude", "agy", "opencode", "codex"}
    assert isinstance(get_reader_by_name("claude"), ClaudeSessionReader)
    with pytest.raises(ValueError):
        get_reader_by_name("unknown")


def test_missing_stores_are_safe(tmp_path):
    readers = [
        ClaudeSessionReader(tmp_path / "x"),
        AgySessionReader(base_dirs=[tmp_path / "y"]),
        OpenCodeSessionReader(tmp_path / "z.db"),
        CodexSessionReader(codex_home=tmp_path / "w"),
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
