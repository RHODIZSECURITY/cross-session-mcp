"""Tests for the CLI `memory` subcommand."""

import pytest

from cross_session_mcp import cli
from cross_session_mcp.db import CrossSessionDB
from cross_session_mcp.service import CrossSessionService
from tests.conftest import WORKSPACE
from tests.test_service_memory import FakeAgent


@pytest.fixture
def patched(stores, tmp_path, monkeypatch):
    agent = FakeAgent()
    svc = CrossSessionService(db=CrossSessionDB(tmp_path / "cli.db"), agent=agent)
    monkeypatch.setattr(cli, "CrossSessionService", lambda: svc)
    return svc


def run_cli(*argv):
    cli.main(list(argv))


def test_memory_help_lists_all_actions(capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli("memory", "--help")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for action in ("show", "update", "export", "pin", "unpin", "forget", "history", "suggest-rules", "search"):
        assert action in out


def test_memory_flow(patched, capsys, tmp_path):
    run_cli("memory", "show", WORKSPACE)
    assert "No memory saved" in capsys.readouterr().out

    run_cli("memory", "update", WORKSPACE, "-n", "3")
    captured = capsys.readouterr()
    assert "# MEMORY" in captured.out and "3 sessions" in captured.out and "LLM calls" in captured.err

    out_file = tmp_path / "m.md"
    run_cli("memory", "export", WORKSPACE, "-o", str(out_file))
    assert out_file.read_text().startswith("# MEMORY")

    first_decision = next(e["id"] for e in patched.db.get_entries(WORKSPACE) if e["kind"] == "D")
    run_cli("memory", "pin", first_decision, WORKSPACE)
    assert "pin applied" in capsys.readouterr().out
    run_cli("memory", "forget", first_decision, WORKSPACE)
    run_cli("memory", "history", WORKSPACE)
    assert "v3" in capsys.readouterr().out
    run_cli("memory", "show", WORKSPACE, "--archived")
    assert first_decision in capsys.readouterr().out
    run_cli("memory", "suggest-rules", WORKSPACE, "--target", "agents")
    assert "A1" in capsys.readouterr().out


def test_memory_errors_exit_nonzero(patched, capsys):
    patched.agent.available = False
    with pytest.raises(SystemExit) as exc:
        run_cli("memory", "update", WORKSPACE)
    assert exc.value.code == 1 and "key" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        run_cli("memory", "pin", "D1", WORKSPACE)
    assert exc.value.code == 1


def test_doctor_reports_memory(patched, capsys):
    with pytest.raises(SystemExit):
        run_cli("doctor")
    out = capsys.readouterr().out
    assert "Project memory" in out and "0 project(s) with memory · schema v4" in out
    assert "Saved handoffs: 0 in the current format" in out and "Maintenance in progress: 0" in out


def test_memory_search_and_handoff_by_position(patched, capsys):
    run_cli("memory", "search", "docker", WORKSPACE)
    assert "No matches" in capsys.readouterr().out
    run_cli("memory", "update", WORKSPACE)
    capsys.readouterr()
    run_cli("memory", "search", "unique decision", WORKSPACE, "--limit", "1")
    out = capsys.readouterr().out
    assert "unique decision" in out and "more results" in out

    run_cli("handoff", "-w", WORKSPACE, "--back", "1")
    captured = capsys.readouterr()
    assert captured.out.startswith("# Extended Handoff —") and "local bank" in captured.err
    with pytest.raises(SystemExit) as exc:
        run_cli("handoff", "-w", WORKSPACE, "--back", "11")
    assert exc.value.code == 2  # argparse: out of range


def test_show_query_and_tools(patched, capsys):
    run_cli("show", "oc-sess-1", "--query", "refactor")
    out = capsys.readouterr().out
    assert "Refactor utils.py" in out and "3 passed" not in out
    run_cli("show", "oc-sess-1", "--tools")
    assert "3 passed" in capsys.readouterr().out


def test_doctor_with_empty_home_and_no_key(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for var in ("CROSS_SESSION_CLAUDE_DIR", "CROSS_SESSION_AGY_DIRS", "CROSS_SESSION_OPENCODE_DB",
                "CROSS_SESSION_CODEX_HOME", "CROSS_SESSION_CURSOR_DIR", "CROSS_SESSION_VSCODE_DIRS"):
        monkeypatch.delenv(var, raising=False)
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CROSS_SESSION_KEY_FILE", str(tmp_path / "missing"))
    svc = CrossSessionService(db=CrossSessionDB(tmp_path / "doctor.db"), agent=FakeAgent())
    with pytest.raises(SystemExit) as exc:
        cli.cmd_doctor(svc, None)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in ("claude", "agy", "opencode", "codex", "cursor", "copilot"):
        assert f"➖ {name:<9} not found" in out
    assert "No key" in out and "the project memory cannot be built" in out
