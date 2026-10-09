"""End-to-end test against the REAL stores of this machine and, with a key, the Gemini API.

Not run by default. Launch with:  pytest -m live -s
Uses a temporary database: ~/.ai/cross_sessions.db is left untouched.
"""

from pathlib import Path

import pytest

from cross_session_mcp.agent import get_gemini_api_key, has_handoff_sections
from cross_session_mcp.db import CrossSessionDB
from cross_session_mcp.server import MCPServer
from cross_session_mcp.service import CrossSessionService

pytestmark = pytest.mark.live


@pytest.fixture
def service(tmp_path):
    return CrossSessionService(db=CrossSessionDB(tmp_path / "live.db"))


def test_live_readers_and_index(service):
    count = service.sync(force=True)
    assert count > 0, "At least one session must exist in some local harness."
    sessions = service.list_sessions(limit=10)
    harnesses = {s.harness for s in sessions}
    print(f"\nIndexed sessions: {count} · harnesses in the top 10: {sorted(harnesses)}")
    for s in sessions:
        assert s.updated_at and s.updated_at.endswith("+00:00")
        detail = service.get_detail(s.session_id, s.harness, max_turns=5)
        assert detail.summary.session_id == s.session_id


def test_live_handoff(service):
    service.sync(force=True)
    target = service.list_sessions(limit=1)[0]
    result = service.get_handoff(target.session_id, target.harness, force_refresh=True)
    print(f"\nHandoff of {target.harness}:{target.session_id} with {result.model_used}")
    assert has_handoff_sections(result.markdown_handoff)
    if get_gemini_api_key():
        assert not result.model_used.startswith("offline"), "There is a key but Gemini did not answer (see stderr)."
        cached = service.get_handoff(target.session_id, target.harness)
        assert cached.from_cache or cached.source_updated_at != result.source_updated_at


def test_live_mcp_tools(service):
    server = MCPServer(service=service)
    assert "Recent Sessions" in server.handle_tool_call("list_cross_sessions", {"limit": 3, "refresh": True})


def test_live_bank_layer(service):
    """Resume by position, background maintenance and search over this very repository."""
    if not get_gemini_api_key():
        pytest.skip("No Gemini key")
    from cross_session_mcp.server import cap_output, max_output_chars
    workspace = str(Path(__file__).resolve().parent.parent)
    server = MCPServer(service=service)  # turns on current-session detection and the background
    service.sync(force=True)
    previous = service.recent_sessions(workspace, 10)
    if not previous:
        pytest.skip("This repository has no indexed sessions")

    def tool(name, args):
        text = cap_output(server.handle_tool_call(name, args), name)
        assert len(text) <= max_output_chars()
        return text

    resumed = tool("inject_session_context", {"workspace_path": workspace})
    assert f"`{previous[0].session_id}`" in resumed and has_handoff_sections(resumed)

    first = tool("get_project_memory", {"workspace_path": workspace})
    print(f"\nFirst read:\n{first[:300]}")
    service.wait_for_maintenance(workspace, timeout=600)
    memory_md = tool("get_project_memory", {"workspace_path": workspace})
    assert "# MEMORY" in memory_md
    assert "AQ." not in memory_md and "sk-" not in memory_md

    entries = service.db.get_entries(workspace)
    word = max((w for e in entries for w in e["text"].split() if len(w) > 6), key=len)
    found = tool("search_project_memory", {"topic": word, "workspace_path": workspace})
    print(f"\nSearch \"{word}\":\n{found[:600]}")
    assert "match(es)" in found
    archived = [e for e in entries if e["status"] == "archived"]
    if archived:
        term = max(archived[0]["text"].split(), key=len)
        assert "(archived" in tool("search_project_memory", {"topic": term, "workspace_path": workspace, "limit": 50})


def test_live_project_memory(service):
    """Updates the memory of a real workspace with Gemini and checks the cap and the absence of secrets."""
    if not get_gemini_api_key():
        pytest.skip("No Gemini key")
    from cross_session_mcp import memory

    service.sync(force=True)
    counts = {}
    for s in service.list_sessions(limit=200):
        if s.workspace_path:
            counts[s.workspace_path] = counts.get(s.workspace_path, 0) + 1
    workspace = max(counts, key=counts.get)
    result = service.update_project_memory(workspace, sessions=3)
    md = service.get_project_memory(workspace)
    print(f"\nMemory of {workspace}: v{result.version}, {result.llm_calls} calls, {len(md.splitlines())} lines\n{md}")
    assert len(result.markdown.splitlines()) <= memory.max_lines()
    assert "AQ." not in md and "sk-" not in md
    again = service.update_project_memory(workspace, sessions=3)
    assert again.up_to_date and again.llm_calls == 0
