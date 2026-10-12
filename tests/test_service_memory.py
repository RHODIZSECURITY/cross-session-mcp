"""Tests for the project memory orchestration with a fake agent (no network)."""

import os
import stat

import pytest

from cross_session_mcp import memory
from cross_session_mcp.db import CrossSessionDB
from cross_session_mcp.service import CrossSessionService, SessionError
from tests.conftest import WORKSPACE


class FakeAgent:
    """Deterministic agent that counts its calls."""

    def __init__(self):
        self.available = True
        self.digest_calls = []
        self.consolidation_calls = 0
        self.fail_digest_for = set()
        self.fail_consolidation = False
        self.on_consolidate = None
        self.extra_entries = []

    def generate_digest(self, detail, focus_area=""):
        sid = detail.summary.session_id
        self.digest_calls.append(sid)
        if sid in self.fail_digest_for:
            return None
        return {"handoff_markdown": f"# handoff {sid}", "relevant": True, "relevance_reason": "x",
                "summary_line": f"summary of {sid}", "decisions": [{"text": f"decision of {sid}", "why": "because"}],
                "pitfalls": [], "commands": [], "pending": [], "agents_md_candidates": [], "suggested_skills": [],
                "model_used": "fake-model"}

    def consolidate_memory(self, entries, digests, ref_by_key, workspace, window_size):
        self.consolidation_calls += 1
        if self.on_consolidate:
            self.on_consolidate()
        if self.fail_consolidation:
            return None
        out = [{"id": None, "kind": "D", "text": f"unique decision of {d['ref']}", "why": "because",
                "evidence_sessions": [d["ref"]]} for d in digests]
        out.append({"id": None, "kind": "A", "text": "Answer in formal English", "scope": "universal",
                    "evidence_sessions": [d["ref"] for d in digests]})
        out += self.extra_entries
        for e in entries:  # keeps the existing ones (the LLM returns them with their ID)
            out.append({"id": e["id"], "kind": e["kind"], "text": e["text"], "evidence_sessions": []})
        return {"state": ["test state"], "entries": out, "archive_ids": [], "model_used": "fake-model"}

    @property
    def calls(self):
        return len(self.digest_calls) + self.consolidation_calls


@pytest.fixture
def svc(stores, tmp_path):
    agent = FakeAgent()
    s = CrossSessionService(db=CrossSessionDB(tmp_path / "mem.db"), agent=agent)
    s.sync_ttl = 10_000
    s.sync(force=True)
    return s


def bump(svc, session_id, when="2099-01-01T00:00:00.000000+00:00"):
    import sqlite3
    conn = sqlite3.connect(svc.db.db_path)
    conn.execute("UPDATE cross_sessions SET updated_at = ? WHERE session_id = ?", (when, session_id))
    conn.commit()
    conn.close()


def test_first_incremental_and_unchanged_updates(svc):
    r1 = svc.update_project_memory(WORKSPACE)
    assert svc.agent.calls == 7 and r1.version == 1 and r1.llm_calls == 7  # 6 sessions + 1 consolidation
    assert "6 sessions" in r1.markdown and "fake-model" in r1.markdown

    svc.agent.digest_calls.clear()
    svc.agent.consolidation_calls = 0
    r2 = svc.update_project_memory(WORKSPACE)
    assert svc.agent.calls == 0 and r2.up_to_date and r2.version == 1  # no changes: 0 calls

    bump(svc, "codex-sess-1")
    svc.update_project_memory(WORKSPACE)
    assert svc.agent.digest_calls == ["codex-sess-1"] and svc.agent.consolidation_calls == 1  # 1 + 1
    assert svc.db.get_memory(WORKSPACE)["version"] == 2

    svc.agent.digest_calls.clear()
    svc.agent.consolidation_calls = 0
    svc.update_project_memory(WORKSPACE, force_refresh=True)
    assert svc.agent.calls == 7


def test_window_limit_and_exact_workspace(svc):
    import sqlite3
    conn = sqlite3.connect(svc.db.db_path)
    conn.execute("INSERT INTO cross_sessions (session_id, harness, workspace_path, title, updated_at) "
                 "VALUES ('sub-1','claude',?, 't', '2099-01-01T00:00:00.000000+00:00')", (WORKSPACE + "/sub",))
    conn.commit()
    conn.close()
    r = svc.update_project_memory(WORKSPACE, sessions=3)
    assert "3 sessions" in r.markdown and "sub-1" not in r.markdown and svc.agent.digest_calls.__len__() == 3
    with pytest.raises(SessionError, match="No sessions indexed"):
        svc.update_project_memory("/does/not/exist")
    with pytest.raises(SessionError, match="absolute"):
        svc.update_project_memory("relative")


def test_no_key_leaves_memory_untouched(svc):
    svc.update_project_memory(WORKSPACE)
    before = svc.db.get_memory(WORKSPACE)
    bump(svc, "codex-sess-1")
    svc.agent.available = False
    with pytest.raises(SessionError, match="credential"):
        svc.update_project_memory(WORKSPACE)
    assert svc.db.get_memory(WORKSPACE)["version"] == before["version"]
    assert svc.db.get_memory(WORKSPACE)["rendered_md"] == before["rendered_md"]


def test_llm_failures_keep_memory_and_good_digests(svc):
    svc.agent.fail_digest_for = {"codex-sess-1"}
    with pytest.raises(SessionError, match="codex:codex-se"):
        svc.update_project_memory(WORKSPACE)
    assert svc.db.get_memory(WORKSPACE) is None
    assert svc.db.get_digest("claude-sess-1", "claude") is not None  # the good ones are kept
    assert svc.db.get_digest("codex-sess-1", "codex") is None

    svc.agent.fail_digest_for = set()
    svc.agent.digest_calls.clear()
    svc.agent.fail_consolidation = True
    with pytest.raises(SessionError, match="consolidation"):
        svc.update_project_memory(WORKSPACE)
    assert svc.agent.digest_calls == ["codex-sess-1"]  # only regenerates what was missing
    assert svc.db.get_memory(WORKSPACE) is None

    svc.agent.fail_consolidation = False
    svc.update_project_memory(WORKSPACE)
    assert svc.db.get_memory(WORKSPACE)["version"] == 1


def test_concurrent_update_retries_once_then_fails(svc):
    svc.update_project_memory(WORKSPACE)
    bump(svc, "codex-sess-1")
    other = CrossSessionDB(svc.db.db_path)

    def interfere():  # another instance saves a version while we consolidate
        cur = other.get_memory(WORKSPACE)
        other.save_memory_version(WORKSPACE, cur["version"], "# other", cur["window"], [], "other", other.get_entries(WORKSPACE))

    calls = {"n": 0}

    def once():
        calls["n"] += 1
        if calls["n"] == 1:
            interfere()

    svc.agent.on_consolidate = once
    r = svc.update_project_memory(WORKSPACE)
    assert r.version == 3 and any("retried" in w for w in r.warnings) and svc.agent.consolidation_calls == 3

    bump(svc, "codex-sess-1", "2099-02-01T00:00:00.000000+00:00")
    svc.agent.on_consolidate = interfere  # always interferes: the retry fails too
    with pytest.raises(SessionError, match="Another update"):
        svc.update_project_memory(WORKSPACE)


def test_get_project_memory_warns_without_calling_llm(svc):
    assert "No memory saved" in svc.get_project_memory(WORKSPACE) and "update_project_memory" in svc.get_project_memory(WORKSPACE)
    svc.update_project_memory(WORKSPACE)
    svc.agent.digest_calls.clear()
    svc.agent.consolidation_calls = 0
    assert "new sessions" not in svc.get_project_memory(WORKSPACE)
    bump(svc, "codex-sess-1")
    bump(svc, "agy-sess-1")
    text = svc.get_project_memory(WORKSPACE)
    assert "2 new sessions since the last update" in text and "# MEMORY" in text
    assert svc.agent.calls == 0


def test_pin_unpin_forget_create_versions_and_forget_sticks(svc):
    svc.update_project_memory(WORKSPACE)
    entries = {e["id"]: e for e in svc.db.get_entries(WORKSPACE)}
    target = next(e for e in entries.values() if e["kind"] == "D")
    msg = svc.edit_memory_entry(WORKSPACE, target["id"].lower(), "pin")
    assert "memory v2" in msg and "📌" in svc.get_project_memory(WORKSPACE)
    svc.edit_memory_entry(WORKSPACE, target["id"], "unpin")
    svc.edit_memory_entry(WORKSPACE, target["id"], "forget")
    assert f"**[{target['id']}]**" not in svc.get_project_memory(WORKSPACE)
    assert [v["version"] for v in svc.db.list_memory_versions(WORKSPACE)] == [4, 3, 2, 1]

    svc.agent.extra_entries = [{"id": None, "kind": "D", "text": target["text"], "evidence_sessions": ["claude:claude-se"]}]
    bump(svc, "codex-sess-1")
    svc.update_project_memory(WORKSPACE)
    texts = [e["text"] for e in svc.db.get_entries(WORKSPACE) if e["status"] != "forgotten"]
    assert target["text"] not in texts  # is not added again
    with pytest.raises(SessionError, match="no entry"):
        svc.edit_memory_entry(WORKSPACE, "D99", "pin")
    with pytest.raises(SessionError, match="forgotten"):
        svc.edit_memory_entry(WORKSPACE, target["id"], "pin")


def test_history_lists_versions_and_archived(svc):
    svc.update_project_memory(WORKSPACE)
    bump(svc, "codex-sess-1")
    svc.update_project_memory(WORKSPACE)
    hist = svc.memory_history(WORKSPACE)
    assert "v2" in hist and "v1" in hist and "fake-model" in hist and "6 sessions" in hist
    assert "No archived entries" in svc.list_archived(WORKSPACE) or "archived" in svc.list_archived(WORKSPACE)


def test_export_is_explicit_mode_600_and_same_content(svc, tmp_path):
    target = tmp_path / "out" / "MEMORY.md"
    r = svc.update_project_memory(WORKSPACE, export_path=str(target))
    assert r.exported_to == str(target) and stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert target.read_text() == svc.get_project_memory(WORKSPACE)
    other = tmp_path / "again.md"
    target.chmod(0o644)
    svc.export_project_memory(WORKSPACE, str(other))
    assert other.read_text() == target.read_text() and stat.S_IMODE(os.stat(other).st_mode) == 0o600
    # nothing is written without an explicit path
    assert not (tmp_path / "MEMORY.md").exists()


def test_redaction_of_llm_entries_is_persisted(svc):
    svc.agent.extra_entries = [{"id": None, "kind": "D", "text": "token ghp_" + "a" * 36 + " mail a@b.com",
                                "evidence_sessions": ["claude:claude-se"]}]
    svc.update_project_memory(WORKSPACE)
    stored = " ".join(e["text"] for e in svc.db.get_entries(WORKSPACE))
    assert "ghp_" not in stored and "a@b.com" not in stored and "[REDACTED]" in stored
    assert "ghp_" not in svc.get_project_memory(WORKSPACE)


# ---------------------------------------------------------------- suggest_agent_rules
@pytest.fixture
def rules_ws(tmp_path):
    db = CrossSessionDB(tmp_path / "r.db")
    ws = tmp_path / "proj"
    ws.mkdir()
    window = [{"key": f"claude:s{i}", "harness": "claude", "session_id": f"s{i}", "updated_at": f"2026-09-0{i + 1}",
               "relevant": True, "summary_line": "x"} for i in range(8)]
    keys = [w["key"] for w in window]

    def entry(eid, text, scope, n):
        return dict(id=eid, kind=eid[0], text=text, why="", scope=scope, evidence=keys[:n], status="active",
                    pinned=False, text_hash=memory.text_hash(text), first_seen="x", last_seen="x", updated_at="x")

    db.save_memory_version(str(ws), 0, "# M", window, [], "m",
                           [entry("A1", "Always answer in formal English", "universal", 5),
                            entry("A2", "Run the handoff skill when closing", "claude", 3)])
    return CrossSessionService(db=db, agent=FakeAgent()), ws


def test_suggest_rules_scenarios(rules_ws):
    svc, ws = rules_ws
    text = svc.suggest_agent_rules(str(ws))
    assert "| A1 |" in text and "5/8" in text and "AGENTS.md" in text
    assert "| A2 |" in text and "CLAUDE.md" in text
    only_claude = svc.suggest_agent_rules(str(ws), "claude")
    assert "| A2 |" in only_claude and "| A1 |" not in only_claude

    (ws / "AGENTS.md").write_text("- Always answer in formal English\n")
    before = sorted(p.name for p in ws.iterdir())
    present = svc.suggest_agent_rules(str(ws))
    assert "already in AGENTS.md" in present
    assert "To paste into `AGENTS.md`" not in present  # A1 left out of the block and nothing else for AGENTS.md
    assert sorted(p.name for p in ws.iterdir()) == before  # read-only
    assert svc.agent.calls == 0


def test_suggest_rules_without_memory(svc):
    text = svc.suggest_agent_rules(WORKSPACE)
    assert "No memory saved" in text and "update_project_memory" in text and svc.agent.calls == 0
    with pytest.raises(SessionError):
        svc.suggest_agent_rules(WORKSPACE, "other")
