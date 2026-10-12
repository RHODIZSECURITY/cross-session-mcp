"""Memory bank layer: resume by position, current session, single Extended Handoff, ephemeral focus,
background maintenance and topic search (fake agent, no network)."""

import sqlite3
import threading
import time

import pytest

from cross_session_mcp import memory
from cross_session_mcp.db import CrossSessionDB
from cross_session_mcp.models import SessionTurn
from cross_session_mcp.service import IDLE_BEFORE_REFRESH, CrossSessionService, SessionError, select_turns
from tests.conftest import WORKSPACE
from tests.test_service_memory import FakeAgent


class BankAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.focus_calls = []
        self.synth_calls = []
        self.digest_gate = None  # threading.Event to pause generation

    def generate_digest(self, detail, focus_area=""):
        if self.digest_gate is not None:
            self.digest_gate.wait(5)
        return super().generate_digest(detail)

    def generate_focused_handoff(self, detail, focus):
        self.focus_calls.append((detail.summary.session_id, focus))
        return f"### 1. Focus {focus} of {detail.summary.session_id}", "fake-model"

    def synthesize_topic(self, topic, results_text):
        self.synth_calls.append(results_text)
        return "### Recurring patterns\n- pattern", "fake-model"


@pytest.fixture
def bank(stores, tmp_path):
    svc = CrossSessionService(db=CrossSessionDB(tmp_path / "bank.db"), agent=BankAgent())
    svc.sync_ttl = 10_000
    svc.sync(force=True)
    return svc


def set_session(svc, session_id, updated=None, created=None):
    conn = sqlite3.connect(svc.db.db_path)
    if updated:
        conn.execute("UPDATE cross_sessions SET updated_at = ? WHERE session_id = ?", (updated, session_id))
    if created:
        conn.execute("UPDATE cross_sessions SET created_at = ? WHERE session_id = ?", (created, session_id))
    conn.commit()
    conn.close()


def iso(offset_seconds):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(time.time() + offset_seconds, timezone.utc).isoformat(timespec="microseconds")


def ordered(svc):
    return [s.session_id for s in svc.recent_sessions(WORKSPACE, 10)]


# ------------------------------------------------------------------ current session and back
def test_current_session_is_excluded_only_when_mcp_started(bank):
    newest = ordered(bank)[0]
    set_session(bank, newest, created=iso(30), updated=iso(40))  # created after startup
    assert ordered(bank)[0] == newest  # CLI: there is no current session
    bank.process_started_at = time.time()
    assert newest not in ordered(bank)
    set_session(bank, newest, created=iso(-7200))  # created two hours before startup
    assert ordered(bank)[0] == newest


def test_resolve_back_positions_and_errors(bank):
    sessions = ordered(bank)
    assert bank.resolve_back(WORKSPACE, 1).session_id == sessions[0]
    assert bank.resolve_back(WORKSPACE, 2).session_id == sessions[1]
    with pytest.raises(SessionError, match=f"Only {len(sessions)} previous"):
        bank.resolve_back(WORKSPACE, len(sessions) + 1)
    with pytest.raises(SessionError):
        bank.resolve_back(WORKSPACE, 11)
    with pytest.raises(SessionError):
        bank.resolve_back("/does/not/exist", 1)


# ------------------------------------------------------------------ single Extended Handoff
def test_handoff_is_generated_saved_and_reused_by_memory(bank):
    agent = bank.agent
    h1 = bank.get_handoff(workspace=WORKSPACE, back=1)
    target = ordered(bank)[0]
    assert h1.session_id == target and not h1.from_cache and h1.markdown_handoff.startswith("# Extended Handoff —")
    assert agent.digest_calls == [target]
    h2 = bank.get_handoff(target)  # by ID: local bank, no LLM
    assert h2.from_cache and agent.digest_calls == [target]
    bank.update_project_memory(WORKSPACE)  # the memory reuses that digest
    assert agent.digest_calls.count(target) == 1


def test_focused_handoff_is_ephemeral(bank):
    before = bank.db.digest_format_counts()
    h = bank.get_handoff(workspace=WORKSPACE, focus_area="tests")
    assert "Focus tests" in h.markdown_handoff and h.focus_area == "tests"
    assert bank.agent.focus_calls and bank.agent.digest_calls == []
    assert bank.db.digest_format_counts() == before  # nothing saved


def test_offline_handoff_without_key_is_not_saved(bank):
    bank.agent.available = False
    h = bank.get_handoff(workspace=WORKSPACE)
    assert h.model_used == "offline-extractor" and "### 7. Immediate plan" in h.markdown_handoff
    assert bank.db.digest_format_counts()["current"] == 0
    assert "GEMINI_API_KEY" in h.markdown_handoff  # explains how to set the key


def test_offline_handoff_after_llm_failure_does_not_ask_for_a_key(bank, monkeypatch):
    bank.agent.fail_digest_for = {s.session_id for s in bank.list_sessions(workspace=WORKSPACE)}
    h = bank.get_handoff(workspace=WORKSPACE)
    assert h.model_used == "offline-fallback" and "the Gemini API failed" in h.markdown_handoff
    assert "GEMINI_API_KEY" not in h.markdown_handoff
    assert bank.db.digest_format_counts()["current"] == 0

    monkeypatch.setattr(bank.agent, "generate_focused_handoff", lambda detail, focus: None)
    focused = bank.get_handoff(workspace=WORKSPACE, focus_area="tests")
    assert focused.model_used == "offline-fallback" and "GEMINI_API_KEY" not in focused.markdown_handoff


def test_offline_focused_handoff_without_key_explains_the_key(bank, monkeypatch):
    bank.agent.available = False
    monkeypatch.setattr(bank.agent, "generate_focused_handoff", lambda detail, focus: None)
    h = bank.get_handoff(workspace=WORKSPACE, focus_area="tests")
    assert h.model_used == "offline-extractor" and "GEMINI_API_KEY" in h.markdown_handoff


def test_resume_regenerates_stale_handoff_even_if_recently_active(bank):
    target = ordered(bank)[0]
    bank.get_handoff(workspace=WORKSPACE, back=1)
    set_session(bank, target, updated=iso(60))  # it just moved on (< 15 min idle)
    bank.get_handoff(workspace=WORKSPACE, back=1)
    assert bank.agent.digest_calls == [target, target]


# ------------------------------------------------------------------ background maintenance
def test_pending_sessions_rules(bank):
    bank.process_started_at = time.time()
    sessions = ordered(bank)
    assert [s.session_id for s in bank.pending_sessions(WORKSPACE)] == sessions  # none has a handoff
    for sid in sessions:
        bank.get_handoff(sid)
    assert bank.pending_sessions(WORKSPACE) == []
    set_session(bank, sessions[0], updated=iso(-60))  # stale but active 1 min ago
    set_session(bank, sessions[1], updated=iso(-IDLE_BEFORE_REFRESH - 60))  # stale and idle
    set_session(bank, sessions[2], created=iso(10), updated=iso(20))  # the current session
    pending = [s.session_id for s in bank.pending_sessions(WORKSPACE)]
    assert pending == [sessions[1]]


def test_maintenance_is_disabled_in_cli_and_without_key(bank):
    assert bank.maybe_schedule_maintenance(WORKSPACE) == "disabled"
    bank.background = True
    bank.agent.available = False
    assert bank.maybe_schedule_maintenance(WORKSPACE) == "no-key"
    assert bank.agent.calls == 0


def test_maintenance_runs_in_background_and_consolidates(bank):
    bank.background = True
    gate = threading.Event()
    bank.agent.digest_gate = gate
    started = time.time()
    text = bank.get_project_memory(WORKSPACE)
    assert time.time() - started < 2  # does not wait for generation
    assert "in the background" in text
    assert bank.maybe_schedule_maintenance(WORKSPACE) == "running"
    gate.set()
    bank.wait_for_maintenance(WORKSPACE, timeout=10)
    mem = bank.db.get_memory(WORKSPACE)
    assert mem is not None and mem["version"] == 1
    assert bank.maybe_schedule_maintenance(WORKSPACE) == "idle"
    calls = bank.agent.calls
    assert "in the background" not in bank.get_project_memory(WORKSPACE)
    assert bank.agent.calls == calls


def test_maintenance_consolidates_handoffs_generated_by_resume(bank):
    bank.background = True
    bank.update_project_memory(WORKSPACE)
    target = ordered(bank)[0]
    set_session(bank, target, updated=iso(60))
    bank.get_handoff(workspace=WORKSPACE, back=1)  # resuming regenerates the handoff: the memory goes stale
    assert bank.maybe_schedule_maintenance(WORKSPACE) == "started"
    bank.wait_for_maintenance(WORKSPACE, timeout=10)
    assert bank.db.get_memory(WORKSPACE)["version"] == 2
    assert bank.maybe_schedule_maintenance(WORKSPACE) == "idle"


def test_only_one_instance_runs_maintenance(bank, tmp_path):
    bank.background = True
    assert bank.db.acquire_maintenance_lock(WORKSPACE, "other-instance")
    assert bank.maybe_schedule_maintenance(WORKSPACE) == "started"
    bank.wait_for_maintenance(WORKSPACE, timeout=10)
    assert bank.agent.calls == 0  # another instance holds the lock
    assert bank.db.get_memory(WORKSPACE) is None


def test_interrupted_maintenance_resumes_with_missing_digests_only(bank):
    sessions = ordered(bank)
    bank.agent.fail_digest_for = set(sessions[1:])  # simulates a cut after the first digest
    bank.background = True
    bank.maybe_schedule_maintenance(WORKSPACE)
    bank.wait_for_maintenance(WORKSPACE, timeout=10)
    assert len(bank.db.get_memory(WORKSPACE)["window"]) == 1  # what did get generated is consolidated
    assert bank.db.digest_format_counts()["current"] == 1
    bank.agent.fail_digest_for = set()
    bank.agent.digest_calls.clear()
    bank._digest_failures.clear()  # simulates a process restart (or the backoff running out)
    bank._consolidation_failures.clear()
    assert bank.maybe_schedule_maintenance(WORKSPACE) == "started"
    bank.wait_for_maintenance(WORKSPACE, timeout=10)
    assert sorted(bank.agent.digest_calls) == sorted(sessions[1:])  # only the missing ones
    mem = bank.db.get_memory(WORKSPACE)
    assert mem["version"] == 2 and len(mem["window"]) == min(len(sessions), memory.default_window())


# ------------------------------------------------------------------ topic search
@pytest.fixture
def searchable(bank):
    bank.agent.extra_entries = [
        {"id": None, "kind": "T", "text": "Docker without network in CI → missing --network host → add it",
         "evidence_sessions": []},
        {"id": None, "kind": "D", "text": "Use Alembic for migrations", "why": "versioned history",
         "evidence_sessions": []},
    ]
    bank.update_project_memory(WORKSPACE)
    return bank


def test_search_finds_active_and_archived_entries_without_llm(searchable):
    t = next(e for e in searchable.db.get_entries(WORKSPACE) if e["kind"] == "T")
    searchable.edit_memory_entry(WORKSPACE, t["id"], "pin")
    entries = searchable.db.get_entries(WORKSPACE)
    for e in entries:
        if e["kind"] == "D" and "Alembic" in e["text"]:
            e["status"] = "archived"
    mem = searchable.db.get_memory(WORKSPACE)
    searchable.db.save_memory_version(WORKSPACE, mem["version"], mem["rendered_md"], mem["window"], mem["state"],
                                      "m", entries)
    calls = searchable.agent.calls
    out = searchable.search_project_memory("DOCKER", WORKSPACE)
    assert f"[{t['id']}]" in out and "(active" in out
    out = searchable.search_project_memory("ALEMBIC migrations", WORKSPACE)  # case-insensitive
    assert "(archived" in out
    assert searchable.agent.calls == calls


def test_search_excludes_forgotten_and_reports_no_matches(searchable):
    t = next(e for e in searchable.db.get_entries(WORKSPACE) if e["kind"] == "T")
    searchable.edit_memory_entry(WORKSPACE, t["id"], "forget")
    assert "No matches" in searchable.search_project_memory("docker", WORKSPACE)


def test_search_includes_handoffs_and_all_projects(searchable):
    sid = ordered(searchable)[0]
    handoff = "### 1. Intent and scope\n- Migrate to Kubernetes\n### 5. Errors → fix\n- Helm failed because of the chart"
    searchable.db.save_digest(sid, "claude", WORKSPACE, iso(0), {"relevant": True}, handoff, "m")
    out = searchable.search_project_memory("helm chart", WORKSPACE)
    assert f"claude:{sid[:8]}" in out and "handoff · 5. Errors → fix" in out and "Kubernetes" not in out
    assert "unique decision" in searchable.search_project_memory("unique", WORKSPACE)
    out = searchable.search_project_memory("docker", WORKSPACE, all_projects=True)
    assert f"`{WORKSPACE}`" in out and "all projects" in out


def test_search_limit_and_omitted_notice(searchable):
    out = searchable.search_project_memory("decision", WORKSPACE, limit=1)
    assert "more results; narrow the topic or use limit" in out


def test_search_synthesis_optional_and_not_saved(searchable):
    versions = len(searchable.db.list_memory_versions(WORKSPACE))
    out = searchable.search_project_memory("docker", WORKSPACE, synthesize=True)
    assert "Pattern synthesis" in out and "not saved" in out
    assert "Docker without network" in searchable.agent.synth_calls[0]
    assert len(searchable.db.list_memory_versions(WORKSPACE)) == versions
    searchable.agent.available = False
    out = searchable.search_project_memory("docker", WORKSPACE, synthesize=True)
    assert "No synthesis" in out and "Docker without network" in out


def test_search_ignores_accents():
    entries = [{"id": "D1", "kind": "D", "status": "active", "evidence": [], "text": "Usar migraciones con Alembic"}]
    out = memory.render_search("migración ALEMBIC", memory.search("migración ALEMBIC", entries, []))
    assert "[D1]" in out


def test_search_redacts_secrets():
    results = memory.search("token", [{"id": "T1", "kind": "T", "status": "active", "evidence": [],
                                       "text": "token ghp_" + "a" * 36 + " leaked"}], [])
    assert "ghp_" not in memory.render_search("token", results)


# ------------------------------------------------------------------ select_turns
def test_select_turns_query_context_and_max():
    turns = [SessionTurn("user", "start"), SessionTurn("tool", "output"), SessionTurn("assistant", "Migración hecha"),
             SessionTurn("user", "something else"), SessionTurn("assistant", "end")]
    shown, omitted = select_turns(turns, None, False, 10)
    assert [t.role for t in shown] == ["user", "assistant", "user", "assistant"] and omitted == 0
    shown, omitted = select_turns(turns, "MIGRACION", False, 10)  # accent- and case-insensitive
    assert [t.content for t in shown] == ["start", "Migración hecha", "something else"] and omitted == 1
    shown, omitted = select_turns(turns, None, True, 2)
    assert len(shown) == 2 and omitted == 3


def test_failing_session_is_not_retried_in_a_loop_and_memory_still_consolidates(bank):
    sessions = ordered(bank)
    bank.background = True
    bank.agent.fail_digest_for = {sessions[0]}  # a session that always fails
    bank.maybe_schedule_maintenance(WORKSPACE)
    bank.wait_for_maintenance(WORKSPACE, timeout=10)
    mem = bank.db.get_memory(WORKSPACE)
    assert mem is not None  # the rest consolidate without the failing one
    assert sessions[0] not in {w["session_id"] for w in mem["window"]}
    calls = bank.agent.calls
    for _ in range(3):  # successive MCP calls: no retry or consolidation loops
        assert bank.maybe_schedule_maintenance(WORKSPACE) == "idle"
    assert bank.agent.calls == calls


def test_failed_consolidation_is_not_retried_in_a_loop(bank):
    bank.background = True
    bank.agent.fail_consolidation = True
    bank.maybe_schedule_maintenance(WORKSPACE)
    bank.wait_for_maintenance(WORKSPACE, timeout=10)
    assert bank.db.get_memory(WORKSPACE) is None
    calls = bank.agent.calls
    assert bank.maybe_schedule_maintenance(WORKSPACE) == "idle"
    assert bank.agent.calls == calls
