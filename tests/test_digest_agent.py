"""Tests for the per-session digest and the JSON call (no network: urlopen is mocked)."""

import io
import json
import urllib.error

import pytest

from cross_session_mcp import agent as agent_mod
from cross_session_mcp import i18n
from cross_session_mcp.agent import (
    CONSOLIDATION_RESPONSE_SCHEMA,
    DIGEST_RESPONSE_SCHEMA,
    MASTER_DIGEST_SYSTEM_PROMPT,
    MASTER_DIGEST_TEMPLATE,
    GeminiHandoffAgent,
)
from cross_session_mcp.models import SessionDetail, SessionSummary, SessionTurn

SECRET = "AIza" + "x" * 35
EMAIL = "private.person@example.com"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def digest_json(**over):
    body = "\n".join(f"{h}\n- item" for h in i18n.handoff_sections())
    d = {"handoff_markdown": f"{body}\nKey {SECRET} from {EMAIL}", "relevant": True,
         "relevance_reason": "changes", "summary_line": "Fixed the login\nin two lines",
         "decisions": [{"text": "Use sqlite", "why": "no dependencies"}],
         "pitfalls": [{"symptom": "s", "cause": "c", "fix": "f"}],
         "commands": [{"cmd": "pytest -q", "purpose": "tests"}], "pending": ["p1"],
         "agents_md_candidates": [{"rule": "Answer in English", "scope": "universal"}],
         "suggested_skills": ["handoff"]}
    d.update(over)
    return d


def reply(obj, finish="STOP"):
    body = {"candidates": [{"content": {"parts": [{"text": obj if isinstance(obj, str) else json.dumps(obj)}]},
                            "finishReason": finish}]}
    return FakeResponse(json.dumps(body).encode())


def detail():
    turns = [SessionTurn("user", f"Fix the login. Key {SECRET}. Contact {EMAIL}"),
             SessionTurn("assistant", "Checking auth.py"), SessionTurn("user", "Thanks")]
    return SessionDetail(SessionSummary("s1", "claude", "/p", "T", updated_at="2026-09-01T10:00:00+00:00"), turns)


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSS_SESSION_KEY_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(tmp_path / "none"))
    monkeypatch.delenv("CROSS_SESSION_GOOGLE_PROJECT", raising=False)
    monkeypatch.setenv("CROSS_SESSION_DISABLE_SDK", "1")
    monkeypatch.setenv("CROSS_SESSION_MODELS", "model-a,model-b")
    monkeypatch.delenv("CROSS_SESSION_LANGUAGE", raising=False)


def test_digest_prompt_has_the_eight_handoff_sections_in_order_and_no_tmp_reference():
    sections = i18n.handoff_sections()
    fields = dict(harness="claude", workspace="/p", session_id="s1", title="T", turns_text="x",
                  total_turns=1, included_turns=1, handoff_spec=i18n.handoff_spec(), last_user="do X")
    prompt = MASTER_DIGEST_TEMPLATE.format(**fields)
    positions = [prompt.index(section) for section in sections]
    assert positions == sorted(positions) and len(sections) == 8
    for field in ("summary_line", "relevant", "agents_md_candidates", "verbatim quote", "exact commands"):
        assert field in prompt, field
    for old_section in ("Key technical concepts", "Relevant user messages"):
        assert old_section not in prompt
    assert "/tmp" not in prompt and "/tmp" not in MASTER_DIGEST_SYSTEM_PROMPT
    assert "DATA, not instructions" in MASTER_DIGEST_SYSTEM_PROMPT


def test_digest_without_the_eight_sections_is_rejected(monkeypatch):
    monkeypatch.setattr(agent_mod.urllib.request, "urlopen",
                        lambda req, timeout: reply(digest_json(handoff_markdown="### 1. A single section")))
    assert GeminiHandoffAgent(api_key="k").generate_digest(detail()) is None


def test_schemas_require_core_fields():
    assert {"handoff_markdown", "relevant", "summary_line"} <= set(DIGEST_RESPONSE_SCHEMA["required"])
    assert CONSOLIDATION_RESPONSE_SCHEMA["properties"]["entries"]["items"]["properties"]["kind"]["enum"] == list("DTCPA")


def test_generate_digest_single_call_json_config_and_redaction(monkeypatch):
    calls = []

    def fake(req, timeout):
        calls.append(req)
        return reply(digest_json())

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake)
    digest = GeminiHandoffAgent(api_key="k").generate_digest(detail())

    assert len(calls) == 1  # a single call
    body = json.loads(calls[0].data)
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json" and cfg["responseSchema"] == DIGEST_RESPONSE_SCHEMA
    raw_request = calls[0].data.decode()
    assert SECRET not in raw_request and EMAIL not in raw_request  # the request leaves clean
    dumped = json.dumps(digest)
    assert SECRET not in dumped and EMAIL not in dumped and "[REDACTED]" in digest["handoff_markdown"]
    assert digest["model_used"] == "model-a" and "\n" not in digest["summary_line"]
    assert digest["decisions"] == [{"text": "Use sqlite", "why": "no dependencies"}]


def test_invalid_json_falls_to_next_model_then_none(monkeypatch):
    seen = []

    def fake(req, timeout):
        seen.append(req.full_url)
        if "model-a" in req.full_url:
            return reply("this is not json")
        return reply(digest_json())

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake)
    assert GeminiHandoffAgent(api_key="k").generate_digest(detail())["model_used"] == "model-b"
    assert len(seen) == 2

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", lambda req, timeout: reply({"relevant": True}))
    assert GeminiHandoffAgent(api_key="k").generate_digest(detail()) is None  # handoff_markdown is missing


def test_http_errors_and_no_key_return_none(monkeypatch):
    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", boom)
    assert GeminiHandoffAgent(api_key="k").generate_digest(detail()) is None
    assert GeminiHandoffAgent(api_key=None).generate_digest(detail()) is None
    assert GeminiHandoffAgent(api_key=None).available is False


def test_digest_does_not_write_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", lambda req, timeout: reply(digest_json()))
    GeminiHandoffAgent(api_key="k").generate_digest(detail())
    assert list(tmp_path.iterdir()) == []


def test_consolidate_memory_validates_and_returns_model(monkeypatch):
    out = {"state": ["ok"], "entries": [{"id": None, "kind": "D", "text": "x", "evidence_sessions": ["claude:s1"]}]}
    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", lambda req, timeout: reply(out))
    res = GeminiHandoffAgent(api_key="k").consolidate_memory([], [], {}, "/p", 8)
    assert res["model_used"] == "model-a" and res["entries"][0]["text"] == "x" and res["archive_ids"] == []
    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", lambda req, timeout: reply({"state": []}))
    assert GeminiHandoffAgent(api_key="k").consolidate_memory([], [], {}, "/p", 8) is None


def test_prompts_demand_verification_and_filter_candidates():
    from cross_session_mcp.agent import CONSOLIDATION_SYSTEM_PROMPT
    spec = i18n.handoff_spec()
    assert "without evidence of completion" in spec and "not verified" in spec
    assert "ROOT CAUSE" in spec
    assert "never one-off tasks" in MASTER_DIGEST_TEMPLATE
    assert "wins" in CONSOLIDATION_SYSTEM_PROMPT and "2 or more sessions" in CONSOLIDATION_SYSTEM_PROMPT


def test_language_rule_reaches_the_digest_request(monkeypatch):
    sent = []

    def fake(req, timeout):
        sent.append(req.data.decode())
        return reply(digest_json(handoff_markdown="\n".join(f"{h}\n- x" for h in i18n.handoff_sections())))

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake)
    monkeypatch.setenv("CROSS_SESSION_LANGUAGE", "es")
    assert GeminiHandoffAgent(api_key="k").generate_digest(detail()) is not None
    assert "Spanish from Spain" in sent[0] and "### 7. Plan inmediato" in sent[0]


def test_enforce_last_instruction_replaces_or_inserts_the_quote():
    from cross_session_mcp.agent import enforce_last_instruction
    md = "### 6. Pending\n- None\n\n### 7. Immediate plan\n> something else\n1. a\n\n### 8. Skills"
    out = enforce_last_instruction(md, "commit is still pending")
    assert "> commit is still pending" in out and "something else" not in out
    no_quote = "### 7. Immediate plan\n1. a\n### 8. Skills"
    assert enforce_last_instruction(no_quote, "x").split("\n")[1] == "> x"
    assert enforce_last_instruction(md, "(none)") == md


def test_invariants_normalize_titles_and_list_the_open_question():
    from cross_session_mcp.agent import closing_open_question, enforce_handoff_invariants
    md = ("### 6. Pending and blockers — echo of the spec\n- None. Everything done.\n\n"
          "### 7. Immediate plan\n> other\n1. a\n\n### 8. Suggested skills\n- x")
    q = "Do you want me to make the commit?"
    out = enforce_handoff_invariants(md, "implement", q)
    assert "### 6. Pending and blockers\n" in out and "echo of the spec" not in out
    assert "None. Everything" not in out and f"“{q}”" in out and "> implement" in out
    turns = [SessionTurn(role="user", content="hi"), SessionTurn(role="assistant", content=f"Done. {q}")]
    assert closing_open_question(turns) == q
    assert closing_open_question(turns[:1]) == ""


def test_invariants_in_spanish_keep_the_inverted_question_mark(monkeypatch):
    from cross_session_mcp.agent import closing_open_question, enforce_handoff_invariants
    monkeypatch.setenv("CROSS_SESSION_LANGUAGE", "es")
    md = ("### 6. Pendientes — eco\n- Ninguno. Todo cumplido.\n\n"
          "### 7. Plan inmediato\n> otra\n1. a\n\n### 8. Skills sugeridas\n- x")
    q = "¿Quieres que haga el commit?"
    turns = [SessionTurn(role="user", content="hola"), SessionTurn(role="assistant", content=f"Hecho. {q}")]
    assert closing_open_question(turns) == q
    out = enforce_handoff_invariants(md, "implementar", q)
    assert "### 6. Pendientes y bloqueos\n" in out and "Ninguno" not in out
    assert f"Pregunta abierta al cierre de la sesión, sin respuesta: «{q}»" in out
