"""Tests for the synthesis agent (no network: urlopen is mocked)."""

import io
import json
import urllib.error

import pytest

from cross_session_mcp import agent as agent_mod
from cross_session_mcp import i18n
from cross_session_mcp.agent import (
    GeminiHandoffAgent,
    build_offline_handoff,
    build_transcript,
    get_gemini_api_key,
    has_handoff_sections,
)
from cross_session_mcp.models import SessionDetail, SessionSummary, SessionTurn

SECRET = "AIza" + "x" * 35


def make_detail(n_turns=6):
    turns = [SessionTurn(role="user", content=f"Initial goal with key {SECRET}")]
    for i in range(n_turns):
        turns.append(SessionTurn(role="assistant", content=f"step {i}"))
        turns.append(SessionTurn(role="tool", content=f"→ bash: command {i}"))
    turns.append(SessionTurn(role="user", content="Last instruction"))
    summary = SessionSummary(session_id="s1", harness="claude", workspace_path="/p", title="T",
                             updated_at="2026-09-01T10:00:00.000000+00:00")
    return SessionDetail(summary=summary, turns=turns)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSS_SESSION_KEY_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(tmp_path / "none"))
    monkeypatch.delenv("CROSS_SESSION_GOOGLE_PROJECT", raising=False)
    monkeypatch.setenv("CROSS_SESSION_DISABLE_SDK", "1")
    monkeypatch.setenv("CROSS_SESSION_MODELS", "model-a,model-b")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CROSS_SESSION_LANGUAGE", raising=False)


def test_build_transcript_small_fits_entirely():
    text, included = build_transcript(make_detail(3).turns, budget_chars=100_000)
    assert included == 8
    assert SECRET not in text and "[REDACTED]" in text


def test_build_transcript_budget_keeps_directives_head_and_tail():
    detail = make_detail(400)
    text, included = build_transcript(detail.turns, budget_chars=4000)
    assert len(text) < 5000
    assert included < len(detail.turns)
    assert "User instructions" in text
    assert "Last instruction" in text  # final directive
    assert "step 0" in text  # genesis
    assert "step 399" in text  # final stretch
    assert "left out for size" in text


def test_key_file_loading(tmp_path, monkeypatch):
    f = tmp_path / ".gemini_env"
    f.write_text("# comment\nexport GEMINI_API_KEY='file-key'\n")
    f.chmod(0o600)
    monkeypatch.setenv("CROSS_SESSION_KEY_FILE", str(f))
    assert get_gemini_api_key() == "file-key"
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert get_gemini_api_key() == "env-key"


def test_offline_handoff_has_the_eight_sections_without_secrets():
    md = build_offline_handoff(make_detail(), focus_area="tests")
    assert has_handoff_sections(md)
    assert all(section in md for section in i18n.handoff_sections())
    assert SECRET not in md
    assert "> Last instruction" in md  # verbatim quote of the last instruction in the immediate plan
    assert "Requested focus: tests" in md


def test_offline_handoff_in_spanish(monkeypatch):
    monkeypatch.setenv("CROSS_SESSION_LANGUAGE", "es")
    md = build_offline_handoff(make_detail(), focus_area="tests")
    assert "### 7. Plan inmediato" in md and "Foco solicitado: tests" in md
    assert has_handoff_sections(md)
    assert "no hay credencial de Gemini" in md and "GEMINI_API_KEY" in md
    failed = build_offline_handoff(make_detail(), reason="llm_failed")
    assert "la API de Gemini falló" in failed and "GEMINI_API_KEY" not in failed


@pytest.mark.parametrize("value,expected", [
    (None, "en"), ("es", "es"), ("ES", "es"), ("es_ES.UTF-8", "es"), ("fr", "en"), ("", "en"),
])
def test_language_selection_falls_back_to_english(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("CROSS_SESSION_LANGUAGE", value)
    assert i18n.language() == expected


def test_section_titles_follow_the_language(monkeypatch):
    assert i18n.handoff_sections()[6] == "### 7. Immediate plan"
    monkeypatch.setenv("CROSS_SESSION_LANGUAGE", "es")
    assert i18n.handoff_sections()[6] == "### 7. Plan inmediato"
    assert len(i18n.handoff_sections()) == 8
    assert "Spanish from Spain" in agent_mod._system(agent_mod.MASTER_DIGEST_SYSTEM_PROMPT)


def test_focused_handoff_key_in_header_and_fallback_between_models(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        if "model-a" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b"{}"))
        body = {"candidates": [{"content": {"parts": [{"text": f"### 1. Handoff with {SECRET}"}]}, "finishReason": "STOP"}]}
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    markdown, model = GeminiHandoffAgent(api_key="k-test").generate_focused_handoff(make_detail(), "tests")

    assert model == "model-b"
    assert SECRET not in markdown  # second barrier on the output
    prompt = json.loads(calls[-1].data.decode())["contents"][0]["parts"][0]["text"]
    assert "**Requested focus:** tests" in prompt and "### 7. Immediate plan" in prompt
    for req in calls:
        assert "k-test" not in req.full_url
        assert req.get_header("X-goog-api-key") == "k-test"
        assert SECRET not in req.data.decode()  # the transcript leaves without secrets


def test_markdown_calls_return_none_when_all_models_fail_or_no_key(monkeypatch):
    def boom(req, timeout):
        raise TimeoutError("slow")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", boom)
    assert GeminiHandoffAgent(api_key="k").generate_focused_handoff(make_detail(), "x") is None
    assert GeminiHandoffAgent(api_key="k").synthesize_topic("docker", "- [T1] something") is None
    assert GeminiHandoffAgent(api_key=None).generate_focused_handoff(make_detail(), "x") is None
    assert GeminiHandoffAgent(api_key=None).synthesize_topic("docker", "x") is None


def test_topic_synthesis_receives_only_results(monkeypatch):
    sent = []

    def fake_urlopen(req, timeout):
        sent.append(json.loads(req.data.decode()))
        body = {"candidates": [{"content": {"parts": [{"text": "### Recurring patterns\n- [T1]"}]}}]}
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    text, model = GeminiHandoffAgent(api_key="k").synthesize_topic("docker", "- [T1] Docker without network")
    prompt = sent[0]["contents"][0]["parts"][0]["text"]
    assert "Docker without network" in prompt and "### Contradictions or changes of criteria" in prompt
    assert text.startswith("### Recurring") and model == "model-a"
