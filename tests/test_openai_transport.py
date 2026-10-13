"""Tests for the OpenAI-compatible transport (no network: urlopen is mocked)."""

import io
import json
import urllib.error

from cross_session_mcp import agent as agent_mod
from cross_session_mcp.agent import GeminiHandoffAgent


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSS_SESSION_OPENAI_BASE", "http://127.0.0.1:10100/v1")
    monkeypatch.delenv("CROSS_SESSION_OPENAI_KEY", raising=False)
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_DISABLE_SDK", "1")
    monkeypatch.setenv("CROSS_SESSION_MODELS", "model-a")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CROSS_SESSION_KEY_FILE", raising=False)
    monkeypatch.delenv("CROSS_SESSION_GOOGLE_PROJECT", raising=False)


def chat_ok(text="hello-oai", finish="stop"):
    return FakeResponse(
        json.dumps({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish}]}).encode(
            "utf-8"
        )
    )


def payload():
    return {
        "contents": [{"role": "user", "parts": [{"text": "do it"}]}],
        "systemInstruction": {"parts": [{"text": "Be brief."}]},
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 50},
    }


def test_message_translation_and_headers(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.headers)
        seen["body"] = json.loads(req.data.decode())
        return chat_ok()

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    agent = GeminiHandoffAgent()
    assert agent.available is True
    text, _ = agent._generate_markdown("sys", "prompt")
    assert text == "hello-oai"
    assert seen["url"] == "http://127.0.0.1:10100/v1/chat/completions"
    assert seen["body"]["model"] == "model-a"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["body"]["messages"][1]["content"] == "prompt"
    assert seen["body"]["temperature"] == 0.2 and seen["body"]["max_tokens"] == 8192
    assert "response_format" not in seen["body"]
    assert "authorization" not in {k.lower() for k in seen["headers"]}


def test_bearer_key_and_json_mode(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    monkeypatch.setenv("CROSS_SESSION_OPENAI_KEY", "proxy-key")
    seen = {}

    def fake_urlopen(req, timeout):
        seen["headers"] = {k.lower(): v for k, v in req.headers.items()}
        seen["body"] = json.loads(req.data.decode())
        return chat_ok('{"a": 1}')

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    agent = GeminiHandoffAgent()
    out = agent._call_model("m", {**payload(), "generationConfig": {"responseMimeType": "application/json"}})
    assert out == ('{"a": 1}', "stop")
    assert seen["headers"]["authorization"] == "Bearer proxy-key"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_length_maps_to_max_tokens_and_errors(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)

    def fake_len(req, timeout):
        return chat_ok("partial", "length")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_len)
    assert GeminiHandoffAgent()._call_model("m", payload()) == ("partial", "MAX_TOKENS")

    def fake_empty(req, timeout):
        return FakeResponse(b"{}")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_empty)
    assert GeminiHandoffAgent()._call_model("m", payload()) is None

    def fake_500(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 500, "err", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_500)
    assert GeminiHandoffAgent()._call_model("m", payload()) is None


def test_base_wins_over_other_transports(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "key-1")
    hits = []

    def fake_urlopen(req, timeout):
        hits.append(req.full_url)
        return chat_ok()

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    text, _ = GeminiHandoffAgent()._generate_markdown("sys", "prompt")
    assert text == "hello-oai"
    assert hits == ["http://127.0.0.1:10100/v1/chat/completions"]


def test_no_base_no_availability(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    monkeypatch.delenv("CROSS_SESSION_OPENAI_BASE")
    assert GeminiHandoffAgent().available is False
