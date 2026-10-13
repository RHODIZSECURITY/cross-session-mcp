"""Tests for OAuth (ADC) authentication (no network: urlopen is mocked)."""

import io
import json
import urllib.error
import urllib.parse

from cross_session_mcp import agent as agent_mod
from cross_session_mcp.agent import (
    GeminiHandoffAgent,
    OAuthTokenCache,
    load_adc,
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_DISABLE_SDK", "1")
    monkeypatch.setenv("CROSS_SESSION_MODELS", "model-a")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CROSS_SESSION_KEY_FILE", raising=False)
    monkeypatch.delenv("CROSS_SESSION_GOOGLE_PROJECT", raising=False)


def write_adc(path, **over):
    data = {
        "type": "authorized_user",
        "client_id": "cid",
        "client_secret": "csec",
        "refresh_token": "rtok",
    }
    data.update(over)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def gemini_ok(text="hello"):
    return FakeResponse(
        json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}).encode("utf-8")
    )


def token_ok(token="tok-1", expires_in=3600):
    return FakeResponse(
        json.dumps({"access_token": token, "expires_in": expires_in, "token_type": "Bearer"}).encode("utf-8")
    )


def test_no_credential_not_available(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    agent = GeminiHandoffAgent()
    assert agent.available is False
    assert agent._auth_headers() is None


def test_malformed_adc_ignored(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    bad = tmp_path / "adc.json"
    bad.write_text("not json", encoding="utf-8")
    assert load_adc(bad) is None
    write_adc(bad, type="service_account")
    assert load_adc(bad) is None
    write_adc(bad, refresh_token="")
    assert load_adc(bad) is None


def test_adc_group_readable_warns(monkeypatch, tmp_path, caplog):
    isolated_env(monkeypatch, tmp_path)
    adc = write_adc(tmp_path / "adc.json")
    adc.chmod(0o640)
    with caplog.at_level("WARNING", logger="cross_session_mcp"):
        assert load_adc(adc) is not None
    assert "chmod 600" in caplog.text


def test_api_key_wins_over_adc(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    adc = write_adc(tmp_path / "adc.json")
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(adc))
    monkeypatch.setenv("GEMINI_API_KEY", "key-123")
    agent = GeminiHandoffAgent()
    headers = agent._auth_headers()
    assert headers["x-goog-api-key"] == "key-123"
    assert "Authorization" not in headers


def test_oauth_bearer_and_user_project_headers(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    adc = write_adc(tmp_path / "adc.json")
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(adc))
    monkeypatch.setenv("CROSS_SESSION_GOOGLE_PROJECT", "my-proj")
    seen = {}

    def fake_urlopen(req, timeout):
        if req.full_url == agent_mod.OAUTH_TOKEN_URL:
            body = urllib.parse.parse_qs(req.data.decode("utf-8"))
            seen["grant"] = body
            return token_ok()
        seen["gemini_headers"] = dict(req.headers)
        return gemini_ok("from-oauth")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    agent = GeminiHandoffAgent()
    assert agent.available is True
    text, model = agent._generate_markdown("sys", "prompt")
    assert text == "from-oauth"
    assert seen["grant"]["grant_type"] == ["refresh_token"]
    assert seen["grant"]["refresh_token"] == ["rtok"]
    assert "client_secret" in seen["grant"]
    headers = {k.lower(): v for k, v in seen["gemini_headers"].items()}
    assert headers["authorization"] == "Bearer tok-1"
    assert headers["x-goog-user-project"] == "my-proj"
    assert "x-goog-api-key" not in headers


def test_non_object_refresh_payload_fails_closed(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    adc = write_adc(tmp_path / "adc.json")
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(adc))

    def fake_urlopen(req, timeout):
        return FakeResponse(json.dumps(["not", "an", "object"]).encode("utf-8"))

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    agent = GeminiHandoffAgent()
    assert agent._generate_markdown("sys", "prompt") == (None, "")


def test_project_with_line_break_is_ignored(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    adc = write_adc(tmp_path / "adc.json")
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(adc))
    monkeypatch.setenv("CROSS_SESSION_GOOGLE_PROJECT", "proj\ninjected: x")
    agent = GeminiHandoffAgent()
    assert agent.google_project == ""
    assert "x-goog-user-project" not in (agent._auth_headers() or {})


def test_cached_token_reused_without_refresh(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    cache = OAuthTokenCache({"client_id": "c", "client_secret": "s", "refresh_token": "r"})
    cache._token = "cached"
    cache._expires_at = 9999999999.0

    def boom(req, timeout):
        raise AssertionError("refresh must not be called")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", boom)
    assert cache.token(90) == "cached"


def test_401_refreshes_once_and_retries(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    adc = write_adc(tmp_path / "adc.json")
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(adc))
    calls = {"gemini": 0, "token": 0}

    def fake_urlopen(req, timeout):
        if req.full_url == agent_mod.OAUTH_TOKEN_URL:
            calls["token"] += 1
            return token_ok(token=f"tok-{calls['token'] + 1}")
        calls["gemini"] += 1
        if calls["gemini"] == 1:
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"{}"))
        auth = {k.lower(): v for k, v in req.headers.items()}["authorization"]
        assert auth == "Bearer tok-3"  # refreshed after the 401
        return gemini_ok("after-retry")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    agent = GeminiHandoffAgent()
    text, _ = agent._generate_markdown("sys", "prompt")
    assert text == "after-retry"
    assert calls == {"gemini": 2, "token": 2}


def test_refresh_failure_returns_none(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    adc = write_adc(tmp_path / "adc.json")
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(adc))

    def boom(req, timeout):
        raise ConnectionError("no network")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", boom)
    agent = GeminiHandoffAgent()
    assert agent.available is True  # configured, but unusable right now
    assert agent._generate_markdown("sys", "prompt") == (None, "")


def test_guards_use_available(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    agent = GeminiHandoffAgent()
    assert agent.generate_focused_handoff(None, "x") is None
    assert agent.synthesize_topic("t", "r") is None
