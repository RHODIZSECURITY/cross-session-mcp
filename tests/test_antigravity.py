"""Tests for Antigravity (Cloud Code Assist) auth + transport (no network: urlopen is mocked)."""

import base64
import hashlib
import io
import json
import urllib.error
import urllib.parse

import pytest

from cross_session_mcp import agent as agent_mod
from cross_session_mcp import antigravity as ag_mod
from cross_session_mcp.agent import GeminiHandoffAgent
from cross_session_mcp.antigravity import (
    AntigravityAuth,
    build_auth_url,
    cca_envelope,
    cca_unwrap,
    discover_project,
    exchange_code,
    extract_code,
    load_cred,
    refresh_access,
    save_cred,
)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("CROSS_SESSION_DISABLE_SDK", "1")
    monkeypatch.setenv("CROSS_SESSION_MODELS", "model-a")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CROSS_SESSION_KEY_FILE", raising=False)
    monkeypatch.delenv("CROSS_SESSION_GOOGLE_PROJECT", raising=False)


@pytest.fixture(autouse=True)
def _dummy_client_secret(monkeypatch):
    # The real secret never belongs in tests or the repo; any non-empty value
    # exercises the same code path (urlopen is always mocked here).
    monkeypatch.setattr(ag_mod, "CLIENT_SECRET", "dummy-secret-for-tests")


def token_payload(**over):
    data = {"access_token": "acc-1", "refresh_token": "ref-1", "expires_in": 3600}
    data.update(over)
    return FakeResponse(json.dumps(data).encode("utf-8"))


def test_auth_url_pkce_consistent():
    url, verifier = build_auth_url()
    assert "code_challenge_method=S256" in url and "1071006060591" in url
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()
    assert query["code_challenge"] == expected


def test_extract_code_url_bare_and_invalid():
    assert extract_code("http://127.0.0.1:51121/callback?code=ABC-123_x.y&scope=x") == "ABC-123_x.y"
    assert extract_code("  ABC-123_x.y  ") == "ABC-123_x.y"
    for bad in ("", "http://x/?nocode=1", "!!!", "x" * 513):
        try:
            extract_code(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_exchange_and_credential_roundtrip(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout):
        assert "oauth2.googleapis.com/token" in req.full_url
        body = dict(urllib.parse.parse_qsl(req.data.decode()))
        assert body["grant_type"] == "authorization_code" and body["code"] == "C0DE"
        assert "client_secret" in body
        return token_payload()

    monkeypatch.setattr(ag_mod.urllib.request, "urlopen", fake_urlopen)
    creds = exchange_code("C0DE", "verifier")
    assert creds["access"] == "acc-1" and creds["refresh"] == "ref-1"
    target = tmp_path / "cred.json"
    saved = save_cred(creds["refresh"], "proj-9", "a@b.c", target)
    import os as _os

    if _os.name != "nt":
        assert _os.stat(saved).st_mode & 0o777 == 0o600
    loaded = load_cred(target)
    assert loaded == {"refresh_token": "ref-1", "project_id": "proj-9", "email": "a@b.c"}


def test_exchange_rejects_bad_payload(monkeypatch):
    monkeypatch.setattr(ag_mod.urllib.request, "urlopen", lambda req, timeout: FakeResponse(b"[1,2]"))
    try:
        exchange_code("C", "V")
    except ValueError:
        return
    raise AssertionError("accepted non-object payload")


def test_missing_client_secret_fails_before_network(monkeypatch, tmp_path):
    from cross_session_mcp import antigravity as _ag

    monkeypatch.setattr(_ag, "CLIENT_SECRET", "")
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_SECRET_FILE", str(tmp_path / "none"))
    calls = []

    def boom(req, timeout):
        calls.append(req)
        raise AssertionError("must not reach the network")

    monkeypatch.setattr(_ag.urllib.request, "urlopen", boom)
    try:
        _ag.refresh_access("ref-x")
    except ValueError as exc:
        assert "GOOGLE_ANTIGRAVITY_CLIENT_SECRET" in str(exc)
        assert not calls
        return
    raise AssertionError("refresh without client secret must fail")


def test_secret_file_fallback(monkeypatch, tmp_path):
    from cross_session_mcp import antigravity as _ag

    monkeypatch.setattr(_ag, "CLIENT_SECRET", "")
    secret_file = tmp_path / "sec.env"
    secret_file.write_text('export GOOGLE_ANTIGRAVITY_CLIENT_SECRET="file-secret"\n')
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_SECRET_FILE", str(secret_file))

    def fake_urlopen(req, timeout):
        body = dict(urllib.parse.parse_qsl(req.data.decode()))
        assert body.get("client_secret") == "file-secret"
        return token_payload()

    monkeypatch.setattr(_ag.urllib.request, "urlopen", fake_urlopen)
    out = _ag.refresh_access("ref-x")
    assert out["access"] == "acc-1"


def test_malformed_cred_ignored(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text("not json", encoding="utf-8")
    assert load_cred(bad) is None
    bad.write_text(json.dumps({"refresh_token": "r"}), encoding="utf-8")
    assert load_cred(bad) is None


def test_discover_load_then_onboard(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        if "loadCodeAssist" in req.full_url:
            return FakeResponse(json.dumps({"cloudaicompanionProject": "p-load"}).encode())
        if "onboardUser" in req.full_url:
            return FakeResponse(json.dumps({"done": True, "response": {"projectId": "p-onboard"}}).encode())
        raise AssertionError(req.full_url)

    monkeypatch.setattr(ag_mod.urllib.request, "urlopen", fake_urlopen)
    assert discover_project("tok") == "p-load"
    calls.clear()

    def fake_empty(req, timeout):
        if "loadCodeAssist" in req.full_url:
            return FakeResponse(b"{}")
        return FakeResponse(json.dumps({"done": True, "response": {"project": {"id": "p-nested"}}}).encode())

    monkeypatch.setattr(ag_mod.urllib.request, "urlopen", fake_empty)
    assert discover_project("tok") == "p-nested"


def test_discover_hard_failure_empty(monkeypatch):
    def fake_403(req, timeout):
        if "loadCodeAssist" in req.full_url:
            return FakeResponse(b"{}")
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(ag_mod.urllib.request, "urlopen", fake_403)
    assert discover_project("tok") == ""


def test_refresh_keeps_old_token_on_missing(monkeypatch):
    monkeypatch.setattr(
        ag_mod.urllib.request,
        "urlopen",
        lambda req, timeout: FakeResponse(json.dumps({"access_token": "new-acc"}).encode()),
    )
    out = refresh_access("old-ref")
    assert out["access"] == "new-acc" and out["refresh"] == "old-ref"


def test_auth_class_caches_and_refreshes(monkeypatch):
    auth = AntigravityAuth({"refresh_token": "r", "project_id": "p"})
    assert auth.configured and auth.project == "p"
    auth._access = "cached"
    auth._expires_at = 9999999999.0

    def boom(req, timeout):
        raise AssertionError("must reuse cache")

    monkeypatch.setattr(ag_mod.urllib.request, "urlopen", boom)
    assert auth.token(30) == "cached"
    assert AntigravityAuth().configured is False
    assert AntigravityAuth({"refresh_token": "r"}).configured is False


def test_envelope_shape_and_ua(monkeypatch):
    env = cca_envelope("gemini-x", {"contents": [{"a": 1}]}, "proj-z")
    assert env["model"] == "gemini-x" and env["userAgent"] == "antigravity"
    assert env["requestType"] == "agent" and env["project"] == "proj-z"
    assert env["requestId"].startswith("agent-") and env["request"]["sessionId"]
    assert env["request"]["contents"] == [{"a": 1}]
    assert ag_mod.antigravity_user_agent().startswith("antigravity/ide/2.5.5 (os_type=")
    assert cca_unwrap({"response": {"candidates": []}}) == {"candidates": []}
    assert cca_unwrap({"nope": 1}) is None
    assert cca_unwrap([1]) is None


def _gemini_ok(text="via-cca"):
    return FakeResponse(
        json.dumps(
            {"response": {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}}
        ).encode()
    )


def _agent_with_cca(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    cred_file = tmp_path / "cca.json"
    save_cred("ref-1", "proj-1", "a@b.c", cred_file)
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(cred_file))
    return GeminiHandoffAgent()


def test_cca_wire_envelope_and_headers(monkeypatch, tmp_path):
    agent = _agent_with_cca(monkeypatch, tmp_path)
    assert agent.available is True
    seen = {}

    def fake_urlopen(req, timeout):
        if "oauth2.googleapis.com/token" in req.full_url:
            return token_payload()
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.headers.items()}
        seen["body"] = json.loads(req.data.decode())
        return _gemini_ok()

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    text, model = agent._generate_markdown("sys", "prompt")
    assert text == "via-cca" and model == "model-a"
    assert seen["url"].endswith("/v1internal:generateContent")
    assert seen["headers"]["authorization"] == "Bearer acc-1"
    assert seen["headers"]["user-agent"].startswith("antigravity/ide/")
    assert "x-goog-api-key" not in seen["headers"]
    body = seen["body"]
    assert body["model"] == "model-a" and body["project"] == "proj-1"
    assert body["userAgent"] == "antigravity" and body["requestType"] == "agent"
    assert body["request"]["contents"][0]["parts"][0]["text"] == "prompt"


def test_cca_401_refreshes_once_and_retries(monkeypatch, tmp_path):
    agent = _agent_with_cca(monkeypatch, tmp_path)
    calls = {"cca": 0, "token": 0}

    def fake_urlopen(req, timeout):
        if "oauth2.googleapis.com/token" in req.full_url:
            calls["token"] += 1
            return token_payload(**{"access_token": f"tok-{calls['token'] + 1}"})
        calls["cca"] += 1
        if calls["cca"] == 1:
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"{}"))
        auth = {k.lower(): v for k, v in req.headers.items()}["authorization"]
        assert auth == "Bearer tok-3"
        return _gemini_ok("after-retry")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    text, _ = agent._generate_markdown("sys", "prompt")
    assert text == "after-retry"
    assert calls == {"cca": 2, "token": 2}


def test_key_wins_over_cca(monkeypatch, tmp_path):
    _agent_with_cca(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "key-1")
    agent2 = GeminiHandoffAgent()
    assert agent2._auth_headers()["x-goog-api-key"] == "key-1"


def test_nothing_configured_returns_none(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    agent = GeminiHandoffAgent()
    assert agent.available is False
    assert agent._generate_markdown("sys", "prompt") == (None, "")


def _write_adc(path):
    import json as _json

    path.write_text(
        _json.dumps({"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r"})
    )
    return path


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(b"{}"))


def test_adc_403_falls_through_to_cca(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    adc = _write_adc(tmp_path / "adc.json")
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(adc))
    cca_file = tmp_path / "cca.json"
    save_cred("ref-1", "proj-1", "a@b.c", cca_file)
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(cca_file))
    hits = []

    def fake_urlopen(req, timeout):
        if "oauth2.googleapis.com/token" in req.full_url:
            hits.append("token")
            body = dict(urllib.parse.parse_qsl(req.data.decode()))
            if body.get("refresh_token") == "r":
                return token_payload(**{"access_token": "adc-tok"})
            return token_payload()
        if "generativelanguage.googleapis.com" in req.full_url:
            hits.append("developer")
            raise _http_error(req.full_url, 403)
        if "cloudcode-pa.googleapis.com" in req.full_url:
            hits.append("cca")
            auth = {k.lower(): v for k, v in req.headers.items()}["authorization"]
            assert auth == "Bearer acc-1"
            return _gemini_ok("via-fallthrough")
        raise AssertionError(req.full_url)

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    agent = GeminiHandoffAgent()
    text, _ = agent._generate_markdown("sys", "prompt")
    assert text == "via-fallthrough"
    assert "developer" in hits and "cca" in hits


def test_adc_400_does_not_fall_through(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    adc = _write_adc(tmp_path / "adc.json")
    monkeypatch.setenv("CROSS_SESSION_ADC_FILE", str(adc))
    cca_file = tmp_path / "cca.json"
    save_cred("ref-1", "proj-1", "a@b.c", cca_file)
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(cca_file))
    hits = []

    def fake_urlopen(req, timeout):
        if "oauth2.googleapis.com/token" in req.full_url:
            return token_payload(**{"access_token": "adc-tok"})
        if "generativelanguage.googleapis.com" in req.full_url:
            hits.append("developer")
            raise _http_error(req.full_url, 400)
        hits.append("other:" + req.full_url)
        raise AssertionError("must not try another transport on 400")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    agent = GeminiHandoffAgent()
    assert agent._generate_markdown("sys", "prompt") == (None, "")
    assert hits == ["developer"]


def test_key_401_never_falls_through(monkeypatch, tmp_path):
    isolated_env(monkeypatch, tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "key-1")
    cca_file = tmp_path / "cca.json"
    save_cred("ref-1", "proj-1", "a@b.c", cca_file)
    monkeypatch.setenv("CROSS_SESSION_ANTIGRAVITY_FILE", str(cca_file))
    hits = []

    def fake_urlopen(req, timeout):
        hits.append(req.full_url)
        raise _http_error(req.full_url, 401)

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    agent = GeminiHandoffAgent()
    assert agent._generate_markdown("sys", "prompt") == (None, "")
    assert all("cloudcode-pa" not in h for h in hits)


def _http_429(url):
    return urllib.error.HTTPError(url, 429, "Too Many Requests", {}, io.BytesIO(b'{"error":{}}'))


def test_cca_429_retries_with_backoff_then_succeeds(monkeypatch, tmp_path):
    agent = _agent_with_cca(monkeypatch, tmp_path)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    calls = []

    def fake_urlopen(req, timeout):
        if "oauth2.googleapis.com/token" in req.full_url:
            return token_payload()
        calls.append(req.full_url)
        if len(calls) <= 2:
            raise _http_429(req.full_url)
        return _gemini_ok("after-backoff")

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    text, _ = agent._generate_markdown("sys", "prompt")
    assert text == "after-backoff"
    assert sleeps == [5, 15]
    assert len(calls) == 3


def test_cca_429_exhausted_gives_up(monkeypatch, tmp_path):
    agent = _agent_with_cca(monkeypatch, tmp_path)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    def fake_urlopen(req, timeout):
        if "oauth2.googleapis.com/token" in req.full_url:
            return token_payload()
        raise _http_429(req.full_url)

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    assert agent._generate_markdown("sys", "prompt") == (None, "")
    assert sleeps == [5, 15]


def test_cca_scrubs_rejected_system_text(monkeypatch, tmp_path):
    agent = _agent_with_cca(monkeypatch, tmp_path)
    seen = {}

    def fake_urlopen(req, timeout):
        if "oauth2.googleapis.com/token" in req.full_url:
            return token_payload()
        seen["body"] = json.loads(req.data.decode())
        return _gemini_ok()

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "billing header talk here is kept"}]}],
        "systemInstruction": {
            "parts": [
                {
                    "text": "x-anthropic-billing-header: abc\nYou are helpful.\n\nYou are a Claude agent, built on Anthropic's Claude Agent SDK.\n\nDo the task."
                }
            ]
        },
    }
    text, _ = agent._call_model_cca("gemini-3.8-flash", payload)
    assert text == "via-cca"
    sent = seen["body"]["request"]["systemInstruction"]["parts"][0]["text"]
    assert "x-anthropic-billing-header" not in sent
    assert "You are a Claude agent" not in sent
    assert "You are helpful." in sent and "Do the task." in sent
    # user turns untouched
    assert "billing header talk here is kept" in seen["body"]["request"]["contents"][0]["parts"][0]["text"]
    # original payload not mutated
    assert "x-anthropic-billing-header" in payload["systemInstruction"]["parts"][0]["text"]


def test_cca_400_does_not_retry(monkeypatch, tmp_path):
    agent = _agent_with_cca(monkeypatch, tmp_path)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    calls = []

    def fake_urlopen(req, timeout):
        if "oauth2.googleapis.com/token" in req.full_url:
            return token_payload()
        calls.append(req.full_url)
        raise _http_error(req.full_url, 400)

    monkeypatch.setattr(agent_mod.urllib.request, "urlopen", fake_urlopen)
    assert agent._generate_markdown("sys", "prompt") == (None, "")
    assert len(calls) == 1 and sleeps == []
