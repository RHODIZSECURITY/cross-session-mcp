"""Tests for date normalization, secret redaction and filters."""

import datetime
import sys
import time

import pytest

from cross_session_mcp.utils import (
    config_home,
    is_safe_session_id,
    redact,
    to_utc_iso,
    workspace_matches,
)

# Fake values built by concatenation so they do not look like real secrets
PEM_HEADER = "-----" + "BEGIN RSA PRIVATE KEY" + "-----"
PEM_FOOTER = "-----" + "END RSA PRIVATE KEY" + "-----"


@pytest.mark.parametrize("value,expected", [
    ("2026-09-30 00:08:48.581713291+00:00", "2026-09-30T00:08:48.581713+00:00"),
    ("2026-09-29T22:18:40Z", "2026-09-29T22:18:40.000000+00:00"),
    ("2026-09-29T18:18:40-04:00", "2026-09-29T22:18:40.000000+00:00"),
    (1790200000, "2026-09-23T21:46:40.000000+00:00"),
    (1790200000000, "2026-09-23T21:46:40.000000+00:00"),
    ("1790200000", "2026-09-23T21:46:40.000000+00:00"),
    (None, None),
    ("garbage", None),
])
def test_to_utc_iso(value, expected):
    assert to_utc_iso(value) == expected


def test_naive_dates_are_local():
    naive = datetime.datetime(2026, 1, 1, 12, 0, 0)
    assert to_utc_iso(naive) == naive.astimezone().astimezone(datetime.timezone.utc).isoformat(timespec="microseconds")


@pytest.mark.parametrize("secret", [
    "AIza" + "x" * 35,
    "AQ." + "x" * 48,
    "sk-ant-api03-" + "x" * 40,
    "sk-proj-" + "x" * 40,
    "ghp_" + "x" * 36,
    "glpat-" + "x" * 20,
    "AKIA" + "X" * 16,
    "xoxb-" + "0" * 10 + "-" + "x" * 10,
    "eyJ" + "x" * 20 + ".eyJ" + "x" * 20 + "." + "x" * 30,
    "sk-or-v1-" + "a1" * 32,  # OpenRouter, covered by the sk- pattern
    "sk-ant-admin01-" + "x" * 40,
    "hf_" + "aB3" * 11 + "x",
    "npm_" + "aB3" * 12,
    "pypi-AgEIcHlwaS5vcmc" + "x" * 60,
    "dop_v1_" + "ab12" * 16,
    "doo_v1_" + "ab12" * 16,
    "sntrys_" + "eyJpYXQiOjE" + "x" * 40,
    "sntryu_" + "ab12" * 16,
    "gsk_" + "x" * 52,
    "xai-" + "x" * 80,
    "r8_" + "x" * 37,
    "lin_api_" + "x" * 40,
    "shpat_" + "ab12" * 8,
    "SG." + "x" * 22 + "." + "y" * 43,
])
def test_redact_known_tokens(secret):
    out = redact(f"text before {secret} text after")
    assert secret not in out
    assert "[REDACTED]" in out
    assert out.startswith("text before") and out.endswith("text after")


@pytest.mark.parametrize("text,leak", [
    ("export GEMINI_API_KEY=xxxxxxxxxxxx", "xxxxxxxxxxxx"),
    ('"password": "xxxxxxxxxxxx"', "xxxxxxxxxxxx"),
    ("DB_PASSWORD: 'xxxx-xxxx-xxxx'", "xxxx-xxxx-xxxx"),
    ("postgres://admin:xxxxxxxx@db:5432/app", "xxxxxxxx"),
    ("https://api.example.com/v1?key=xxxxxxxxxxxx&x=1", "xxxxxxxxxxxx"),
    ("Authorization: Bearer " + "x" * 30, "x" * 30),
    (PEM_HEADER + "\nMIIExxxx\n" + PEM_FOOTER, "MIIExxxx"),
])
def test_redact_assignments(text, leak):
    assert leak not in redact(text)


@pytest.mark.parametrize("text", [
    "maxOutputTokens: 8192",
    "def get_token(): return None",
    "The user asked to use design tokens",
    "/path/to/cross-session-mcp",
    "from huggingface_hub import hf_hub_download",
    "npm_config_cache=/tmp/npm",
    "pip install --index-url https://example.org/pypi-simple/ pkg",
    "dop_v1_test is a fixture name",
    "r8_client and gsk_settings are variable names",
    "commit 3f2c9a1b7d4e5f60718293a4b5c6d7e8f9012345",
    "id 123e4567-e89b-12d3-a456-426614174000",
])
def test_redact_keeps_normal_text(text):
    assert redact(text) == text


def test_redact_never_fails_on_empty_input():
    assert redact(None) == ""
    assert redact("") == ""


def test_redact_is_fast_on_large_input():
    chunk = "token: abc hf_hub sk- npm_ SG. https://u@h/ eyJx.y " * 20
    text = (chunk + "x" * 200 + "\n") * (1_000_000 // (len(chunk) + 201))
    start = time.perf_counter()
    redact(text)
    assert time.perf_counter() - start < 2


def test_workspace_matches():
    assert workspace_matches(None, "/a")
    assert workspace_matches("/a/b", "/a/b/c")
    assert workspace_matches("/a/b/", "/a/b")
    assert not workspace_matches("/a/b", "/a/bc")
    assert workspace_matches("my-app", "/x/MY-APP/y")
    assert workspace_matches("file:///a/b", "/a/b")


@pytest.mark.parametrize("sid,ok", [
    ("4cbb572e-c928-4b15-94af-48353e500530", True),
    ("ses_abc123", True),
    ("../etc", False),
    ("a/b", False),
    ("*", False),
    ("", False),
])
def test_is_safe_session_id(sid, ok):
    assert is_safe_session_id(sid) is ok


# ---------------------------------------------------------------- redact_pii
from cross_session_mcp.utils import redact_pii  # noqa: E402


@pytest.mark.parametrize("text", [
    "write to john.doe@example.com now",
    "call +34 612 345 678",  # Spanish mobile
    "or 612 345 678",
    "or 612-345-678.",
    "or 612345678",
    "US +1 (415) 555-2671",
])
def test_redact_pii_positive(text):
    out = redact_pii(text)
    assert "[REDACTED]" in out
    assert "@" not in out
    assert not any(ch.isdigit() for ch in out)


@pytest.mark.parametrize("text", [
    "sha 1234567890abcdef12",
    "session 4cbb572e-1234-5678-9abc-def012345678",
    "version v1.2.3 and 10.20.30",
    "date 2026-09-01 10:00:00",
    "epoch 1790300000",
    "ip 192.168.100.200",
    "2026-09-01T10:00:00.000000+00:00",
    "v3 (schema 3)",
])
def test_redact_pii_no_false_positives(text):
    assert redact_pii(text) == text


@pytest.mark.parametrize("platform, expected", [("darwin", "Library/Application Support"), ("linux", ".config"),
                                                ("win32", ".config")])
def test_config_home(tmp_path, monkeypatch, platform, expected):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", platform)
    assert config_home() == tmp_path / expected
