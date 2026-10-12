"""Shared utilities: date normalization, secret redaction and path filters."""

import datetime
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

logger = logging.getLogger("cross_session_mcp")


def setup_logging() -> None:
    """Sends logging to stderr (stdout is reserved for the MCP protocol)."""
    level = os.environ.get("CROSS_SESSION_LOG_LEVEL", "WARNING").upper()
    if not logger.handlers:
        handler = logging.StreamHandler()  # stderr by default
        handler.setFormatter(logging.Formatter("[cross-session-mcp] %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.WARNING))


# -----------------------------------------------------------------------------
# Fechas
# -----------------------------------------------------------------------------
_FRACTION_RE = re.compile(r"(\.\d{6})\d+")


def to_utc_iso(value: Any) -> str | None:
    """Normalizes epoch (s/ms/ns), datetime or ISO strings to 'YYYY-MM-DDTHH:MM:SS.ffffff+00:00'.

    Dates without a time zone are read as the machine's local time.
    Always returning fixed-format UTC lets SQLite sort by string.
    """
    if value is None or value == "":
        return None

    dt: datetime.datetime | None = None
    if isinstance(value, datetime.datetime):
        dt = value
    elif isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\d+(\.\d+)?", value.strip())):
        num = float(value)
        if num > 1e17:  # nanosegundos
            num /= 1e9
        elif num > 1e14:  # microsegundos
            num /= 1e6
        elif num > 1e11:  # milisegundos
            num /= 1e3
        dt = datetime.datetime.fromtimestamp(num, tz=datetime.timezone.utc)
    elif isinstance(value, str):
        text = value.strip().replace(" ", "T", 1)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        text = _FRACTION_RE.sub(r"\1", text)  # Python only supports microseconds
        try:
            dt = datetime.datetime.fromisoformat(text)
        except ValueError:
            return None

    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.astimezone()  # local time -> zone-aware
    return dt.astimezone(datetime.timezone.utc).isoformat(timespec="microseconds")


def now_utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds")


def iso_to_epoch(value: str | None) -> float | None:
    """Seconds since epoch of an ISO 8601 date (as normalized by to_utc_iso); None if unreadable."""
    if not value:
        return None
    try:
        dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.timestamp()


# -----------------------------------------------------------------------------
# Secret redaction
# -----------------------------------------------------------------------------
REDACTED = "[REDACTED]"

_SECRET_PATTERNS = [
    # Whole private key blocks
    (re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"), REDACTED),
    # Credentials embedded in URLs: scheme://user:password@host
    (re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s:/@]+:)[^\s@/]+(@)"), r"\1" + REDACTED + r"\2"),
    # Sensitive query parameters: ?key=... &token=...
    (re.compile(r"([?&](?:key|api_key|apikey|token|access_token|secret)=)[^&\s\"']+", re.I), r"\1" + REDACTED),
    # Cabeceras Authorization: Bearer ...
    (re.compile(r"(\bBearer\s+)[A-Za-z0-9._~+/=-]{16,}", re.I), r"\1" + REDACTED),
    # Proveedores conocidos
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), REDACTED),  # Google API key
    (re.compile(r"\bAQ\.[A-Za-z0-9_-]{20,}"), REDACTED),  # Google (formato AQ.)
    (re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"), REDACTED),  # Google OAuth access token
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"), REDACTED),  # Anthropic / OpenAI
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), REDACTED),  # GitHub
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"), REDACTED),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), REDACTED),  # GitLab
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), REDACTED),  # AWS access key id
    (re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"), REDACTED),  # Slack
    (re.compile(r"\b(?:sk|rk|pk)_live_[A-Za-z0-9]{20,}"), REDACTED),  # Stripe
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), REDACTED),  # JWT
    # Generic assignments: FOO_API_KEY=value, password: value, "secret": "value"
    (
        re.compile(
            r"(?i)(\b[\w.-]*(?:api[_-]?key|secret|token|passw(?:or)?d|pwd|credential)[\w.-]*[\"']?\s*[:=]\s*[\"']?)"
            r"(?!\[REDACTED\])([^\s\"',;]{8,})"
        ),
        r"\1" + REDACTED,
    ),
]


def redact(text: str | None) -> str:
    """Replaces known secrets with [REDACTED]. Must never raise."""
    if not text:
        return text or ""
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
# Phone candidate: digits with spaces, dashes or parentheses (and an optional '+'). Validated in code.
_PHONE_CANDIDATE_RE = re.compile(r"(?<![\w.:/-])\+?\(?\d[\d ()-]{7,20}\d(?![\w:-]|\.\d)")
_DATE_LIKE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _phone_replacement(match: "re.Match") -> str:
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if not 9 <= len(digits) <= 15 or _DATE_LIKE_RE.match(raw):
        return raw
    if raw.startswith("+"):
        return REDACTED
    if re.fullmatch(r"\d+", raw):  # no separators: only Spanish mobile/landline numbers (9 digits)
        return REDACTED if re.fullmatch(r"[6-9]\d{8}", raw) else raw
    return REDACTED if re.search(r"[ -]", raw) else raw


def redact_pii(text: str | None) -> str:
    """Replaces email addresses and phone numbers with [REDACTED].

    Applied to digests and memory (not in `redact()`, so handoffs and turns stay untouched).
    Conservative: it leaves hashes, session IDs, dates and version numbers alone.
    """
    if not text:
        return text or ""
    text = _EMAIL_RE.sub(REDACTED, text)
    return _PHONE_CANDIDATE_RE.sub(_phone_replacement, text)


def clean_text(text: str | None) -> str:
    """Secrets + personal data: the full barrier for everything persisted in memory."""
    return redact_pii(redact(text))


# -----------------------------------------------------------------------------
# Text and paths
# -----------------------------------------------------------------------------
def truncate(text: str | None, limit: int) -> str:
    text = (text or "").strip()
    if limit and len(text) > limit:
        return text[:limit] + f"... [truncated, {len(text) - limit} characters omitted]"
    return text


def file_uri_to_path(uri: str) -> str:
    """Turns 'file:///path%20with%20spaces' into '/path with spaces'."""
    uri = (uri or "").strip()
    if uri.startswith("file://"):
        return unquote(urlparse(uri).path)
    return uri


def normalize_workspace_filter(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    value = file_uri_to_path(value.strip())
    if value.startswith("~"):
        value = os.path.expanduser(value)
    return value.rstrip("/") or "/"


def workspace_matches(filter_value: str | None, workspace: str) -> bool:
    """Absolute path -> matches the path or its subdirectories. Relative text -> substring."""
    f = normalize_workspace_filter(filter_value)
    if not f:
        return True
    workspace = (workspace or "").rstrip("/")
    if f.startswith("/"):
        return workspace == f or workspace.startswith(f + "/")
    return f.lower() in workspace.lower()


def config_home() -> Path:
    """Folder where desktop apps keep their settings: `~/Library/Application Support` on macOS, `~/.config` elsewhere."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".config"


def env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def is_safe_session_id(session_id: str) -> bool:
    """Keeps a tampered ID ('../', '*') from being used to build paths or globs."""
    return bool(session_id) and bool(_SAFE_ID_RE.match(session_id)) and ".." not in session_id


DEFAULT_EXCLUDED_WORKSPACES = ["~/.claude-mem/observer-sessions"]


def excluded_workspaces() -> list:
    """Workspaces that are not indexed (internal plugin sessions, e.g. claude-mem).

    CROSS_SESSION_EXCLUDE_WORKSPACES replaces the default list (':' separator; empty = none).
    """
    raw = os.environ.get("CROSS_SESSION_EXCLUDE_WORKSPACES")
    items = DEFAULT_EXCLUDED_WORKSPACES if raw is None else [p for p in raw.split(os.pathsep) if p.strip()]
    return [normalize_workspace_filter(p) for p in items]


def is_excluded_workspace(workspace: str) -> bool:
    return any(workspace_matches(ex, workspace) for ex in excluded_workspaces())
