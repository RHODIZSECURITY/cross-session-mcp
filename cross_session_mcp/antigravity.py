"""Google Antigravity (Cloud Code Assist) OAuth + transport (no API key).

Mechanism mirrored from lidge-jun/opencodex (MIT), which mirrors CLIProxyAPI:
standard Google OAuth (PKCE) with the public OAuth client identifiers embedded
in the Antigravity desktop client (overridable via env, not user secrets),
then Cloud Code Assist project discovery via ``loadCodeAssist`` (``onboardUser``
fallback). Requests go to ``cloudcode-pa.googleapis.com/v1internal`` — never
to the Developer API — with the IDE User-Agent fingerprint and the project in
the envelope. Tokens/refresh are never logged.

Kept in this fork only: it impersonates a first-party client signature, so it
is intentionally NOT proposed upstream.
"""

import base64
import hashlib
import json
import os
import secrets
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from cross_session_mcp.utils import logger

# Public OAuth client identifiers embedded in the Antigravity desktop client
# (overridable via env). These identify the app to Google; they are not secrets.
CLIENT_ID = (
    os.environ.get("GOOGLE_ANTIGRAVITY_CLIENT_ID")
    or "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
)
# No default: push protection (and hygiene) forbids shipping it. Same
# public value as the Antigravity desktop client; export it (see README).
CLIENT_SECRET = os.environ.get("GOOGLE_ANTIGRAVITY_CLIENT_SECRET") or ""


def _secret_from_file() -> str:
    """Reads the client secret from a KEY=VALUE file (default ~/.ai/.antigravity_env). Empty when absent."""
    secret_file = Path(os.environ.get(SECRET_FILE_ENV_VAR, Path.home() / ".ai" / ".antigravity_env")).expanduser()
    try:
        if not secret_file.is_file():
            return ""
        for line in secret_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if line.startswith("GOOGLE_ANTIGRAVITY_CLIENT_SECRET="):
                return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def _client_secret() -> str:
    if CLIENT_SECRET:
        return CLIENT_SECRET
    from_file = _secret_from_file()
    if from_file:
        return from_file
    raise ValueError("set GOOGLE_ANTIGRAVITY_CLIENT_SECRET or store it in ~/.ai/.antigravity_env (see README)")


AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
PROD_API = "https://cloudcode-pa.googleapis.com"
DAILY_API = "https://daily-cloudcode-pa.googleapis.com"
API_VERSION = "v1internal"
API_BASE_ENV_VAR = "GOOGLE_ANTIGRAVITY_API_BASE"
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]
CALLBACK_PORT = 51121
CALLBACK_PATH = "/callback"
IDE_VERSION = "2.5.5"
CRED_ENV_VAR = "CROSS_SESSION_ANTIGRAVITY_FILE"
SECRET_FILE_ENV_VAR = "CROSS_SESSION_ANTIGRAVITY_SECRET_FILE"
REQUEST_TIMEOUT = 30
_REFRESH_SKEW = 5 * 60
_ONBOARD_ATTEMPTS = 5
_ONBOARD_POLL_SECS = 2


def api_base() -> str:
    return (os.environ.get(API_BASE_ENV_VAR) or PROD_API).rstrip("/")


def antigravity_user_agent() -> str:
    """IDE User-Agent fingerprint (env override wins)."""
    override = (os.environ.get("GOOGLE_ANTIGRAVITY_USER_AGENT") or "").strip()
    if override:
        return override
    import platform as _platform

    os_type = {"linux": "linux", "darwin": "darwin", "win32": "windows"}.get(sys.platform, sys.platform)
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(_platform.machine(), "amd64")
    return f"antigravity/ide/{IDE_VERSION} (os_type={os_type}; arch={arch}; aidev_client; auth_method=oauth)"


def default_cred_path() -> Path:
    return Path(os.environ.get(CRED_ENV_VAR, Path.home() / ".ai" / "antigravity_adc.json")).expanduser()


def load_cred(path: Path | None = None) -> dict[str, str] | None:
    """Reads the stored Antigravity credential. None when absent/invalid. Never raises, never logs secrets."""
    cred_file = path or default_cred_path()
    try:
        if not cred_file.is_file():
            return None
        mode = cred_file.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            logger.warning("%s is readable by other users; run: chmod 600 %s", cred_file, cred_file)
        data = json.loads(cred_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read Antigravity credential %s (%s)", cred_file, type(exc).__name__)
        return None
    if not isinstance(data, dict):
        return None
    creds = {key: data.get(key) for key in ("refresh_token", "project_id")}
    if not all(isinstance(value, str) and value for value in creds.values()):
        return None
    email = data.get("email")
    if isinstance(email, str) and email:
        creds["email"] = email
    return creds


def save_cred(refresh_token: str, project_id: str, email: str = "", path: Path | None = None) -> Path:
    """Persists the credential with mode 600. Returns the file path."""
    cred_file = path or default_cred_path()
    cred_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "refresh_token": refresh_token,
            "project_id": project_id,
            "email": email,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    with open(os.open(str(cred_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as handle:
        handle.write(payload)
    return cred_file


def _post(url: str, body: dict[str, str] | str, headers: dict[str, str], timeout: int) -> Any:
    """POSTs and returns decoded JSON. Raises on transport/HTTP errors (status only in logs)."""
    data = body.encode("utf-8") if isinstance(body, str) else urllib.parse.urlencode(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("Antigravity request to %s failed: HTTP %s", url.split("/")[2], exc.code)
        raise


def build_auth_url() -> tuple[str, str]:
    """Returns (auth_url, pkce_verifier). The user opens the URL, consents, pastes back the code."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    params = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": f"http://127.0.0.1:{CALLBACK_PORT}{CALLBACK_PATH}",
            "scope": " ".join(SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
        }
    )
    return f"{AUTH_ENDPOINT}?{params}", verifier


def extract_code(pasted: str) -> str:
    """Accepts the full redirect URL or the bare code pasted by the user."""
    text = (pasted or "").strip().strip("\"'")
    if "code=" in text:
        query = urllib.parse.urlsplit(text).query if "://" in text else text
        for key, value in urllib.parse.parse_qsl(query, keep_blank_values=True):
            if key == "code" and value:
                text = value.strip()
                break
    if not text or len(text) > 512 or not all(c.isalnum() or c in "-_./" for c in text):
        raise ValueError("no authorization code found in pasted text")
    return text


def _email_from_id_token(id_token: str | None) -> str:
    if not id_token or id_token.count(".") < 2:
        return ""
    try:
        payload = json.loads(base64.urlsafe_b64decode(id_token.split(".")[1] + "==").decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return ""
    email = payload.get("email") if isinstance(payload, dict) else None
    return email.lower() if isinstance(email, str) and email else ""


def _credentials_from_payload(payload: Any, refresh_fallback: str = "") -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("token response was not a JSON object")
    access = payload.get("access_token")
    if not isinstance(access, str) or not access:
        raise ValueError("token response did not include an access token")
    refresh = payload.get("refresh_token")
    if not isinstance(refresh, str) or not refresh:
        refresh = refresh_fallback
    if not refresh:
        raise ValueError("token response did not include a refresh token")
    try:
        lifetime = int(payload.get("expires_in", 3600))
    except (TypeError, ValueError):
        lifetime = 3600
    id_token = payload.get("id_token")
    return {
        "access": access,
        "refresh": refresh,
        "expires_at": str(time.time() + max(lifetime, _REFRESH_SKEW + 1)),
        "email": _email_from_id_token(id_token if isinstance(id_token, str) else None),
    }


def exchange_code(code: str, verifier: str, timeout: int = REQUEST_TIMEOUT) -> dict[str, str]:
    """Exchanges a pasted authorization code for tokens. Fail-closed, no secrets in errors."""
    secret = _client_secret()
    try:
        payload = _post(
            TOKEN_ENDPOINT,
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "client_secret": secret,
                "code": code,
                "redirect_uri": f"http://127.0.0.1:{CALLBACK_PORT}{CALLBACK_PATH}",
                "code_verifier": verifier,
            },
            {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            timeout,
        )
        return _credentials_from_payload(payload)
    except Exception as exc:
        raise ValueError(f"Antigravity code exchange failed ({type(exc).__name__})") from None


def _extract_project_id(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("cloudaicompanionProject", "projectId", "project"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"]:
            return value["id"]
    return ""


def _load_project(access: str, timeout: int) -> str:
    try:
        data = _post(
            f"{api_base()}/{API_VERSION}:loadCodeAssist",
            json.dumps({"metadata": {"ideType": "ANTIGRAVITY"}}),
            {
                "Authorization": f"Bearer {access}",
                "Accept": "*/*",
                "Content-Type": "application/json",
                "User-Agent": antigravity_user_agent(),
            },
            timeout,
        )
    except Exception:
        return ""
    return _extract_project_id(data)


def _onboard_project(access: str, timeout: int) -> str:
    for _ in range(_ONBOARD_ATTEMPTS):
        try:
            data = _post(
                f"{DAILY_API}/{API_VERSION}:onboardUser",
                json.dumps(
                    {
                        "tier_id": "free-tier",
                        "metadata": {"ide_type": "ANTIGRAVITY", "ide_name": "antigravity", "ide_version": IDE_VERSION},
                    }
                ),
                {
                    "Authorization": f"Bearer {access}",
                    "Accept": "*/*",
                    "Content-Type": "application/json",
                    "User-Agent": antigravity_user_agent(),
                },
                timeout,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                time.sleep(_ONBOARD_POLL_SECS)
                continue
            return ""
        except Exception:
            return ""
        if isinstance(data, dict) and data.get("done") is True:
            project = _extract_project_id(data.get("response"))
            if project:
                return project
        time.sleep(_ONBOARD_POLL_SECS)
    return ""


def discover_project(access: str, timeout: int = REQUEST_TIMEOUT) -> str:
    """Discovers the Cloud Code Assist project (loadCodeAssist, onboardUser fallback). Empty when none."""
    return _load_project(access, timeout) or _onboard_project(access, timeout)


def refresh_access(refresh_token: str, timeout: int = REQUEST_TIMEOUT) -> dict[str, str]:
    """Refreshes the access token. Fail-closed, no secrets in errors."""
    if not refresh_token:
        raise ValueError("no refresh token stored")
    secret = _client_secret()
    try:
        payload = _post(
            TOKEN_ENDPOINT,
            {
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "client_secret": secret,
                "refresh_token": refresh_token,
            },
            {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
            timeout,
        )
        return _credentials_from_payload(payload, refresh_token)
    except Exception as exc:
        raise ValueError(f"Antigravity token refresh failed ({type(exc).__name__})") from None


class AntigravityAuth:
    """Cached Cloud Code Assist credential (same duck-type as OAuthTokenCache)."""

    def __init__(self, creds: dict[str, str] | None = None, timeout: int = REQUEST_TIMEOUT):
        self._refresh = (creds or {}).get("refresh_token", "")
        self._project = (creds or {}).get("project_id", "")
        self._timeout = timeout
        self._access = ""
        self._expires_at = 0.0

    @property
    def configured(self) -> bool:
        return bool(self._refresh and self._project)

    @property
    def project(self) -> str:
        return self._project

    def invalidate(self) -> None:
        self._access = ""
        self._expires_at = 0.0

    def token(self, timeout: int) -> str:
        """A valid access token, refreshing when missing or expiring. Empty string on failure."""
        if self._access and time.time() < self._expires_at - _REFRESH_SKEW:
            return self._access
        self.invalidate()
        try:
            creds = refresh_access(self._refresh, timeout)
        except ValueError as exc:
            logger.warning("%s", exc)
            return ""
        self._access = creds["access"]
        self._expires_at = float(creds["expires_at"])
        return self._access

    def refresh(self, timeout: int) -> bool:
        return bool(self.token(timeout))


def cca_envelope(model: str, request_body: dict[str, Any], project: str) -> dict[str, Any]:
    """Wraps the flat Gemini body in the Cloud Code Assist envelope."""
    request = dict(request_body)
    request.setdefault("sessionId", str(uuid.uuid4()))
    return {
        "model": model,
        "userAgent": "antigravity",
        "requestType": "agent",
        "project": project,
        "requestId": f"agent-{uuid.uuid4()}",
        "request": request,
    }


def cca_headers(access: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access}",
        "User-Agent": antigravity_user_agent(),
    }


def cca_unwrap(data: Any) -> dict[str, Any] | None:
    """Unwraps the CCA `response` envelope. None when the shape is wrong."""
    if not isinstance(data, dict):
        return None
    wrapped = data.get("response")
    if not isinstance(wrapped, dict):
        return None
    return wrapped
