"""Business logic shared by the MCP server and the CLI."""

import dataclasses
import os
import socket
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from cross_session_mcp import i18n, memory
from cross_session_mcp.agent import GeminiHandoffAgent, build_offline_handoff
from cross_session_mcp.db import ConcurrentUpdateError, CrossSessionDB
from cross_session_mcp.models import (
    HARNESSES,
    HandoffResult,
    MemoryUpdateResult,
    SessionDetail,
    SessionSummary,
    SessionTurn,
)
from cross_session_mcp.readers import get_all_readers
from cross_session_mcp.readers.base import BaseSessionReader
from cross_session_mcp.utils import (
    clean_text,
    is_safe_session_id,
    iso_to_epoch,
    logger,
    normalize_workspace_filter,
    now_utc_iso,
    redact,
    redact_pii,
)

MAX_BACK = 10  # previous sessions that can be resumed and that the bank maintains
CURRENT_SESSION_MARGIN = 60  # s: a session created after the MCP started (minus this margin) is the current one
IDLE_BEFORE_REFRESH = 15 * 60  # s: idle time required to regenerate a stale handoff in the background
RETRY_BACKOFF = 30 * 60  # s: after an LLM failure, the background does not retry that session/consolidation sooner


def select_turns(
    turns: Sequence[SessionTurn], query: str | None, include_tools: bool, max_turns: int,
) -> tuple[list[SessionTurn], int]:
    """Turns to show and how many are left out. With `query`: only those containing the text (ignoring case
    and accents) with one turn of context on each side; `max_turns` applies after the filter."""
    base = [t for t in turns if include_tools or t.role != "tool"]
    if query:
        needle = memory.normalize(query)
        hits = [i for i, t in enumerate(base) if needle in memory.normalize(t.content)]
        keep = sorted({j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(base)})
        selected = [base[j] for j in keep]
    else:
        selected = list(base)
    shown = selected[-max_turns:]
    return shown, len(base) - len(shown)


class SessionError(Exception):
    """Expected error (invalid input, missing session...) with a message for the user."""


class CrossSessionService:
    def __init__(
        self,
        db: CrossSessionDB | None = None,
        agent: GeminiHandoffAgent | None = None,
        readers_factory: Callable[[], list[BaseSessionReader]] = get_all_readers,
    ):
        self.db = db or CrossSessionDB()
        self._agent = agent
        self.readers_factory = readers_factory
        self.sync_ttl = float(os.environ.get("CROSS_SESSION_SYNC_TTL", "60"))
        self.sync_limit = int(os.environ.get("CROSS_SESSION_SYNC_LIMIT", "50"))
        self._sync_lock = threading.Lock()
        self._memory_locks: dict[str, threading.Lock] = {}
        self._memory_locks_guard = threading.Lock()
        self.digest_workers = 3
        # Only the MCP server sets these two: the CLI has no "current session" and no background threads.
        self.process_started_at: float | None = None
        self.background = False
        self._maintenance: dict[str, threading.Thread] = {}
        self._maintenance_guard = threading.Lock()
        # Recent LLM failures (in process memory): they stop the background from retrying in a loop
        self._digest_failures: dict[tuple[str, str, str], float] = {}
        self._consolidation_failures: dict[str, float] = {}

    @property
    def agent(self) -> GeminiHandoffAgent:
        if self._agent is None:  # lazy: credentials are not read until needed
            self._agent = GeminiHandoffAgent()
        return self._agent

    # ------------------------------------------------------------------ index
    def sync(self, force: bool = False) -> int:
        with self._sync_lock:
            if not force and self.db.seconds_since_last_sync() < self.sync_ttl:
                return 0
            return self.db.sync_from_readers(self.sync_limit, readers=self.readers_factory())

    def list_sessions(
        self,
        workspace: str | None = None,
        harness: str | None = None,
        limit: int = 15,
        refresh: bool = False,
    ) -> list[SessionSummary]:
        if harness:
            self._validate_harness(harness)
        self.sync(force=refresh)
        return self.db.query_sessions(workspace_filter=workspace, harness_filter=harness, limit=limit)

    # ------------------------------------------------------------------ resolution
    def _validate_harness(self, harness: str) -> None:
        if harness not in HARNESSES:
            raise SessionError(f"Unsupported harness: '{harness}'. Options: {', '.join(HARNESSES)}.")

    def reader(self, harness: str) -> BaseSessionReader:
        self._validate_harness(harness)
        return next(r for r in self.readers_factory() if r.name == harness)

    def resolve(self, session_id: str, harness: str | None = None) -> tuple[str, str]:
        """Resolves (full session_id, harness), accepting ID prefixes and an optional harness."""
        session_id = (session_id or "").strip()
        if not is_safe_session_id(session_id):
            raise SessionError(f"Invalid session ID: '{session_id}'.")
        if harness:
            self._validate_harness(harness)

        matches = self.db.find_sessions(session_id, harness)
        if not matches:
            self.sync(force=True)
            matches = self.db.find_sessions(session_id, harness)

        if not matches:
            if harness:
                return session_id, harness  # it may be an old session outside the index
            raise SessionError(
                f"Session '{session_id}' not found in the index. Pass the 'harness' parameter "
                "or use list_cross_sessions to see the available IDs."
            )
        exact = [m for m in matches if m.session_id == session_id]
        candidates = exact or matches
        if len({(m.session_id, m.harness) for m in candidates}) > 1:
            options = ", ".join(f"{m.harness}:{m.session_id}" for m in candidates[:5])
            raise SessionError(f"The ID '{session_id}' is ambiguous. Matches: {options}. Give more characters or the harness.")
        return candidates[0].session_id, candidates[0].harness

    def get_detail(self, session_id: str, harness: str | None = None, max_turns: int | None = 25) -> SessionDetail:
        sid, hname = self.resolve(session_id, harness)
        detail = self.reader(hname).get_session_detail(sid, max_turns=max_turns)
        if detail is None:
            raise SessionError(f"Could not read session '{sid}' from the '{hname}' store.")
        return detail

    # ------------------------------------------------------------------ project sessions
    def _is_current(self, s: SessionSummary) -> bool:
        """Session created after this MCP process started: it is the agent's ongoing session."""
        if self.process_started_at is None:
            return False
        created = iso_to_epoch(s.created_at) or iso_to_epoch(s.updated_at)
        return created is not None and created >= self.process_started_at - CURRENT_SESSION_MARGIN

    def recent_sessions(self, workspace: str, n: int) -> list[SessionSummary]:
        """Last `n` sessions of the EXACT workspace, from any harness, excluding the current session."""
        self.sync()
        candidates = self.db.window_sessions(workspace, n + 10)
        return [s for s in candidates if not self._is_current(s)][:n]

    def resolve_back(self, workspace: str, back: int = 1) -> SessionSummary:
        ws = self._workspace(workspace)
        if not 1 <= int(back) <= MAX_BACK:
            raise SessionError(f"'back' must be between 1 and {MAX_BACK}.")
        sessions = self.recent_sessions(ws, MAX_BACK)
        if not sessions:
            raise SessionError(f"No previous sessions indexed for the exact workspace '{ws}'.")
        if back > len(sessions):
            raise SessionError(f"Only {len(sessions)} previous session(s) in '{ws}'; ask for back ≤ {len(sessions)}.")
        return sessions[back - 1]

    # ------------------------------------------------------------------ Extended Handoff
    @staticmethod
    def _handoff_header(s: SessionSummary) -> str:
        when = (s.updated_at or "")[:16].replace("T", " ")
        title = clean_text(s.title) or i18n.t("untitled")
        return f"# {i18n.t('handoff_title')} — {title} ({s.harness} · {s.session_id[:8]} · {when} UTC)\n\n"

    def _save_digest(self, s: SessionSummary, digest: dict[str, Any]) -> dict[str, Any]:
        handoff = digest.pop("handoff_markdown")
        model = digest.pop("model_used", "")
        self.db.save_digest(s.session_id, s.harness, s.workspace_path, s.updated_at, digest, handoff, model)
        return {"handoff_md": handoff, "model_used": model}

    def get_handoff(
        self,
        session_id: str | None = None,
        harness: str | None = None,
        focus_area: str | None = None,
        force_refresh: bool = False,
        workspace: str | None = None,
        back: int = 1,
    ) -> HandoffResult:
        """Extended Handoff of a session, by ID or by position (`workspace` + `back`; the ID wins).

        Without focus: the saved one if it is up to date; otherwise it is generated in this call and saved (it
        also feeds the memory). With focus: a separate call that is not saved. Without LLM: local extractor,
        not saved."""
        if session_id:
            detail = self.get_detail(session_id, harness, max_turns=None)
        elif workspace:
            target = self.resolve_back(workspace, back)
            detail = self.reader(target.harness).get_session_detail(target.session_id, max_turns=None)
            if detail is None:
                raise SessionError(f"Could not read session '{target.session_id}' from the '{target.harness}' store.")
            # The index may lag behind the store or run ahead of it (sync): the most recent date wins
            if target.updated_at and (target.updated_at > (detail.summary.updated_at or "")):
                detail.summary = dataclasses.replace(detail.summary, updated_at=target.updated_at)
        else:
            raise SessionError("Pass session_id, or workspace_path (and optionally back).")
        s = detail.summary
        focus = (focus_area or "").strip()
        header = self._handoff_header(s)

        def result(markdown: str, model: str, from_cache: bool = False, generated_at: str | None = None) -> HandoffResult:
            return HandoffResult(s.session_id, s.harness, s.workspace_path, header + clean_text(markdown), model,
                                 generated_at or now_utc_iso(), from_cache=from_cache, focus_area=focus,
                                 source_updated_at=s.updated_at)

        has_key = getattr(self.agent, "available", True)
        reason, offline_model = ("llm_failed", "offline-fallback") if has_key else ("no_key", "offline-extractor")
        if focus:
            out = self.agent.generate_focused_handoff(detail, focus)
            if out:
                return result(out[0], out[1])
            return result(build_offline_handoff(detail, focus, reason), offline_model)

        if not force_refresh:
            cached = self.db.get_digest(s.session_id, s.harness, s.updated_at)
            if cached:
                return result(cached["handoff_md"], cached["model_used"], from_cache=True, generated_at=cached["generated_at"])
        digest = self.agent.generate_digest(detail) if has_key else None
        if digest:
            saved = self._save_digest(s, digest)
            return result(saved["handoff_md"], saved["model_used"])
        return result(build_offline_handoff(detail, reason=reason), offline_model)

    # ------------------------------------------------------------------ project memory
    @staticmethod
    def _workspace(value: str | None) -> str:
        ws = normalize_workspace_filter(value)
        if not ws or not ws.startswith("/"):
            raise SessionError("workspace_path must be an absolute path (e.g. /home/user/project).")
        return ws

    def _memory_lock(self, workspace: str) -> threading.Lock:
        with self._memory_locks_guard:
            return self._memory_locks.setdefault(workspace, threading.Lock())

    @staticmethod
    def _refs(window: list[dict[str, Any]]) -> dict[str, str]:
        """short ref (harness:first 8 characters of the ID) -> full key; uses the full ID on a collision."""
        refs: dict[str, str] = {}
        for w in window:
            ref = f"{w['harness']}:{w['session_id'][:8]}"
            if ref in refs:
                ref = w["key"]
            refs[ref] = w["key"]
        return refs

    def _require_llm(self) -> None:
        if not getattr(self.agent, "available", True):
            raise SessionError(
                "The project memory needs the LLM and there is no Gemini credential: set GEMINI_API_KEY, "
                "~/.ai/.gemini_env, or OAuth via `gcloud auth application-default-login`. "
                "The existing memory was not modified."
            )

    def _generate_missing_digests(self, pending: list[SessionSummary]) -> tuple[int, list[str], list[SessionSummary]]:
        """Generates the pending digests in parallel and saves each one as soon as it finishes.

        Returns (generated, LLM failures, unreadable or skipped sessions)."""
        failures: list[str] = []
        unreadable: list[SessionSummary] = []

        def work(s: SessionSummary) -> bool:
            detail = self.reader(s.harness).get_session_detail(s.session_id, max_turns=None)
            if detail is None or not detail.turns:
                unreadable.append(s)  # purged or empty in its store: it does not count for the window
                return False
            digest = self.agent.generate_digest(detail)
            if digest is None:
                failures.append(f"{s.harness}:{s.session_id[:8]}")
                self._digest_failures[(s.harness, s.session_id, s.updated_at or "")] = time.time()
                return False
            self._save_digest(s, digest)
            return True

        with ThreadPoolExecutor(max_workers=self.digest_workers, thread_name_prefix="digest") as pool:
            done = list(pool.map(work, pending))
        return sum(done), failures, unreadable

    def _persist(
        self, workspace: str, base_version: int, entries: list[dict[str, Any]], window: list[dict[str, Any]],
        state: list[str], model_used: str, note: str | None = None,
    ) -> tuple[int, str]:
        now = now_utc_iso()
        version = base_version + 1
        md = memory.render_markdown(workspace, entries, window, state, model_used, version, now)
        self.db.save_memory_version(workspace, base_version, md, window, state, model_used, entries, note=note)
        return version, md

    def _consolidate_once(
        self, workspace: str, window: list[dict[str, Any]], digests: dict[str, dict[str, Any]], n: int,
    ) -> tuple[int, str, str]:
        """One consolidation over the current memory. Returns (version, markdown, model)."""
        current = self.db.get_memory(workspace)
        base = current["version"] if current else 0
        entries = self.db.get_entries(workspace)
        refs = self._refs(window)
        ref_by_key = {v: k for k, v in refs.items()}
        prompt_digests = [
            {"ref": ref_by_key[w["key"]], "harness": w["harness"], "updated_at": w["updated_at"],
             "digest": digests[w["key"]]["digest"]}
            for w in window if w["relevant"]
        ]
        out = self.agent.consolidate_memory(entries, prompt_digests, ref_by_key, workspace, n)
        if out is None:
            raise SessionError(
                "No model returned a valid consolidation. The existing memory was not modified "
                "and the generated digests are kept: try again."
            )
        model = out.get("model_used", "")
        new_entries, state = memory.apply_consolidation(entries, out, window, refs, now_utc_iso())
        version, md = self._persist(workspace, base, new_entries, window, state, model)
        return version, md, model

    def update_project_memory(
        self,
        workspace: str,
        sessions: int | None = None,
        force_refresh: bool = False,
        export_path: str | None = None,
        reuse_stale: bool = False,
    ) -> MemoryUpdateResult:
        """With `reuse_stale` (background maintenance) a stale digest is reused as is and the window records
        its source date: only the missing digests are generated."""
        ws = self._workspace(workspace)
        n = max(1, min(20, int(sessions))) if sessions else memory.default_window()
        with self._memory_lock(ws):
            summaries = self.recent_sessions(ws, n)
            if not summaries:
                raise SessionError(f"No sessions indexed for the exact workspace '{ws}'.")

            if reuse_stale:  # the background does not retry too soon what has just failed
                summaries = [s for s in summaries if not self._recently_failed(s)]
                if not summaries:
                    raise SessionError(f"The sessions of '{ws}' failed recently; they will be retried later.")

            digests: dict[str, dict[str, Any]] = {}
            pending: list[SessionSummary] = []
            for i, s in enumerate(summaries):
                cached = None
                if not force_refresh:
                    cached = self.db.get_digest(s.session_id, s.harness, None if reuse_stale else s.updated_at)
                if cached:
                    digests[memory.session_key(s.harness, s.session_id)] = cached
                    if reuse_stale:
                        summaries[i] = dataclasses.replace(s, updated_at=cached["source_updated_at"])
                else:
                    pending.append(s)

            current = self.db.get_memory(ws)
            if current and not pending and not force_refresh:
                signature = memory.window_signature([memory.window_entry(s, digests[memory.session_key(s.harness, s.session_id)])
                                                     for s in summaries])
                if signature == memory.window_signature(current["window"]):
                    result = MemoryUpdateResult(ws, current["rendered_md"], current["version"], up_to_date=True,
                                                digests_cached=len(summaries), model_used=current["model_used"])
                    result.exported_to = self._export(current["rendered_md"], export_path)
                    return result

            self._require_llm()
            generated, failures, unreadable = self._generate_missing_digests(pending)
            if failures:
                raise SessionError(
                    f"Could not generate the digest of {len(failures)} session(s): {', '.join(failures)}. "
                    "The existing memory was not modified; the good digests are kept, "
                    "run the update again to complete the missing ones."
                )
            skipped = {(s.session_id, s.harness) for s in unreadable}
            summaries = [s for s in summaries if (s.session_id, s.harness) not in skipped]
            pending = [s for s in pending if (s.session_id, s.harness) not in skipped]
            if not summaries:
                raise SessionError(f"No session of '{ws}' could be read from its source store.")
            for s in pending:
                digests[memory.session_key(s.harness, s.session_id)] = self.db.get_digest(s.session_id, s.harness)

            window = [memory.window_entry(s, digests[memory.session_key(s.harness, s.session_id)]) for s in summaries]
            llm_calls = len(pending)
            warnings: list[str] = []
            if unreadable:
                warnings.append(f"{len(unreadable)} unreadable session(s) left out of the window.")
            try:
                llm_calls += 1
                version, md, model = self._consolidate_once(ws, window, digests, n)
            except ConcurrentUpdateError:
                warnings.append("The memory changed during the consolidation: it was retried over the new base.")
                try:
                    llm_calls += 1
                    version, md, model = self._consolidate_once(ws, window, digests, n)
                except ConcurrentUpdateError as exc:
                    raise SessionError(
                        f"Another update modified the memory at the same time ({exc}). Try again."
                    ) from exc

            result = MemoryUpdateResult(
                ws, md, version, digests_generated=generated, digests_cached=len(summaries) - len(pending),
                llm_calls=llm_calls, model_used=model, warnings=warnings,
            )
            result.exported_to = self._export(md, export_path)
            return result

    def _export(self, markdown: str, export_path: str | None) -> str | None:
        """Writes the memory only to an explicit path, with mode 600."""
        if not export_path:
            return None
        path = Path(export_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(redact_pii(redact(markdown)))
            path.chmod(0o600)  # in case the file already existed with another mode
        except OSError as exc:
            raise SessionError(f"Could not export to '{path}': {exc}") from exc
        return str(path)

    def export_project_memory(self, workspace: str, export_path: str) -> str:
        ws = self._workspace(workspace)
        mem = self.db.get_memory(ws)
        if not mem:
            raise SessionError(self._no_memory_text(ws))
        return self._export(mem["rendered_md"], export_path) or ""

    @staticmethod
    def _no_memory_text(ws: str) -> str:
        return f"No memory saved for `{ws}`. Run `update_project_memory` (CLI: `memory update`) to create it."

    def get_project_memory(self, workspace: str) -> str:
        """Current memory, never waiting for the LLM. With unrecorded sessions and a key, it starts the
        background maintenance and says so; otherwise it warns."""
        ws = self._workspace(workspace)
        status = self.maybe_schedule_maintenance(ws)
        updating = status in ("started", "running")
        mem = self.db.get_memory(ws)
        if not mem:
            if updating:
                return (f"No memory for `{ws}` yet: it is being generated in the background. "
                        "Check again in a few minutes.")
            if not getattr(self.agent, "available", True):
                return (f"No memory saved for `{ws}` and no Gemini credential to generate it: set "
                        "GEMINI_API_KEY, ~/.ai/.gemini_env, or OAuth via "
                        "`gcloud auth application-default-login`.")
            return self._no_memory_text(ws)
        saved = {(w["key"], w.get("updated_at") or "") for w in mem["window"]}
        current = self.recent_sessions(ws, max(len(mem["window"]), 1))
        new = sum(1 for s in current if (memory.session_key(s.harness, s.session_id), s.updated_at or "") not in saved)
        text = mem["rendered_md"]
        if new:
            word = "new session" if new == 1 else "new sessions"
            if updating:
                note = f"> ⚠️ {new} {word}; updating in the background."
            else:
                note = (f"> ⚠️ {new} {word} since the last update. "
                        "Run `update_project_memory` to add them.")
            text = note + "\n\n" + text
        elif updating:
            text = "> ℹ️ Updating in the background.\n\n" + text
        return text

    def list_archived(self, workspace: str) -> str:
        ws = self._workspace(workspace)
        rows = [e for e in self.db.get_entries(ws) if e["status"] != "active"]
        if not rows:
            return "No archived entries."
        rows.sort(key=lambda e: (e["kind"], memory._num(e["id"])))
        return "\n".join(f"- [{e['id']}] ({e['status']}) {e['text']}" for e in rows)

    def memory_history(self, workspace: str) -> str:
        ws = self._workspace(workspace)
        versions = self.db.list_memory_versions(ws)
        if not versions:
            return self._no_memory_text(ws)
        lines = [f"### History of `{ws}`"]
        for v in versions:
            ids = ", ".join(f"{w['harness']}:{w['session_id'][:8]}" for w in v["window"])
            lines.append(f"- v{v['version']} · {v['created_at'][:16].replace('T', ' ')} UTC · {v['model_used']} · "
                         f"{len(v['window'])} sessions ({ids})")
        return "\n".join(lines)

    def edit_memory_entry(self, workspace: str, entry_id: str, action: str) -> str:
        """pin | unpin | forget on an entry; produces a new rendered version."""
        ws = self._workspace(workspace)
        with self._memory_lock(ws):
            mem = self.db.get_memory(ws)
            if not mem:
                raise SessionError(self._no_memory_text(ws))
            entries = self.db.get_entries(ws)
            eid = (entry_id or "").strip().upper()
            entry = next((e for e in entries if e["id"] == eid), None)
            if entry is None:
                raise SessionError(f"There is no entry '{entry_id}' in the memory of {ws}.")
            if entry["status"] == "forgotten":
                raise SessionError(f"Entry {eid} was already forgotten.")
            if action == "pin":
                entry.update(pinned=True, status="active")
            elif action == "unpin":
                entry["pinned"] = False
            elif action == "forget":
                entry.update(pinned=False, status="forgotten", text_hash=memory.text_hash(entry["text"]))
            else:
                raise SessionError(f"Unknown action: {action}")
            entry["updated_at"] = now_utc_iso()
            try:
                version, _ = self._persist(ws, mem["version"], entries, mem["window"], mem["state"],
                                           mem["model_used"], note=f"manual: {action} {eid}")
            except ConcurrentUpdateError as exc:
                raise SessionError(f"The memory changed while it was being edited ({exc}). Try again.") from exc
        return f"{eid}: {action} applied (memory v{version})."

    def suggest_agent_rules(self, workspace: str, target: str = "both") -> str:
        """Memory rules that can be promoted to AGENTS.md / CLAUDE.md. No LLM and read-only."""
        ws = self._workspace(workspace)
        if target not in ("agents", "claude", "both"):
            raise SessionError("target must be 'agents', 'claude' or 'both'.")
        mem = self.db.get_memory(ws)
        if not mem:
            return self._no_memory_text(ws)
        candidates = memory.suggest_rules(self.db.get_entries(ws), mem["window"], memory.read_rule_files(ws), target)
        return memory.render_suggestions(ws, candidates, target)

    # ------------------------------------------------------------------ background maintenance
    def _recently_failed(self, s: SessionSummary) -> bool:
        failed_at = self._digest_failures.get((s.harness, s.session_id, s.updated_at or ""))
        return failed_at is not None and time.time() - failed_at < RETRY_BACKOFF

    def pending_sessions(self, workspace: str) -> list[SessionSummary]:
        """Sessions among the last MAX_BACK (excluding the current one) without an up-to-date handoff, or stale
        and idle for at least IDLE_BEFORE_REFRESH seconds. Skips those that failed less than RETRY_BACKOFF ago."""
        idle_limit = time.time() - IDLE_BEFORE_REFRESH
        pending = []
        for s in self.recent_sessions(workspace, MAX_BACK):
            if self._recently_failed(s):
                continue
            d = self.db.get_digest(s.session_id, s.harness)
            if d is None:
                pending.append(s)
            elif (s.updated_at or "") > (d["source_updated_at"] or ""):
                updated = iso_to_epoch(s.updated_at)
                if updated is not None and updated <= idle_limit:
                    pending.append(s)
        return pending

    def _memory_needs_consolidation(self, workspace: str) -> bool:
        """True if the memory window no longer matches the saved handoffs (without calling the LLM)."""
        failed_at = self._consolidation_failures.get(workspace)
        if failed_at is not None and time.time() - failed_at < RETRY_BACKOFF:
            return False
        mem = self.db.get_memory(workspace)
        window = []
        for s in self.recent_sessions(workspace, memory.default_window()):
            d = self.db.get_digest(s.session_id, s.harness)
            if d is not None:
                window.append(memory.window_entry(dataclasses.replace(s, updated_at=d["source_updated_at"]), d))
        if not window:
            return False
        return mem is None or memory.window_signature(window) != memory.window_signature(mem["window"])

    def maybe_schedule_maintenance(self, workspace: str | None) -> str:
        """Starts (without waiting) the regeneration of pending handoffs and the memory consolidation.

        Returns: 'disabled' (CLI), 'no-key', 'idle' (nothing to do), 'running' or 'started'."""
        if not self.background:
            return "disabled"
        ws = normalize_workspace_filter(workspace)
        if not ws or not ws.startswith("/"):
            return "idle"
        if not getattr(self.agent, "available", True):
            return "no-key"
        with self._maintenance_guard:
            thread = self._maintenance.get(ws)
            if thread is not None and thread.is_alive():
                return "running"
        if not self.pending_sessions(ws) and not self._memory_needs_consolidation(ws):
            return "idle"
        with self._maintenance_guard:
            thread = self._maintenance.get(ws)
            if thread is not None and thread.is_alive():
                return "running"
            thread = threading.Thread(target=self._run_maintenance, args=(ws,), name=f"maintenance:{ws}", daemon=True)
            self._maintenance[ws] = thread
            thread.start()
        return "started"

    def _run_maintenance(self, workspace: str) -> None:
        owner = f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"
        try:
            if not self.db.acquire_maintenance_lock(workspace, owner):
                return  # another MCP instance is already doing it
            try:
                pending = self.pending_sessions(workspace)
                if pending:
                    _, failures, _ = self._generate_missing_digests(pending)
                    if failures:
                        logger.warning("Maintenance of %s: %s failed", workspace, ", ".join(failures))
                if self._memory_needs_consolidation(workspace):
                    try:
                        self.update_project_memory(workspace, reuse_stale=True)
                    except SessionError:
                        self._consolidation_failures[workspace] = time.time()
                        raise
            finally:
                self.db.release_maintenance_lock(workspace, owner)
        except SessionError as exc:
            logger.warning("Maintenance of %s: %s", workspace, exc)
        except Exception:
            logger.exception("Unexpected failure in the maintenance of %s", workspace)

    def wait_for_maintenance(self, workspace: str, timeout: float | None = None) -> None:
        """Waits for the maintenance of `workspace` to finish (tests and CLI)."""
        with self._maintenance_guard:
            thread = self._maintenance.get(normalize_workspace_filter(workspace) or workspace)
        if thread is not None:
            thread.join(timeout)

    # ------------------------------------------------------------------ topic search
    def search_project_memory(
        self, topic: str, workspace: str, all_projects: bool = False, limit: int = 20, synthesize: bool = False,
    ) -> str:
        """Local search (no LLM) in active/archived entries and handoffs; optional synthesis, never saved."""
        ws = self._workspace(workspace)
        if not memory.normalize(topic).split():
            raise SessionError("The topic cannot be empty.")
        scope = None if all_projects else ws
        results = memory.search(topic, self.db.get_entries(scope), self.db.search_digests(scope), all_projects)
        if not all_projects:
            for r in results:
                r["workspace_path"] = ws
        text = memory.render_search(topic, results, limit, all_projects, None if all_projects else ws)
        if synthesize and results:
            out = None
            if getattr(self.agent, "available", True):
                out = self.agent.synthesize_topic(topic, memory.render_search(topic, results, limit, all_projects, None))
            if out:
                text += f"\n\n## Pattern synthesis ({out[1]}, not saved)\n{out[0]}"
            else:
                text += "\n\n> ⚠️ No synthesis: there is no Gemini credential or the API failed. Only the matches are shown."
        return text
