"""Deterministic project memory logic: merging, pruning, limits, rendering and rules.

The LLM proposes (typed entries); this module decides: IDs, evidence, pruning and limits
are computed here and never left to the model's obedience.
"""

import hashlib
import os
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cross_session_mcp import i18n
from cross_session_mcp.utils import clean_text, redact, redact_pii

KINDS = ("D", "T", "C", "P", "A")
SECTION_LIMITS = {"D": 15, "T": 12, "C": 8, "P": 6, "A": 5}
STATE_LIMIT = 8
JACCARD_MERGE = 0.8  # merges a new entry into an active one of the same kind
JACCARD_PRESENT = 0.6  # rule already present in AGENTS.md / CLAUDE.md
KEEP_A_EVIDENCE = 3  # A# entries with this much accumulated evidence are never pruned
RULE_FILE_MAX_BYTES = 200 * 1024
DEFAULT_MAX_LINES = 100
DEFAULT_WINDOW = 8

_STOPWORDS = frozenset(
    "el la los las un una unos unas de del al y o u e a en con por para que se su sus lo le les es son "
    "the a an of to and or in on for with is are be as at by it this that"
    .split()
)
_CLAUDE_KEYWORDS = ("skill", "hook", "slash command", "/compact", "claude.md", "~/.claude", ".claude/",
                    "claude-mem", "claude code", "plugin")


def max_lines() -> int:
    try:
        return max(20, int(os.environ.get("CROSS_SESSION_MEMORY_MAX_LINES", DEFAULT_MAX_LINES)))
    except ValueError:
        return DEFAULT_MAX_LINES


def default_window() -> int:
    try:
        return max(1, min(20, int(os.environ.get("CROSS_SESSION_MEMORY_WINDOW", DEFAULT_WINDOW))))
    except ValueError:
        return DEFAULT_WINDOW


# ------------------------------------------------------------------ similarity
def normalize(text: str | None) -> str:
    """Lowercase and without accents, for searches insensitive to both."""
    decomposed = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"\w+", (text or "").lower()) if w not in _STOPWORDS)


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def text_hash(text: str) -> str:
    ordered = [w for w in re.findall(r"\w+", (text or "").lower()) if w not in _STOPWORDS]
    return hashlib.sha1(" ".join(ordered).encode("utf-8")).hexdigest()


def _num(entry_id: str) -> int:
    m = re.search(r"(\d+)$", entry_id or "")
    return int(m.group(1)) if m else 0


def _one_line(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ------------------------------------------------------------------ window
def session_key(harness: str, session_id: str) -> str:
    return f"{harness}:{session_id}"


def window_entry(summary: Any, digest: dict[str, Any] | None) -> dict[str, Any]:
    """`window_json` item for a session (with what is needed to re-render without the LLM)."""
    d = (digest or {}).get("digest") or {}
    return {
        "key": session_key(summary.harness, summary.session_id),
        "harness": summary.harness,
        "session_id": summary.session_id,
        "updated_at": summary.updated_at or "",
        "relevant": bool((digest or {}).get("relevant", False)),
        "summary_line": _one_line(d.get("summary_line")),
    }


def window_signature(window: Sequence[dict[str, Any]]) -> frozenset[tuple[str, str]]:
    return frozenset((w["key"], w.get("updated_at") or "") for w in window)


# ------------------------------------------------------------------ merging
def evidence_in_window(entry: dict[str, Any], window_keys: set) -> list[str]:
    return [k for k in entry.get("evidence", []) if k in window_keys]


def _ranked(entries, window_keys: set) -> list[dict[str, Any]]:
    """Priority order: pinned, most evidence in the window, most recent, lowest ID."""
    items = sorted(entries, key=lambda e: _num(e["id"]))
    items.sort(key=lambda e: e.get("last_seen") or "", reverse=True)
    items.sort(key=lambda e: (0 if e.get("pinned") else 1, -len(evidence_in_window(e, window_keys))))
    return items


def apply_consolidation(
    prev_entries: Sequence[dict[str, Any]],
    llm_output: dict[str, Any],
    window: Sequence[dict[str, Any]],
    ref_to_key: dict[str, str],
    now_iso: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Applies the LLM output over the previous entries. Returns (all entries, state)."""
    window_keys = {w["key"] for w in window}
    updated_by_key = {w["key"]: w.get("updated_at") or "" for w in window}
    fallback_keys = [w["key"] for w in window if w.get("relevant")][:1]  # the most recent relevant one

    entries: dict[str, dict[str, Any]] = {e["id"]: dict(e, evidence=list(e.get("evidence", []))) for e in prev_entries}
    forgotten = [e for e in entries.values() if e["status"] == "forgotten"]
    next_num = {k: max([_num(i) for i in entries if i.startswith(k)] or [0]) for k in KINDS}
    touched: set = set()

    def is_forgotten(text: str, kind: str) -> bool:
        h = text_hash(text)
        return any(f["kind"] == kind and (f.get("text_hash") == h or jaccard(f["text"], text) >= JACCARD_MERGE)
                   for f in forgotten)

    for raw in llm_output.get("entries") or []:
        if not isinstance(raw, dict):
            continue
        rid = str(raw.get("id") or "").strip().upper()
        target = entries.get(rid) if rid else None
        kind = target["kind"] if target else str(raw.get("kind") or "").strip().upper()[:1]
        text = _one_line(clean_text(raw.get("text")))
        if kind not in KINDS or not text:
            continue
        why = _one_line(clean_text(raw.get("why")))
        scope = raw.get("scope") if raw.get("scope") in ("universal", "claude") else None
        keys = [ref_to_key[r] for r in (raw.get("evidence_sessions") or []) if r in ref_to_key]

        if target is None:
            if is_forgotten(text, kind):
                continue
            target = next((e for e in entries.values()
                           if e["kind"] == kind and e["status"] == "active" and jaccard(e["text"], text) >= JACCARD_MERGE),
                          None)
        if target is None:
            next_num[kind] += 1
            target = {"id": f"{kind}{next_num[kind]}", "kind": kind, "text": text, "why": why, "scope": scope,
                      "evidence": [], "status": "active", "pinned": False, "text_hash": text_hash(text),
                      "first_seen": now_iso, "last_seen": None, "updated_at": now_iso}
            entries[target["id"]] = target
            keys = keys or list(fallback_keys)
        else:
            if target["status"] in ("forgotten", "superseded"):
                continue
            if not target["pinned"]:
                if target["status"] == "archived":
                    target["status"] = "active"
                target.update(text=text, why=why or target.get("why", ""), text_hash=text_hash(text),
                              updated_at=now_iso)
                if scope:
                    target["scope"] = scope
        target["evidence"] = sorted(set(target["evidence"]) | set(keys))
        touched.add(target["id"])

        sup = str(raw.get("supersedes") or "").strip().upper()
        old = entries.get(sup)
        if old and old is not target and not old["pinned"] and old["status"] == "active":
            old.update(status="superseded", updated_at=now_iso)

    for aid in llm_output.get("archive_ids") or []:
        e = entries.get(str(aid).strip().upper())
        if e and e["id"] not in touched and not e["pinned"] and e["status"] == "active":
            e.update(status="archived", updated_at=now_iso)

    # Window pruning: no evidence in the current window -> archived (except pinned and accumulated A#)
    for e in entries.values():
        if e["status"] != "active" or e["pinned"]:
            continue
        if not evidence_in_window(e, window_keys) and not (e["kind"] == "A" and len(e["evidence"]) >= KEEP_A_EVIDENCE):
            e.update(status="archived", updated_at=now_iso)

    for e in entries.values():
        in_win = evidence_in_window(e, window_keys)
        if e["status"] == "active" and in_win:
            e["last_seen"] = max(updated_by_key.get(k, "") for k in in_win) or e.get("last_seen")

    # Per-section limits: what does not fit is archived (pinned entries never are)
    for kind in KINDS:
        active = _ranked([e for e in entries.values() if e["kind"] == kind and e["status"] == "active"], window_keys)
        for e in active[SECTION_LIMITS[kind]:]:
            if not e["pinned"]:
                e.update(status="archived", updated_at=now_iso)

    state = [_one_line(clean_text(s)) for s in (llm_output.get("state") or []) if isinstance(s, str) and s.strip()]
    return list(entries.values()), state[:STATE_LIMIT]


# ------------------------------------------------------------------ render
_SECTIONS = ["state", "D", "T", "C", "P", "A", "log"]
_TRIM_ORDER = ["log", "A", "P", "C", "T", "D", "state"]  # lowest to highest priority


def _evidence_text(entry: dict[str, Any], window_keys: set, n: int) -> str:
    return f"{len(evidence_in_window(entry, window_keys))}/{n}"


def _entry_line(e: dict[str, Any], window_keys: set, n: int) -> str:
    pin = "📌 " if e.get("pinned") else ""
    text, why = _one_line(e["text"]), _one_line(e.get("why"))
    ev = _evidence_text(e, window_keys, n)
    if e["kind"] == "C":
        return f"- {pin}**[{e['id']}]** `{text.replace('`', '')}`" + (f" — {why}" if why else "")
    line = f"- {pin}**[{e['id']}]** {text}"
    if why:
        line += f" — {why}"
    if e["kind"] == "A":
        dest = "CLAUDE.md" if classify_target(e) == "claude" else "AGENTS.md"
        return f"{line} _({i18n.t('memory_evidence')} {ev} · {dest})_"
    if e["kind"] in ("D", "T"):
        return f"{line} _({i18n.t('memory_evidence')} {ev})_"
    return line


def render_markdown(
    workspace: str,
    entries: Sequence[dict[str, Any]],
    window: Sequence[dict[str, Any]],
    state: Sequence[str],
    model_used: str,
    version: int,
    now_iso: str,
    limit_lines: int | None = None,
) -> str:
    limit_lines = limit_lines or max_lines()
    window_keys = {w["key"] for w in window}
    n = len(window)
    per_harness: dict[str, int] = {}
    for w in window:
        per_harness[w["harness"]] = per_harness.get(w["harness"], 0) + 1
    harness_txt = ", ".join(f"{h} {c}" for h, c in sorted(per_harness.items()))
    t = i18n.t
    sessions_txt = f"{n} {t('memory_session') if n == 1 else t('memory_sessions')}"
    sessions_txt += f" ({harness_txt})" if harness_txt else ""
    header = [
        f"# MEMORY — {Path(workspace).name or workspace}",
        f"> {t('memory_project')}: `{workspace}` · {t('memory_updated')}: {now_iso[:16].replace('T', ' ')} UTC · v{version}",
        f"> {sessions_txt} · {t('memory_model')}: {model_used or t('memory_unknown')}",
        "",
    ]

    bodies: dict[str, list[str]] = {"state": [f"- {_one_line(s)}" for s in state[:STATE_LIMIT]]}
    for kind in KINDS:
        active = _ranked([e for e in entries if e["kind"] == kind and e["status"] == "active"], window_keys)
        bodies[kind] = [_entry_line(e, window_keys, n) for e in active[:SECTION_LIMITS[kind]]]
    bodies["log"] = [
        f"- {(w.get('updated_at') or '')[:10]} · {w['harness']} · {w['session_id'][:8]} · {w.get('summary_line') or t('memory_no_summary')}"
        for w in sorted(window, key=lambda w: w.get("updated_at") or "", reverse=True)
        if w.get("relevant")
    ]

    def total() -> int:
        return len(header) + sum(2 + max(len(bodies[k]), 1) for k in _SECTIONS)

    while total() > limit_lines:
        victim = next((k for k in _TRIM_ORDER if bodies[k]), None)
        if victim is None:
            break
        bodies[victim].pop()

    lines = list(header)
    for key in _SECTIONS:
        lines.append(f"## {t('section_' + key)}")
        lines.extend(bodies[key] or [t('memory_empty')])
        lines.append("")
    md = "\n".join(lines[:limit_lines]).rstrip() + "\n"
    return redact_pii(redact(md))


# ------------------------------------------------------------------ rules for AGENTS.md / CLAUDE.md
def classify_target(entry: dict[str, Any]) -> str:
    """'agents' (valid for any agent) or 'claude' (specific to Claude Code)."""
    if entry.get("scope") == "universal":
        return "agents"
    if entry.get("scope") == "claude":
        return "claude"
    blob = f"{entry.get('text', '')} {entry.get('why', '')}".lower()
    return "claude" if any(k in blob for k in _CLAUDE_KEYWORDS) else "agents"


_SUGGESTED_SECTION = {"A": "Conventions", "D": "Design decisions", "T": "Known pitfalls",
                      "C": "Commands", "P": "Pending"}


def read_rule_files(workspace: str) -> dict[str, list[str]]:
    """Lines (without list markers) of <workspace>/AGENTS.md and CLAUDE.md. Read-only, at most 200 KB."""
    found: dict[str, list[str]] = {}
    for name in ("AGENTS.md", "CLAUDE.md"):
        path = Path(workspace) / name
        try:
            if not path.is_file():
                continue
            with path.open("rb") as fh:
                raw = fh.read(RULE_FILE_MAX_BYTES).decode("utf-8", "replace")
        except OSError:
            continue
        lines = [re.sub(r"^[\s>#*\-\d.)]+", "", ln).strip() for ln in raw.splitlines()]
        found[name] = [ln for ln in lines if ln]
    return found


def suggest_rules(
    entries: Sequence[dict[str, Any]],
    window: Sequence[dict[str, Any]],
    rule_files: dict[str, list[str]],
    target: str = "both",
) -> list[dict[str, Any]]:
    window_keys = {w["key"] for w in window}
    n = len(window)
    out: list[dict[str, Any]] = []
    for e in _ranked([e for e in entries if e["status"] == "active"], window_keys):
        ev = evidence_in_window(e, window_keys)
        if not (e["kind"] == "A" or (e["kind"] in ("D", "T") and (len(ev) >= 3 or e.get("pinned")))):
            continue
        dest = classify_target(e)
        if target in ("agents", "claude") and dest != target:
            continue
        present_in = present_line = None
        for fname, lines in rule_files.items():
            for ln in lines:
                if jaccard(e["text"], ln) >= JACCARD_PRESENT:
                    present_in, present_line = fname, ln
                    break
            if present_in:
                break
        out.append({
            "id": e["id"], "rule": e["text"], "evidence": f"{len(ev)}/{n}", "sessions": ev,
            "target": dest, "file": "CLAUDE.md" if dest == "claude" else "AGENTS.md",
            "section": _SUGGESTED_SECTION.get(e["kind"], "Conventions"),
            "present": present_in is not None, "present_in": present_in, "present_line": present_line,
        })
    return out


def render_suggestions(workspace: str, candidates: Sequence[dict[str, Any]], target: str = "both") -> str:
    if not candidates:
        return (f"No candidate rules for `{workspace}`"
                + (f" targeting {target}" if target != "both" else "")
                + ". They need `A#` entries, or `D#`/`T#` entries with evidence in ≥ 3 sessions or pinned.")
    lines = [f"### Candidate rules for `{workspace}`\n",
             "| ID | Rule | Evidence | Target | Status |", "|---|---|---|---|---|"]
    for c in candidates:
        status = f"already in {c['present_in']}: “{c['present_line'][:60]}”" if c["present"] else "new"
        lines.append(f"| {c['id']} | {c['rule'].replace('|', '/')} | {c['evidence']} | {c['file']} | {status} |")
    for fname in ("AGENTS.md", "CLAUDE.md"):
        new = [c for c in candidates if c["file"] == fname and not c["present"]]
        if not new:
            continue
        lines += ["", f"#### To paste into `{fname}`", "```markdown"]
        by_section: dict[str, list[str]] = {}
        for c in new:
            by_section.setdefault(c["section"], []).append(f"- {c['rule']}")
        for section, items in by_section.items():
            lines += [f"## {section}", *items, ""]
        lines[-1:] = ["```"]
    return "\n".join(lines)


# ------------------------------------------------------------------ topic search
SEARCH_DEFAULT_LIMIT = 20
SEARCH_MAX_LIMIT = 50
_SNIPPET = 260
_DIGEST_ITEMS = (
    ("decisions", "decision", lambda i: f"{i.get('text', '')} — {i.get('why', '')}"),
    ("pitfalls", "pitfall", lambda i: f"{i.get('symptom', '')} → {i.get('cause', '')} → {i.get('fix', '')}"),
    ("commands", "command", lambda i: f"{i.get('cmd', '')} — {i.get('purpose', '')}"),
    ("pending", "pending", lambda i: str(i)),
    ("agents_md_candidates", "rule", lambda i: str(i.get("rule", ""))),
)


def search_terms(topic: str) -> list[str]:
    words = re.findall(r"\w+", normalize(topic))
    meaningful = [w for w in words if w not in _STOPWORDS]
    return meaningful or words


def _score(text: str, terms: Sequence[str]) -> int:
    """Total matches if ALL the terms appear; 0 if any is missing."""
    norm = normalize(text)
    counts = [norm.count(t) for t in terms]
    return sum(counts) if counts and all(counts) else 0


def _snippet(text: str, terms: Sequence[str]) -> str:
    flat = _one_line(text)
    norm = normalize(flat)  # NFKD + filtering keeps the length for accented Latin text (one letter per letter)
    idx = min((norm.find(t) for t in terms if norm.find(t) >= 0), default=0)
    start = max(0, idx - _SNIPPET // 3)
    piece = flat[start:start + _SNIPPET]
    return ("…" if start else "") + piece + ("…" if start + _SNIPPET < len(flat) else "")


def _handoff_sections(markdown: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^### ", markdown or "")
    out = []
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        out.append((title.strip(), body))
    return out


def search(
    topic: str,
    entries: Sequence[dict[str, Any]],
    digests: Sequence[dict[str, Any]],
    all_projects: bool = False,
) -> list[dict[str, Any]]:
    """Searches a topic in the entries (any status except forgotten) and in the digests (structured data
    and Extended Handoff sections). No LLM. Results sorted by matches and recency."""
    terms = search_terms(topic)
    if not terms:
        return []
    results: list[dict[str, Any]] = []
    for e in entries:
        if e.get("status") == "forgotten":
            continue
        blob = f"{e.get('text', '')} — {e.get('why', '')}" if e.get("why") else e.get("text", "")
        score = _score(blob, terms)
        if score:
            results.append({
                "source": "entry", "id": e["id"], "kind": e["kind"], "status": e["status"],
                "evidence": len(e.get("evidence") or []), "sessions": list(e.get("evidence") or []),
                "date": e.get("last_seen") or e.get("updated_at") or "", "snippet": _snippet(blob, terms),
                "text": e.get("text", ""), "score": score, "workspace_path": e.get("workspace_path", ""),
            })
    entry_results = [r for r in results if r["source"] == "entry"]
    for d in digests:
        key = session_key(d["harness"], d["session_id"])
        ref = f"{d['harness']}:{d['session_id'][:8]}"
        items = []
        for field, label, fmt in _DIGEST_ITEMS:
            for item in (d.get("digest") or {}).get(field) or []:
                items.append((label, fmt(item)))
        for title, body in _handoff_sections(d.get("handoff_md") or ""):
            items.append((f"handoff · {title}", body))
        for label, text in items:
            score = _score(text, terms)
            if not score:
                continue
            # Already represented by a memory entry that comes from this session
            if any(key in r["sessions"] and jaccard(r["text"], text) >= JACCARD_MERGE for r in entry_results):
                continue
            results.append({
                "source": "session", "ref": ref, "label": label, "date": d.get("updated_at") or "",
                "snippet": _snippet(text, terms), "score": score, "workspace_path": d.get("workspace_path", ""),
            })
    results.sort(key=lambda r: r["date"], reverse=True)
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def render_search(
    topic: str, results: Sequence[dict[str, Any]], limit: int = SEARCH_DEFAULT_LIMIT,
    all_projects: bool = False, workspace: str | None = None,
) -> str:
    scope = "all projects" if all_projects else f"`{workspace}`" if workspace else "the bank"
    if not results:
        return (f"No matches for “{clean_text(topic)}” in {scope}. Try other terms"
                + ("" if all_projects else " or all_projects=true") + ".")
    shown = results[:limit]
    lines = [f"### Memory bank — “{clean_text(topic)}” in {scope}: {len(results)} match(es)", ""]
    for r in shown:
        date = (r.get("date") or "")[:10] or "n/d"
        project = f" · `{r['workspace_path']}`" if all_projects and r.get("workspace_path") else ""
        if r["source"] == "entry":
            lines.append(f"- **[{r['id']}]** ({r['status']}, evidence {r['evidence']} sess., {date}){project}: {r['snippet']}")
        else:
            lines.append(f"- **{r['ref']}** · {r['label']} ({date}){project}: {r['snippet']}")
    if len(results) > len(shown):
        lines += ["", f"_{len(results) - len(shown)} more results; narrow the topic or use limit._"]
    return redact_pii(redact("\n".join(lines)))
