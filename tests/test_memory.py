"""Tests for the deterministic memory logic: merging, pruning, limits, rendering and rules."""

from cross_session_mcp import memory
from cross_session_mcp.agent import build_consolidation_prompt
from cross_session_mcp.memory import apply_consolidation, render_markdown, suggest_rules

NOW = "2026-10-04T12:00:00+00:00"


def win(n=8, relevant=True):
    return [
        {
            "key": f"claude:s{i}",
            "harness": "claude",
            "session_id": f"sess{i:04d}xx",
            "updated_at": f"2026-09-{20 - i:02d}T10:00:00+00:00",
            "relevant": relevant,
            "summary_line": f"summary {i}",
        }
        for i in range(n)
    ]


def refs(window):
    return {f"r{i}": w["key"] for i, w in enumerate(window)}


def ent(eid, text, **kw):
    base = dict(
        id=eid,
        kind=eid[0],
        text=text,
        why="",
        scope=None,
        evidence=["claude:s0"],
        status="active",
        pinned=False,
        text_hash=memory.text_hash(text),
        first_seen="2026-09-01",
        last_seen="2026-09-19",
        updated_at="2026-09-01",
    )
    base.update(kw)
    return base


def raw(kind, text, ev=("r0",), **kw):
    return dict(id=None, kind=kind, text=text, why=kw.pop("why", ""), evidence_sessions=list(ev), **kw)


def run(prev, entries, window=None, **extra):
    window = window or win()
    return apply_consolidation(
        prev, dict(state=extra.pop("state", []), entries=entries, **extra), window, refs(window), NOW
    )


def by_id(entries):
    return {e["id"]: e for e in entries}


def test_new_ids_are_stable_and_never_reuse_archived():
    prev = [ent("D1", "one", status="archived"), ent("D2", "two totally different")]
    out, _ = run(
        prev, [dict(raw("D", "new decision about cache"), id=None), dict(raw("D", "two totally different"), id="D2")]
    )
    ids = by_id(out)
    assert "D3" in ids and ids["D3"]["text"] == "new decision about cache"  # D1 is not reused
    assert ids["D2"]["status"] == "active"


def test_existing_id_kept_when_text_changes():
    out, _ = run([ent("D3", "use sqlite")], [dict(raw("D", "use sqlite with WAL and busy_timeout"), id="D3")])
    assert by_id(out)["D3"]["text"] == "use sqlite with WAL and busy_timeout"


def test_jaccard_merges_with_active_entry_of_same_kind():
    out, _ = run(
        [ent("D1", "use sqlite in WAL mode for the cache")], [raw("D", "Use sqlite in WAL mode for the cache.")]
    )
    assert [e["id"] for e in out if e["kind"] == "D"] == ["D1"]


def test_pinned_entries_are_protected():
    prev = [ent("D2", "original pinned text", pinned=True, evidence=[])]
    out, _ = run(prev, [dict(raw("D", "different text"), id="D2")], archive_ids=["D2"])
    e = by_id(out)["D2"]
    assert e["text"] == "original pinned text" and e["status"] == "active"  # neither modified nor pruned (no evidence)


def test_forgotten_text_is_not_readded():
    prev = [ent("T4", "the login fails with cookies", status="forgotten")]
    out, _ = run(prev, [raw("T", "The login fails with cookies"), dict(raw("T", "other"), id="T4")])
    assert all(e["text"] != "The login fails with cookies" for e in out if e["id"] != "T4")
    assert by_id(out)["T4"]["status"] == "forgotten"


def test_evidence_recomputed_from_window_not_from_llm():
    window = win(8)
    out, _ = run([], [raw("D", "decision", ev=("r0", "r1", "made-up"))], window)
    e = out[0]
    assert sorted(e["evidence"]) == ["claude:s0", "claude:s1"]
    md = render_markdown("/p/app", out, window, [], "m", 1, NOW)
    assert "evidence 2/8" in md


def test_new_entry_without_valid_refs_gets_latest_relevant_session():
    out, _ = run([], [raw("P", "pending without refs", ev=())])
    assert out[0]["evidence"] == ["claude:s0"]


def test_rule_that_stops_appearing_is_archived():
    prev = [ent("T2", "old pitfall", evidence=["claude:old1", "claude:old2"])]
    out, _ = run(prev, [])
    assert by_id(out)["T2"]["status"] == "archived"
    md = render_markdown("/p/app", out, win(), [], "m", 2, NOW)
    assert "old pitfall" not in md


def test_a_candidates_with_high_cumulative_evidence_are_exempt():
    prev = [
        ent("A1", "repeated rule", evidence=["claude:o1", "claude:o2", "claude:o3"]),
        ent("A2", "rule with little evidence", evidence=["claude:o1"]),
    ]
    out, _ = run(prev, [])
    assert by_id(out)["A1"]["status"] == "active" and by_id(out)["A2"]["status"] == "archived"


def test_reverted_decision_is_superseded():
    prev = [ent("D1", "use redis for the queue")]
    out, _ = run(
        prev, [dict(raw("D", "use sqlite for the queue, redis dropped", why="fewer dependencies"), supersedes="D1")]
    )
    ids = by_id(out)
    assert ids["D1"]["status"] == "superseded"
    md = render_markdown("/p/app", out, win(), [], "m", 2, NOW)
    assert "use sqlite for the queue" in md and "use redis for the queue" not in md


def test_too_many_entries_are_limited_and_archived():
    window = win(8)
    llm = [
        raw("D", f"decision number {i} about topic{i} different{i}", ev=tuple(f"r{j}" for j in range(i % 8 + 1)))
        for i in range(25)
    ]
    out, _ = run([], llm, window)
    active = [e for e in out if e["kind"] == "D" and e["status"] == "active"]
    assert len(active) == 15 and len([e for e in out if e["status"] == "archived"]) == 10
    # the ones with the least evidence are archived
    assert min(len(e["evidence"]) for e in active) >= max(len(e["evidence"]) for e in out if e["status"] == "archived")
    md = render_markdown("/p/app", out, window, [], "m", 1, NOW)
    assert md.count("**[D") == 15


def test_secrets_and_pii_are_redacted_in_entries():
    out, state = run(
        [], [raw("D", "use token ghp_" + "a" * 36 + " and write to x@y.com")], state=["contact 612 345 678"]
    )
    assert "ghp_" not in out[0]["text"] and "x@y.com" not in out[0]["text"] and "[REDACTED]" in out[0]["text"]
    assert "612" not in state[0]


def test_render_structure_header_and_log():
    window = win(3)
    window[2]["relevant"] = False
    out, _ = run(
        [],
        [
            raw("D", "dec", why="because"),
            raw("T", "symptom → cause → fix"),
            raw("C", "pytest -q", why="tests"),
            raw("P", "finish x"),
            raw("A", "rule", scope="claude"),
        ],
        window,
    )
    md = render_markdown("/home/u/my-app", out, window, ["state one"], "gemini-x", 3, NOW)
    titles = [line for line in md.splitlines() if line.startswith("## ")]
    assert titles == [
        "## Current state",
        "## Active decisions",
        "## Pitfalls → fix",
        "## Commands that work",
        "## Pending",
        "## AGENTS.md candidates",
        "## Log",
    ]
    assert "3 sessions (claude 3)" in md and "gemini-x" in md and "v3" in md
    assert "**[D1]** dec — because _(evidence 1/3)_" in md
    assert "`pytest -q` — tests" in md and "CLAUDE.md" in md
    assert "summary 0" in md and "summary 1" in md and "summary 2" not in md  # irrelevant: not in the log
    assert md.count("sess0000") == 1


def test_render_never_exceeds_max_lines():
    window = win(20)
    llm = []
    for kind, lim in memory.SECTION_LIMITS.items():
        llm += [raw(kind, f"entry {kind} {i} unique{kind}{i}", ev=("r0",)) for i in range(lim)]
    out, _ = run([], llm, window, state=[f"state {i}" for i in range(8)])
    for limit in (100, 60, 40, 20):
        md = render_markdown("/p", out, window, [f"state {i}" for i in range(8)], "m", 1, NOW, limit_lines=limit)
        assert len(md.splitlines()) <= limit, limit
    full = render_markdown("/p", out, window, [], "m", 1, NOW)
    assert len(full.splitlines()) <= memory.max_lines()


def test_consolidation_prompt_marks_pinned_as_immutable():
    entries = [ent("D1", "pinned", pinned=True), ent("D2", "free")]
    prompt = build_consolidation_prompt(
        entries,
        [
            {
                "ref": "r0",
                "harness": "claude",
                "updated_at": "2026-09-19",
                "digest": {"summary_line": "something", "decisions": [{"text": "t", "why": "w"}]},
            }
        ],
        {"claude:s0": "r0"},
        "/p",
        2,
        "2026-10-04",
    )
    pinned_line = next(line for line in prompt.splitlines() if line.startswith("[D1]"))
    free_line = next(line for line in prompt.splitlines() if line.startswith("[D2]"))
    assert "IMMUTABLE" in pinned_line and "IMMUTABLE" not in free_line
    assert "ref `r0`" in prompt and "evidence: r0" in pinned_line


# ---------------------------------------------------------------- suggest_rules
def rules_fixture():
    window = win(8)
    keys = [w["key"] for w in window]
    return window, [
        ent("A1", "Always answer in formal English", scope="universal", evidence=keys[:5]),
        ent("A2", "Use the handoff skill before closing", scope="claude", evidence=keys[:3]),
        ent("D1", "Use sqlite WAL for the cache", evidence=keys[:4]),
        ent("D2", "poorly backed decision", evidence=keys[:1]),
        ent("T1", "The claude-mem hook blocks startup", evidence=keys[:3]),
        ent("A3", "archived rule", status="archived"),
    ]


def test_suggest_rules_candidates_and_targets():
    window, entries = rules_fixture()
    got = {c["id"]: c for c in suggest_rules(entries, window, {})}
    assert set(got) == {"A1", "A2", "D1", "T1"}
    assert got["A1"]["file"] == "AGENTS.md" and got["A1"]["evidence"] == "5/8"
    assert got["A2"]["file"] == "CLAUDE.md"
    assert got["T1"]["file"] == "CLAUDE.md"  # heuristic: hook / claude-mem
    assert got["D1"]["file"] == "AGENTS.md"
    claude_only = suggest_rules(entries, window, {}, "claude")
    assert {c["id"] for c in claude_only} == {"A2", "T1"}


def test_suggest_rules_marks_present_and_excludes_from_block():
    window, entries = rules_fixture()
    files = {"AGENTS.md": ["Always answer in formal English."]}
    cands = suggest_rules(entries, window, files)
    a1 = next(c for c in cands if c["id"] == "A1")
    assert a1["present"] and a1["present_in"] == "AGENTS.md"
    text = memory.render_suggestions("/p", cands)
    assert "already in AGENTS.md" in text
    block = text.split("#### To paste into `AGENTS.md`")[1]
    assert "Always answer" not in block and "sqlite WAL" in block
    assert "#### To paste into `CLAUDE.md`" in text


def test_read_rule_files_read_only_and_limited(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Rules\n- Use tabs\n\n* Another rule\n")
    (tmp_path / "CLAUDE.md").write_bytes(b"x" * (300 * 1024))
    files = memory.read_rule_files(str(tmp_path))
    assert files["AGENTS.md"] == ["Rules", "Use tabs", "Another rule"]
    assert len(files["CLAUDE.md"][0]) == 200 * 1024
    assert sorted(p.name for p in tmp_path.iterdir()) == ["AGENTS.md", "CLAUDE.md"]


def test_render_in_spanish(monkeypatch):
    monkeypatch.setenv("CROSS_SESSION_LANGUAGE", "es")
    window = win(3)
    out, _ = run([], [raw("D", "dec", why="motivo")], window)
    md = render_markdown("/home/u/my-app", out, window, ["estado"], "gemini-x", 1, NOW)
    titles = [line for line in md.splitlines() if line.startswith("## ")]
    assert titles[:2] == ["## Estado actual", "## Decisiones vigentes"] and titles[-1] == "## Bitácora"
    assert "3 sesiones (claude 3)" in md and "_(evidencia 1/3)_" in md
