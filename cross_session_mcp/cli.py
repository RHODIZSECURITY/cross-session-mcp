"""Standalone CLI for cross-session-mcp."""

import argparse
import os
import shutil
import sys
from pathlib import Path

from cross_session_mcp import __version__
from cross_session_mcp import antigravity as antigravity_mod
from cross_session_mcp.agent import HAS_ANTIGRAVITY_SDK, configured_models, get_gemini_api_key, load_adc
from cross_session_mcp.models import HARNESSES
from cross_session_mcp.readers import get_all_readers
from cross_session_mcp.service import MAX_BACK, CrossSessionService, SessionError, select_turns
from cross_session_mcp.utils import redact, setup_logging, truncate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = PROJECT_ROOT / "bin" / "cross-session-mcp-server"


def cmd_list(svc: CrossSessionService, args) -> None:
    sessions = svc.list_sessions(workspace=args.workspace, harness=args.harness, limit=args.limit, refresh=args.sync)
    if not sessions:
        print("No sessions found.")
        return
    print(f"\n📋 Sessions ({len(sessions)} shown):")
    print("-" * 80)
    for s in sessions:
        when = (s.updated_at or "---")[:16].replace("T", " ")
        print(f"[{s.harness.upper():<8}] {s.session_id:<36} | {when} UTC | {redact(s.title)[:40]}")
        print(f"           Path: {s.workspace_path}")
        if s.last_prompt:
            print(f"           Last: {truncate(redact(s.last_prompt).replace(chr(10), ' '), 70)}")
        print()


def cmd_show(svc: CrossSessionService, args) -> None:
    detail = svc.get_detail(args.session_id, args.harness, max_turns=None)
    turns, omitted = select_turns(detail.turns, args.query, args.tools, args.turns)
    s = detail.summary
    print(f"\n🔍 {redact(s.title)}")
    print(f"Harness: {s.harness} | ID: {s.session_id} | Path: {s.workspace_path}")
    print(f"Turns shown: {len(turns)} · omitted: {omitted}\n" + "=" * 80)
    for i, t in enumerate(turns, 1):
        print(f"[{i}] {t.role.upper()} ({t.timestamp or ''}):")
        print(redact(t.content))
        print("-" * 40)


def cmd_handoff(svc: CrossSessionService, args) -> None:
    print("Generating handoff...", file=sys.stderr)
    workspace = None if args.session_id else (args.workspace or os.getcwd())
    h = svc.get_handoff(args.session_id, args.harness, focus_area=args.focus, force_refresh=args.force,
                        workspace=workspace, back=args.back)
    origin = "local bank" if h.from_cache else h.model_used
    print(f"[Source: {origin} · {h.generated_at[:16]} UTC]", file=sys.stderr)
    if args.output:
        out = Path(args.output)
        out.write_text(h.markdown_handoff + "\n", encoding="utf-8")
        out.chmod(0o600)
        print(f"Handoff saved to {out}", file=sys.stderr)
    else:
        print(h.markdown_handoff)


def cmd_sync(svc: CrossSessionService, args) -> None:
    count = svc.sync(force=True)
    print(f"✅ {count} sessions indexed in {svc.db.db_path}.")


def cmd_clear_cache(svc: CrossSessionService, args) -> None:
    print(f"🧹 {svc.db.clear_digests()} saved handoffs deleted (consolidated memory is kept).")


def _memory_workspace(args) -> str:
    return args.workspace or os.getcwd()


def cmd_memory(svc: CrossSessionService, args) -> None:
    action, ws = args.memory_action, _memory_workspace(args)
    if action == "show":
        print(svc.list_archived(ws) if args.archived else svc.get_project_memory(ws))
    elif action == "update":
        print("Updating memory (may take a while: 1 call per new session + 1 for consolidation)...", file=sys.stderr)
        result = svc.update_project_memory(ws, sessions=args.sessions, force_refresh=args.force, export_path=args.export)
        print(result.markdown)
        print(f"[{result.summary()}]", file=sys.stderr)
    elif action == "export":
        print(f"Memory exported to {svc.export_project_memory(ws, args.output)}", file=sys.stderr)
    elif action in ("pin", "unpin", "forget"):
        print(svc.edit_memory_entry(ws, args.entry_id, action))
    elif action == "history":
        print(svc.memory_history(ws))
    elif action == "suggest-rules":
        print(svc.suggest_agent_rules(ws, args.target))
    elif action == "search":
        print(svc.search_project_memory(args.topic, ws, all_projects=args.all, limit=args.limit,
                                        synthesize=args.synthesize))


def cmd_serve(svc: CrossSessionService, args) -> None:
    from cross_session_mcp.server import MCPServer
    MCPServer(service=svc).run_stdio()


def cmd_doctor(svc: CrossSessionService, args) -> None:
    """Diagnostics that reveal no secrets."""
    ok = True
    print(f"cross-session-mcp {__version__} · Python {sys.version.split()[0]} ({sys.executable})\n")
    print("Session stores:")
    for r in get_all_readers():
        try:
            readable = r.can_read()
            count = len(r.list_sessions(limit=5)) if readable else 0
            print(f"  {'✅' if readable else '➖'} {r.name:<9} {'readable' if readable else 'not found'}"
                  f"{f' ({count}+ sessions)' if readable else ''}")
        except Exception as exc:
            ok = False
            print(f"  ❌ {r.name:<9} error: {exc}")

    print("\nDatabase:")
    db_path = svc.db.db_path
    mode = oct(db_path.stat().st_mode & 0o777) if db_path.exists() else "-"
    print(f"  {db_path} (permissions {mode})")

    print("\nProject memory:")
    print(f"  {svc.db.count_memory_projects()} project(s) with memory · schema v{svc.db.schema_version()}")
    formats = svc.db.digest_format_counts()
    print(f"  Saved handoffs: {formats['current']} in the current format, {formats['legacy']} legacy")
    locks = svc.db.active_maintenance_locks()
    print(f"  Maintenance in progress: {len(locks)}"
          + "".join(f"\n    - {lk['workspace_path']} ({lk['owner']})" for lk in locks))

    print("\nGemini synthesis:")
    key = get_gemini_api_key()
    if key:
        print("  ✅ Key configured (value not shown)")
    elif load_adc() is not None:
        print("  ✅ OAuth (ADC) configured, no API key needed (secrets not shown)")
    else:
        cca = antigravity_mod.load_cred()
        if cca is not None:
            print(f"  ✅ OAuth (Antigravity) configured for {cca.get('email') or 'account'} "
                  f"(project {cca['project_id']}, secrets not shown)")
        else:
            print("  ⚠️  No key: handoffs fall back to a basic local extractor and the project memory cannot be built.\n"
                  "      Set GEMINI_API_KEY or ~/.ai/.gemini_env, log in with "
                  "`gcloud auth application-default-login` (OAuth), or run "
                  "`cross-session-mcp antigravity-login` (Google account, no key).")
    print(f"  Models: {', '.join(configured_models())}")
    print(f"  Antigravity SDK: {'available' if HAS_ANTIGRAVITY_SDK else 'not installed (REST is used)'}")

    print("\nLauncher for the harnesses:")
    installed = shutil.which("cross-session-mcp-server")
    print(f"  Repository script: {LAUNCHER} {'✅' if LAUNCHER.exists() else '❌'}")
    print(f"  On PATH: {installed or 'no (optional: uv tool install -e .)'}")
    sys.exit(0 if ok else 1)


def cmd_antigravity_login(svc: CrossSessionService, args) -> None:
    """Google login for Cloud Code Assist (Antigravity OAuth, no API key)."""
    url, verifier = antigravity_mod.build_auth_url()
    print("Open this URL in your browser and consent with your Google account,")
    print("then paste back the `code=` value from the address bar (nothing is shown yet).\n")
    print(f"  {url}\n")
    try:
        pasted = input("Paste the redirect URL or code: ")
    except (EOFError, KeyboardInterrupt):
        print("\nLogin cancelled.")
        sys.exit(1)
    try:
        code = antigravity_mod.extract_code(pasted)
        creds = antigravity_mod.exchange_code(code, verifier)
    except ValueError as exc:
        print(f"Login failed: {exc}")
        sys.exit(1)
    print("Code exchanged. Discovering Cloud Code Assist project...")
    project = antigravity_mod.discover_project(creds["access"])
    if not project:
        print("Login failed: no Cloud Code Assist project for this account. "
              "Ensure the account has Antigravity/Cloud Code Assist access and try again.")
        sys.exit(1)
    cred_file = antigravity_mod.save_cred(creds["refresh"], project, creds.get("email", ""))
    print(f"✅ Antigravity login saved for {creds.get('email') or 'account'} "
          f"(project {project}) to {cred_file}")
    sys.exit(0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cross-session-mcp",
        description="Session index, viewer and handoff across Claude Code, agy, OpenCode, Codex, Cursor and Copilot (VS Code).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="List recorded sessions")
    p.add_argument("--workspace", "-w", help="Filter by project path (absolute = subdirectories included)")
    p.add_argument("--harness", "-H", choices=HARNESSES)
    p.add_argument("--limit", "-n", type=int, default=15)
    p.add_argument("--sync", action="store_true", help="Force a sync before listing")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="Show the turns of a session (accepts an ID prefix)")
    p.add_argument("session_id")
    p.add_argument("--harness", "-H", choices=HARNESSES)
    p.add_argument("--turns", "-t", type=int, default=15)
    p.add_argument("--query", "-q", help="Only turns containing this text (with ±1 of context)")
    p.add_argument("--tools", action="store_true", help="Include tool outputs")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("handoff", help="Extended Handoff of a session (by ID or by position in the project)")
    p.add_argument("session_id", nargs="?", help="ID or prefix; if omitted, --workspace and --back are used")
    p.add_argument("--workspace", "-w", help="Exact project path (default: the current directory)")
    p.add_argument("--back", "-b", type=int, default=1, choices=range(1, MAX_BACK + 1), metavar=f"1-{MAX_BACK}",
                   help="1 = last session, 2 = the one before...")
    p.add_argument("--harness", "-H", choices=HARNESSES)
    p.add_argument("--focus", "-f", help="Focus area or special directive")
    p.add_argument("--force", action="store_true", help="Ignore the cache")
    p.add_argument("--output", "-o", help="Save the handoff to a file (permissions 600)")
    p.set_defaults(func=cmd_handoff)

    p = sub.add_parser("memory", help="Consolidated memory per project (MEMORY.md)")
    msub = p.add_subparsers(dest="memory_action", required=True)

    def memory_cmd(name: str, help_text: str, entry: bool = False):
        mp = msub.add_parser(name, help=help_text)
        if entry:
            mp.add_argument("entry_id", help="Entry ID (e.g. D2, T4, A1)")
        mp.add_argument("workspace", nargs="?", help="Exact project path (default: the current directory)")
        mp.set_defaults(func=cmd_memory)
        return mp

    mp = memory_cmd("show", "Show the current memory (no LLM call)")
    mp.add_argument("--archived", action="store_true", help="List archived entries instead of the memory")
    mp = memory_cmd("update", "Rebuild the memory from the last N sessions (uses Gemini)")
    mp.add_argument("--sessions", "-n", type=int, choices=range(1, 21), metavar="1-20", help="Window size")
    mp.add_argument("--force", action="store_true", help="Also regenerate cached digests")
    mp.add_argument("--export", help="Also write MEMORY.md to this path (permissions 600)")
    mp = memory_cmd("export", "Write the memory to a file (permissions 600)")
    mp.add_argument("--output", "-o", required=True, help="File path")
    memory_cmd("pin", "Pin an entry: never pruned or modified by the LLM", entry=True)
    memory_cmd("unpin", "Unpin an entry", entry=True)
    memory_cmd("forget", "Archive an entry and keep it from being added again", entry=True)
    memory_cmd("history", "List memory versions")
    mp = memory_cmd("suggest-rules", "Rules worth promoting to AGENTS.md / CLAUDE.md (no LLM)")
    mp.add_argument("--target", choices=("agents", "claude", "both"), default="both")
    mp = msub.add_parser("search", help="Search a topic in active and archived memory and handoffs (no LLM)")
    mp.add_argument("topic")
    mp.add_argument("workspace", nargs="?", help="Exact project path (default: the current directory)")
    mp.add_argument("--all", action="store_true", help="Search every project")
    mp.add_argument("--synthesize", action="store_true", help="Add a pattern synthesis with Gemini (not saved)")
    mp.add_argument("--limit", type=int, default=20, choices=range(1, 51), metavar="1-50")
    mp.set_defaults(func=cmd_memory)

    p = sub.add_parser("sync", help="Re-sync the session index")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("clear-cache", help="Delete saved handoffs (regenerated on demand)")
    p.set_defaults(func=cmd_clear_cache)

    p = sub.add_parser("doctor", help="Diagnose the installation without showing secrets")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("antigravity-login", help="Log in with a Google account for Cloud Code Assist (no API key)")
    p.set_defaults(func=cmd_antigravity_login)
    p = sub.add_parser("serve", help="Start the MCP server over stdio")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv=None) -> None:
    setup_logging()
    args = build_parser().parse_args(argv)
    try:
        args.func(CrossSessionService(), args)
    except SessionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
