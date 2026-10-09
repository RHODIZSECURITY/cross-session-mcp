# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.3.0] - 2026-10-09

### Added
- Session readers for Claude Code, Google Antigravity (`agy` and the IDE),
  OpenCode, OpenAI Codex, Cursor and GitHub Copilot Chat in VS Code.
- SQLite session index with workspace filters, ID prefixes and excluded
  workspaces.
- 8-section Extended Handoff generated with Gemini, with model fallback and
  a local extractor that never blocks the agent.
- Project memory consolidated from the last sessions of every harness, with
  stable IDs, pinning, forgetting, versions and archiving.
- Background upkeep of the memory bank with a cross-process lock and
  backoff after LLM failures.
- Topic search over active and archived memory and saved handoffs, with an
  optional pattern synthesis.
- `suggest_agent_rules` to promote memory rules to `AGENTS.md` or
  `CLAUDE.md`.
- `CROSS_SESSION_LANGUAGE` (`en` or `es`) for handoffs, the resume wrapper
  and MEMORY.md.
- Standard-library MCP server over stdio and a command line interface.

### Security
- Secrets, email addresses and phone numbers are redacted before anything
  is sent to Gemini or stored, and the API key travels only in the
  `x-goog-api-key` header.
