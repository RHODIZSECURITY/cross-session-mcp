# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- OAuth authentication with a Google account (no API key): Application Default
  Credentials from `gcloud auth application-default-login`, with automatic token
  refresh, `Authorization: Bearer` transport, optional `CROSS_SESSION_GOOGLE_PROJECT`
  billing, one 401-triggered refresh-and-retry, and `doctor` reporting. An explicit
  API key still takes precedence. Standard library only, no new dependency.

## [1.4.0] - 2026-10-12

### Added
- macOS support: the Cursor and VS Code / Copilot readers look under
  `~/Library/Application Support`. Tested with synthetic stores only;
  please report any macOS store that is not found.
- README: context savings measured on real sessions, a quick start and
  an example Extended Handoff.

### Changed
- A handoff built without a model now says why. With no key it explains
  how to set one; when Gemini failed it does not ask for a key.
- `doctor` warns that the project memory needs the Gemini key.
- CI uses `actions/checkout` v7 and `astral-sh/setup-uv` v10.

### Fixed
- Focused handoffs built after a Gemini failure were labelled
  `offline-extractor`; they are now `offline-fallback`.
- The source distribution no longer ships an incomplete test suite.

### Security
- Hugging Face, npm, PyPI, DigitalOcean, Sentry, Groq, xAI, Replicate,
  Linear, Shopify and SendGrid tokens are now redacted.

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
