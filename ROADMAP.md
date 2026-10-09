# Roadmap

Planned improvements, in priority order. Each one widens who can use cross-session-mcp without giving up what it is
today: standard library only at runtime, read-only on the harness stores, and secrets redacted before anything leaves
the machine.

Every phase ships on its own branch and commit. Each one starts with a short spike with an explicit go / no-go
criterion. If the spike fails, the phase is dropped or redesigned. A phase is done when it has: the code, tests on
synthetic data (no network), a README update, a CHANGELOG entry, and a green `ruff check` and `pytest` on Python 3.10
to 3.13.

## Phase 1: publish on PyPI and run with `uvx`

Today you clone the repository and point each harness at `bin/cross-session-mcp-server`. A package on PyPI turns
that into a single line (`uvx cross-session-mcp-server`) and makes upgrades trivial.

**Spike.** Build the wheel and the sdist with `uv build`, install them in a clean environment and launch the server
from every harness config in the README. Go if the stdio handshake and the tools work exactly as with the launcher
script.

- Release workflow on tags with PyPI trusted publishing (no long-lived tokens).
- README installation section rewritten around `uvx`, keeping the launcher script as the no-install option.

Cost: low. No code changes beyond packaging.

## Phase 2: other LLM providers

Gemini is the only backend. Many users already pay for Claude or OpenAI, and others want everything local.

**Spike.** Generate the digest and the consolidation for the same ten sessions with each candidate model and compare
them with Gemini's. Go if the JSON is valid on the first try in most calls and the handoffs keep all 8 sections.

- A small provider interface in `agent.py` with the current Gemini client as the first implementation, still over
  `urllib` so the runtime stays dependency-free.
- Claude (Messages API) and OpenAI (Responses API) with structured output; Ollama over its local HTTP API for fully
  offline use.
- Same guarantees for every provider: the key travels in a header, transcripts are redacted before the call, and the
  local extractor remains the fallback.

Cost: medium. Each provider is independent and can ship on its own.

## Phase 3: more harnesses

Six harnesses are read today. The next ones, in order of demand: Windsurf, Zed, Aider and Gemini CLI.

**Spike.** For each one, locate where it stores sessions on Linux and macOS, and confirm the format is stable across
two releases. Go if the index and the transcript can be read without writing to its store.

- One reader per harness in `readers/`, following the `BaseSessionReader` contract.
- Synthetic store fixtures in `tests/conftest.py`, like the existing ones.
- A row in the README harness table and connection instructions where the harness supports MCP.

Cost: low to medium per harness. A harness whose format turns out to be unstable is left out.

## Phase 4: optional semantic search

`search_project_memory` matches words, ignoring case and accents. It misses a pitfall described with other words.

**Spike.** Embed the memory entries and handoff sections of a real project and compare the results with keyword
search on twenty real queries. Go if it finds clearly more relevant matches without slowing down the tool.

- Opt-in with an environment variable; keyword search stays the default and needs nothing extra.
- Embeddings computed through the configured provider and stored in SQLite next to the digests, regenerated only
  when an entry changes.
- Results keep the current format, so agents need no changes.

Cost: medium. If the gain is small, this phase is removed.

## Out of scope

- A hosted service or any sync of sessions between machines: the bank stays local.
- Writing to harness stores (for example, injecting a summary into another agent's history).
- A graphical interface: the CLI and the MCP tools cover the use cases.
