# Contributing

Thanks for helping improve cross-session-mcp.

## Setup

```bash
git clone https://github.com/reiarseni/cross-session-mcp
cd cross-session-mcp
uv sync
uv run pytest
uv run ruff check .
```

The live tests (`uv run pytest -m live -s`) read the real session stores of
your machine and call Gemini if a key is configured. They are never run in
CI.

## Guidelines

- Keep the runtime on the Python standard library and Python 3.10+. Test
  and lint tools go in the `dev` dependency group only.
- Readers never write to harness stores and never raise on corrupt data:
  they log it to stderr and skip the record.
- Pass every text that may hold keys, tokens or credentials through
  `redact()` before it is stored in a handoff or sent to an external API.
  Memory and digests also go through `redact_pii()`.
- Send the Gemini key in the `x-goog-api-key` header, never in the URL,
  where it would end up in logs.
- Never add a broad `DROP TABLE` to the migrations in `db.py`: the session
  index is rebuildable, but the digest and memory tables are not, and
  dropping them destroys the consolidated memory.
- Leave invariants to code, not to the LLM: IDs, pruning, limits and the
  quote of the last user instruction are enforced deterministically.
- Add a test for every new behavior. Tests use synthetic stores and a
  mocked `urlopen`; they must not need the network.
- Write commits as `type: description` (`feat`, `fix`, `docs`, `chore`,
  `refactor`, `test`), in English, imperative mood, 72 characters at most.
- Open an issue before large changes.
