"""Multi-harness session readers."""

from cross_session_mcp.readers.base import BaseSessionReader
from cross_session_mcp.readers.claude import ClaudeSessionReader
from cross_session_mcp.readers.codex import CodexSessionReader


def get_all_readers() -> list[BaseSessionReader]:
    """Returns instances of every available reader."""
    return [
        ClaudeSessionReader(),
        CodexSessionReader(),
    ]


def get_reader_by_name(name: str) -> BaseSessionReader:
    """Returns a reader by harness name."""
    readers = {r.name: r for r in get_all_readers()}
    if name not in readers:
        raise ValueError(f"Unsupported harness: '{name}'. Options: {list(readers.keys())}")
    return readers[name]
