"""Multi-harness session readers."""

from cross_session_mcp.readers.agy import AgySessionReader
from cross_session_mcp.readers.base import BaseSessionReader
from cross_session_mcp.readers.claude import ClaudeSessionReader
from cross_session_mcp.readers.codex import CodexSessionReader
from cross_session_mcp.readers.copilot import CopilotSessionReader
from cross_session_mcp.readers.cursor import CursorSessionReader
from cross_session_mcp.readers.opencode import OpenCodeSessionReader


def get_all_readers() -> list[BaseSessionReader]:
    """Returns instances of every available reader."""
    return [
        ClaudeSessionReader(),
        AgySessionReader(),
        OpenCodeSessionReader(),
        CodexSessionReader(),
        CursorSessionReader(),
        CopilotSessionReader(),
    ]


def get_reader_by_name(name: str) -> BaseSessionReader:
    """Returns a reader by harness name."""
    readers = {r.name: r for r in get_all_readers()}
    if name not in readers:
        raise ValueError(f"Unsupported harness: '{name}'. Options: {list(readers.keys())}")
    return readers[name]
