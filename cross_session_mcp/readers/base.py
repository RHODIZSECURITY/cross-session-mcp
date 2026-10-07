"""Base class for the session readers of each harness."""

import json
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cross_session_mcp.models import SessionDetail, SessionSummary, SessionTurn
from cross_session_mcp.utils import logger, truncate

# Per-turn size limits when extracting transcripts
TEXT_TURN_LIMIT = 6000
TOOL_TURN_LIMIT = 1200


class BaseSessionReader(ABC):
    """Abstract interface every harness extractor implements.

    Contract: readers never write to harness stores, never raise on corrupt data
    (they log it to stderr) and return dates normalized to ISO UTC.
    """

    name: str = "base"

    @abstractmethod
    def can_read(self) -> bool:
        """Whether the harness store exists and is readable on this machine."""

    @abstractmethod
    def list_sessions(
        self,
        workspace_filter: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        """Returns the most recent sessions of this harness."""

    @abstractmethod
    def get_session_detail(
        self,
        session_id: str,
        max_turns: int | None = 25,
    ) -> SessionDetail | None:
        """Fetches the session and its latest turns (max_turns=None -> all)."""

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tail(turns: list[SessionTurn], max_turns: int | None) -> list[SessionTurn]:
        return turns[-max_turns:] if max_turns else turns

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterator[Any]:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)

    @staticmethod
    def _connect_ro(db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _tool_call_text(name: str, args: Any) -> str:
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(args)
        return f"→ {name}: {truncate(args, 400)}"
