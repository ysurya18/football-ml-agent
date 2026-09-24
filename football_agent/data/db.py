"""DuckDB connection helper."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def db_path() -> Path:
    """Location of the DuckDB file, overridable with DUCKDB_PATH."""
    configured = os.environ.get("DUCKDB_PATH", "football.duckdb")
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the match database.

    Defaults to read-only: everything downstream of the build step only reads,
    and read-only connections can be held concurrently.
    """
    path = db_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n  uv run python -m football_agent.data.build"
        )
    return duckdb.connect(str(path), read_only=read_only)
