"""PostgreSQL access for NEXUS Intelligence (settings, engine, schema)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

from nexus.config import get_settings  # importing nexus.config also loads the .env file

CORE_SCHEMA = "core"


@dataclass(frozen=True)
class DbSettings:
    """Connection settings. The password comes from the environment and is never logged."""

    host: str
    port: int
    name: str
    user: str
    password: str


def get_db_settings() -> DbSettings:
    """Read the NEXUS_DB_* environment variables (loaded from .env by nexus.config)."""
    password = os.getenv("NEXUS_DB_PASSWORD", "")
    if not password:
        raise RuntimeError("NEXUS_DB_PASSWORD is not set. Put it in .env (see .env.example).")
    port_raw = os.getenv("NEXUS_DB_PORT", "5432").strip()
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ValueError(f"NEXUS_DB_PORT must be an integer, got {port_raw!r}") from exc
    return DbSettings(
        host=os.getenv("NEXUS_DB_HOST", "localhost").strip(),
        port=port,
        name=os.getenv("NEXUS_DB_NAME", "nexus_intelligence").strip(),
        user=os.getenv("NEXUS_DB_USER", "nexus_app").strip(),
        password=password,
    )


def get_engine() -> Engine:
    """Build a SQLAlchemy engine. It does not connect until it is used."""
    s = get_db_settings()
    url = URL.create(
        "postgresql+psycopg",
        username=s.user,
        password=s.password,
        host=s.host,
        port=s.port,
        database=s.name,
    )
    return create_engine(url, pool_pre_ping=True)


def schema_path() -> Path:
    return get_settings().sql_dir / "schema.sql"


def split_sql(sql: str) -> list[str]:
    """Split a SQL script into statements.

    Comment lines (starting with --) are removed. The script must not contain a semicolon
    inside a string or a comment, nor a percent sign, which schema.sql respects.
    """
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def apply_schema(engine: Engine, reset: bool = False) -> int:
    """Apply sql/schema.sql in one transaction. With reset=True the core schema is dropped first.

    Returns the number of statements in the script. The script is idempotent without reset.
    """
    statements = split_sql(schema_path().read_text(encoding="utf-8"))
    with engine.begin() as conn:
        if reset:
            conn.exec_driver_sql(f"DROP SCHEMA IF EXISTS {CORE_SCHEMA} CASCADE")
        for statement in statements:
            conn.exec_driver_sql(statement)
    return len(statements)


def list_core_tables(engine: Engine) -> list[str]:
    """Names of the base tables in the core schema, sorted."""
    query = text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = :schema AND table_type = 'BASE TABLE' ORDER BY table_name"
    )
    with engine.connect() as conn:
        return [row[0] for row in conn.execute(query, {"schema": CORE_SCHEMA})]