"""Load the as-delivered CSV files into the raw schema as text (SYNTHETIC data).

Every data column is text, so defective values load without being rejected. Each row keeps
its position in the source file in _row_id, and raw.load_log records each file's hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from nexus.db import CORE_SCHEMA
from nexus.loader import LOAD_ORDER, csv_row_count

RAW_SCHEMA = "raw"
ROW_ID = "_row_id"

LOAD_LOG_SQL = (
    f"CREATE TABLE {RAW_SCHEMA}.load_log ("
    "table_name text PRIMARY KEY, "
    "source_file text NOT NULL, "
    "file_sha256 text NOT NULL, "
    "rows_loaded bigint NOT NULL, "
    "loaded_at timestamptz NOT NULL DEFAULT now())"
)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def create_table_sql(table: str, columns: list[str]) -> str:
    """DDL for one raw table: an identity row id, then every CSV column as text."""
    cols = ",\n    ".join(f"{quote_ident(c)} text" for c in columns)
    return (
        f"CREATE TABLE {RAW_SCHEMA}.{table} (\n"
        f"    {ROW_ID} bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\n"
        f"    {cols}\n)"
    )


def _copy_file(raw_conn, table: str, path: Path, columns: list[str]) -> None:
    column_sql = ", ".join(quote_ident(c) for c in columns)
    statement = (
        f"COPY {RAW_SCHEMA}.{table} ({column_sql}) FROM STDIN WITH (FORMAT csv, HEADER true)"
    )
    with raw_conn.cursor() as cur:
        with cur.copy(statement) as copy:
            with open(path, "rb") as handle:
                while chunk := handle.read(1 << 20):
                    copy.write(chunk)


def load_raw(engine: Engine, dirty_dir: Path) -> dict[str, int]:
    """Drop, recreate and load all raw tables in one transaction. Returns rows per table.

    Any error rolls the whole load back, leaving the previous raw layer untouched.
    """
    missing = [t for t in LOAD_ORDER if not (dirty_dir / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files in {dirty_dir}: {', '.join(missing)}")

    loaded: dict[str, int] = {}
    with engine.begin() as conn:
        conn.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}")
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {RAW_SCHEMA}.load_log")
        for table in LOAD_ORDER:
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {RAW_SCHEMA}.{table}")
        conn.exec_driver_sql(LOAD_LOG_SQL)

        raw_conn = conn.connection.driver_connection
        for table in LOAD_ORDER:
            path = dirty_dir / f"{table}.csv"
            columns = list(pd.read_csv(path, nrows=0).columns)
            conn.exec_driver_sql(create_table_sql(table, columns))
            _copy_file(raw_conn, table, path, columns)
            n = int(conn.execute(text(f"SELECT count(*) FROM {RAW_SCHEMA}.{table}")).scalar())
            loaded[table] = n
            conn.execute(
                text(
                    f"INSERT INTO {RAW_SCHEMA}.load_log (table_name, source_file, file_sha256, rows_loaded) "
                    "VALUES (:t, :f, :h, :n)"
                ),
                {"t": table, "f": path.name, "h": file_sha256(path), "n": n},
            )
    return loaded


def _columns(conn: Connection, schema: str) -> dict[str, list[str]]:
    query = text(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = :s AND column_name <> '_row_id' "
        "ORDER BY table_name, ordinal_position"
    )
    out: dict[str, list[str]] = {}
    for table, column in conn.execute(query, {"s": schema}):
        out.setdefault(table, []).append(column)
    return out


def reconcile_raw(engine: Engine, dirty_dir: Path) -> list[str]:
    """Compare the raw layer with the files and with core. Returns problems (empty is good)."""
    problems: list[str] = []
    with engine.connect() as conn:
        for table in LOAD_ORDER:
            expected = csv_row_count(dirty_dir / f"{table}.csv")
            actual = int(conn.execute(text(f"SELECT count(*) FROM {RAW_SCHEMA}.{table}")).scalar())
            if actual != expected:
                problems.append(f"{table}: {actual} rows in raw, {expected} in the file")

        raw_cols = _columns(conn, RAW_SCHEMA)
        core_cols = _columns(conn, CORE_SCHEMA)
        for table in LOAD_ORDER:
            if raw_cols.get(table) != core_cols.get(table):
                problems.append(f"{table}: raw columns differ from core columns")

        log = {
            row[0]: (row[1], int(row[2]))
            for row in conn.execute(
                text(f"SELECT table_name, file_sha256, rows_loaded FROM {RAW_SCHEMA}.load_log")
            )
        }
        for table in LOAD_ORDER:
            if table not in log:
                problems.append(f"{table}: missing from load_log")
                continue
            sha, rows = log[table]
            path = dirty_dir / f"{table}.csv"
            if sha != file_sha256(path):
                problems.append(f"{table}: file changed since it was loaded")
            if rows != csv_row_count(path):
                problems.append(f"{table}: load_log row count differs from the file")
    return problems