import hashlib

import pandas as pd
import pytest
from sqlalchemy import text

from nexus.config import get_settings
from nexus.db import apply_schema, get_engine
from nexus.loader import LOAD_ORDER, csv_row_count
from nexus.raw_loader import create_table_sql, file_sha256, load_raw

DIRTY = get_settings().raw_dir / "dirty"


@pytest.fixture(scope="module")
def engine():
    missing = [t for t in LOAD_ORDER if not (DIRTY / f"{t}.csv").exists()]
    if missing:
        pytest.skip(f"dirty CSV files are missing: {missing}")
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")
    apply_schema(eng)
    load_raw(eng, DIRTY)
    yield eng
    eng.dispose()


def _scalar(engine, sql):
    with engine.connect() as conn:
        return conn.execute(text(sql)).scalar()


def _columns(engine, schema):
    query = text(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = :s AND column_name <> '_row_id' ORDER BY table_name, ordinal_position"
    )
    out = {}
    with engine.connect() as conn:
        for table, column in conn.execute(query, {"s": schema}):
            out.setdefault(table, []).append(column)
    return out


def test_sha256_of_known_content(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == hashlib.sha256(b"abc").hexdigest()
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_create_table_sql_uses_identity_and_text():
    sql = create_table_sql("demo", ["date", "amount"])
    assert "CREATE TABLE raw.demo" in sql
    assert "_row_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY" in sql
    assert '"date" text' in sql
    assert '"amount" text' in sql


def test_raw_tables_exist_and_manifest_is_not_loaded(engine):
    query = text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'raw' ORDER BY table_name")
    with engine.connect() as conn:
        names = [r[0] for r in conn.execute(query)]
    assert sorted(names) == sorted([*LOAD_ORDER, "load_log"])
    assert "defect_manifest" not in names


def test_row_counts_match_the_dirty_csv_files(engine):
    for table in LOAD_ORDER:
        assert _scalar(engine, f"SELECT count(*) FROM raw.{table}") == csv_row_count(DIRTY / f"{table}.csv")


def test_columns_match_the_core_schema(engine):
    raw, core = _columns(engine, "raw"), _columns(engine, "core")
    for table in LOAD_ORDER:
        assert raw[table] == core[table], table


def test_all_data_columns_are_text(engine):
    other = _scalar(
        engine,
        "SELECT count(*) FROM information_schema.columns WHERE table_schema = 'raw' "
        "AND table_name <> 'load_log' AND column_name <> '_row_id' AND data_type <> 'text'",
    )
    total = _scalar(
        engine,
        "SELECT count(*) FROM information_schema.columns WHERE table_schema = 'raw' "
        "AND table_name <> 'load_log' AND column_name <> '_row_id'",
    )
    assert other == 0
    assert total > 50


def test_row_ids_are_sequential_in_file_order(engine):
    for table, key in (("customers", "customer_id"), ("orders", "order_id")):
        csv = pd.read_csv(DIRTY / f"{table}.csv", dtype=str, keep_default_na=False)
        with engine.connect() as conn:
            ids = [r[0] for r in conn.execute(text(f"SELECT {key} FROM raw.{table} ORDER BY _row_id"))]
        assert ids == csv[key].tolist()
        assert _scalar(engine, f"SELECT min(_row_id) FROM raw.{table}") == 1
        assert _scalar(engine, f"SELECT max(_row_id) FROM raw.{table}") == len(csv)


def test_dirty_values_are_preserved_as_text(engine):
    orders = pd.read_csv(DIRTY / "orders.csv", dtype=str)
    customers = pd.read_csv(DIRTY / "customers.csv", dtype=str)
    items = pd.read_csv(DIRTY / "order_items.csv", dtype=str)

    non_iso = int((~orders["order_date"].str.match(r"^\d{4}-\d{2}-\d{2}$")).sum())
    assert non_iso > 0
    assert _scalar(engine, "SELECT count(*) FROM raw.orders WHERE order_date !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'") == non_iso

    assert int(customers["email"].isna().sum()) > 0
    assert _scalar(engine, "SELECT count(*) FROM raw.customers WHERE email IS NULL") == int(customers["email"].isna().sum())

    assert int((orders["discount_pct"] == "1.5").sum()) > 0
    assert _scalar(engine, "SELECT count(*) FROM raw.orders WHERE discount_pct = '1.5'") == int((orders["discount_pct"] == "1.5").sum())

    negative = int((items["quantity"].astype(int) < 1).sum())
    assert negative > 0
    assert _scalar(engine, "SELECT count(*) FROM raw.order_items WHERE quantity::integer < 1") == negative


def test_load_log_records_hash_and_rows(engine):
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT table_name, source_file, file_sha256, rows_loaded FROM raw.load_log")).all()
    assert {r[0] for r in rows} == set(LOAD_ORDER)
    for name, source, sha, n in rows:
        path = DIRTY / f"{name}.csv"
        assert source == path.name
        assert sha == file_sha256(path)
        assert n == csv_row_count(path)


def test_reloading_gives_identical_counts(engine):
    before = {t: _scalar(engine, f"SELECT count(*) FROM raw.{t}") for t in LOAD_ORDER}
    load_raw(engine, DIRTY)
    after = {t: _scalar(engine, f"SELECT count(*) FROM raw.{t}") for t in LOAD_ORDER}
    assert before == after
    assert _scalar(engine, "SELECT count(*) FROM raw.load_log") == len(LOAD_ORDER)


def test_a_corrupt_file_rolls_back_the_load(engine, tmp_path):
    for table in LOAD_ORDER:
        (tmp_path / f"{table}.csv").write_bytes((DIRTY / f"{table}.csv").read_bytes())
    with open(tmp_path / "customers.csv", "ab") as handle:
        handle.write(b"only,two\n")
    before = _scalar(engine, "SELECT count(*) FROM raw.customers")
    assert before > 0
    with pytest.raises(Exception):
        load_raw(engine, tmp_path)
    assert _scalar(engine, "SELECT count(*) FROM raw.customers") == before
    assert _scalar(engine, "SELECT count(*) FROM raw.load_log") == len(LOAD_ORDER)