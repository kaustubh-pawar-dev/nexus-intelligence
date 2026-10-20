"""Cleaning pipeline part 1: from the raw schema to the staging schema (SYNTHETIC data).

Reads only the raw tables. Every row-level decision is logged in staging.row_issues:
fixed, flagged, dropped (duplicate) or quarantined (kept out of staging). Nothing is
deleted silently: raw rows are always accounted for.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

STAGING_SCHEMA = "staging"
TABLES = ["customers", "orders", "order_items", "payments"]

CANONICAL_COUNTRIES = (
    "India", "United States", "United Kingdom", "Germany",
    "Australia", "United Arab Emirates", "Singapore", "Canada",
)
COUNTRY_ALIASES = {"usa": "United States", "uk": "United Kingdom", "uae": "United Arab Emirates"}
PAYMENT_METHODS = (
    "Credit Card", "Debit Card", "UPI", "Net Banking", "Digital Wallet", "Bank Transfer",
)


def country_map() -> dict[str, str]:
    mapping = {c.lower(): c for c in CANONICAL_COUNTRIES}
    mapping.update(COUNTRY_ALIASES)
    return mapping


def method_map() -> dict[str, str]:
    return {m.lower(): m for m in PAYMENT_METHODS}


def _values(mapping: dict[str, str]) -> str:
    return ", ".join(f"('{alias}', '{canonical}')" for alias, canonical in sorted(mapping.items()))


def infer_slash_date_format(conn: Connection) -> str:
    """Infer whether slash dates are dd/mm/yyyy or mm/dd/yyyy from the whole column.

    One part never exceeds 12 and the other does, which settles the format. If the evidence
    is ambiguous the pipeline stops instead of guessing. Returns 'dmy', 'mdy' or 'none'.
    """
    row = conn.execute(
        text(
            "SELECT count(*), max(split_part(order_date, '/', 1)::int), "
            "max(split_part(order_date, '/', 2)::int) "
            "FROM raw.orders WHERE order_date ~ '^[0-9]{2}/[0-9]{2}/[0-9]{4}$'"
        )
    ).one()
    n, first_max, second_max = int(row[0]), row[1], row[2]
    if n == 0:
        return "none"
    if second_max <= 12 < first_max:
        return "dmy"
    if first_max <= 12 < second_max:
        return "mdy"
    raise ValueError(
        f"Cannot infer the slash date format (max first part {first_max}, max second part {second_max})"
    )


_NUM = "'^[0-9]+([.][0-9]+)?$'"
_ISO = "'^[0-9]{4}-[0-9]{2}-[0-9]{2}$'"
_SLASH = "'^[0-9]{2}/[0-9]{2}/[0-9]{4}$'"

_TEMPLATES = [
    "CREATE SCHEMA IF NOT EXISTS staging",
    "CREATE INDEX IF NOT EXISTS ix_raw_customers_id ON raw.customers (customer_id)",
    "CREATE INDEX IF NOT EXISTS ix_raw_orders_id ON raw.orders (order_id)",
    "CREATE INDEX IF NOT EXISTS ix_raw_order_items_key ON raw.order_items (order_id, line_number)",
    "CREATE INDEX IF NOT EXISTS ix_raw_payments_id ON raw.payments (payment_id)",
    "CREATE INDEX IF NOT EXISTS ix_raw_products_id ON raw.products (product_id)",
    "ANALYZE raw.customers",
    "ANALYZE raw.orders",
    "ANALYZE raw.order_items",
    "ANALYZE raw.payments",
    "ANALYZE raw.products",
    "DROP VIEW IF EXISTS staging.quarantine",
    "DROP VIEW IF EXISTS staging.cleaning_summary",
    "DROP TABLE IF EXISTS staging.payments, staging.order_items, staging.orders, "
    "staging.customers, staging.row_issues",
    "CREATE TABLE staging.row_issues ("
    "table_name text NOT NULL, source_row_id bigint NOT NULL, row_key text NOT NULL, "
    "rule text NOT NULL, action text NOT NULL "
    "CHECK (action IN ('fixed', 'flagged', 'dropped', 'quarantined')))",
    "CREATE TEMP TABLE w_country_map (alias text PRIMARY KEY, canonical text NOT NULL) ON COMMIT DROP",
    "INSERT INTO w_country_map (alias, canonical) VALUES @COUNTRY_VALUES@",
    "CREATE TEMP TABLE w_method_map (alias text PRIMARY KEY, canonical text NOT NULL) ON COMMIT DROP",
    "INSERT INTO w_method_map (alias, canonical) VALUES @METHOD_VALUES@",
    # ---------------------------------------------------------------- customers
    "INSERT INTO staging.row_issues (table_name, source_row_id, row_key, rule, action) "
    "SELECT 'customers', d._row_id, d.customer_id, 'duplicate_key', 'dropped' FROM ("
    "SELECT _row_id, customer_id, row_number() OVER (PARTITION BY customer_id ORDER BY _row_id) AS rn "
    "FROM raw.customers) d WHERE d.rn > 1",
    """
    CREATE TEMP TABLE w_customers ON COMMIT DROP AS
    WITH ranked AS (
        SELECT c.*, row_number() OVER (PARTITION BY c.customer_id ORDER BY c._row_id) AS rn
        FROM raw.customers c
    ), kept AS (
        SELECT * FROM ranked WHERE rn = 1
    ), parsed AS (
        SELECT k.*,
            CASE WHEN k.signup_date ~ @ISO@ THEN
                CASE WHEN k.signup_date::date BETWEEN DATE '2021-01-01' AND DATE '2025-12-31'
                     THEN k.signup_date::date END
            END AS valid_signup
        FROM kept k
    )
    SELECT p.*,
        cm.canonical AS country_std,
        max(p.valid_signup) OVER (ORDER BY p.customer_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS filled_signup
    FROM parsed p
    LEFT JOIN w_country_map cm ON cm.alias = lower(trim(p.country))
    """,
    """
    INSERT INTO staging.row_issues (table_name, source_row_id, row_key, rule, action)
    SELECT 'customers', _row_id, customer_id, 'unknown_country', 'quarantined'
        FROM w_customers WHERE country_std IS NULL
    UNION ALL SELECT 'customers', _row_id, customer_id, 'unrecoverable_signup_date', 'quarantined'
        FROM w_customers WHERE COALESCE(valid_signup, filled_signup) IS NULL
    UNION ALL SELECT 'customers', _row_id, customer_id, 'country_standardised', 'fixed'
        FROM w_customers WHERE country_std IS NOT NULL AND country_std <> country
            AND COALESCE(valid_signup, filled_signup) IS NOT NULL
    UNION ALL SELECT 'customers', _row_id, customer_id, 'signup_date_imputed_from_neighbours', 'fixed'
        FROM w_customers WHERE valid_signup IS NULL AND filled_signup IS NOT NULL AND country_std IS NOT NULL
    UNION ALL SELECT 'customers', _row_id, customer_id, 'missing_email', 'flagged'
        FROM w_customers WHERE email IS NULL
    """,
    """
    CREATE TABLE staging.customers AS
    SELECT customer_id, email, country_std AS country, region, segment, acquisition_channel,
           COALESCE(valid_signup, filled_signup) AS signup_date,
           marketing_opt_in::boolean AS marketing_opt_in,
           _row_id AS source_row_id
    FROM w_customers
    WHERE country_std IS NOT NULL AND COALESCE(valid_signup, filled_signup) IS NOT NULL
    """,
    "ALTER TABLE staging.customers ADD PRIMARY KEY (customer_id)",
    # ------------------------------------------------------------------- orders
    """
    CREATE TEMP TABLE w_item_disc ON COMMIT DROP AS
    SELECT i.order_id,
        max(CASE WHEN i.discount_pct ~ @NUM@ THEN
                CASE WHEN i.discount_pct::numeric <= 0.20 THEN i.discount_pct::numeric END
            END) AS disc_items
    FROM raw.order_items i
    GROUP BY i.order_id
    """,
    "INSERT INTO staging.row_issues (table_name, source_row_id, row_key, rule, action) "
    "SELECT 'orders', d._row_id, d.order_id, 'duplicate_key', 'dropped' FROM ("
    "SELECT _row_id, order_id, row_number() OVER (PARTITION BY order_id ORDER BY _row_id) AS rn "
    "FROM raw.orders) d WHERE d.rn > 1",
    """
    CREATE TEMP TABLE w_orders ON COMMIT DROP AS
    WITH ranked AS (
        SELECT o.*, row_number() OVER (PARTITION BY o.order_id ORDER BY o._row_id) AS rn
        FROM raw.orders o
    ), kept AS (
        SELECT * FROM ranked WHERE rn = 1
    ), parsed AS (
        SELECT k.*,
            CASE WHEN k.order_date ~ @ISO@ THEN k.order_date::date END AS iso_date,
            CASE WHEN k.order_date ~ @SLASH@ THEN split_part(k.order_date, '/', 1)::int END AS p1,
            CASE WHEN k.order_date ~ @SLASH@ THEN split_part(k.order_date, '/', 2)::int END AS p2,
            CASE WHEN k.order_date ~ @SLASH@ THEN split_part(k.order_date, '/', 3)::int END AS yr
        FROM kept k
    ), bounds AS (
        SELECT p.*,
            max(p.iso_date) OVER (ORDER BY p.order_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS prev_date,
            min(p.iso_date) OVER (ORDER BY p.order_id DESC ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
                AS next_date
        FROM parsed p
    ), slash AS (
        SELECT b.*,
            CASE WHEN b.@MONTH@ BETWEEN 1 AND 12 THEN
                CASE WHEN b.@DAY@ BETWEEN 1 AND
                          EXTRACT(DAY FROM (make_date(b.yr, b.@MONTH@, 1) + INTERVAL '1 month'
                                            - INTERVAL '1 day'))::int
                     THEN make_date(b.yr, b.@MONTH@, b.@DAY@) END
            END AS slash_date
        FROM bounds b
    ), resolved AS (
        SELECT s.*,
            CASE WHEN s.iso_date IS NOT NULL THEN
                    CASE WHEN s.iso_date BETWEEN DATE '2023-01-01' AND DATE '2025-12-31'
                         THEN s.iso_date END
                 WHEN s.slash_date BETWEEN COALESCE(s.prev_date, DATE '2023-01-01')
                                       AND COALESCE(s.next_date, DATE '2025-12-31')
                      THEN s.slash_date
            END AS order_date_std,
            CASE lower(trim(s.order_status))
                WHEN 'completed' THEN 'Completed' WHEN 'cancelled' THEN 'Cancelled' END AS status_std,
            CASE WHEN s.discount_pct ~ @NUM@ THEN
                CASE WHEN s.discount_pct::numeric <= 0.20 THEN s.discount_pct::numeric END
            END AS disc_valid
        FROM slash s
    ), merged AS (
        SELECT r.*, COALESCE(r.disc_valid, di.disc_items) AS disc_std
        FROM resolved r LEFT JOIN w_item_disc di ON di.order_id = r.order_id
    ), checked AS (
        SELECT m.*,
            COALESCE(
                CASE WHEN m.customer_id IS NULL THEN 'missing_customer_id'
                     WHEN EXISTS (SELECT 1 FROM staging.customers c WHERE c.customer_id = m.customer_id)
                          THEN NULL
                     WHEN EXISTS (SELECT 1 FROM raw.customers rc WHERE rc.customer_id = m.customer_id)
                          THEN 'customer_quarantined'
                     ELSE 'unknown_customer_id' END,
                CASE WHEN m.order_date_std IS NULL THEN 'unresolved_order_date'
                     WHEN m.status_std IS NULL THEN 'unknown_order_status'
                     WHEN m.disc_std IS NULL THEN 'unresolved_discount' END
            ) AS reject_reason
        FROM merged m
    )
    SELECT * FROM checked
    """,
    """
    INSERT INTO staging.row_issues (table_name, source_row_id, row_key, rule, action)
    SELECT 'orders', _row_id, order_id, reject_reason, 'quarantined'
        FROM w_orders WHERE reject_reason IS NOT NULL
    UNION ALL SELECT 'orders', _row_id, order_id, 'date_format_standardised', 'fixed'
        FROM w_orders WHERE reject_reason IS NULL AND iso_date IS NULL
    UNION ALL SELECT 'orders', _row_id, order_id, 'status_case_standardised', 'fixed'
        FROM w_orders WHERE reject_reason IS NULL AND status_std <> order_status
    UNION ALL SELECT 'orders', _row_id, order_id, 'discount_recovered_from_items', 'fixed'
        FROM w_orders WHERE reject_reason IS NULL AND disc_valid IS NULL
    """,
    """
    CREATE TABLE staging.orders AS
    SELECT order_id, customer_id, order_date_std AS order_date, sales_channel,
           status_std AS order_status, disc_std::numeric(4,2) AS discount_pct,
           _row_id AS source_row_id
    FROM w_orders WHERE reject_reason IS NULL
    """,
    "ALTER TABLE staging.orders ADD PRIMARY KEY (order_id)",
    # -------------------------------------------------------------- order_items
    "INSERT INTO staging.row_issues (table_name, source_row_id, row_key, rule, action) "
    "SELECT 'order_items', d._row_id, d.order_id || '|' || d.line_number, 'duplicate_key', 'dropped' FROM ("
    "SELECT _row_id, order_id, line_number, "
    "row_number() OVER (PARTITION BY order_id, line_number ORDER BY _row_id) AS rn "
    "FROM raw.order_items) d WHERE d.rn > 1",
    """
    CREATE TEMP TABLE w_items ON COMMIT DROP AS
    WITH ranked AS (
        SELECT i.*, row_number() OVER (PARTITION BY i.order_id, i.line_number ORDER BY i._row_id) AS rn
        FROM raw.order_items i
    ), kept AS (
        SELECT * FROM ranked WHERE rn = 1
    ), typed AS (
        SELECT k.*,
            k.line_number::integer AS ln,
            k.quantity::integer AS qty_raw,
            k.unit_price::numeric AS up,
            k.discount_pct::numeric AS disc,
            k.line_total::numeric AS lt,
            pr.list_price::numeric AS list_price
        FROM kept k LEFT JOIN raw.products pr ON pr.product_id = k.product_id
    ), priced AS (
        SELECT t.*, CASE WHEN t.up > 0 THEN t.up ELSE t.list_price END AS price_std
        FROM typed t
    ), qtyd AS (
        SELECT s.*,
            CASE WHEN s.qty_raw BETWEEN 1 AND 20 THEN s.qty_raw
                 WHEN s.qty_raw < 0 AND (-s.qty_raw) BETWEEN 1 AND 20
                      AND s.lt IS NOT NULL AND s.price_std IS NOT NULL
                      AND abs(round((-s.qty_raw) * s.price_std * (1 - s.disc), 2) - s.lt) <= 0.01
                      THEN -s.qty_raw
            END AS qty_std
        FROM priced s
    ), totalled AS (
        SELECT u.*,
            COALESCE(u.lt, CASE WHEN u.qty_std IS NOT NULL AND u.price_std IS NOT NULL
                                THEN round(u.qty_std * u.price_std * (1 - u.disc), 2) END) AS lt_std
        FROM qtyd u
    ), checked AS (
        SELECT f.*,
            CASE
                WHEN NOT EXISTS (SELECT 1 FROM staging.orders o WHERE o.order_id = f.order_id)
                     THEN 'order_not_in_staging'
                WHEN f.list_price IS NULL THEN 'unknown_product'
                WHEN f.price_std IS NULL THEN 'unresolved_unit_price'
                WHEN f.qty_std IS NULL THEN 'unresolved_quantity'
                WHEN f.lt_std IS NULL THEN 'unresolved_line_total'
                WHEN abs(round(f.qty_std * f.price_std * (1 - f.disc), 2) - f.lt_std) > 0.01
                     THEN 'line_total_mismatch'
            END AS reject_reason
        FROM totalled f
    )
    SELECT * FROM checked
    """,
    """
    INSERT INTO staging.row_issues (table_name, source_row_id, row_key, rule, action)
    SELECT 'order_items', _row_id, order_id || '|' || line_number, reject_reason, 'quarantined'
        FROM w_items WHERE reject_reason IS NOT NULL
    UNION ALL SELECT 'order_items', _row_id, order_id || '|' || line_number, 'quantity_sign_corrected', 'fixed'
        FROM w_items WHERE reject_reason IS NULL AND qty_raw < 0
    UNION ALL SELECT 'order_items', _row_id, order_id || '|' || line_number, 'unit_price_from_catalogue', 'fixed'
        FROM w_items WHERE reject_reason IS NULL AND up <= 0
    UNION ALL SELECT 'order_items', _row_id, order_id || '|' || line_number, 'line_total_recomputed', 'fixed'
        FROM w_items WHERE reject_reason IS NULL AND lt IS NULL
    """,
    """
    CREATE TABLE staging.order_items AS
    SELECT order_id, ln AS line_number, product_id, qty_std AS quantity,
           price_std::numeric(12,2) AS unit_price, unit_cost::numeric(12,2) AS unit_cost,
           disc::numeric(4,2) AS discount_pct, lt_std::numeric(14,2) AS line_total,
           _row_id AS source_row_id
    FROM w_items WHERE reject_reason IS NULL
    """,
    "ALTER TABLE staging.order_items ADD PRIMARY KEY (order_id, line_number)",
    "ALTER TABLE staging.order_items ADD CONSTRAINT staging_items_unique_product UNIQUE (order_id, product_id)",
    # ----------------------------------------------------------------- payments
    "INSERT INTO staging.row_issues (table_name, source_row_id, row_key, rule, action) "
    "SELECT 'payments', d._row_id, d.payment_id, 'duplicate_key', 'dropped' FROM ("
    "SELECT _row_id, payment_id, row_number() OVER (PARTITION BY payment_id ORDER BY _row_id) AS rn "
    "FROM raw.payments) d WHERE d.rn > 1",
    """
    CREATE TEMP TABLE w_item_totals ON COMMIT DROP AS
    SELECT order_id, count(*) AS staged_lines, sum(line_total) AS staged_total
    FROM staging.order_items GROUP BY order_id
    """,
    """
    CREATE TEMP TABLE w_item_counts ON COMMIT DROP AS
    SELECT order_id, count(DISTINCT line_number) AS raw_lines
    FROM raw.order_items GROUP BY order_id
    """,
    """
    CREATE TEMP TABLE w_payments ON COMMIT DROP AS
    WITH ranked AS (
        SELECT p.*, row_number() OVER (PARTITION BY p.payment_id ORDER BY p._row_id) AS rn
        FROM raw.payments p
    ), kept AS (
        SELECT * FROM ranked WHERE rn = 1
    ), typed AS (
        SELECT k.*,
            CASE WHEN k.amount ~ @NUM@ THEN k.amount::numeric END AS amt,
            CASE WHEN k.payment_date ~ @ISO@ THEN k.payment_date::date END AS pay_date,
            mm.canonical AS method_std,
            so.order_status AS order_status_std,
            it.staged_total, it.staged_lines, ic.raw_lines
        FROM kept k
        LEFT JOIN w_method_map mm ON mm.alias = lower(trim(k.payment_method))
        LEFT JOIN staging.orders so ON so.order_id = k.order_id
        LEFT JOIN w_item_totals it ON it.order_id = k.order_id
        LEFT JOIN w_item_counts ic ON ic.order_id = k.order_id
    ), merged AS (
        SELECT t.*,
            COALESCE(t.amt, CASE WHEN t.staged_lines = t.raw_lines THEN t.staged_total END) AS amount_std
        FROM typed t
    ), checked AS (
        SELECT f.*,
            CASE
                WHEN f.order_status_std IS NULL THEN 'order_not_in_staging'
                WHEN f.order_status_std <> 'Completed' THEN 'payment_for_non_completed_order'
                WHEN f.method_std IS NULL THEN 'unknown_payment_method'
                WHEN f.payment_status IS NULL OR f.payment_status NOT IN ('Captured', 'Failed')
                     THEN 'unknown_payment_status'
                WHEN f.pay_date IS NULL THEN 'invalid_payment_date'
                WHEN f.amount_std IS NULL OR f.amount_std <= 0 THEN 'unresolved_amount'
            END AS reject_reason
        FROM merged f
    )
    SELECT * FROM checked
    """,
    """
    INSERT INTO staging.row_issues (table_name, source_row_id, row_key, rule, action)
    SELECT 'payments', _row_id, payment_id, reject_reason, 'quarantined'
        FROM w_payments WHERE reject_reason IS NOT NULL
    UNION ALL SELECT 'payments', _row_id, payment_id, 'payment_method_case_standardised', 'fixed'
        FROM w_payments WHERE reject_reason IS NULL AND method_std <> payment_method
    UNION ALL SELECT 'payments', _row_id, payment_id, 'amount_recomputed_from_items', 'fixed'
        FROM w_payments WHERE reject_reason IS NULL AND amt IS NULL
    UNION ALL SELECT 'payments', _row_id, payment_id, 'amount_differs_from_items', 'flagged'
        FROM w_payments WHERE reject_reason IS NULL AND amt IS NOT NULL
            AND staged_lines = raw_lines AND abs(amt - staged_total) > 0.01
    """,
    """
    CREATE TABLE staging.payments AS
    SELECT payment_id, order_id, pay_date AS payment_date, method_std AS payment_method,
           payment_status, amount_std::numeric(14,2) AS amount, _row_id AS source_row_id
    FROM w_payments WHERE reject_reason IS NULL
    """,
    "ALTER TABLE staging.payments ADD PRIMARY KEY (payment_id)",
    "ALTER TABLE staging.payments ADD CONSTRAINT staging_payments_unique_order UNIQUE (order_id)",
    # -------------------------------------------------------------------- views
    "CREATE VIEW staging.quarantine AS SELECT * FROM staging.row_issues WHERE action = 'quarantined'",
    "CREATE VIEW staging.cleaning_summary AS "
    "SELECT table_name, rule, action, count(*) AS rows FROM staging.row_issues GROUP BY 1, 2, 3",
]


def staging_statements(fmt: str) -> list[str]:
    """The SQL statements of the pipeline, in execution order, for a slash date format."""
    day, month = ("p2", "p1") if fmt == "mdy" else ("p1", "p2")
    out = []
    for template in _TEMPLATES:
        sql = (
            template.replace("@DAY@", day)
            .replace("@MONTH@", month)
            .replace("@COUNTRY_VALUES@", _values(country_map()))
            .replace("@METHOD_VALUES@", _values(method_map()))
            .replace("@NUM@", _NUM)
            .replace("@ISO@", _ISO)
            .replace("@SLASH@", _SLASH)
        )
        out.append(" ".join(sql.split()))
    return out


def build_staging(engine: Engine) -> tuple[dict[str, int], str]:
    """Rebuild the staging layer in one transaction. Returns (rows per table, slash format).

    Any error rolls everything back, leaving the previous staging layer untouched.
    """
    with engine.begin() as conn:
        fmt = infer_slash_date_format(conn)
        for statement in staging_statements(fmt):
            conn.exec_driver_sql(statement)
        counts = {
            t: int(conn.execute(text(f"SELECT count(*) FROM {STAGING_SCHEMA}.{t}")).scalar())
            for t in TABLES
        }
    return counts, fmt


_INTEGRITY_CHECKS = [
    ("orders without a staged customer",
     "SELECT count(*) FROM staging.orders o LEFT JOIN staging.customers c USING (customer_id) "
     "WHERE c.customer_id IS NULL"),
    ("order lines without a staged order",
     "SELECT count(*) FROM staging.order_items i LEFT JOIN staging.orders o USING (order_id) "
     "WHERE o.order_id IS NULL"),
    ("order lines without a catalogue product",
     "SELECT count(*) FROM staging.order_items i LEFT JOIN raw.products p USING (product_id) "
     "WHERE p.product_id IS NULL"),
    ("payments without a staged order",
     "SELECT count(*) FROM staging.payments p LEFT JOIN staging.orders o USING (order_id) "
     "WHERE o.order_id IS NULL"),
    ("staged orders without any staged line",
     "SELECT count(*) FROM staging.orders o LEFT JOIN staging.order_items i USING (order_id) "
     "WHERE i.order_id IS NULL"),
    ("completed staged orders without a staged payment",
     "SELECT count(*) FROM staging.orders o LEFT JOIN staging.payments p USING (order_id) "
     "WHERE o.order_status = 'Completed' AND p.payment_id IS NULL"),
    ("staged payments whose amount differs from their order lines",
     "SELECT count(*) FROM staging.payments p JOIN (SELECT order_id, sum(line_total) AS total "
     "FROM staging.order_items GROUP BY order_id) t USING (order_id) WHERE abs(p.amount - t.total) > 0.01"),
    ("order lines whose discount differs from their order",
     "SELECT count(*) FROM staging.order_items i JOIN staging.orders o USING (order_id) "
     "WHERE i.discount_pct <> o.discount_pct"),
]


def check_staging(engine: Engine) -> list[str]:
    """Accounting and integrity checks. Returns problems (an empty list means all good)."""
    problems: list[str] = []
    with engine.connect() as conn:
        for table in TABLES:
            raw_n = int(conn.execute(text(f"SELECT count(*) FROM raw.{table}")).scalar())
            staged = int(conn.execute(text(f"SELECT count(*) FROM staging.{table}")).scalar())
            removed = int(
                conn.execute(
                    text(
                        "SELECT count(DISTINCT source_row_id) FROM staging.row_issues "
                        "WHERE table_name = :t AND action IN ('dropped', 'quarantined')"
                    ),
                    {"t": table},
                ).scalar()
            )
            if raw_n != staged + removed:
                problems.append(
                    f"{table}: {raw_n} raw rows, {staged} staged, {removed} removed "
                    f"({raw_n - staged - removed} unaccounted)"
                )
        for name, sql in _INTEGRITY_CHECKS:
            n = int(conn.execute(text(sql)).scalar())
            if n:
                problems.append(f"{n} {name}")
    return problems