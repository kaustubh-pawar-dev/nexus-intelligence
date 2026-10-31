import pandas as pd
import pytest

from nexus.db import get_engine
from nexus.quality_data import (
    RevenueCheck,
    accounting,
    analytics_vs_ground_truth,
    cleaning_log,
    ground_truth_checks,
    load_manifest,
    manifest_path,
    revenue_check,
)
from nexus.quality_view import (
    DEFECT_RULES,
    QUALITY_DEFINITIONS,
    accounting_shares,
    action_chart_frame,
    defect_comparison,
    knock_on_rows,
    quality_cards,
)


def sample_accounting() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "table_name": ["customers", "orders"],
            "raw_rows": [100, 200],
            "staged_rows": [90, 170],
            "dropped_duplicates": [10, 20],
            "quarantined": [0, 10],
            "unaccounted": [0, 0],
        }
    )


def sample_log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "table_name": ["customers", "customers", "orders", "orders", "order_items"],
            "action": ["dropped", "flagged", "dropped", "fixed", "quarantined"],
            "rule": ["duplicate_key", "missing_email", "duplicate_key", "status_case_standardised", "order_not_in_staging"],
            "rows": [10, 8, 20, 30, 5],
        }
    )


def test_defect_comparison_lines_up_injected_defects_with_pipeline_rows():
    manifest = pd.DataFrame(
        {
            "table": ["customers"] * 11 + ["orders"] * 31,
            "defect_type": ["exact_duplicate_row"] * 10 + ["missing_email"] + ["exact_duplicate_row"] * 20 + ["status_case_variant"] * 11,
        }
    )
    c = defect_comparison(manifest, sample_log()).set_index(["table", "defect"])
    assert len(c) == len(DEFECT_RULES) == 17
    assert c.loc[("customers", "exact_duplicate_row"), ["injected", "pipeline_rows", "difference"]].tolist() == [10, 10, 0]
    assert c.loc[("customers", "missing_email"), ["injected", "pipeline_rows", "difference"]].tolist() == [1, 8, -7]
    assert c.loc[("orders", "status_case_variant"), ["injected", "pipeline_rows", "difference"]].tolist() == [11, 30, -19]
    assert c.loc[("orders", "status_case_variant"), "action"] == "fixed"
    assert c.loc[("payments", "missing_amount"), ["injected", "pipeline_rows", "action"]].tolist() == [0, 0, ""]


def test_cards_use_the_accounting_and_revenue_results():
    cards = {c.label: c for c in quality_cards(sample_accounting(), 2266, 0.9194)}
    assert list(cards) == list(QUALITY_DEFINITIONS)
    assert cards["Raw rows cleaned"].value == "300" and cards["Unaccounted rows"].value == "0"
    assert cards["Defects injected"].value == "2,266" and cards["Revenue held back"].value == "0.92%"
    assert quality_cards(sample_accounting(), None, None)[2].value == "n/a"
    assert quality_cards(sample_accounting(), None, None)[3].value == "n/a"


def test_accounting_shares_add_up_to_100_percent():
    s = accounting_shares(sample_accounting())
    total = s["staged_rows_pct"] + s["dropped_duplicates_pct"] + s["quarantined_pct"] + s["unaccounted_pct"]
    assert total.round(9).tolist() == [100.0, 100.0]


def test_knock_on_rows_and_chart_frame():
    log = sample_log()
    assert list(knock_on_rows(log)["rule"]) == ["order_not_in_staging"]
    frame = action_chart_frame(log)
    assert frame["rows"].is_monotonic_decreasing and frame.loc[0, "label"] == "orders: status_case_standardised"


def test_revenue_check_arithmetic():
    r = RevenueCheck(staged_revenue=100.0, ground_truth_revenue=120.0, quarantined_revenue=20.0)
    assert r.unexplained_gap == 0.0 and r.held_back_pct == pytest.approx(100 * 20 / 120)
    assert RevenueCheck(0.0, 0.0, 0.0).held_back_pct is None
    assert RevenueCheck(100.0, 120.0, 15.0).unexplained_gap == 5.0
    assert str(RevenueCheck(12337953.99, 12452447.91, 114493.92).unexplained_gap) == "0.0"  # never shown as -0.00


# ---- reconciliation against the database ----

@pytest.fixture(scope="module")
def engine():
    return get_engine()


def test_every_raw_row_is_accounted_for_and_matches_a_direct_count(engine):
    acc = accounting(engine).set_index("table_name")
    assert (acc["unaccounted"] == 0).all()
    with engine.connect() as conn:
        for table in acc.index:
            raw = int(conn.exec_driver_sql(f"SELECT count(*) FROM raw.{table}").scalar())
            assert raw == acc.loc[table, "raw_rows"]
            assert acc.loc[table, ["staged_rows", "dropped_duplicates", "quarantined"]].sum() == raw


def test_cleaning_log_totals_match_the_accounting(engine):
    acc = accounting(engine).set_index("table_name")
    log = cleaning_log(engine)
    for table in acc.index:
        part = log[log["table_name"] == table]
        assert part.loc[part["action"] == "dropped", "rows"].sum() == acc.loc[table, "dropped_duplicates"]
        assert part.loc[part["action"] == "quarantined", "rows"].sum() == acc.loc[table, "quarantined"]


@pytest.mark.skipif(not manifest_path().exists(), reason="generate the dirty data first (scripts/generate_dirty_data.py)")
def test_the_rule_mapping_covers_every_injected_defect_and_the_pipeline_handles_almost_all(engine):
    manifest = load_manifest()
    assert set(map(tuple, manifest[["table", "defect_type"]].drop_duplicates().to_numpy())) == set(DEFECT_RULES)
    comparison = defect_comparison(manifest, cleaning_log(engine))
    assert (comparison["pipeline_rows"] <= comparison["injected"]).all()
    assert comparison["pipeline_rows"].sum() >= 0.98 * comparison["injected"].sum()


def test_staging_matches_the_ground_truth_except_for_imputed_dates(engine):
    checks = ground_truth_checks(engine).set_index("check_name")["rows"]
    must_match = [name for name in checks.index if "differs (imputed" not in name and "largest" not in name]
    assert len(must_match) == 6
    assert (checks[must_match] == 0).all()
    assert checks["customers: largest signup_date error in days"] <= 31  # imputed dates are estimates, but close


def test_revenue_gap_is_fully_explained_by_quarantine(engine):
    r = revenue_check(engine)
    assert abs(r.unexplained_gap) < 0.01
    assert 0 <= r.held_back_pct < 5


def test_analytics_layer_is_a_subset_of_the_ground_truth_where_rows_are_only_removed(engine):
    t = analytics_vs_ground_truth(engine).set_index("table")
    assert len(t) == 11
    assert (t["analytics_rows"] <= t["ground_truth_rows"]).all()
    for table in ("suppliers", "products", "purchase_orders", "inventory_daily"):
        assert t.loc[table, ["in_analytics_not_in_truth", "in_truth_not_in_analytics"]].sum() == 0
    for table in ("orders", "order_items", "payments", "refunds", "shipments"):
        assert t.loc[table, "in_analytics_not_in_truth"] == 0