"""View model of the Data Quality page: cards, defect comparison and chart tables (SYNTHETIC data).

No Streamlit or Plotly imports here, so everything can be tested directly.
"""

from __future__ import annotations

import pandas as pd

from nexus.dashboard_format import format_number, format_pct
from nexus.dashboard_view import Card

# Which cleaning rule handles which injected defect type: (table, defect_type) -> rule.
DEFECT_RULES = {
    ("customers", "exact_duplicate_row"): "duplicate_key",
    ("customers", "missing_email"): "missing_email",
    ("customers", "country_variant"): "country_standardised",
    ("customers", "future_signup_date"): "signup_date_imputed_from_neighbours",
    ("orders", "exact_duplicate_row"): "duplicate_key",
    ("orders", "date_format_dmy"): "date_format_standardised",
    ("orders", "missing_customer_id"): "missing_customer_id",
    ("orders", "unknown_customer_id"): "unknown_customer_id",
    ("orders", "invalid_discount_pct"): "discount_recovered_from_items",
    ("orders", "status_case_variant"): "status_case_standardised",
    ("order_items", "exact_duplicate_row"): "duplicate_key",
    ("order_items", "negative_quantity"): "quantity_sign_corrected",
    ("order_items", "missing_line_total"): "line_total_recomputed",
    ("order_items", "zero_unit_price"): "unit_price_from_catalogue",
    ("payments", "exact_duplicate_row"): "duplicate_key",
    ("payments", "method_case_variant"): "payment_method_case_standardised",
    ("payments", "missing_amount"): "amount_recomputed_from_items",
}

QUALITY_DEFINITIONS = {
    "Raw rows cleaned": "Rows delivered in the four tables that go through the cleaning pipeline: customers, orders, order items and payments.",
    "Unaccounted rows": "Raw rows that are neither staged, dropped as a duplicate nor quarantined. It must be zero.",
    "Defects injected": "Defects deliberately added to the raw data and recorded in the manifest, so the cleaning can be measured against a known answer.",
    "Revenue held back": "Share of ground-truth revenue that sits in quarantined orders and is therefore missing from the cleaned layer.",
}


def quality_cards(accounting: pd.DataFrame, injected_total: int | None, held_back_pct: float | None) -> list[Card]:
    d = QUALITY_DEFINITIONS
    return [
        Card("Raw rows cleaned", format_number(int(accounting["raw_rows"].sum())), None, "off", d["Raw rows cleaned"]),
        Card("Unaccounted rows", format_number(int(accounting["unaccounted"].sum())), None, "off", d["Unaccounted rows"]),
        Card("Defects injected", format_number(injected_total), None, "off", d["Defects injected"]),
        Card("Revenue held back", format_pct(held_back_pct, 2), None, "off", d["Revenue held back"]),
    ]


def accounting_shares(accounting: pd.DataFrame) -> pd.DataFrame:
    """Share of the raw rows of each table that was staged, dropped as a duplicate or quarantined."""
    out = accounting.copy()
    for col in ("staged_rows", "dropped_duplicates", "quarantined", "unaccounted"):
        out[f"{col}_pct"] = 100 * out[col] / out["raw_rows"]
    return out


def defect_comparison(manifest: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Injected defects against what the pipeline did, one row per defect type.

    A defect that sat on a row that was also dropped or quarantined cannot be repaired, so a small
    positive difference is expected.
    """
    injected = manifest.groupby(["table", "defect_type"]).size()
    handled = log.groupby(["table_name", "rule"])["rows"].sum()
    rows = []
    for (table, defect), rule in DEFECT_RULES.items():
        actions = log.loc[(log["table_name"] == table) & (log["rule"] == rule), "action"]
        inj = int(injected.get((table, defect), 0))
        done = int(handled.get((table, rule), 0))
        rows.append(
            {
                "table": table,
                "defect": defect,
                "pipeline_rule": rule,
                "action": actions.iloc[0] if len(actions) else "",
                "injected": inj,
                "pipeline_rows": done,
                "difference": inj - done,
            }
        )
    return pd.DataFrame(rows)


def knock_on_rows(log: pd.DataFrame) -> pd.DataFrame:
    """Cleaning actions that are consequences of other actions (for example an item whose order was quarantined)."""
    return log[~log["rule"].isin(set(DEFECT_RULES.values()))].reset_index(drop=True)


def action_chart_frame(log: pd.DataFrame) -> pd.DataFrame:
    """Cleaning actions with a label for charts, largest first."""
    out = log.copy()
    out["label"] = out["table_name"] + ": " + out["rule"]
    return out.sort_values("rows", ascending=False).reset_index(drop=True)