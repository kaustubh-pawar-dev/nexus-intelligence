from datetime import date

import pandas as pd
import pytest

from nexus.dashboard_data import Kpis
from nexus.dashboard_view import (
    DEFINITIONS,
    build_cards,
    category_table,
    month_label,
    month_options,
    preset_window,
    trend_for_chart,
)

D = date


def kpis(**changes) -> Kpis:
    base = dict(
        start=D(2025, 1, 1), end=D(2025, 12, 31), orders=12, customers=9, revenue=1200.0,
        cogs=700.0, shipping=60.0, refunds=36.0, delivered=10, late=1,
    )
    base.update(changes)
    return Kpis(**base)


PREVIOUS = dict(orders=10, customers=8, revenue=1000.0, cogs=600.0, shipping=50.0, refunds=30.0, delivered=9, late=3)


def test_cards_show_values_and_changes_against_the_previous_period():
    cards = {c.label: c for c in build_cards(kpis(), kpis(**PREVIOUS))}
    assert list(cards) == list(DEFINITIONS)
    assert cards["Revenue"].value == "1.20K" and cards["Revenue"].delta == "+20.0%"
    assert cards["Orders"].value == "12" and cards["Orders"].delta == "+20.0%"
    assert cards["Customers"].value == "9" and cards["Customers"].delta == "+12.5%"
    assert cards["Average order value"].value == "100.00" and cards["Average order value"].delta == "+0.0%"
    assert cards["Gross margin"].value == "41.7%" and cards["Gross margin"].delta == "+1.7 pp"
    assert cards["Contribution margin"].value == "33.7%" and cards["Contribution margin"].delta == "+1.7 pp"
    assert cards["Refund rate"].value == "3.00%" and cards["Refund rate"].delta == "+0.0 pp"
    assert cards["On-time delivery"].value == "90.0%" and cards["On-time delivery"].delta == "+23.3 pp"
    assert cards["Refund rate"].delta_color == "inverse"
    assert all(c.delta_color == "normal" for c in cards.values() if c.label != "Refund rate")
    assert all(c.help == DEFINITIONS[c.label] for c in cards.values())


def test_cards_without_a_comparison_or_without_data():
    assert all(c.delta is None for c in build_cards(kpis(), None))
    empty = kpis(orders=0, customers=0, revenue=0.0, cogs=0.0, shipping=0.0, refunds=0.0, delivered=0, late=0)
    cards = {c.label: c for c in build_cards(empty, kpis(**PREVIOUS))}
    assert cards["Average order value"].value == "n/a" and cards["Gross margin"].value == "n/a"
    assert cards["On-time delivery"].value == "n/a" and cards["On-time delivery"].delta is None
    assert cards["Revenue"].delta == "-100.0%"


def test_preset_windows():
    first, last = D(2023, 1, 5), D(2025, 12, 31)
    assert preset_window("Full period", first, last) == (D(2023, 1, 1), D(2025, 12, 31))
    assert preset_window("Last 12 months", first, last) == (D(2025, 1, 1), D(2025, 12, 31))
    assert preset_window("Last 6 months", first, last) == (D(2025, 7, 1), D(2025, 12, 31))
    assert preset_window("Custom range", first, last) is None
    assert preset_window("Last 12 months", D(2025, 10, 3), last) == (D(2025, 10, 1), D(2025, 12, 31))  # short data
    with pytest.raises(ValueError, match="unknown"):
        preset_window("Yesterday", first, last)


def test_month_options_and_labels():
    months = month_options(D(2023, 1, 5), D(2025, 12, 31))
    assert len(months) == 36 and months[0] == D(2023, 1, 1) and months[-1] == D(2025, 12, 1)
    assert month_label(D(2024, 3, 1)) == "2024-03"
    assert month_options(D(2025, 5, 20), D(2025, 5, 25)) == [D(2025, 5, 1)]


def test_trend_for_chart_hides_incomplete_moving_averages_only():
    trend = pd.DataFrame(
        {
            "month": pd.date_range("2023-01-01", periods=4, freq="MS"),
            "revenue": [10.0, 20.0, 30.0, 40.0],
            "moving_avg_3m": [10.0, 15.0, 20.0, 30.0],
            "window_months": [1, 2, 3, 3],
        }
    )
    out = trend_for_chart(trend)
    assert out["moving_avg_3m"].isna().tolist() == [True, True, False, False]
    assert list(trend["moving_avg_3m"]) == [10.0, 15.0, 20.0, 30.0]  # the input is not changed


def test_category_table_shares_and_margins():
    mix = pd.DataFrame({"category": ["A", "B", "C"], "revenue": [100.0, 300.0, 600.0], "contribution": [20.0, 90.0, 120.0]})
    t = category_table(mix)
    assert list(t["category"]) == ["C", "B", "A"]
    assert t["revenue_share_pct"].sum() == pytest.approx(100.0)
    assert list(t["contribution_pct"].round(1)) == [20.0, 30.0, 20.0]