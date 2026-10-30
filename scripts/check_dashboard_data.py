"""Print the dashboard KPIs per year and for the full range, to compare with the earlier SQL summaries (SYNTHETIC data)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from nexus.dashboard_data import (
    category_mix,
    data_range,
    fetch_kpis,
    month_window,
    pct_change,
    previous_period,
)
from nexus.dashboard_format import format_compact, format_delta, format_number, format_pct
from nexus.db import get_engine


def main() -> int:
    pd.set_option("display.width", 200)
    engine = get_engine()
    first, last = data_range(engine)
    print(f"SYNTHETIC DATA: DASHBOARD KPIs (completed orders {first} to {last})")

    rows = []
    for year in (2023, 2024, 2025):
        start, end = month_window(date(year, 1, 1), date(year, 12, 1))
        k = fetch_kpis(engine, start, end)
        prev = previous_period(start, end, first)
        change = pct_change(k.revenue, fetch_kpis(engine, *prev).revenue) if prev else None
        rows.append(
            {
                "year": year,
                "revenue": format_number(round(k.revenue)),
                "vs_prior": format_delta(change) or "n/a",
                "orders": format_number(k.orders),
                "customers": format_number(k.customers),
                "aov": format_number(k.aov, 2),
                "gross_margin": format_pct(k.gross_margin_pct),
                "contribution": format_pct(k.contribution_pct),
                "refund_rate": format_pct(k.refund_rate_pct, 2),
                "on_time": format_pct(k.on_time_pct),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    start, end = month_window(date(2025, 1, 1), date(2025, 12, 1))
    mix = category_mix(engine, start, end)
    mix["revenue_share"] = (100 * mix["revenue"] / mix["revenue"].sum()).round(1)
    print("\n2025 revenue and contribution by category:")
    print(mix.assign(revenue=mix["revenue"].map(format_compact), contribution=mix["contribution"].map(format_compact)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())