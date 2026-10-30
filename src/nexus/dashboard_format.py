"""Number formatting for the dashboard. None always means "not available" and shows as n/a (SYNTHETIC data)."""

from __future__ import annotations

NA = "n/a"


def format_number(value, decimals: int = 0) -> str:
    return NA if value is None else f"{value:,.{decimals}f}"


def format_compact(value) -> str:
    """12,337,954 -> 12.34M, 4,100 -> 4.10K. The dataset defines no currency, so no symbol is shown."""
    if value is None:
        return NA
    sign = "-" if value < 0 else ""
    v = abs(float(value))
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= limit:
            return f"{sign}{v / limit:.2f}{suffix}"
    return f"{sign}{v:.0f}"


def format_pct(value, decimals: int = 1) -> str:
    return NA if value is None else f"{value:.{decimals}f}%"


def format_delta(change_pct) -> str | None:
    """Signed percentage change for st.metric, or None when there is no comparison."""
    return None if change_pct is None else f"{change_pct:+.1f}%"