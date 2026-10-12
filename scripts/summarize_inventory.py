"""Print a summary of the generated daily inventory (reads data/raw; SYNTHETIC data)."""

from __future__ import annotations

import pandas as pd

from nexus.config import get_settings
from nexus.datagen.constants import DATA_END

HORIZON_DAYS = 14


def _future_start(x: pd.Series) -> pd.Series:
    """1 if a stockout episode starts on any of the next HORIZON_DAYS days (per product)."""
    rev = x.iloc[::-1]
    forward = rev.rolling(HORIZON_DAYS, min_periods=1).max().iloc[::-1]
    return forward.shift(-1)


def main() -> int:
    raw = get_settings().raw_dir
    inv = pd.read_csv(raw / "inventory_daily.csv", parse_dates=["date"])
    products = pd.read_csv(raw / "products.csv")
    suppliers = pd.read_csv(raw / "suppliers.csv")

    inv = (
        inv.merge(products[["product_id", "category", "supplier_id", "unit_cost"]], on="product_id")
        .merge(suppliers[["supplier_id", "region", "lead_time_mean_days"]], on="supplier_id")
        .sort_values(["product_id", "date"])
        .reset_index(drop=True)
    )
    inv["year"] = inv["date"].dt.year
    prev_flag = inv.groupby("product_id")["stockout_flag"].shift(1).fillna(0)
    inv["episode_start"] = ((inv["stockout_flag"] == 1) & (prev_flag == 0)).astype(int)

    print("SYNTHETIC DATA SUMMARY: DAILY INVENTORY")
    print(
        f"rows: {len(inv):,}   products: {inv['product_id'].nunique()}   "
        f"dates: {inv['date'].min().date()} to {inv['date'].max().date()}"
    )

    print(f"\nStockout share of product-days (overall): {inv['stockout_flag'].mean():.4f}")
    print("By year:")
    print(inv.groupby("year")["stockout_flag"].mean().round(4).to_string())

    with_stockout = inv.groupby("product_id")["stockout_flag"].max().sum()
    print(f"\nProducts with at least one stockout day: {int(with_stockout)} of {inv['product_id'].nunique()}")
    episodes = inv.groupby("year")["episode_start"].sum()
    days = inv.groupby("year")["stockout_flag"].sum()
    print("\nStockout episodes and mean episode length in days (episodes cut by year ends are split):")
    print(pd.DataFrame({"episodes": episodes, "stockout_days": days, "mean_days": (days / episodes).round(2)}).to_string())

    print("\nStockout share by supplier region:")
    print(inv.groupby("region")["stockout_flag"].mean().round(4).to_string())

    print("\nStockout share by category:")
    print(inv.groupby("category")["stockout_flag"].mean().round(4).to_string())

    inv["lead_band"] = pd.cut(
        inv["lead_time_mean_days"], bins=[0, 10, 15, 100], labels=["<=10 days", "11-15 days", ">15 days"]
    )
    print("\nStockout share by supplier planned lead time:")
    print(inv.groupby("lead_band", observed=True)["stockout_flag"].mean().round(4).to_string())

    by_supplier = inv.groupby("supplier_id").agg(
        products=("product_id", "nunique"), stockout_share=("stockout_flag", "mean")
    )
    print("\nFive suppliers with the highest stockout share:")
    print(by_supplier.sort_values("stockout_share", ascending=False).head(5).round(4).to_string())

    inv["cogs"] = inv["units_fulfilled"] * inv["unit_cost"]
    inv["writeoff_cost"] = inv["defective_writeoff_units"] * inv["unit_cost"]
    cogs = inv.groupby("year")["cogs"].sum()
    avg_value = inv.groupby(["year", "date"])["inventory_value"].sum().groupby("year").mean()
    table = pd.DataFrame(
        {
            "cogs": cogs,
            "avg_inventory_value": avg_value,
            "turnover": cogs / avg_value,
            "defect_writeoff_cost": inv.groupby("year")["writeoff_cost"].sum(),
        }
    )
    print("\nTurnover and defect write-offs by year (SYNTHETIC, at cost):")
    print(table.round(2).to_string())

    inv["label"] = inv.groupby("product_id")["episode_start"].transform(_future_start)
    cutoff = pd.Timestamp(DATA_END) - pd.Timedelta(days=HORIZON_DAYS)
    usable = inv[(inv["date"] <= cutoff) & (inv["stockout_flag"] == 0)]
    positives = int(usable["label"].sum())
    print(f"\nLabel 'a stockout starts within the next {HORIZON_DAYS} days', among product-days not in stockout:")
    print(f"positives: {positives:,} of {len(usable):,}   positive rate: {positives / len(usable):.4f}")
    print("positive rate by year:")
    print(usable.groupby("year")["label"].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())