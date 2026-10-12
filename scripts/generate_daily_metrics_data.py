"""Generate the synthetic daily business metrics table into data/raw.

Upstream tables are regenerated in memory from the same seed, so the output always matches
the CSVs from the earlier scripts.
"""

from __future__ import annotations

import argparse

from nexus.config import get_settings
from nexus.datagen.customers import generate_customers
from nexus.datagen.daily_metrics import build_daily_metrics, validate_daily_metrics
from nexus.datagen.inventory import build_inventory_daily
from nexus.datagen.orders import generate_order_items, generate_orders
from nexus.datagen.payments import generate_payments, generate_refunds
from nexus.datagen.purchasing import generate_purchase_orders
from nexus.datagen.reference import generate_products, generate_suppliers
from nexus.datagen.shipments import generate_shipments
from nexus.logging_config import get_logger


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="random seed")
    args = parser.parse_args(argv)

    log = get_logger("nexus.datagen")
    log.info("Generating SYNTHETIC daily business metrics with seed=%s", args.seed)

    suppliers = generate_suppliers(args.seed)
    products = generate_products(suppliers, args.seed)
    customers = generate_customers(args.seed)
    orders = generate_orders(customers, args.seed)
    order_items = generate_order_items(orders, customers, products, args.seed)
    payments = generate_payments(orders, order_items, customers, args.seed)
    refunds = generate_refunds(payments, order_items, products, suppliers, args.seed)
    purchase_orders = generate_purchase_orders(products, suppliers, orders, order_items, args.seed)
    inventory = build_inventory_daily(products, orders, order_items, purchase_orders)
    shipments = generate_shipments(orders, order_items, customers, inventory, args.seed)

    daily = build_daily_metrics(
        customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory
    )
    problems = validate_daily_metrics(
        daily, customers, orders, order_items, payments, refunds, shipments, purchase_orders, inventory
    )
    if problems:
        for problem in problems:
            log.error("Validation failed: %s", problem)
        return 1

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    path = settings.raw_dir / "daily_metrics.csv"
    daily.to_csv(path, index=False, date_format="%Y-%m-%d")
    log.info("Wrote %s (%d rows, %d columns)", path, len(daily), daily.shape[1])

    print("RESULT: daily_metrics generated and validated (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())