"""Generate synthetic shipments into data/raw.

Upstream tables are regenerated in memory from the same seed, so the output always matches
the CSVs from the earlier scripts.
"""

from __future__ import annotations

import argparse

from nexus.config import get_settings
from nexus.datagen.customers import generate_customers
from nexus.datagen.inventory import build_inventory_daily
from nexus.datagen.orders import generate_order_items, generate_orders
from nexus.datagen.purchasing import generate_purchase_orders
from nexus.datagen.reference import generate_products, generate_suppliers
from nexus.datagen.shipments import generate_shipments, validate_shipments
from nexus.logging_config import get_logger


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="random seed")
    args = parser.parse_args(argv)

    log = get_logger("nexus.datagen")
    log.info("Generating SYNTHETIC shipments with seed=%s", args.seed)

    suppliers = generate_suppliers(args.seed)
    products = generate_products(suppliers, args.seed)
    customers = generate_customers(args.seed)
    orders = generate_orders(customers, args.seed)
    order_items = generate_order_items(orders, customers, products, args.seed)
    purchase_orders = generate_purchase_orders(products, suppliers, orders, order_items, args.seed)
    inventory = build_inventory_daily(products, orders, order_items, purchase_orders)

    shipments = generate_shipments(orders, order_items, customers, inventory, args.seed)
    problems = validate_shipments(shipments, orders)
    if problems:
        for problem in problems:
            log.error("Validation failed: %s", problem)
        return 1

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    path = settings.raw_dir / "shipments.csv"
    shipments.to_csv(path, index=False, date_format="%Y-%m-%d")
    log.info("Wrote %s (%d rows, %d columns)", path, len(shipments), shipments.shape[1])

    print("RESULT: shipments generated and validated (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())