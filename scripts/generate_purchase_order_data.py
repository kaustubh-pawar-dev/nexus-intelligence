"""Generate synthetic purchase orders into data/raw.

Upstream tables (suppliers, products, customers, orders, order lines) are regenerated in
memory from the same seed, so the output always matches the CSVs from the earlier scripts.
"""

from __future__ import annotations

import argparse

from nexus.config import get_settings
from nexus.datagen.customers import generate_customers
from nexus.datagen.orders import generate_order_items, generate_orders
from nexus.datagen.purchasing import generate_purchase_orders, validate_purchase_orders
from nexus.datagen.reference import generate_products, generate_suppliers
from nexus.logging_config import get_logger


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="random seed")
    args = parser.parse_args(argv)

    log = get_logger("nexus.datagen")
    log.info("Generating SYNTHETIC purchase orders with seed=%s", args.seed)

    suppliers = generate_suppliers(args.seed)
    products = generate_products(suppliers, args.seed)
    customers = generate_customers(args.seed)
    orders = generate_orders(customers, args.seed)
    order_items = generate_order_items(orders, customers, products, args.seed)

    purchase_orders = generate_purchase_orders(products, suppliers, orders, order_items, args.seed)

    problems = validate_purchase_orders(purchase_orders, products, suppliers)
    if problems:
        for problem in problems:
            log.error("Validation failed: %s", problem)
        return 1

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    path = settings.raw_dir / "purchase_orders.csv"
    purchase_orders.to_csv(path, index=False, date_format="%Y-%m-%d")
    log.info(
        "Wrote %s (%d rows, %d columns)", path, len(purchase_orders), purchase_orders.shape[1]
    )

    print("RESULT: purchase orders generated and validated (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())