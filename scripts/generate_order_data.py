"""Generate synthetic orders and order lines into data/raw.

Suppliers, products and customers are regenerated in memory from the same seed,
so the output always matches the CSVs written by the earlier generator scripts.
"""

from __future__ import annotations

import argparse

from nexus.config import get_settings
from nexus.datagen.customers import generate_customers
from nexus.datagen.orders import generate_order_items, generate_orders, validate_orders
from nexus.datagen.reference import generate_products, generate_suppliers
from nexus.logging_config import get_logger


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="random seed")
    args = parser.parse_args(argv)

    log = get_logger("nexus.datagen")
    log.info("Generating SYNTHETIC orders with seed=%s", args.seed)

    suppliers = generate_suppliers(args.seed)
    products = generate_products(suppliers, args.seed)
    customers = generate_customers(args.seed)

    orders = generate_orders(customers, args.seed)
    order_items = generate_order_items(orders, customers, products, args.seed)

    problems = validate_orders(orders, order_items, customers, products)
    if problems:
        for problem in problems:
            log.error("Validation failed: %s", problem)
        return 1

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("orders", orders), ("order_items", order_items)):
        path = settings.raw_dir / f"{name}.csv"
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        log.info("Wrote %s (%d rows, %d columns)", path, len(frame), frame.shape[1])

    print("RESULT: orders and order_items generated and validated (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())