"""Generate synthetic reference tables (suppliers, products) into data/raw."""

from __future__ import annotations

import argparse

from nexus.config import get_settings
from nexus.datagen.reference import (
    generate_products,
    generate_suppliers,
    validate_reference_tables,
)
from nexus.logging_config import get_logger


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="random seed")
    args = parser.parse_args(argv)

    log = get_logger("nexus.datagen")
    log.info("Generating SYNTHETIC reference data with seed=%s", args.seed)

    suppliers = generate_suppliers(args.seed)
    products = generate_products(suppliers, args.seed)

    problems = validate_reference_tables(suppliers, products)
    if problems:
        for problem in problems:
            log.error("Validation failed: %s", problem)
        return 1

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("suppliers", suppliers), ("products", products)):
        path = settings.raw_dir / f"{name}.csv"
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        log.info("Wrote %s (%d rows, %d columns)", path, len(frame), frame.shape[1])

    print("RESULT: reference data generated and validated (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())