"""Generate the synthetic customers table into data/raw."""

from __future__ import annotations

import argparse

from nexus.config import get_settings
from nexus.datagen.customers import generate_customers, validate_customers
from nexus.logging_config import get_logger


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="random seed")
    args = parser.parse_args(argv)

    log = get_logger("nexus.datagen")
    log.info("Generating SYNTHETIC customers with seed=%s", args.seed)

    customers = generate_customers(args.seed)

    problems = validate_customers(customers)
    if problems:
        for problem in problems:
            log.error("Validation failed: %s", problem)
        return 1

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    path = settings.raw_dir / "customers.csv"
    customers.to_csv(path, index=False, date_format="%Y-%m-%d")
    log.info("Wrote %s (%d rows, %d columns)", path, len(customers), customers.shape[1])

    print("RESULT: customers generated and validated (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())