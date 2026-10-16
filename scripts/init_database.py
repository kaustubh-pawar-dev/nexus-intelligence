"""Create (or recreate with --reset) the core schema in PostgreSQL."""

from __future__ import annotations

import argparse

from nexus.db import apply_schema, get_engine, list_core_tables
from nexus.logging_config import get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="drop the core schema first (DESTROYS any loaded data)",
    )
    args = parser.parse_args(argv)

    log = get_logger("nexus.db")
    try:
        engine = get_engine()
        if args.reset:
            log.warning("Resetting: the core schema and its data will be dropped first")
        count = apply_schema(engine, reset=args.reset)
        tables = list_core_tables(engine)
    except Exception as exc:
        log.error("Database initialisation failed: %s: %s", type(exc).__name__, exc)
        return 1

    log.info("Applied %d statements; core has %d tables", count, len(tables))
    for name in tables:
        print(f"  core.{name}")
    print(f"RESULT: core schema applied ({count} statements, {len(tables)} tables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())