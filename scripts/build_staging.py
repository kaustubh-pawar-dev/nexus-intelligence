"""Build the staging layer from the raw layer and check it (SYNTHETIC data)."""

from __future__ import annotations

from nexus.cleaning import TABLES, build_staging, check_staging
from nexus.db import get_engine
from nexus.logging_config import get_logger


def main() -> int:
    log = get_logger("nexus.cleaning")
    try:
        engine = get_engine()
        counts, fmt = build_staging(engine)
    except Exception as exc:
        log.error("Staging build failed and was rolled back: %s: %s", type(exc).__name__, exc)
        return 1

    log.info("Slash date format inferred from the data: %s", fmt)
    for table in TABLES:
        log.info("Staged staging.%s (%d rows)", table, counts[table])

    problems = check_staging(engine)
    if problems:
        for problem in problems:
            log.error("Staging check failed: %s", problem)
        return 1

    print(f"RESULT: {len(TABLES)} staging tables built, every raw row accounted for (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())