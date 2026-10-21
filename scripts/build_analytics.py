"""Build the analytics schema from staging and raw, then check it (SYNTHETIC data)."""

from __future__ import annotations

from nexus.db import get_engine
from nexus.loader import LOAD_ORDER
from nexus.logging_config import get_logger
from nexus.promotion import build_analytics, check_analytics


def main() -> int:
    log = get_logger("nexus.promotion")
    try:
        engine = get_engine()
        counts = build_analytics(engine)
    except Exception as exc:
        log.error("Analytics build failed and was rolled back: %s: %s", type(exc).__name__, exc)
        return 1

    for table in LOAD_ORDER:
        log.info("Built analytics.%s (%d rows)", table, counts[table])

    problems = check_analytics(engine)
    if problems:
        for problem in problems:
            log.error("Analytics check failed: %s", problem)
        return 1

    print(f"RESULT: {len(LOAD_ORDER)} analytics tables built, constraints enforced and metrics reconciled (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())