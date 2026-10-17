"""Load the clean CSVs from data/raw into the core schema and reconcile (SYNTHETIC data)."""

from __future__ import annotations

from nexus.config import get_settings
from nexus.db import apply_schema, get_engine
from nexus.loader import LOAD_ORDER, load_core, reconcile
from nexus.logging_config import get_logger


def main() -> int:
    log = get_logger("nexus.loader")
    raw = get_settings().raw_dir
    try:
        engine = get_engine()
        apply_schema(engine)
        loaded = load_core(engine, raw)
    except Exception as exc:
        log.error("Load failed and was rolled back: %s: %s", type(exc).__name__, exc)
        return 1

    for table in LOAD_ORDER:
        log.info("Loaded core.%s (%d rows)", table, loaded[table])

    problems = reconcile(engine, raw)
    if problems:
        for problem in problems:
            log.error("Reconciliation failed: %s", problem)
        return 1

    print(f"RESULT: {len(LOAD_ORDER)} tables loaded and reconciled with the CSV files (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())