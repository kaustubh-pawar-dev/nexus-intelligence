"""Load the as-delivered files from data/raw/dirty into the raw schema as text (SYNTHETIC data)."""

from __future__ import annotations

from nexus.config import get_settings
from nexus.db import apply_schema, get_engine
from nexus.logging_config import get_logger
from nexus.loader import LOAD_ORDER
from nexus.raw_loader import load_raw, reconcile_raw


def main() -> int:
    log = get_logger("nexus.raw_loader")
    dirty = get_settings().raw_dir / "dirty"
    try:
        engine = get_engine()
        apply_schema(engine)
        loaded = load_raw(engine, dirty)
    except Exception as exc:
        log.error("Raw load failed and was rolled back: %s: %s", type(exc).__name__, exc)
        return 1

    for table in LOAD_ORDER:
        log.info("Loaded raw.%s (%d rows)", table, loaded[table])

    problems = reconcile_raw(engine, dirty)
    if problems:
        for problem in problems:
            log.error("Reconciliation failed: %s", problem)
        return 1

    print(f"RESULT: {len(LOAD_ORDER)} raw tables loaded as text and reconciled (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())