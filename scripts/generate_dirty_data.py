"""Build the "as delivered" raw layer with injected defects (SYNTHETIC data).

Reads the clean CSVs in data/raw (ground truth), injects known defects into four tables and
writes data/raw/dirty. The --seed argument controls only the injection. The other tables are
copied unchanged so the dirty folder is a complete set.
"""

from __future__ import annotations

import argparse
import shutil

import pandas as pd

from nexus.config import get_settings
from nexus.datagen.defects import DEFECT_RATES, inject_defects, validate_defects
from nexus.logging_config import get_logger

PASSTHROUGH = [
    "suppliers",
    "products",
    "refunds",
    "purchase_orders",
    "inventory_daily",
    "shipments",
    "daily_metrics",
]


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.random_seed, help="injection seed")
    args = parser.parse_args(argv)

    log = get_logger("nexus.datagen")
    raw = settings.raw_dir
    missing = [t for t in [*DEFECT_RATES, *PASSTHROUGH] if not (raw / f"{t}.csv").exists()]
    if missing:
        log.error("Clean CSV files are missing in %s: %s", raw, ", ".join(missing))
        return 1

    log.info("Injecting SYNTHETIC data-quality defects with seed=%s", args.seed)
    clean = {t: pd.read_csv(raw / f"{t}.csv") for t in DEFECT_RATES}
    dirty, manifest = inject_defects(clean, args.seed)

    problems = validate_defects(clean, dirty, manifest)
    if problems:
        for problem in problems:
            log.error("Validation failed: %s", problem)
        return 1

    out = raw / "dirty"
    out.mkdir(parents=True, exist_ok=True)
    for table, frame in dirty.items():
        path = out / f"{table}.csv"
        frame.to_csv(path, index=False)
        log.info("Wrote %s (%d rows, %d columns)", path, len(frame), frame.shape[1])
    manifest_path = out / "defect_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    log.info("Wrote %s (%d entries)", manifest_path, len(manifest))
    for table in PASSTHROUGH:
        shutil.copy2(raw / f"{table}.csv", out / f"{table}.csv")
    log.info("Copied %d unchanged tables", len(PASSTHROUGH))

    print("RESULT: dirty raw layer generated and validated (SYNTHETIC DATA)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())