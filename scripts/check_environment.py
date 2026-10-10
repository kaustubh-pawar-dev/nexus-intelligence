"""Environment and project-layout check for NEXUS Intelligence."""

from __future__ import annotations

import sys
from importlib import metadata

from nexus import __version__
from nexus.config import get_settings
from nexus.logging_config import get_logger

MIN_PYTHON = (3, 10)
REQUIRED_DIRS = ["src/nexus", "app", "sql", "scripts", "tests", "docs", "reports", "data/raw", "logs"]
INSTALLED_PACKAGES = ["python-dotenv", "pytest"]


def main() -> int:
    log = get_logger("nexus.check")
    settings = get_settings()
    problems: list[str] = []

    print(f"NEXUS Intelligence v{__version__}")
    print(f"Python        : {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] < MIN_PYTHON:
        problems.append(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required")

    print(f"Project root  : {settings.project_root}")
    print(f"Environment   : {settings.environment}")
    print(f"Log level     : {settings.log_level}")
    print(f"Random seed   : {settings.random_seed}")

    for rel in REQUIRED_DIRS:
        path = settings.project_root / rel
        status = "ok" if path.is_dir() else "MISSING"
        print(f"  dir {rel:<12}: {status}")
        if status == "MISSING":
            problems.append(f"missing directory {rel}")

    for pkg in INSTALLED_PACKAGES:
        try:
            print(f"  pkg {pkg:<14}: {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            print(f"  pkg {pkg:<14}: NOT INSTALLED")
            problems.append(f"package {pkg} not installed")

    if problems:
        log.error("Environment check FAILED: %s", "; ".join(problems))
        return 1

    log.info("Environment check passed")
    print("RESULT: environment check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())