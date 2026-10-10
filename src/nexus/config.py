"""Central configuration for NEXUS Intelligence.

Settings are read from environment variables, optionally loaded from a local
``.env`` file at the project root. No secrets are stored in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# src/nexus/config.py -> parents[0]=nexus, [1]=src, [2]=project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    project_root: Path
    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    logs_dir: Path
    reports_dir: Path
    sql_dir: Path
    random_seed: int
    log_level: str
    environment: str


def get_settings() -> Settings:
    """Build Settings from the current environment."""
    root = PROJECT_ROOT
    data_dir = root / "data"

    log_level = os.getenv("NEXUS_LOG_LEVEL", "INFO").strip().upper()
    if log_level not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"NEXUS_LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {log_level!r}"
        )

    return Settings(
        project_root=root,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        interim_dir=data_dir / "interim",
        processed_dir=data_dir / "processed",
        logs_dir=root / "logs",
        reports_dir=root / "reports",
        sql_dir=root / "sql",
        random_seed=_env_int("NEXUS_RANDOM_SEED", 42),
        log_level=log_level,
        environment=os.getenv("NEXUS_ENV", "development").strip().lower(),
    )