import pytest

from nexus.config import PROJECT_ROOT, get_settings


def test_defaults(monkeypatch):
    for name in ("NEXUS_ENV", "NEXUS_LOG_LEVEL", "NEXUS_RANDOM_SEED"):
        monkeypatch.delenv(name, raising=False)
    s = get_settings()
    assert s.random_seed == 42
    assert s.log_level == "INFO"
    assert s.environment == "development"


def test_paths_are_consistent():
    s = get_settings()
    assert s.project_root == PROJECT_ROOT
    assert (s.project_root / "src" / "nexus").is_dir()
    assert s.raw_dir == s.data_dir / "raw"
    assert s.logs_dir == s.project_root / "logs"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("NEXUS_RANDOM_SEED", "7")
    monkeypatch.setenv("NEXUS_LOG_LEVEL", "debug")
    s = get_settings()
    assert s.random_seed == 7
    assert s.log_level == "DEBUG"


def test_invalid_seed_rejected(monkeypatch):
    monkeypatch.setenv("NEXUS_RANDOM_SEED", "abc")
    with pytest.raises(ValueError):
        get_settings()


def test_invalid_log_level_rejected(monkeypatch):
    monkeypatch.setenv("NEXUS_LOG_LEVEL", "LOUD")
    with pytest.raises(ValueError):
        get_settings()