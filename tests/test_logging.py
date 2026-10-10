import logging

from nexus.config import get_settings
from nexus.logging_config import get_logger


def test_logger_name_and_no_duplicate_handlers():
    first = get_logger()
    count = len(first.handlers)
    second = get_logger("nexus.test")
    assert second.name == "nexus.test"
    assert len(get_logger().handlers) == count
    assert count >= 2


def test_log_message_reaches_file():
    log = get_logger("nexus.test")
    log.warning("pytest-log-check-12345")
    for handler in logging.getLogger("nexus").handlers:
        handler.flush()
    log_file = get_settings().logs_dir / "nexus.log"
    assert log_file.exists()
    assert "pytest-log-check-12345" in log_file.read_text(encoding="utf-8")