"""Structured (key=value or JSON) logging built on the standard library."""

from __future__ import annotations

import json
import logging
import sys

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class StructuredFormatter(logging.Formatter):
    """Render records as ``ts level logger event key=value ...`` or as one JSON object."""

    def __init__(self, as_json: bool = False) -> None:
        super().__init__(datefmt="%Y-%m-%dT%H:%M:%S")
        self.as_json = as_json

    def format(self, record: logging.LogRecord) -> str:
        fields = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        ts = self.formatTime(record, self.datefmt)
        if self.as_json:
            payload = {
                "ts": ts,
                "level": record.levelname,
                "logger": record.name,
                "event": record.getMessage(),
                **fields,
            }
            return json.dumps(payload, default=str)
        kv = " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"{ts} {record.levelname:<7} {record.name} {record.getMessage()}"
        return f"{line} {kv}".rstrip()


def configure_logging(level: str = "INFO", as_json: bool = False) -> None:
    """Install a single structured stderr handler on the ``eegkit`` logger."""
    logger = logging.getLogger("eegkit")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(StructuredFormatter(as_json))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``eegkit`` logger; pass context via ``extra={...}``."""
    return logging.getLogger(name if name.startswith("eegkit") else f"eegkit.{name}")
