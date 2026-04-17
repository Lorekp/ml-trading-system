"""Structured JSON logger for the paper-trading engine.

One log record per line in ``deribit_paper/logs/YYYY-MM-DD.jsonl``.
Fields include timestamp, level, logger name, message, and any
``extra`` dict passed to ``logger.log()``.

Backward-compatible with the existing ``logging`` module — just call
:func:`setup_json_logging` once at engine startup and all loggers
will produce JSON output.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge any ``extra`` dict attributes the caller attached
        for key, value in record.__dict__.items():
            if key in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


def setup_json_logging(
    *,
    logs_dir: str | Path = "deribit_paper/logs",
    level: str = "INFO",
    also_console: bool = True,
) -> None:
    """Configure the root logger to emit JSON to a rotating daily file.

    Idempotent: calling it twice does not double-install handlers.
    """
    root = logging.getLogger()
    # Only install if not already configured
    if any(isinstance(h, logging.handlers.TimedRotatingFileHandler) for h in root.handlers):
        return

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(logs_dir) / "deribit_paper.jsonl"

    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=30, utc=True, encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    if also_console:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
        ))
        console.setLevel(level)
        root.addHandler(console)

    root.setLevel(level)
    logging.getLogger(__name__).info(
        "JSON logging configured at %s (level %s)", log_path, level,
    )
