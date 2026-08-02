"""Structured JSON logging for request metrics."""

from __future__ import annotations

import json
import logging
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        metrics = getattr(record, "metrics", None)
        if isinstance(metrics, dict):
            payload.update(metrics)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_metrics_logger(name: str = "omnivoice.metrics") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(logging.INFO)
    return logger


def log_metrics(logger: logging.Logger, **metrics: Any) -> None:
    logger.info("request_metrics", extra={"metrics": metrics})
