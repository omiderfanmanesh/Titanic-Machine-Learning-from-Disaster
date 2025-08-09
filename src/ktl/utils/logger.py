from __future__ import annotations

import logging
from logging import Logger


class LoggerFactory:
    """Structured logger factory for KTL.

    Ensures consistent formatting without relying on global state.
    """

    _configured = False

    @classmethod
    def _configure(cls) -> None:
        if cls._configured:
            return
        handler = logging.StreamHandler()
        fmt = (
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(logging.Formatter(fmt))
        root = logging.getLogger("ktl")
        root.setLevel(logging.INFO)
        if not root.handlers:
            root.addHandler(handler)
        cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> Logger:
        cls._configure()
        return logging.getLogger(name)
