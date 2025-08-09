from __future__ import annotations


class KTLException(Exception):
    """Base exception for KTL user-facing errors."""


class TrainingError(KTLException):
    """Raised for training-related errors with actionable hints."""
