from __future__ import annotations


class KTLException(Exception):
    """Base exception for KTL user-facing errors."""


class TrainingError(KTLException):
    """Raised for training-related errors with actionable hints."""


class InferenceError(KTLException):
    """Raised for inference-related errors with actionable hints."""
