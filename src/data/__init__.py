"""Data loading and validation components."""

from .loader import TitanicDataLoader, KaggleDataLoader, CachedDataLoader, MultiSourceDataLoader

try:
    from .validate import TitanicDataValidator, DataLeakageDetector
except ImportError:
    # Handle case where validation dependencies are missing
    TitanicDataValidator = None
    DataLeakageDetector = None

__all__ = [
    "TitanicDataLoader",
    "KaggleDataLoader", 
    "CachedDataLoader",
    "MultiSourceDataLoader",
    "TitanicDataValidator",
    "DataLeakageDetector"
]
