"""Core utilities for logging, seeding, paths, and configuration management."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from pydantic import BaseModel, Field, root_validator


class LoggerFactory:
    """Factory for creating structured loggers with consistent formatting."""
    
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, name: str, level: int = logging.INFO) -> logging.Logger:
        """Get or create a logger with standard formatting."""
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            
            # Don't add handlers if they already exist
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                
            cls._loggers[name] = logger
            
        return cls._loggers[name]


class SeedManager:
    """Manages global random seeds for reproducibility."""
    
    _current_seed: Optional[int] = None
    
    @classmethod
    def set_seed(cls, seed: int) -> None:
        """Set global random seed for all libraries."""
        cls._current_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Set seeds for ML libraries if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
            
        try:
            import sklearn
            from sklearn.utils import check_random_state
        except ImportError:
            pass
    
    @classmethod
    def get_seed(cls) -> Optional[int]:
        """Get current seed."""
        return cls._current_seed


class PathManager:
    """Manages project paths and directory creation."""
    
    def __init__(self, project_root: Optional[Path] = None):
        # Resolve project root: use provided path or infer from this file location
        if project_root is None:
            # utils.py -> core -> src -> project root
            inferred = Path(__file__).resolve().parents[2]
            self.project_root = inferred
        else:
            self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.config_dir = self.project_root / "configs"
        self.artifacts_dir = self.project_root / "artifacts"
        
    def create_run_directory(self, timestamp: Optional[str] = None) -> Path:
        """Create timestamped run directory for artifacts."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        run_dir = self.artifacts_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for dir_path in [
            self.data_dir / "raw",
            self.data_dir / "interim", 
            self.data_dir / "processed",
            self.config_dir,
            self.artifacts_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}
    
    def load_config(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if use_cache and config_name in self._cache:
            return self._cache[config_name]

        config_path = Path(config_name)
        # Use config_name directly if it's absolute or starts with "configs/"
        if not config_path.is_absolute() and not str(config_path).startswith("configs/"):
            config_path = self.config_dir / config_path
        if not config_path.suffix == ".yaml":
            config_path = config_path.with_suffix(".yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r") as f:
            config = yaml.safe_load(f) or {}

        if use_cache:
            self._cache[config_name] = config

        return config
    
    def get_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash of configuration for reproducibility."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def validate_config(self, config: Dict[str, Any], schema: BaseModel) -> BaseModel:
        """Validate configuration against Pydantic schema."""
        return schema(**config)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[float] = None
        
    def __enter__(self):
        self.start_time = datetime.now().timestamp()
        self.logger.info(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = datetime.now().timestamp() - self.start_time
            self.logger.info(f"Completed {self.operation} in {duration:.2f}s")


class CacheKeyGenerator:
    """Generates deterministic cache keys for pipeline steps."""
    
    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Convert all arguments to strings and sort for consistency
        all_args = list(map(str, args))
        all_args.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        key_string = "|".join(all_args)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]


# Configuration schemas using Pydantic
class ExperimentConfig(BaseModel):
    """Schema for experiment configuration."""
    model_config = {'protected_namespaces': ()}
    name: str = Field(..., description="Experiment name")
    seed: int = Field(42, description="Random seed for reproducibility")
    debug_mode: bool = Field(False, description="Whether to run in debug mode")
    debug_n_rows: Optional[int] = Field(None, description="Number of rows for debug mode")
    
    model_name: str = Field(..., description="Model name to use")
    model_params: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    cv_strategy: str = Field("stratified", description="CV strategy")
    cv_shuffle: bool = Field(True, description="Whether to shuffle CV splits")
    cv_random_state: int = Field(42, description="Random state for CV splits")
    
    early_stopping_rounds: Optional[int] = Field(None, description="Early stopping rounds")
    logging_level: str = Field("INFO", description="Logging level")


class InferenceConfig(BaseModel):
    """Schema for inference configuration."""
    model_config = {'protected_namespaces': ()}
    model_paths: List[str] = Field(..., description="Paths to trained models")
    ensemble_method: str = Field("average", description="Ensemble method")
    ensemble_weights: Optional[List[float]] = Field(None, description="Ensemble weights")
    
    use_tta: bool = Field(False, description="Whether to use test-time augmentation")
    tta_rounds: int = Field(5, description="Number of TTA rounds")
    
    output_path: str = Field(..., description="Output path for predictions")
    submission_path: str = Field(..., description="Output path for submission file")






class DataConfig(BaseModel):
    # ---- Core ----
    train_path: str
    test_path: str
    target_column: str
    id_column: str
    task_type: str

    required_columns: List[str]
    numeric_columns: List[str]
    categorical_columns: List[str]

    # ---- Global switches ----
    handle_missing: bool
    scale_features: bool
    encode_categorical: bool
    feature_importance: bool = False  # Add feature importance switch

    # ---- Other params used by transforms ----
    log_transform_fare: bool
    age_bins: int
    rare_title_threshold: Optional[int]
    title_map_override: Optional[Dict[str, str]] = None
    skip_encoding_columns: List[str]

    # ---- Encoding ----
    encoding: Dict[str, Any]

    # ---- Imputation config ----
    imputation: Dict[str, Any]

    # ---- Feature engineering stages ----
    feature_engineering: Dict[str, List[str]]

    # ---- Feature toggles (per-transform) ----
    # Use transform class names as keys, boolean to enable/disable.
    # Example:
    #   feature_toggles:
    #     TitleTransform: true
    #     TicketParseTransform: false
    feature_toggles: Dict[str, bool] = Field(default_factory=dict)

    # ---- Output controls ----
    # Include original raw columns (preprocessed sources) in final dataset
    add_original_columns: bool = Field(False, description="Keep original raw columns alongside engineered ones")

    # ---------- Convenience helpers ----------
    def pre_impute_transforms(self) -> List[str]:
        """List of transform class names to run before imputation."""
        return list(self.feature_engineering.get("pre_impute", []))

    def post_impute_transforms(self) -> List[str]:
        """List of transform class names to run after imputation."""
        return list(self.feature_engineering.get("post_impute", []))

    def to_dict(self, include_none: bool = False) -> dict:
        """Return this config as a plain dictionary."""
        return self.dict(exclude_none=not include_none, by_alias=True)
