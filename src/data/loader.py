"""Data loading components with multiple sources and caching support."""

from __future__ import annotations


from typing import Optional, Union, Tuple
from pathlib import Path
import pandas as pd

from core.utils import LoggerFactory
# from .interfaces import IDataLoader  # assuming you have this; keep your original import

import pickle
from pathlib import Path
from typing import Optional, Tuple, Union


from core.interfaces import IDataLoader
from core.utils import LoggerFactory


class TitanicDataLoader(IDataLoader):
    """Basic Titanic data loader for CSV files."""
    
    def __init__(self, train_file: Optional[Union[str, Path]] = None, 
                 test_file: Optional[Union[str, Path]] = None):
        self.train_file = Path(train_file) if train_file else None
        self.test_file = Path(test_file) if test_file else None
        self.logger = LoggerFactory.get_logger(__name__)
    
    def load(self, path: Optional[Union[str, Path]] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load data from CSV files."""
        if path is not None:
            # Load single file
            return self._load_single_file(Path(path))
        
        if self.train_file is None:
            raise ValueError("No training file specified")
        
        # Load train data
        train_df = self._load_single_file(self.train_file)
        self.logger.info(f"Loaded training data: {train_df.shape}")
        
        # Load test data if available
        if self.test_file is not None:
            test_df = self._load_single_file(self.test_file)
            self.logger.info(f"Loaded test data: {test_df.shape}")
            return train_df, test_df
        
        return train_df
    
    def _load_single_file(self, file_path: Path) -> pd.DataFrame:
        """Load single CSV file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Basic schema validation."""
        # Check for PassengerId column
        if "PassengerId" not in df.columns:
            return False
        
        # Check for basic expected columns
        expected_cols = {"Pclass", "Name", "Sex"}
        if not expected_cols.issubset(set(df.columns)):
            return False
        
        return True


class KaggleDataLoader(IDataLoader):
    """Data loader that downloads from Kaggle competitions."""

    def __init__(self, competition: str, download_path: Union[str, Path] = "data/raw", quiet: bool = True):
        self.competition = competition
        self.download_path = Path(download_path)
        self.quiet = quiet
        self.logger = LoggerFactory.get_logger(__name__)

    # --- Public API ---

    def download(self, dest: Optional[Union[str, Path]] = None) -> Path:
        """
        Download (and extract) competition files into dest (or self.download_path).
        Returns the destination directory.
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as e:
            raise ImportError("Kaggle API not available. Install with: pip install kaggle") from e

        dest_path = Path(dest) if dest else self.download_path
        dest_path.mkdir(parents=True, exist_ok=True)

        api = KaggleApi()
        api.authenticate()

        self.logger.info(f"Downloading '{self.competition}' competition data to {dest_path} ...")
        api.competition_download_files(
            self.competition,
            path=str(dest_path),
            quiet=self.quiet is True  # Kaggle API expects bool
        )

        self._extract_files(dest_path)
        self.logger.info(f"Download completed: {dest_path}")
        return dest_path

    # Backward compatibility for existing CLI usage
    def download_competition_data(self, dest: Optional[Union[str, Path]] = None) -> Path:
        """Alias to `download()` for backward-compat with CLI."""
        return self.download(dest)

    def load(self, path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train/test CSVs from path (or self.download_path).
        Does NOT download; call `.download()` first if needed.
        """
        base = Path(path) if path else self.download_path

        train_file = base / "train.csv"
        test_file  = base / "test.csv"

        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                f"Expected train.csv/test.csv in {base}. "
                f"Run download() first or point to the correct directory."
            )

        train_df = pd.read_csv(train_file)
        test_df  = pd.read_csv(test_file)
        self.logger.info(f"Loaded Kaggle data - Train: {train_df.shape}, Test: {test_df.shape}")
        return train_df, test_df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate Kaggle competition data schema via the Titanic loader."""
        loader = TitanicDataLoader()
        return loader.validate_schema(df)

    # --- Internals ---

    def _extract_files(self, base: Path) -> None:
        """Extract any zip archives downloaded by the Kaggle API into base."""
        import zipfile

        zip_files = list(base.glob("*.zip"))
        for zf in zip_files:
            with zipfile.ZipFile(zf, "r") as z:
                z.extractall(base)
                self.logger.info(f"Extracted {zf.name}")
            try:
                zf.unlink()
            except Exception:
                # Non-fatal; leave the zip if we can't remove it
                pass

class CachedDataLoader(IDataLoader):
    """Data loader with caching support."""
    
    def __init__(self, base_loader: IDataLoader, cache_file: Union[str, Path]):
        self.base_loader = base_loader
        self.cache_file = Path(cache_file)
        self.logger = LoggerFactory.get_logger(__name__)
    
    def load(self, path: Optional[Union[str, Path]] = None, force_refresh: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load data with caching."""
        if not force_refresh and self.cache_file.exists():
            self.logger.info(f"Loading cached data from {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}. Loading fresh data.")
        
        # Load fresh data
        self.logger.info("Loading fresh data...")
        data = self.base_loader.load(path)
        
        # Cache the data
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Cached data to {self.cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data: {e}")
        
        return data
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate schema using base loader."""
        return self.base_loader.validate_schema(df)


class MultiSourceDataLoader(IDataLoader):
    """Data loader that can handle multiple data sources."""
    
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)
    
    def load_from_csv(self, train_path: Union[str, Path], 
                     test_path: Optional[Union[str, Path]] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load from CSV files."""
        loader = TitanicDataLoader(train_path, test_path)
        return loader.load()
    
    def load_from_kaggle(self, competition: str, 
                        download_path: Union[str, Path] = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load from Kaggle competition."""
        loader = KaggleDataLoader(competition, download_path)
        return loader.load()
    
    def load_from_database(self, connection_string: str, 
                          train_query: str, test_query: Optional[str] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Load from database (placeholder for future implementation)."""
        raise NotImplementedError("Database loading not yet implemented")
    
    def load(self, path: Optional[Union[str, Path]] = None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Generic load method."""
        if path is None:
            raise ValueError("Path must be specified for generic load")
        
        path = Path(path)
        if path.suffix == '.csv':
            return self.load_from_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Basic schema validation."""
        loader = TitanicDataLoader()
        return loader.validate_schema(df)
