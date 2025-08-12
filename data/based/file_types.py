"""File type constants and utilities.

This module defines supported file types and provides utilities for
file type detection and validation.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class FileType(Enum):
    """Supported file types for data loading."""
    
    CSV = "csv"
    TSV = "tsv"
    TXT = "txt"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    FEATHER = "feather"
    PICKLE = "pickle"
    EXCEL = "xlsx"
    HDF5 = "h5"
    
    @classmethod
    def from_extension(cls, extension: str) -> FileType:
        """Get FileType from file extension.
        
        Args:
            extension: File extension (with or without dot)
            
        Returns:
            Corresponding FileType
            
        Raises:
            ValueError: If extension is not supported
        """
        # Remove leading dot if present
        ext = extension.lstrip('.').lower()
        
        # Handle special cases
        if ext in ('tab', 'tsv'):
            return cls.TSV
        elif ext in ('xls', 'xlsx'):
            return cls.EXCEL
        elif ext in ('pkl', 'pickle'):
            return cls.PICKLE
        elif ext in ('hdf', 'h5', 'hdf5'):
            return cls.HDF5
        
        # Try direct mapping
        try:
            return cls(ext)
        except ValueError:
            supported = [e.value for e in cls]
            raise ValueError(f"Unsupported file extension: {ext}. Supported: {supported}")
    
    @classmethod
    def from_path(cls, path: Path | str) -> FileType:
        """Get FileType from file path.
        
        Args:
            path: File path
            
        Returns:
            Corresponding FileType
        """
        path_obj = Path(path)
        return cls.from_extension(path_obj.suffix)
    
    @property
    def pandas_reader(self) -> str:
        """Get the corresponding pandas reader function name.
        
        Returns:
            Name of pandas reader function
        """
        mapping = {
            self.CSV: "read_csv",
            self.TSV: "read_csv",  # with sep='\t'
            self.TXT: "read_csv",
            self.JSON: "read_json",
            self.JSONL: "read_json",  # with lines=True
            self.PARQUET: "read_parquet",
            self.FEATHER: "read_feather", 
            self.PICKLE: "read_pickle",
            self.EXCEL: "read_excel",
            self.HDF5: "read_hdf",
        }
        return mapping[self]
    
    @property
    def pandas_writer(self) -> str:
        """Get the corresponding pandas writer function name.
        
        Returns:
            Name of pandas writer function
        """
        mapping = {
            self.CSV: "to_csv",
            self.TSV: "to_csv",  # with sep='\t'
            self.TXT: "to_csv",
            self.JSON: "to_json",
            self.JSONL: "to_json",  # with lines=True
            self.PARQUET: "to_parquet",
            self.FEATHER: "to_feather",
            self.PICKLE: "to_pickle", 
            self.EXCEL: "to_excel",
            self.HDF5: "to_hdf",
        }
        return mapping[self]
    
    def get_read_kwargs(self) -> Dict[str, Any]:
        """Get default kwargs for reading this file type.
        
        Returns:
            Dictionary of default keyword arguments
        """
        defaults = {
            self.CSV: {"encoding": "utf-8"},
            self.TSV: {"sep": "\t", "encoding": "utf-8"},
            self.TXT: {"encoding": "utf-8"},
            self.JSON: {"encoding": "utf-8"},
            self.JSONL: {"lines": True, "encoding": "utf-8"},
            self.PARQUET: {},
            self.FEATHER: {},
            self.PICKLE: {},
            self.EXCEL: {},
            self.HDF5: {},
        }
        return defaults.get(self, {})
    
    def get_write_kwargs(self) -> Dict[str, Any]:
        """Get default kwargs for writing this file type.
        
        Returns:
            Dictionary of default keyword arguments
        """
        defaults = {
            self.CSV: {"index": False, "encoding": "utf-8"},
            self.TSV: {"sep": "\t", "index": False, "encoding": "utf-8"},
            self.TXT: {"index": False, "encoding": "utf-8"},
            self.JSON: {"orient": "records", "lines": False},
            self.JSONL: {"orient": "records", "lines": True},
            self.PARQUET: {"index": False"},
            self.FEATHER: {},
            self.PICKLE: {},
            self.EXCEL: {"index": False},
            self.HDF5: {},
        }
        return defaults.get(self, {})


class CompressionType(Enum):
    """Supported compression types."""
    
    NONE = None
    GZIP = "gzip"
    BZIP2 = "bz2"
    XZ = "xz"
    ZIP = "zip"
    
    @classmethod
    def from_extension(cls, extension: str) -> CompressionType:
        """Get CompressionType from file extension.
        
        Args:
            extension: File extension (with or without dot)
            
        Returns:
            Corresponding CompressionType
        """
        ext = extension.lstrip('.').lower()
        
        mapping = {
            "gz": cls.GZIP,
            "gzip": cls.GZIP,
            "bz2": cls.BZIP2,
            "bzip2": cls.BZIP2,
            "xz": cls.XZ,
            "zip": cls.ZIP,
        }
        
        return mapping.get(ext, cls.NONE)


def detect_file_type(path: Path | str) -> tuple[FileType, CompressionType]:
    """Detect file type and compression from path.
    
    Args:
        path: File path
        
    Returns:
        Tuple of (FileType, CompressionType)
    """
    path_obj = Path(path)
    
    # Handle compressed files
    if path_obj.suffix.lower() in {'.gz', '.bz2', '.xz', '.zip'}:
        compression = CompressionType.from_extension(path_obj.suffix)
        # Get the actual file extension (before compression)
        stem_path = Path(path_obj.stem)
        file_type = FileType.from_extension(stem_path.suffix)
    else:
        compression = CompressionType.NONE
        file_type = FileType.from_extension(path_obj.suffix)
    
    return file_type, compression


def get_supported_extensions() -> Set[str]:
    """Get all supported file extensions.
    
    Returns:
        Set of supported file extensions (including dot)
    """
    extensions = set()
    
    # Add primary extensions
    for file_type in FileType:
        extensions.add(f".{file_type.value}")
    
    # Add aliases
    aliases = {
        ".tab": ".tsv",
        ".xls": ".xlsx", 
        ".pkl": ".pickle",
        ".hdf": ".h5",
        ".hdf5": ".h5",
    }
    extensions.update(aliases.keys())
    
    return extensions


def validate_file_path(path: Path | str, 
                      allowed_types: Optional[List[FileType]] = None) -> bool:
    """Validate if file path has supported type.
    
    Args:
        path: File path to validate
        allowed_types: Optional list of allowed file types
        
    Returns:
        True if file type is supported/allowed
    """
    try:
        file_type, _ = detect_file_type(path)
        
        if allowed_types is None:
            return True
        
        return file_type in allowed_types
        
    except ValueError:
        return False
