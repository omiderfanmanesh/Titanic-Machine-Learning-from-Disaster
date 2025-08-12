"""Tests for data loading and validation components."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from data.loader import (
    TitanicDataLoader,
    KaggleDataLoader,
    CachedDataLoader
)
from data.validate import TitanicDataValidator


class TestTitanicDataLoader:
    """Test basic Titanic data loading functionality."""
    
    def test_load_data_success(self, sample_titanic_data, sample_test_data, temp_dir):
        """Test successful data loading."""
        # Create temporary files
        train_file = temp_dir / "train.csv"
        test_file = temp_dir / "test.csv"
        
        sample_titanic_data.to_csv(train_file, index=False)
        sample_test_data.to_csv(test_file, index=False)
        
        loader = TitanicDataLoader(train_file, test_file)
        train_data, test_data = loader.load()
        
        assert len(train_data) == len(sample_titanic_data)
        assert len(test_data) == len(sample_test_data)
        assert "Survived" in train_data.columns
        assert "Survived" not in test_data.columns
        
    def test_load_data_missing_files(self, temp_dir):
        """Test error handling for missing files."""
        train_file = temp_dir / "missing_train.csv"
        test_file = temp_dir / "missing_test.csv"
        
        loader = TitanicDataLoader(train_file, test_file)
        
        with pytest.raises(FileNotFoundError):
            loader.load()
            
    def test_load_data_malformed_csv(self, temp_dir):
        """Test error handling for malformed CSV."""
        train_file = temp_dir / "bad_train.csv"
        
        # Create malformed CSV
        with open(train_file, "w") as f:
            f.write("bad,csv,content\n")
            f.write("missing,quotes and commas\n")
            
        test_file = temp_dir / "test.csv"
        pd.DataFrame({"col1": [1]}).to_csv(test_file, index=False)
        
        loader = TitanicDataLoader(train_file, test_file)
        
        # Should handle parsing errors gracefully
        with pytest.raises(Exception):  # Could be various pandas parsing errors
            loader.load()


class TestKaggleDataLoader:
    """Test Kaggle API data loading functionality."""
    
    @patch('titanic_ml.data.loader.kaggle')
    def test_download_and_load_success(self, mock_kaggle, temp_dir):
        """Test successful Kaggle data download and loading."""
        # Mock successful API call
        mock_api = Mock()
        mock_kaggle.KaggleApi.return_value = mock_api
        mock_api.authenticate.return_value = None
        mock_api.competition_download_files.return_value = None
        
        # Create mock downloaded files
        train_data = pd.DataFrame({
            "PassengerId": [1, 2],
            "Survived": [0, 1],
            "Name": ["Test1", "Test2"]
        })
        test_data = pd.DataFrame({
            "PassengerId": [3, 4],
            "Name": ["Test3", "Test4"]
        })
        
        train_file = temp_dir / "train.csv"
        test_file = temp_dir / "test.csv"
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        
        loader = KaggleDataLoader(competition="titanic", download_path=temp_dir)
        
        with patch.object(loader, '_extract_files') as mock_extract:
            mock_extract.return_value = None
            train_result, test_result = loader.load()
            
        assert len(train_result) == 2
        assert len(test_result) == 2
        mock_api.authenticate.assert_called_once()
        mock_api.competition_download_files.assert_called_once()
        
    @patch('titanic_ml.data.loader.kaggle')
    def test_kaggle_authentication_error(self, mock_kaggle, temp_dir):
        """Test Kaggle authentication error handling."""
        mock_api = Mock()
        mock_kaggle.KaggleApi.return_value = mock_api
        mock_api.authenticate.side_effect = Exception("Authentication failed")
        
        loader = KaggleDataLoader(competition="titanic", download_path=temp_dir)
        
        with pytest.raises(Exception, match="Authentication failed"):
            loader.load()


class TestCachedDataLoader:
    """Test cached data loading functionality."""
    
    def test_cache_miss_then_hit(self, sample_titanic_data, sample_test_data, temp_dir):
        """Test cache miss followed by cache hit."""
        # Create base loader
        train_file = temp_dir / "train.csv"
        test_file = temp_dir / "test.csv"
        sample_titanic_data.to_csv(train_file, index=False)
        sample_test_data.to_csv(test_file, index=False)
        
        base_loader = TitanicDataLoader(train_file, test_file)
        cache_file = temp_dir / "cache.pkl"
        
        cached_loader = CachedDataLoader(base_loader, cache_file)
        
        # First load (cache miss)
        train1, test1 = cached_loader.load()
        assert cache_file.exists()
        
        # Modify original files to ensure we're loading from cache
        pd.DataFrame({"different": [1, 2]}).to_csv(train_file, index=False)
        
        # Second load (cache hit)
        train2, test2 = cached_loader.load()
        
        # Should be identical to first load (from cache)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)
        
    def test_cache_invalidation(self, sample_titanic_data, sample_test_data, temp_dir):
        """Test cache invalidation with force_refresh."""
        train_file = temp_dir / "train.csv"
        test_file = temp_dir / "test.csv"
        sample_titanic_data.to_csv(train_file, index=False)
        sample_test_data.to_csv(test_file, index=False)
        
        base_loader = TitanicDataLoader(train_file, test_file)
        cache_file = temp_dir / "cache.pkl"
        
        cached_loader = CachedDataLoader(base_loader, cache_file)
        
        # First load
        train1, test1 = cached_loader.load()
        
        # Modify data
        modified_data = sample_titanic_data.copy()
        modified_data["NewColumn"] = 999
        modified_data.to_csv(train_file, index=False)
        
        # Load with force refresh
        train2, test2 = cached_loader.load(force_refresh=True)
        
        # Should have new column
        assert "NewColumn" in train2.columns
        assert "NewColumn" not in train1.columns


@pytest.mark.skipif(True, reason="Requires Pandera dependency")
class TestTitanicDataValidator:
    """Test data validation functionality."""
    
    def test_valid_data_passes(self, sample_titanic_data, sample_test_data):
        """Test that valid data passes validation."""
        validator = TitanicDataValidator()
        
        # Should not raise any exceptions
        validator.validate_train(sample_titanic_data)
        validator.validate_test(sample_test_data)
        validator.validate_consistency(sample_titanic_data, sample_test_data)
        
    def test_invalid_survived_values(self, sample_titanic_data):
        """Test validation fails for invalid Survived values."""
        validator = TitanicDataValidator()
        
        # Add invalid Survived value
        invalid_data = sample_titanic_data.copy()
        invalid_data.loc[0, "Survived"] = 2  # Should be 0 or 1
        
        with pytest.raises(Exception):  # Pandera validation error
            validator.validate_train(invalid_data)
            
    def test_missing_required_columns(self, sample_titanic_data):
        """Test validation fails for missing required columns."""
        validator = TitanicDataValidator()
        
        # Remove required column
        invalid_data = sample_titanic_data.drop(columns=["PassengerId"])
        
        with pytest.raises(Exception):
            validator.validate_train(invalid_data)
            
    def test_inconsistent_passenger_ids(self, sample_titanic_data, sample_test_data):
        """Test detection of overlapping PassengerIds."""
        validator = TitanicDataValidator()
        
        # Make PassengerIds overlap
        test_with_overlap = sample_test_data.copy()
        test_with_overlap.loc[0, "PassengerId"] = sample_titanic_data.loc[0, "PassengerId"]
        
        with pytest.raises(ValueError, match="overlapping PassengerId"):
            validator.validate_consistency(sample_titanic_data, test_with_overlap)
