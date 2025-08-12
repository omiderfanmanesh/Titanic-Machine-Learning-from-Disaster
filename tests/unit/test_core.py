"""Tests for core interfaces and utilities."""

import pytest
import pandas as pd
import numpy as np

from core.utils import SeedManager, PathManager, ConfigManager, CacheKeyGenerator


class TestSeedManager:
    """Test seed management functionality."""
    
    def test_set_seed(self):
        """Test setting random seed."""
        SeedManager.set_seed(42)
        
        # Test numpy seed
        np.random.seed(42)
        expected = np.random.random(5)
        
        SeedManager.set_seed(42)
        actual = np.random.random(5)
        
        np.testing.assert_array_equal(actual, expected)
        
    def test_get_seed(self):
        """Test getting current seed."""
        SeedManager.set_seed(123)
        assert SeedManager.get_seed() == 123


class TestPathManager:
    """Test path management."""
    
    def test_path_manager_initialization(self, temp_dir):
        """Test PathManager initialization."""
        pm = PathManager(temp_dir)
        
        assert pm.project_root == temp_dir
        assert pm.data_dir == temp_dir / "data"
        assert pm.config_dir == temp_dir / "configs"
        assert pm.artifacts_dir == temp_dir / "artifacts"
        
    def test_create_run_directory(self, temp_dir):
        """Test run directory creation."""
        pm = PathManager(temp_dir)
        
        run_dir = pm.create_run_directory("test_run")
        
        assert run_dir.exists()
        assert run_dir.name == "test_run"
        
    def test_ensure_directories(self, temp_dir):
        """Test directory creation."""
        pm = PathManager(temp_dir)
        pm.ensure_directories()
        
        assert (temp_dir / "data" / "raw").exists()
        assert (temp_dir / "data" / "interim").exists()
        assert (temp_dir / "data" / "processed").exists()
        assert (temp_dir / "configs").exists()
        assert (temp_dir / "artifacts").exists()


class TestConfigManager:
    """Test configuration management."""
    
    def test_config_manager(self, temp_dir):
        """Test basic config operations."""
        import yaml
        
        config_dir = temp_dir / "configs"
        config_dir.mkdir()
        
        # Create a test config
        test_config = {"model": "test", "params": {"n_estimators": 100}}
        
        config_path = config_dir / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(test_config, f)
        
        # Test loading
        cm = ConfigManager(config_dir)
        loaded_config = cm.load_config("test")
        
        assert loaded_config == test_config
        
    def test_config_hash(self, temp_dir):
        """Test configuration hashing."""
        cm = ConfigManager(temp_dir)
        
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 2, "a": 1}  # Same content, different order
        config3 = {"a": 1, "b": 3}  # Different content
        
        hash1 = cm.get_config_hash(config1)
        hash2 = cm.get_config_hash(config2)
        hash3 = cm.get_config_hash(config3)
        
        assert hash1 == hash2  # Order shouldn't matter
        assert hash1 != hash3  # Different content should give different hash


class TestCacheKeyGenerator:
    """Test cache key generation."""
    
    def test_generate_key(self):
        """Test cache key generation."""
        key1 = CacheKeyGenerator.generate_key("arg1", "arg2", param1="value1", param2="value2")
        key2 = CacheKeyGenerator.generate_key("arg1", "arg2", param2="value2", param1="value1")
        key3 = CacheKeyGenerator.generate_key("arg1", "arg3", param1="value1", param2="value2")
        
        # Same arguments should produce same key
        assert key1 == key2
        
        # Different arguments should produce different key
        assert key1 != key3
        
        # Keys should be reasonably short
        assert len(key1) == 16
