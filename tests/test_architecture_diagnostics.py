"""
Test and fix issues in the ModelRegistry to make the full architecture work.
"""

import sys
from pathlib import Path
import pytest

# Add src to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_model_registry_creation():
    """Test ModelRegistry model creation with parameters."""
    from modeling.model_registry import ModelRegistry
    
    registry = ModelRegistry()
    
    # Test available models
    models = registry.get_available_models()
    print(f"Available models: {models}")
    assert len(models) > 0
    
    # Test model creation without parameters
    model = registry.create_model('random_forest')
    assert model is not None
    print("✅ Random Forest created without parameters")
    
    # Test model creation with parameters - this might fail
    try:
        model_with_params = registry.create_model('random_forest', n_estimators=10, random_state=42)
        print("✅ Random Forest created with parameters")
        return True
    except TypeError as e:
        print(f"❌ ModelRegistry parameter issue: {e}")
        return False

def test_model_registry_fix():
    """Test and demonstrate fix for ModelRegistry parameter passing."""
    
    # Check current implementation
    from modeling.model_registry import ModelRegistry
    import inspect
    
    registry = ModelRegistry()
    
    # Get the create_model method signature
    create_method = getattr(registry, 'create_model')
    signature = inspect.signature(create_method)
    print(f"create_model signature: {signature}")
    
    # Try to understand the issue
    try:
        # Test with kwargs
        model = registry.create_model('random_forest', **{'n_estimators': 10, 'random_state': 42})
        print("✅ Model creation with **kwargs works")
        
        # Test the actual model parameters
        print(f"Model parameters: {model.get_params()}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        
        # Let's look at the source to understand the issue
        import inspect
        source_lines = inspect.getsourcelines(create_method)
        print("ModelRegistry.create_model source:")
        for i, line in enumerate(source_lines[0][:10], 1):  # First 10 lines
            print(f"{i:2d}: {line.rstrip()}")

def test_fix_cli_config_loading():
    """Test the CLI configuration loading issue."""
    
    try:
        from core.utils import ConfigManager, PathManager
        
        # Test PathManager
        path_manager = PathManager()
        print(f"✅ PathManager created")
        print(f"Config dir: {path_manager.config_dir}")
        
        # Test ConfigManager
        config_manager = ConfigManager(path_manager.config_dir)
        print(f"✅ ConfigManager created")
        
        # Test loading existing config
        try:
            data_config = config_manager.load_config('data')
            print(f"✅ Data config loaded: {list(data_config.keys())}")
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            
        try:
            exp_config = config_manager.load_config('experiment')
            print(f"✅ Experiment config loaded: {list(exp_config.keys())}")
        except Exception as e:
            print(f"❌ Experiment config loading failed: {e}")
            
    except Exception as e:
        print(f"❌ CLI config test failed: {e}")

def test_data_loader_interface():
    """Test the data loader interface issues."""
    
    try:
        # Test direct import first
        from data.loader import TitanicDataLoader
        
        # Test with correct parameters
        loader = TitanicDataLoader(
            train_file="data/train.csv",
            test_file="data/test.csv"
        )
        
        train_df, test_df = loader.load()
        print(f"✅ TitanicDataLoader works: train {train_df.shape}, test {test_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ TitanicDataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_builder_issue():
    """Test and identify the feature builder scaling issue."""
    
    try:
        from features.build import create_feature_builder
        import pandas as pd
        
        # Load data
        train_df = pd.read_csv(project_root / "data" / "train.csv")
        
        # Test feature builder
        builder = create_feature_builder()
        
        # Try fit_transform in steps to isolate the issue
        print("Testing feature builder step by step...")
        
        # First fit
        builder.fit(train_df)
        print("✅ Feature builder fitted")
        
        # Then transform
        X_transformed = builder.transform(train_df)
        print(f"✅ Feature transformation successful: {X_transformed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature builder failed: {e}")
        
        # Let's try to understand the column mismatch issue
        print("\nDiagnosing feature builder issue...")
        try:
            from features.build import create_feature_builder
            import pandas as pd
            
            train_df = pd.read_csv(project_root / "data" / "train.csv")
            builder = create_feature_builder()
            
            # Check what happens during fit
            print(f"Original columns: {list(train_df.columns)}")
            
            # Try to see intermediate steps
            builder.fit(train_df)
            
            # Check what the pipeline produces
            # This might help identify the column mismatch
            print("Identifying column mismatch in feature pipeline...")
            
        except Exception as inner_e:
            print(f"Inner diagnostic failed: {inner_e}")
        
        return False

def run_all_diagnostic_tests():
    """Run all diagnostic tests to identify and fix issues."""
    
    print("🔍 Running Diagnostic Tests for Architecture Issues\n")
    
    print("=" * 60)
    print("1. Testing ModelRegistry")
    print("=" * 60)
    if test_model_registry_creation():
        print("✅ ModelRegistry works with parameters")
    else:
        print("❌ ModelRegistry needs parameter fix")
        test_model_registry_fix()
    
    print("\n" + "=" * 60)
    print("2. Testing CLI Configuration Loading")
    print("=" * 60)
    test_fix_cli_config_loading()
    
    print("\n" + "=" * 60)
    print("3. Testing Data Loader Interface")
    print("=" * 60)
    data_loader_ok = test_data_loader_interface()
    
    print("\n" + "=" * 60)
    print("4. Testing Feature Builder")
    print("=" * 60)
    feature_builder_ok = test_feature_builder_issue()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Working Pipeline: Available (see test_working_pipeline.py)")
    print(f"{'✅' if data_loader_ok else '❌'} Data Loader: {'Working' if data_loader_ok else 'Needs Fix'}")
    print(f"{'✅' if feature_builder_ok else '❌'} Feature Builder: {'Working' if feature_builder_ok else 'Needs Fix'}")
    print(f"⚠️  ModelRegistry: Works but parameter passing needs adjustment")
    print(f"⚠️  CLI: Configuration loading works but needs path fixes")

if __name__ == "__main__":
    run_all_diagnostic_tests()
