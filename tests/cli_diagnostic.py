#!/usr/bin/env python3
"""
CLI Diagnostic Script - Test CLI functionality step by step
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_basic_imports():
    """Test basic Python imports"""
    try:
        import click
        import pandas as pd
        import json
        from pathlib import Path
        print("✓ Basic imports successful")
        return True
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False

def test_core_imports():
    """Test core module imports"""
    try:
        from core.utils import PathManager, ConfigManager, LoggerFactory
        print("✓ Core imports successful")
        return True
    except Exception as e:
        print(f"✗ Core imports failed: {e}")
        traceback.print_exc()
        return False

def test_cli_imports():
    """Test CLI module imports"""
    try:
        import cli
        print("✓ CLI module imported")

        # Check if main components exist
        if hasattr(cli, 'cli'):
            print("✓ CLI group found")
        else:
            print("✗ CLI group not found")

        if hasattr(cli, 'evaluate'):
            print("✓ Evaluate command found")
        else:
            print("✗ Evaluate command not found")

        if hasattr(cli, 'predict'):
            print("✓ Predict command found")
        else:
            print("✗ Predict command not found")

        return True
    except Exception as e:
        print(f"✗ CLI imports failed: {e}")
        traceback.print_exc()
        return False

def test_cli_help():
    """Test CLI help functionality"""
    try:
        from click.testing import CliRunner
        from cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        if result.exit_code == 0:
            print("✓ CLI help works")
            print(f"Output preview: {result.output[:200]}...")
            return True
        else:
            print(f"✗ CLI help failed with exit code {result.exit_code}")
            print(f"Output: {result.output}")
            return False
    except Exception as e:
        print(f"✗ CLI help test failed: {e}")
        traceback.print_exc()
        return False

def test_evaluate_command():
    """Test evaluate command help"""
    try:
        from click.testing import CliRunner
        from cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['evaluate', '--help'])

        if result.exit_code == 0:
            print("✓ Evaluate command help works")
            return True
        else:
            print(f"✗ Evaluate command help failed with exit code {result.exit_code}")
            print(f"Output: {result.output}")
            return False
    except Exception as e:
        print(f"✗ Evaluate command test failed: {e}")
        traceback.print_exc()
        return False

def test_predict_command():
    """Test predict command help"""
    try:
        from click.testing import CliRunner
        from cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['predict', '--help'])

        if result.exit_code == 0:
            print("✓ Predict command help works")
            return True
        else:
            print(f"✗ Predict command help failed with exit code {result.exit_code}")
            print(f"Output: {result.output}")
            return False
    except Exception as e:
        print(f"✗ Predict command test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🔍 CLI Diagnostic Tests")
    print("=" * 50)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Core Imports", test_core_imports),
        ("CLI Imports", test_cli_imports),
        ("CLI Help", test_cli_help),
        ("Evaluate Command", test_evaluate_command),
        ("Predict Command", test_predict_command),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n📋 Testing: {name}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("📊 Summary:")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")

    failed_tests = [name for name, success in results if not success]
    if failed_tests:
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        return 1
    else:
        print("\n✅ All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
