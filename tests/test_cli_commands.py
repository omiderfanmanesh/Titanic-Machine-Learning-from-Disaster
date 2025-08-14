"""
Comprehensive test suite for CLI commands.

This test suite will identify and fix issues with the evaluate and predict commands,
and ensure all CLI functions are working properly.
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import tempfile
import shutil
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from cli import cli, evaluate, predict
    CLI_IMPORT_SUCCESS = True
    CLI_IMPORT_ERROR = None
except Exception as e:
    CLI_IMPORT_SUCCESS = False
    CLI_IMPORT_ERROR = str(e)


class TestCLIImport:
    """Test CLI module can be imported successfully."""

    def test_cli_import(self):
        """Test that CLI module imports without errors."""
        if not CLI_IMPORT_SUCCESS:
            pytest.fail(f"CLI import failed: {CLI_IMPORT_ERROR}")

        assert cli is not None
        assert evaluate is not None
        assert predict is not None


class TestCLIBasics:
    """Test basic CLI functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command works."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "Titanic ML Pipeline" in result.output

    def test_cli_commands_available(self):
        """Test that evaluate and predict commands are available."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert "evaluate" in result.output
        assert "predict" in result.output


class TestEvaluateCommand:
    """Test the evaluate command functionality."""

    def setup_method(self):
        """Setup test environment for evaluate command."""
        self.runner = CliRunner()
        self.temp_dir = None

    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def create_mock_run_dir(self):
        """Create a mock run directory with required files."""
        self.temp_dir = tempfile.mkdtemp()
        run_path = Path(self.temp_dir)

        # Create mock OOF predictions
        oof_data = pd.DataFrame({
            'prediction': np.random.rand(100),
            'target': np.random.randint(0, 2, 100),
            'fold': np.random.randint(0, 5, 100)
        })
        oof_path = run_path / "oof_predictions.csv"
        oof_data.to_csv(oof_path, index=False)

        # Create mock CV scores
        cv_scores = {
            "fold_scores": [0.85, 0.87, 0.83, 0.86, 0.84],
            "mean_score": 0.85,
            "std_score": 0.015,
            "oof_score": 0.854
        }
        scores_path = run_path / "cv_scores.json"
        with open(scores_path, 'w') as f:
            json.dump(cv_scores, f)

        return str(run_path)

    def test_evaluate_help(self):
        """Test evaluate command help."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        result = self.runner.invoke(cli, ['evaluate', '--help'])
        assert result.exit_code == 0
        assert "--run-dir" in result.output

    def test_evaluate_missing_run_dir(self):
        """Test evaluate command with missing run directory."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        result = self.runner.invoke(cli, ['evaluate', '--run-dir', '/nonexistent/path'])
        assert result.exit_code != 0

    def test_evaluate_missing_files(self):
        """Test evaluate command with missing required files."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.runner.invoke(cli, ['evaluate', '--run-dir', temp_dir])
            # Should handle missing files gracefully
            assert "Required files not found" in result.output or result.exit_code != 0

    @patch('cli.config_manager')
    @patch('cli.TitanicEvaluator')
    def test_evaluate_with_mock_data(self, mock_evaluator_class, mock_config_manager):
        """Test evaluate command with mocked dependencies."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        # Setup mocks
        mock_config_manager.load_config.return_value = {
            "threshold": {"optimizer": False, "print": True}
        }

        mock_evaluator = Mock()
        mock_evaluator.evaluate_cv.return_value = {
            "oof_metrics": {"auc": 0.85, "accuracy": 0.82, "f1": 0.78},
            "cv_statistics": {"cv_mean": 0.85, "cv_std": 0.015},
            "stability": {"is_stable": True, "coefficient_of_variation": 0.02}
        }
        mock_evaluator_class.return_value = mock_evaluator

        run_dir = self.create_mock_run_dir()

        result = self.runner.invoke(cli, ['evaluate', '--run-dir', run_dir])

        # Should complete without error
        assert result.exit_code == 0 or "Evaluation Results" in result.output


class TestPredictCommand:
    """Test the predict command functionality."""

    def setup_method(self):
        """Setup test environment for predict command."""
        self.runner = CliRunner()
        self.temp_dir = None

    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def create_mock_run_dir_with_models(self):
        """Create a mock run directory with model files."""
        self.temp_dir = tempfile.mkdtemp()
        run_path = Path(self.temp_dir)

        # Create mock model files
        for i in range(5):
            model_path = run_path / f"fold_{i}_model.joblib"
            model_path.write_text(f"mock_model_{i}")

        return str(run_path)

    def create_mock_processed_data(self, data_dir):
        """Create mock processed test data."""
        processed_dir = Path(data_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Create mock test features
        test_data = pd.DataFrame({
            'PassengerId': range(892, 1310),
            'feature_1': np.random.rand(418),
            'feature_2': np.random.rand(418),
            'feature_3': np.random.randint(0, 3, 418)
        })
        test_path = processed_dir / "test_features.csv"
        test_data.to_csv(test_path, index=False)

        return str(test_path)

    def test_predict_help(self):
        """Test predict command help."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        result = self.runner.invoke(cli, ['predict', '--help'])
        assert result.exit_code == 0
        assert "--run-dir" in result.output
        assert "--threshold" in result.output

    def test_predict_missing_run_dir(self):
        """Test predict command with missing run directory."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        result = self.runner.invoke(cli, ['predict', '--run-dir', '/nonexistent/path'])
        assert result.exit_code != 0

    @patch('cli.path_manager')
    @patch('cli.config_manager')
    @patch('cli.ModelLoader')
    @patch('cli.create_predictor')
    def test_predict_with_mocks(self, mock_create_predictor, mock_model_loader_class,
                               mock_config_manager, mock_path_manager):
        """Test predict command with mocked dependencies."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        # Setup mocks
        run_dir = self.create_mock_run_dir_with_models()

        # Mock path manager
        mock_path_manager.data_dir = Path(self.temp_dir)
        test_path = self.create_mock_processed_data(self.temp_dir)

        # Mock config manager
        mock_config_manager.load_config.return_value = {
            "threshold": {"value": 0.5, "optimizer": False},
            "model_paths": []
        }

        # Mock model loader
        mock_model_loader = Mock()
        mock_model_loader.load_fold_models.return_value = [Mock() for _ in range(5)]
        mock_model_loader_class.return_value = mock_model_loader

        # Mock predictor
        mock_predictor = Mock()
        mock_predictions = pd.DataFrame({
            'PassengerId': range(892, 1310),
            'prediction': np.random.randint(0, 2, 418),
            'prediction_proba': np.random.rand(418)
        })
        mock_predictor.predict.return_value = mock_predictions
        mock_predictor._resolve_threshold.return_value = 0.5
        mock_create_predictor.return_value = mock_predictor

        result = self.runner.invoke(cli, ['predict', '--run-dir', run_dir])

        # Should complete without error or show expected messages
        assert result.exit_code == 0 or "Predictions saved" in result.output


class TestCLIIntegration:
    """Test CLI integration and error handling."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()

    def test_cli_commands_exist(self):
        """Test that all expected CLI commands exist."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0

        expected_commands = ['evaluate', 'predict', 'train', 'features', 'validate']
        for command in expected_commands:
            assert command in result.output

    def test_evaluate_command_structure(self):
        """Test evaluate command structure and options."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        result = self.runner.invoke(cli, ['evaluate', '--help'])
        assert result.exit_code == 0
        assert "--run-dir" in result.output
        assert "required" in result.output.lower()

    def test_predict_command_structure(self):
        """Test predict command structure and options."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        result = self.runner.invoke(cli, ['predict', '--help'])
        assert result.exit_code == 0
        assert "--run-dir" in result.output
        assert "--threshold" in result.output
        assert "--output-path" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()

    def test_evaluate_graceful_error_handling(self):
        """Test that evaluate command handles errors gracefully."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory but no required files
            result = self.runner.invoke(cli, ['evaluate', '--run-dir', temp_dir])

            # Should not crash, should show error message
            assert "Required files not found" in result.output or result.exit_code != 0

    def test_predict_graceful_error_handling(self):
        """Test that predict command handles errors gracefully."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory but no required files
            result = self.runner.invoke(cli, ['predict', '--run-dir', temp_dir])

            # Should not crash, should show error message
            assert result.exit_code != 0 or "not found" in result.output


class TestCLIFixesAndImprovements:
    """Test fixes for CLI issues and verify improvements."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner()

    def test_cli_module_structure(self):
        """Test that CLI module has proper structure."""
        if not CLI_IMPORT_SUCCESS:
            pytest.skip(f"CLI import failed: {CLI_IMPORT_ERROR}")

        # Import the CLI module and check its structure
        import cli

        # Check that main functions exist
        assert hasattr(cli, 'cli')
        assert hasattr(cli, 'evaluate')
        assert hasattr(cli, 'predict')

        # Check that CLI group is properly configured
        assert cli.cli.name == 'cli' or hasattr(cli.cli, 'commands')

    def test_import_error_handling(self):
        """Test that import errors are handled properly."""
        if not CLI_IMPORT_SUCCESS:
            # This test specifically checks import error handling
            assert CLI_IMPORT_ERROR is not None
            print(f"Expected import error detected: {CLI_IMPORT_ERROR}")
        else:
            # If imports work, verify no silent failures
            import cli
            assert cli is not None


if __name__ == "__main__":
    # Run the tests when script is executed directly
    pytest.main([__file__, "-v"])
