"""Validation tests to verify testing infrastructure is set up correctly."""

import sys
from pathlib import Path

import pytest


class TestInfrastructure:
    """Tests to validate the testing infrastructure setup."""

    def test_python_version(self):
        """Verify Python version meets requirements."""
        assert sys.version_info >= (3, 11), "Python 3.11 or higher is required"

    def test_pytest_working(self):
        """Verify pytest is working correctly."""
        assert True, "Basic pytest assertion works"

    @pytest.mark.unit
    def test_unit_marker(self):
        """Verify unit test marker is configured."""
        # This test validates that the 'unit' marker is properly configured
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Verify integration test marker is configured."""
        # This test validates that the 'integration' marker is properly configured
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Verify slow test marker is configured."""
        # This test validates that the 'slow' marker is properly configured
        assert True

    def test_project_structure(self):
        """Verify the project directory structure exists."""
        project_root = Path(__file__).parent.parent
        ai_scientist_dir = project_root / "ai_scientist"

        assert project_root.exists(), "Project root should exist"
        assert ai_scientist_dir.exists(), "ai_scientist package should exist"

    def test_test_directories_exist(self):
        """Verify test directories are properly created."""
        tests_dir = Path(__file__).parent
        unit_dir = tests_dir / "unit"
        integration_dir = tests_dir / "integration"

        assert tests_dir.exists(), "tests directory should exist"
        assert unit_dir.exists(), "tests/unit directory should exist"
        assert integration_dir.exists(), "tests/integration directory should exist"

    def test_conftest_exists(self):
        """Verify conftest.py exists for shared fixtures."""
        conftest_path = Path(__file__).parent / "conftest.py"
        assert conftest_path.exists(), "conftest.py should exist"


class TestFixtures:
    """Tests to validate shared fixtures work correctly."""

    def test_temp_dir_fixture(self, temp_dir):
        """Verify temp_dir fixture creates a usable directory."""
        assert temp_dir.exists(), "Temporary directory should exist"
        assert temp_dir.is_dir(), "temp_dir should be a directory"

        # Test writing to temp directory
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists(), "Should be able to create files in temp_dir"
        assert test_file.read_text() == "test content"

    def test_temp_file_fixture(self, temp_file):
        """Verify temp_file fixture creates a usable file."""
        assert temp_file.exists(), "Temporary file should exist"
        assert temp_file.is_file(), "temp_file should be a file"

        # Test writing and reading
        temp_file.write_text("test data")
        assert temp_file.read_text() == "test data"

    def test_mock_config_fixture(self, mock_config):
        """Verify mock_config fixture provides expected structure."""
        assert isinstance(mock_config, dict), "mock_config should be a dictionary"
        assert "api_key" in mock_config, "mock_config should contain api_key"
        assert "timeout" in mock_config, "mock_config should contain timeout"
        assert "max_retries" in mock_config, "mock_config should contain max_retries"
        assert mock_config["debug"] is True, "mock_config debug should be True"

    def test_sample_data_dir_fixture(self, sample_data_dir):
        """Verify sample_data_dir fixture creates test data."""
        assert sample_data_dir.exists(), "Sample data directory should exist"
        assert sample_data_dir.is_dir(), "sample_data_dir should be a directory"

        files = list(sample_data_dir.glob("*.txt"))
        assert len(files) > 0, "sample_data_dir should contain test files"

    def test_mock_env_vars_fixture(self, mock_env_vars):
        """Verify mock_env_vars fixture sets environment variables."""
        import os

        assert isinstance(mock_env_vars, dict), "mock_env_vars should be a dictionary"
        assert os.getenv("TEST_API_KEY") == "test_key", "Environment variable should be set"
        assert os.getenv("TEST_DEBUG") == "true"
        assert os.getenv("TEST_ENVIRONMENT") == "testing"


class TestMocking:
    """Tests to validate pytest-mock is working."""

    def test_mocker_fixture_available(self, mocker):
        """Verify pytest-mock's mocker fixture is available."""
        # Create a simple mock
        mock_func = mocker.Mock(return_value=42)
        result = mock_func()

        assert result == 42, "Mock should return the configured value"
        mock_func.assert_called_once()

    def test_patch_functionality(self, mocker):
        """Verify mocker.patch works correctly."""
        # Patch a built-in function
        mock_open = mocker.patch("builtins.open", mocker.mock_open(read_data="mocked data"))

        # Use the patched function
        with open("dummy_file.txt") as f:
            content = f.read()

        assert content == "mocked data", "Patched function should return mocked data"
        mock_open.assert_called_once_with("dummy_file.txt")
