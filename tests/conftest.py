"""Shared pytest fixtures for AI Scientist tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Yields:
        Path: Path to the temporary directory

    Example:
        def test_file_creation(temp_dir):
            test_file = temp_dir / "test.txt"
            test_file.write_text("content")
            assert test_file.exists()
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing.

    Args:
        temp_dir: Temporary directory fixture

    Yields:
        Path: Path to the temporary file

    Example:
        def test_file_reading(temp_file):
            temp_file.write_text("test content")
            assert temp_file.read_text() == "test content"
    """
    test_file = temp_dir / "test_file.txt"
    test_file.touch()
    yield test_file


@pytest.fixture
def mock_config() -> dict:
    """Provide a mock configuration dictionary for testing.

    Returns:
        dict: Mock configuration with common test settings

    Example:
        def test_config_loading(mock_config):
            assert "api_key" in mock_config
            assert mock_config["timeout"] == 30
    """
    return {
        "api_key": "test_api_key_123",
        "timeout": 30,
        "max_retries": 3,
        "debug": True,
    }


@pytest.fixture
def sample_data_dir(temp_dir: Path) -> Path:
    """Create a sample data directory with test files.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Path to the sample data directory

    Example:
        def test_data_loading(sample_data_dir):
            files = list(sample_data_dir.glob("*.txt"))
            assert len(files) > 0
    """
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create some sample files
    (data_dir / "sample1.txt").write_text("Sample data 1")
    (data_dir / "sample2.txt").write_text("Sample data 2")

    return data_dir


@pytest.fixture
def mock_env_vars(monkeypatch) -> dict:
    """Set up mock environment variables for testing.

    Args:
        monkeypatch: Pytest's monkeypatch fixture

    Returns:
        dict: Dictionary of set environment variables

    Example:
        def test_env_usage(mock_env_vars):
            assert os.getenv("TEST_API_KEY") == "test_key"
    """
    env_vars = {
        "TEST_API_KEY": "test_key",
        "TEST_DEBUG": "true",
        "TEST_ENVIRONMENT": "testing",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def capture_logs(caplog):
    """Capture log messages during tests.

    Args:
        caplog: Pytest's caplog fixture

    Yields:
        LogCaptureFixture: The caplog fixture for log assertions

    Example:
        def test_logging(capture_logs):
            logger.info("Test message")
            assert "Test message" in capture_logs.text
    """
    yield caplog


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment state before each test.

    This fixture runs automatically before each test to ensure
    a clean state. Add any cleanup logic here.

    Yields:
        None: Yields control to the test, then performs cleanup
    """
    # Setup: Add any pre-test setup here

    yield

    # Teardown: Add any post-test cleanup here
    pass
