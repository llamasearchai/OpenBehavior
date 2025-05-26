"""Pytest configuration and fixtures."""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List
import yaml

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# Test data
SAMPLE_PROMPT = "Explain the importance of ethical AI development."
SAMPLE_RESPONSE = "Ethical AI development is crucial for ensuring..."

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = AsyncMock(spec=AsyncOpenAI)
    return client

@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = AsyncMock(spec=AsyncAnthropic)
    return client

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "openai": {"api_key": "test-openai-key"},
        "anthropic": {"api_key": "test-anthropic-key"},
        "models": {"default": "gpt-4"},
        "cache": {"enabled": True, "ttl": 3600},
        "database": {"url": "sqlite:///test.db"},
        "api": {"host": "localhost", "port": 8000}
    }

@pytest.fixture
def sample_test_prompts():
    """Create sample test prompts for evaluation."""
    return [
        "Hello, how are you today?",
        "Can you help me write a story?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How do I bake a chocolate cake?"
    ]

@pytest.fixture
def temp_config(temp_dir):
    """Create a temporary configuration file."""
    config_data = {
        "openai": {
            "api_key": "test-openai-key",
            "default_model": "gpt-4"
        },
        "anthropic": {
            "api_key": "test-anthropic-key"
        },
        "engine": {
            "max_concurrent": 5,
            "timeout": 30
        },
        "cache": {
            "enabled": True,
            "ttl": 3600
        },
        "database": {
            "url": "sqlite:///test.db"
        },
        "api": {
            "host": "localhost",
            "port": 8000
        }
    }
    
    config_file = temp_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    return config_file

@pytest.fixture
def runner():
    """Create a CLI test runner."""
    from click.testing import CliRunner
    return CliRunner() 