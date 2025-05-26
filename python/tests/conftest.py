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

from openinterpretability.models.evaluation import EvaluationResult, SafetyScore, EthicalScore, AlignmentScore
from openinterpretability.utils.config import ConfigManager
from openinterpretability.utils.cache import CacheManager
from openinterpretability.core.evaluator import BehaviorEvaluator
from openinterpretability.core.analyzer import ModelAnalyzer
from openinterpretability.core.engine import InterpretabilityEngine, EngineConfig

# Test data
SAMPLE_PROMPT = "Explain the importance of ethical AI development."
SAMPLE_RESPONSE = "Ethical AI development is crucial for ensuring..."

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

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
    config = ConfigManager()
    config._config = {
        "openai": {"api_key": "test-openai-key"},
        "anthropic": {"api_key": "test-anthropic-key"},
        "models": {"default": "gpt-4"},
        "cache": {"enabled": True, "ttl": 3600},
        "database": {"url": "sqlite:///test.db"},
        "api": {"host": "localhost", "port": 8000}
    }
    return config

@pytest.fixture
def sample_cache_manager():
    """Create a sample cache manager for testing."""
    cache = CacheManager()
    cache._cache = {}
    return cache

@pytest.fixture
def sample_evaluation_result():
    """Create a sample evaluation result for testing."""
    from datetime import datetime
    from openinterpretability.models.evaluation import RiskLevel, SafetyCategory, EthicalDimension, AlignmentCriteria
    
    return EvaluationResult(
        id="test-eval-123",
        text="This is a test text for evaluation.",
        model="gpt-4",
        safety_score=SafetyScore(
            overall_score=0.85,
            risk_level=RiskLevel.LOW_RISK,
            category_scores={
                SafetyCategory.VIOLENCE: 0.1,
                SafetyCategory.HATE_SPEECH: 0.05,
                SafetyCategory.SELF_HARM: 0.0,
                SafetyCategory.DANGEROUS_INSTRUCTIONS: 0.0
            },
            detected_issues=[],
            confidence=0.9,
            explanation="Text appears safe with minimal risk indicators."
        ),
        ethical_score=EthicalScore(
            overall_score=0.9,
            dimension_scores={
                EthicalDimension.FAIRNESS: 0.95,
                EthicalDimension.HONESTY: 0.85,
                EthicalDimension.HARM_PREVENTION: 0.9,
                EthicalDimension.PRIVACY: 0.9
            },
            ethical_concerns=[],
            recommendations=[],
            confidence=0.88,
            explanation="Text demonstrates good ethical alignment."
        ),
        alignment_score=AlignmentScore(
            overall_score=0.88,
            criteria_scores={
                AlignmentCriteria.HELPFULNESS: 0.9,
                AlignmentCriteria.HARMLESSNESS: 0.85,
                AlignmentCriteria.HONESTY: 0.9
            },
            alignment_issues=[],
            strengths=["Clear communication", "Helpful content"],
            confidence=0.87,
            explanation="Text shows strong alignment with human values."
        ),
        metadata={"test": True},
        timestamp=datetime.now(),
        evaluation_types=["safety", "ethical", "alignment"]
    )

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
def behavior_evaluator(mock_openai_client, mock_anthropic_client, sample_config, sample_cache_manager):
    """Create a BehaviorEvaluator instance for testing."""
    return BehaviorEvaluator(
        openai_client=mock_openai_client,
        anthropic_client=mock_anthropic_client,
        config=sample_config,
        cache_manager=sample_cache_manager
    )

@pytest.fixture
def model_analyzer(behavior_evaluator, mock_openai_client, mock_anthropic_client, sample_config, sample_cache_manager):
    """Create a ModelAnalyzer instance for testing."""
    return ModelAnalyzer(
        behavior_evaluator=behavior_evaluator,
        openai_client=mock_openai_client,
        anthropic_client=mock_anthropic_client,
        config=sample_config,
        cache_manager=sample_cache_manager
    )

@pytest.fixture
def interpretability_engine():
    """Create an InterpretabilityEngine instance for testing."""
    config = EngineConfig(
        openai_api_key="test-openai-key",
        anthropic_api_key="test-anthropic-key",
        default_model="gpt-4",
        max_concurrent_evaluations=5,
        enable_caching=True,
        cache_ttl=3600,
        metrics_enabled=True
    )
    return InterpretabilityEngine(config)

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
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    return str(config_file)

@pytest.fixture
def runner():
    """Click test runner for CLI testing."""
    from click.testing import CliRunner
    return CliRunner() 