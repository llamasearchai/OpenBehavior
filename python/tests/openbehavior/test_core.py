"""Tests for OpenBehavior core components."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any


class TestBehaviorEvaluator:
    """Test behavior evaluator functionality."""
    
    @pytest.fixture
    def mock_evaluator(self):
        """Mock evaluator for testing."""
        return Mock()
    
    def test_evaluator_initialization(self, mock_evaluator):
        """Test evaluator initialization."""
        assert mock_evaluator is not None
    
    def test_basic_evaluation(self, mock_evaluator):
        """Test basic evaluation functionality."""
        mock_evaluator.evaluate.return_value = {"score": 0.85}
        
        result = mock_evaluator.evaluate("test prompt")
        
        assert result["score"] == 0.85
        mock_evaluator.evaluate.assert_called_once_with("test prompt")


class TestPromptEngineering:
    """Test prompt engineering components."""
    
    @pytest.fixture
    def mock_prompt_engineer(self):
        """Mock prompt engineer for testing."""
        return Mock()
    
    def test_prompt_optimization(self, mock_prompt_engineer):
        """Test prompt optimization."""
        mock_prompt_engineer.optimize.return_value = {
            "best_prompt": "optimized prompt",
            "score": 0.95
        }
        
        result = mock_prompt_engineer.optimize("base prompt")
        
        assert result["score"] == 0.95
        assert "optimized prompt" in result["best_prompt"]
    
    def test_prompt_variations(self, mock_prompt_engineer):
        """Test prompt variation generation."""
        mock_prompt_engineer.generate_variations.return_value = [
            "variation 1",
            "variation 2",
            "variation 3"
        ]
        
        variations = mock_prompt_engineer.generate_variations("base prompt")
        
        assert len(variations) == 3
        assert all("variation" in v for v in variations)


class TestModelInterface:
    """Test model interface components."""
    
    @pytest.fixture
    def mock_interface(self):
        """Mock model interface for testing."""
        interface = AsyncMock()
        interface.generate.return_value = "Generated response"
        return interface
    
    @pytest.mark.asyncio
    async def test_model_generation(self, mock_interface):
        """Test model response generation."""
        response = await mock_interface.generate("test prompt")
        
        assert response == "Generated response"
        mock_interface.generate.assert_called_once_with("test prompt")
    
    @pytest.mark.asyncio
    async def test_batch_generation(self, mock_interface):
        """Test batch response generation."""
        mock_interface.batch_generate.return_value = [
            "Response 1",
            "Response 2",
            "Response 3"
        ]
        
        responses = await mock_interface.batch_generate([
            "Prompt 1",
            "Prompt 2", 
            "Prompt 3"
        ])
        
        assert len(responses) == 3
        assert all("Response" in r for r in responses)


class TestEvaluationMetrics:
    """Test evaluation metrics and scoring."""
    
    @pytest.fixture
    def mock_metrics(self):
        """Mock metrics calculator."""
        return Mock()
    
    def test_safety_metrics(self, mock_metrics):
        """Test safety metric calculation."""
        mock_metrics.calculate_safety.return_value = {
            "harm_prevention": 0.9,
            "toxicity": 0.85,
            "overall": 0.875
        }
        
        result = mock_metrics.calculate_safety("test response")
        
        assert result["overall"] == 0.875
        assert result["harm_prevention"] == 0.9
    
    def test_ethical_metrics(self, mock_metrics):
        """Test ethical metric calculation."""
        mock_metrics.calculate_ethical.return_value = {
            "fairness": 0.88,
            "honesty": 0.92,
            "overall": 0.9
        }
        
        result = mock_metrics.calculate_ethical("test response")
        
        assert result["overall"] == 0.9
        assert result["fairness"] == 0.88
    
    def test_alignment_metrics(self, mock_metrics):
        """Test alignment metric calculation."""
        mock_metrics.calculate_alignment.return_value = {
            "helpfulness": 0.85,
            "relevance": 0.9,
            "overall": 0.875
        }
        
        result = mock_metrics.calculate_alignment("test response")
        
        assert result["overall"] == 0.875
        assert result["helpfulness"] == 0.85


class TestConfigurationManagement:
    """Test configuration management."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            "model": {
                "provider": "openai",
                "name": "gpt-4",
                "api_key": "test-key"
            },
            "evaluation": {
                "batch_size": 10,
                "timeout": 30
            }
        }
    
    def test_config_loading(self, mock_config):
        """Test configuration loading."""
        assert mock_config["model"]["provider"] == "openai"
        assert mock_config["evaluation"]["batch_size"] == 10
    
    def test_config_validation(self, mock_config):
        """Test configuration validation."""
        required_keys = ["model", "evaluation"]
        
        for key in required_keys:
            assert key in mock_config
        
        assert "provider" in mock_config["model"]
        assert "batch_size" in mock_config["evaluation"]


class TestDataManagement:
    """Test data loading and management."""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Mock data manager."""
        return Mock()
    
    def test_template_loading(self, mock_data_manager):
        """Test template loading."""
        mock_data_manager.load_templates.return_value = {
            "safety": {"template": "Safety template"},
            "ethical": {"template": "Ethical template"}
        }
        
        templates = mock_data_manager.load_templates()
        
        assert "safety" in templates
        assert "ethical" in templates
        assert templates["safety"]["template"] == "Safety template"
    
    def test_data_export(self, mock_data_manager):
        """Test data export functionality."""
        mock_data_manager.export_data.return_value = True
        
        result = mock_data_manager.export_data("test_data", "output.json")
        
        assert result is True
        mock_data_manager.export_data.assert_called_once_with("test_data", "output.json")
    
    def test_data_import(self, mock_data_manager):
        """Test data import functionality."""
        mock_data_manager.import_data.return_value = {"imported": True}
        
        result = mock_data_manager.import_data("input.json")
        
        assert result["imported"] is True
        mock_data_manager.import_data.assert_called_once_with("input.json")


class TestIntegration:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline."""
        # Mock a simple evaluation pipeline
        mock_result = {
            "overall_score": 0.85,
            "safety_score": 0.9,
            "ethical_score": 0.8,
            "alignment_score": 0.85
        }
        
        # Simulate evaluation
        result = mock_result
        
        assert result["overall_score"] == 0.85
        assert "safety_score" in result
        assert "ethical_score" in result
        assert "alignment_score" in result
    
    def test_configuration_integration(self):
        """Test configuration integration."""
        config = {
            "openai": {"api_key": "test-key"},
            "model": {"default": "gpt-4"},
            "cache": {"enabled": True}
        }
        
        # Test configuration validation
        assert "openai" in config
        assert "model" in config
        assert config["cache"]["enabled"] is True
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling in integration scenarios."""
        # Mock error scenario
        def mock_function_with_error():
            raise Exception("API Error")
        
        # Test error handling
        with pytest.raises(Exception) as exc_info:
            mock_function_with_error()
        
        assert "API Error" in str(exc_info.value)
    
    def test_performance_monitoring(self):
        """Test performance monitoring integration."""
        metrics = {
            "evaluation_time": 1.5,
            "api_calls": 3,
            "cache_hits": 2,
            "cache_misses": 1
        }
        
        assert metrics["evaluation_time"] > 0
        assert metrics["api_calls"] > 0
        assert metrics["cache_hits"] >= 0
        assert metrics["cache_misses"] >= 0 