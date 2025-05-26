"""
Tests for OpenInterpretability core components.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime

from openinterpretability.core.evaluator import BehaviorEvaluator
from openinterpretability.core.analyzer import ModelAnalyzer
from openinterpretability.core.engine import InterpretabilityEngine
from openinterpretability.models.evaluation import (
    EvaluationResult, SafetyScore, EthicalScore, AlignmentScore,
    RiskLevel, SafetyCategory, EthicalDimension, AlignmentCriteria
)


class TestBehaviorEvaluator:
    """Test cases for BehaviorEvaluator."""
    
    @pytest.mark.asyncio
    async def test_evaluate_text_basic(self, behavior_evaluator, sample_evaluation_result):
        """Test basic text evaluation."""
        # Mock the individual evaluators
        with patch.object(behavior_evaluator.safety_evaluator, 'evaluate', new_callable=AsyncMock) as mock_safety, \
             patch.object(behavior_evaluator.ethical_evaluator, 'evaluate', new_callable=AsyncMock) as mock_ethical, \
             patch.object(behavior_evaluator.alignment_evaluator, 'evaluate', new_callable=AsyncMock) as mock_alignment:
            
            mock_safety.return_value = sample_evaluation_result.safety_score
            mock_ethical.return_value = sample_evaluation_result.ethical_score
            mock_alignment.return_value = sample_evaluation_result.alignment_score
            
            result = await behavior_evaluator.evaluate_text("Test text")
            
            assert isinstance(result, EvaluationResult)
            assert result.text == "Test text"
            assert result.safety_score is not None
            assert result.ethical_score is not None
            assert result.alignment_score is not None
    
    @pytest.mark.asyncio
    async def test_evaluate_text_selective_dimensions(self, behavior_evaluator, sample_evaluation_result):
        """Test evaluation with selective dimensions."""
        with patch.object(behavior_evaluator.safety_evaluator, 'evaluate', new_callable=AsyncMock) as mock_safety:
            mock_safety.return_value = sample_evaluation_result.safety_score
            
            result = await behavior_evaluator.evaluate_text(
                "Test text",
                include_safety=True,
                include_ethical=False,
                include_alignment=False
            )
            
            assert result.safety_score is not None
            assert result.ethical_score is None
            assert result.alignment_score is None
    
    @pytest.mark.asyncio
    async def test_evaluate_text_empty_input(self, behavior_evaluator):
        """Test evaluation with empty input."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await behavior_evaluator.evaluate_text("")
    
    @pytest.mark.asyncio
    async def test_evaluate_batch(self, behavior_evaluator, sample_test_prompts, sample_evaluation_result):
        """Test batch evaluation."""
        with patch.object(behavior_evaluator, 'evaluate_text', new_callable=AsyncMock) as mock_evaluate:
            mock_evaluate.return_value = sample_evaluation_result
            
            results = await behavior_evaluator.evaluate_batch(sample_test_prompts[:3])
            
            assert len(results.results) == 3
            assert all(isinstance(r, EvaluationResult) for r in results.results)
            assert results.total_items == 3
            assert results.completed_items == 3
    
    @pytest.mark.asyncio
    async def test_evaluate_batch_with_failures(self, behavior_evaluator, sample_test_prompts):
        """Test batch evaluation with some failures."""
        async def mock_evaluate_with_failure(text, **kwargs):
            if "error" in text:
                raise Exception("Evaluation failed")
            return EvaluationResult(
                id="test",
                text=text,
                model="gpt-4",
                safety_score=None,
                ethical_score=None,
                alignment_score=None,
                metadata={},
                timestamp=datetime.now(),
                evaluation_types=[]
            )
        
        with patch.object(behavior_evaluator, 'evaluate_text', side_effect=mock_evaluate_with_failure):
            test_texts = ["good text", "error text", "another good text"]
            results = await behavior_evaluator.evaluate_batch(test_texts)
            
            assert results.total_items == 3
            assert results.completed_items == 2
            assert results.failed_items == 1
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, behavior_evaluator, sample_evaluation_result):
        """Test caching functionality."""
        with patch.object(behavior_evaluator.cache_manager, 'get', new_callable=AsyncMock) as mock_get, \
             patch.object(behavior_evaluator.cache_manager, 'set', new_callable=AsyncMock) as mock_set, \
             patch.object(behavior_evaluator.safety_evaluator, 'evaluate', new_callable=AsyncMock) as mock_safety:
            
            # First call - cache miss
            mock_get.return_value = None
            mock_safety.return_value = sample_evaluation_result.safety_score
            
            result1 = await behavior_evaluator.evaluate_text("Test text", use_cache=True)
            
            # Verify cache was checked and result was stored
            mock_get.assert_called_once()
            mock_set.assert_called_once()
            
            # Second call - cache hit
            mock_get.reset_mock()
            mock_set.reset_mock()
            mock_get.return_value = {
                'id': 'cached',
                'text': 'Test text',
                'model': 'gpt-4',
                'safety_score': None,
                'ethical_score': None,
                'alignment_score': None,
                'metadata': {},
                'timestamp': datetime.now().isoformat(),
                'evaluation_types': []
            }
            
            result2 = await behavior_evaluator.evaluate_text("Test text", use_cache=True)
            
            # Verify cache was used
            mock_get.assert_called_once()
            mock_set.assert_not_called()


class TestModelAnalyzer:
    """Test cases for ModelAnalyzer."""
    
    @pytest.mark.asyncio
    async def test_analyze_behavior_patterns(self, model_analyzer, sample_evaluation_result):
        """Test behavior pattern analysis."""
        evaluation_results = [sample_evaluation_result] * 5
        
        with patch.object(model_analyzer, '_analyze_safety_patterns', new_callable=AsyncMock) as mock_safety, \
             patch.object(model_analyzer, '_analyze_ethical_patterns', new_callable=AsyncMock) as mock_ethical, \
             patch.object(model_analyzer, '_analyze_alignment_patterns', new_callable=AsyncMock) as mock_alignment, \
             patch.object(model_analyzer, '_llm_pattern_analysis', new_callable=AsyncMock) as mock_llm:
            
            mock_safety.return_value = []
            mock_ethical.return_value = []
            mock_alignment.return_value = []
            mock_llm.return_value = []
            
            patterns = await model_analyzer.analyze_behavior_patterns(evaluation_results)
            
            assert isinstance(patterns, list)
            mock_safety.assert_called_once()
            mock_ethical.assert_called_once()
            mock_alignment.assert_called_once()
            mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_interpretability_report(self, model_analyzer, sample_evaluation_result):
        """Test interpretability report generation."""
        evaluation_results = [sample_evaluation_result] * 3
        
        with patch.object(model_analyzer, 'analyze_behavior_patterns', new_callable=AsyncMock) as mock_patterns, \
             patch.object(model_analyzer, '_generate_insights', new_callable=AsyncMock) as mock_insights:
            
            mock_patterns.return_value = []
            mock_insights.return_value = []
            
            report = await model_analyzer.generate_interpretability_report(evaluation_results)
            
            assert hasattr(report, 'model')
            assert hasattr(report, 'analysis_timestamp')
            assert hasattr(report, 'behavior_patterns')
            assert hasattr(report, 'insights')
            assert hasattr(report, 'risk_assessment')
            assert hasattr(report, 'recommendations')
            assert hasattr(report, 'confidence_score')
    
    @pytest.mark.asyncio
    async def test_compare_models(self, model_analyzer, sample_test_prompts):
        """Test model comparison functionality."""
        with patch.object(model_analyzer.behavior_evaluator, 'evaluate_batch', new_callable=AsyncMock) as mock_batch:
            # Mock batch evaluation results for both models with proper evaluation results
            mock_result = AsyncMock()
            mock_results = []
            for i in range(3):
                mock_eval = AsyncMock()
                mock_eval.safety_score = AsyncMock()
                mock_eval.safety_score.overall_score = 0.8
                mock_eval.ethical_score = AsyncMock()
                mock_eval.ethical_score.overall_score = 0.9
                mock_eval.alignment_score = AsyncMock()
                mock_eval.alignment_score.overall_score = 0.85
                mock_results.append(mock_eval)
            
            mock_result.evaluations = mock_results
            mock_batch.return_value = mock_result
            
            comparison = await model_analyzer.compare_models(
                sample_test_prompts[:3],
                "gpt-4",
                "claude-3"
            )
            
            assert hasattr(comparison, 'model_a')
            assert hasattr(comparison, 'model_b')
            assert hasattr(comparison, 'safety_comparison')
            assert hasattr(comparison, 'ethical_comparison')
            assert hasattr(comparison, 'alignment_comparison')
            assert hasattr(comparison, 'overall_winner')
            assert comparison.model_a == "gpt-4"
            assert comparison.model_b == "claude-3"


class TestInterpretabilityEngine:
    """Test cases for InterpretabilityEngine."""
    
    @pytest.mark.asyncio
    async def test_evaluate_text(self, interpretability_engine):
        """Test text evaluation through engine."""
        with patch.object(interpretability_engine.safety_evaluator, 'evaluate', new_callable=AsyncMock) as mock_safety, \
             patch.object(interpretability_engine.ethical_evaluator, 'evaluate', new_callable=AsyncMock) as mock_ethical, \
             patch.object(interpretability_engine.alignment_evaluator, 'evaluate', new_callable=AsyncMock) as mock_alignment:
            
            mock_safety.return_value = SafetyScore(
                overall_score=0.8,
                risk_level=RiskLevel.LOW_RISK,
                category_scores={},
                detected_issues=[],
                confidence=0.9,
                explanation="Safe"
            )
            mock_ethical.return_value = EthicalScore(
                overall_score=0.9,
                dimension_scores={},
                ethical_concerns=[],
                recommendations=[],
                confidence=0.85,
                explanation="Ethical"
            )
            mock_alignment.return_value = AlignmentScore(
                overall_score=0.85,
                criteria_scores={},
                alignment_issues=[],
                strengths=[],
                confidence=0.8,
                explanation="Aligned"
            )
            
            result = await interpretability_engine.evaluate_text("Test text")
            
            assert isinstance(result, EvaluationResult)
            assert result.text == "Test text"
            assert result.safety_score is not None
            assert result.ethical_score is not None
            assert result.alignment_score is not None
    
    @pytest.mark.asyncio
    async def test_batch_evaluate(self, interpretability_engine, sample_test_prompts):
        """Test batch evaluation through engine."""
        with patch.object(interpretability_engine, 'evaluate_text', new_callable=AsyncMock) as mock_evaluate:
            mock_result = EvaluationResult(
                id="test",
                text="test",
                model="gpt-4",
                safety_score=None,
                ethical_score=None,
                alignment_score=None,
                metadata={},
                timestamp=datetime.now(),
                evaluation_types=[]
            )
            mock_evaluate.return_value = mock_result
            
            results = await interpretability_engine.batch_evaluate(sample_test_prompts[:3])
            
            assert len(results) == 3
            assert all(isinstance(r, EvaluationResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_analyze_model_behavior(self, interpretability_engine, sample_test_prompts):
        """Test model behavior analysis."""
        with patch.object(interpretability_engine, 'batch_evaluate', new_callable=AsyncMock) as mock_batch, \
             patch.object(interpretability_engine, '_analyze_behavior_patterns', new_callable=AsyncMock) as mock_analyze:
            
            mock_batch.return_value = []
            mock_analyze.return_value = {"patterns": [], "insights": []}
            
            analysis = await interpretability_engine.analyze_model_behavior(
                model="gpt-4",
                test_prompts=sample_test_prompts[:3]
            )
            
            assert isinstance(analysis, dict)
            mock_batch.assert_called_once()
            mock_analyze.assert_called_once()


class TestIntegration:
    """Integration tests for core components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_evaluation(self, behavior_evaluator, model_analyzer, sample_test_prompts):
        """Test end-to-end evaluation workflow."""
        # Mock all the underlying evaluators
        with patch.object(behavior_evaluator.safety_evaluator, 'evaluate', new_callable=AsyncMock) as mock_safety, \
             patch.object(behavior_evaluator.ethical_evaluator, 'evaluate', new_callable=AsyncMock) as mock_ethical, \
             patch.object(behavior_evaluator.alignment_evaluator, 'evaluate', new_callable=AsyncMock) as mock_alignment:
            
            mock_safety.return_value = SafetyScore(
                overall_score=0.8,
                risk_level=RiskLevel.LOW_RISK,
                category_scores={},
                detected_issues=[],
                confidence=0.9,
                explanation="Safe"
            )
            mock_ethical.return_value = EthicalScore(
                overall_score=0.9,
                dimension_scores={},
                ethical_concerns=[],
                recommendations=[],
                confidence=0.85,
                explanation="Ethical"
            )
            mock_alignment.return_value = AlignmentScore(
                overall_score=0.85,
                criteria_scores={},
                alignment_issues=[],
                strengths=[],
                confidence=0.8,
                explanation="Aligned"
            )
            
            # Step 1: Evaluate texts
            evaluation_results = []
            for prompt in sample_test_prompts[:3]:
                result = await behavior_evaluator.evaluate_text(prompt)
                evaluation_results.append(result)
            
            assert len(evaluation_results) == 3
            
            # Step 2: Analyze patterns
            with patch.object(model_analyzer, 'analyze_behavior_patterns', new_callable=AsyncMock) as mock_patterns:
                mock_patterns.return_value = []
                patterns = await model_analyzer.analyze_behavior_patterns(evaluation_results)
                assert isinstance(patterns, list)
            
            # Step 3: Generate report
            with patch.object(model_analyzer, 'generate_interpretability_report', new_callable=AsyncMock) as mock_report:
                mock_report.return_value = AsyncMock()
                report = await model_analyzer.generate_interpretability_report(evaluation_results)
                assert report is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, behavior_evaluator, sample_test_prompts):
        """Test concurrent evaluation handling."""
        with patch.object(behavior_evaluator.safety_evaluator, 'evaluate', new_callable=AsyncMock) as mock_safety:
            mock_safety.return_value = SafetyScore(
                overall_score=0.8,
                risk_level=RiskLevel.LOW_RISK,
                category_scores={},
                detected_issues=[],
                confidence=0.9,
                explanation="Safe"
            )
            
            # Run multiple evaluations concurrently
            tasks = [
                behavior_evaluator.evaluate_text(prompt, include_ethical=False, include_alignment=False)
                for prompt in sample_test_prompts[:5]
            ]
            
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(isinstance(r, EvaluationResult) for r in results)
            assert all(r.safety_score is not None for r in results) 