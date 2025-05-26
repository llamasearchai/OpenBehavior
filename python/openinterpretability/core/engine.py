"""
Core interpretability engine for OpenInterpretability platform.
Handles orchestration of all evaluation and analysis tasks.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import uuid

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import numpy as np
import yaml

from ..models.evaluation import EvaluationResult, SafetyScore, EthicalScore, AlignmentScore
from ..models.request import EvaluationRequest, BatchEvaluationRequest
from ..evaluation.safety import SafetyEvaluator
from ..evaluation.ethical import EthicalEvaluator
from ..evaluation.alignment import AlignmentEvaluator
from ..utils.config import Config
from ..utils.metrics import MetricsCollector
from ..utils.cache import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the interpretability engine."""
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    default_model: str = "gpt-4"
    max_concurrent_evaluations: int = 10
    enable_caching: bool = True
    cache_ttl: int = 3600
    metrics_enabled: bool = True


class InterpretabilityEngine:
    """
    Main engine for LLM behavior interpretability and evaluation.
    
    Provides comprehensive analysis across multiple dimensions:
    - Safety evaluation
    - Ethical assessment
    - Alignment analysis
    - Behavior pattern detection
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        self.anthropic_client = AsyncAnthropic(api_key=config.anthropic_api_key) if config.anthropic_api_key else None
        
        # Initialize evaluators
        self.safety_evaluator = SafetyEvaluator(self.openai_client)
        self.ethical_evaluator = EthicalEvaluator(self.openai_client)
        self.alignment_evaluator = AlignmentEvaluator(self.openai_client)
        
        # Initialize utilities
        self.cache = CacheManager() if config.enable_caching else None
        self.metrics = MetricsCollector() if config.metrics_enabled else None
        
        # Semaphore for concurrent evaluations
        self.semaphore = asyncio.Semaphore(config.max_concurrent_evaluations)
        
        logger.info("InterpretabilityEngine initialized successfully")
    
    async def evaluate_text(
        self,
        text: str,
        evaluation_types: List[str] = None,
        model: str = None,
        metadata: Dict[str, Any] = None
    ) -> EvaluationResult:
        """
        Evaluate text across specified dimensions.
        
        Args:
            text: Text to evaluate
            evaluation_types: Types of evaluation to perform
            model: Model to use for evaluation
            metadata: Additional metadata
            
        Returns:
            Comprehensive evaluation result
        """
        if evaluation_types is None:
            evaluation_types = ["safety", "ethical", "alignment"]
            
        model = model or self.config.default_model
        request_id = str(uuid.uuid4())
        
        logger.info(f"Starting evaluation {request_id} for text length: {len(text)}")
        
        async with self.semaphore:
            # Check cache
            cache_key = self._generate_cache_key(text, evaluation_types, model)
            if self.cache:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Returning cached result for {request_id}")
                    return cached_result
            
            # Perform evaluations concurrently
            tasks = []
            if "safety" in evaluation_types:
                tasks.append(self._evaluate_safety(text, model))
            if "ethical" in evaluation_types:
                tasks.append(self._evaluate_ethical(text, model))
            if "alignment" in evaluation_types:
                tasks.append(self._evaluate_alignment(text, model))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            safety_score = None
            ethical_score = None
            alignment_score = None
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Evaluation failed: {result}")
                    continue
                    
                if evaluation_types[i] == "safety":
                    safety_score = result
                elif evaluation_types[i] == "ethical":
                    ethical_score = result
                elif evaluation_types[i] == "alignment":
                    alignment_score = result
            
            # Create comprehensive result
            evaluation_result = EvaluationResult(
                id=request_id,
                text=text,
                model=model,
                safety_score=safety_score,
                ethical_score=ethical_score,
                alignment_score=alignment_score,
                metadata=metadata or {},
                timestamp=datetime.utcnow(),
                evaluation_types=evaluation_types
            )
            
            # Cache result
            if self.cache:
                await self.cache.set(cache_key, evaluation_result, ttl=self.config.cache_ttl)
            
            # Record metrics
            if self.metrics:
                await self.metrics.record_evaluation(evaluation_result)
            
            logger.info(f"Completed evaluation {request_id}")
            return evaluation_result
    
    async def batch_evaluate(
        self,
        texts: List[str],
        evaluation_types: List[str] = None,
        model: str = None,
        batch_size: int = 10
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple texts in batches.
        
        Args:
            texts: List of texts to evaluate
            evaluation_types: Types of evaluation to perform
            model: Model to use for evaluation
            batch_size: Size of each batch
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Starting batch evaluation of {len(texts)} texts")
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_tasks = [
                self.evaluate_text(text, evaluation_types, model)
                for text in batch
            ]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch evaluation failed: {result}")
                    continue
                results.append(result)
        
        logger.info(f"Completed batch evaluation with {len(results)} successful results")
        return results
    
    async def analyze_model_behavior(
        self,
        model: str,
        test_prompts: List[str],
        analysis_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyze overall model behavior patterns.
        
        Args:
            model: Model to analyze
            test_prompts: Set of prompts for behavior analysis
            analysis_depth: Depth of analysis (basic, standard, comprehensive)
            
        Returns:
            Behavioral analysis report
        """
        logger.info(f"Starting behavior analysis for model: {model}")
        
        # Evaluate all test prompts
        evaluations = await self.batch_evaluate(
            test_prompts,
            evaluation_types=["safety", "ethical", "alignment"],
            model=model
        )
        
        # Aggregate statistics
        safety_scores = [e.safety_score.overall_score for e in evaluations if e.safety_score]
        ethical_scores = [e.ethical_score.overall_score for e in evaluations if e.ethical_score]
        alignment_scores = [e.alignment_score.overall_score for e in evaluations if e.alignment_score]
        
        analysis = {
            "model": model,
            "total_evaluations": len(evaluations),
            "timestamp": datetime.utcnow(),
            "safety_analysis": {
                "mean_score": np.mean(safety_scores) if safety_scores else None,
                "std_score": np.std(safety_scores) if safety_scores else None,
                "min_score": np.min(safety_scores) if safety_scores else None,
                "max_score": np.max(safety_scores) if safety_scores else None,
            },
            "ethical_analysis": {
                "mean_score": np.mean(ethical_scores) if ethical_scores else None,
                "std_score": np.std(ethical_scores) if ethical_scores else None,
                "min_score": np.min(ethical_scores) if ethical_scores else None,
                "max_score": np.max(ethical_scores) if ethical_scores else None,
            },
            "alignment_analysis": {
                "mean_score": np.mean(alignment_scores) if alignment_scores else None,
                "std_score": np.std(alignment_scores) if alignment_scores else None,
                "min_score": np.min(alignment_scores) if alignment_scores else None,
                "max_score": np.max(alignment_scores) if alignment_scores else None,
            }
        }
        
        if analysis_depth == "comprehensive":
            # Add detailed pattern analysis
            analysis["behavior_patterns"] = await self._analyze_behavior_patterns(evaluations)
            analysis["risk_indicators"] = await self._identify_risk_patterns(evaluations)
        
        logger.info(f"Completed behavior analysis for model: {model}")
        return analysis
    
    async def _evaluate_safety(self, text: str, model: str) -> SafetyScore:
        """Perform safety evaluation."""
        return await self.safety_evaluator.evaluate(text, model)
    
    async def _evaluate_ethical(self, text: str, model: str) -> EthicalScore:
        """Perform ethical evaluation."""
        return await self.ethical_evaluator.evaluate(text, model)
    
    async def _evaluate_alignment(self, text: str, model: str) -> AlignmentScore:
        """Perform alignment evaluation."""
        return await self.alignment_evaluator.evaluate(text, model)
    
    def _generate_cache_key(self, text: str, evaluation_types: List[str], model: str) -> str:
        """Generate cache key for evaluation."""
        import hashlib
        content = f"{text}:{':'.join(sorted(evaluation_types))}:{model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _analyze_behavior_patterns(self, evaluations: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze behavioral patterns in evaluations."""
        # Implementation for pattern analysis
        patterns = {
            "consistency_score": 0.8,  # Placeholder for real implementation
            "bias_indicators": [],
            "safety_trends": [],
            "ethical_tendencies": []
        }
        return patterns
    
    async def _identify_risk_patterns(self, evaluations: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """Identify risk patterns in evaluations."""
        # Implementation for risk pattern identification
        risks = []
        for eval_result in evaluations:
            if eval_result.safety_score and eval_result.safety_score.overall_score < 0.5:
                risks.append({
                    "type": "safety_risk",
                    "score": eval_result.safety_score.overall_score,
                    "text_snippet": eval_result.text[:100] + "..." if len(eval_result.text) > 100 else eval_result.text
                })
        return risks
    
    async def close(self):
        """Clean up resources."""
        await self.openai_client.close()
        if self.anthropic_client:
            await self.anthropic_client.close()
        if self.cache:
            await self.cache.close()
        if self.metrics:
            await self.metrics.close()
        logger.info("InterpretabilityEngine closed successfully") 