"""
Behavior Evaluator - Orchestrates comprehensive behavior evaluation

This module provides the BehaviorEvaluator class that coordinates
safety, ethical, and alignment evaluations for comprehensive model behavior analysis.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import asdict
from datetime import datetime

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from ..models.evaluation import (
    EvaluationResult, 
    BatchEvaluationResult,
    SafetyScore,
    EthicalScore, 
    AlignmentScore
)
from ..evaluation.safety import SafetyEvaluator
from ..evaluation.ethical import EthicalEvaluator
from ..evaluation.alignment import AlignmentEvaluator
from ..utils.config import ConfigManager
from ..utils.cache import CacheManager

logger = logging.getLogger(__name__)


class BehaviorEvaluator:
    """
    Comprehensive behavior evaluator that orchestrates safety, ethical, and alignment evaluations.
    
    This class provides a unified interface for evaluating text across multiple dimensions
    of model behavior, combining results into comprehensive evaluation reports.
    """
    
    def __init__(
        self,
        openai_client: Optional[AsyncOpenAI] = None,
        anthropic_client: Optional[AsyncAnthropic] = None,
        config: Optional[ConfigManager] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """
        Initialize the BehaviorEvaluator.
        
        Args:
            openai_client: Optional OpenAI client instance
            anthropic_client: Optional Anthropic client instance  
            config: Optional configuration manager
            cache_manager: Optional cache manager for results
        """
        self.config = config or ConfigManager()
        self.cache_manager = cache_manager or CacheManager()
        
        # Initialize clients
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=self.config.get("openai.api_key")
        )
        self.anthropic_client = anthropic_client or AsyncAnthropic(
            api_key=self.config.get("anthropic.api_key")
        )
        
        # Initialize evaluators
        self.safety_evaluator = SafetyEvaluator(self.openai_client)
        
        self.ethical_evaluator = EthicalEvaluator(self.openai_client)
        
        self.alignment_evaluator = AlignmentEvaluator(self.openai_client)
        
        logger.info("BehaviorEvaluator initialized successfully")
    
    async def evaluate_text(
        self,
        text: str,
        include_safety: bool = True,
        include_ethical: bool = True,
        include_alignment: bool = True,
        model: str = "gpt-4",
        use_cache: bool = True
    ) -> EvaluationResult:
        """
        Evaluate text across multiple behavioral dimensions.
        
        Args:
            text: Text to evaluate
            include_safety: Whether to include safety evaluation
            include_ethical: Whether to include ethical evaluation
            include_alignment: Whether to include alignment evaluation
            model: Model to use for evaluation
            use_cache: Whether to use cached results
            
        Returns:
            EvaluationResult with comprehensive scores
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache first
        cache_key = f"behavior_eval:{hash(text)}:{model}:{include_safety}:{include_ethical}:{include_alignment}"
        if use_cache:
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for behavior evaluation")
                return EvaluationResult(**cached_result)
        
        # Run evaluations concurrently
        tasks = []
        
        if include_safety:
            tasks.append(self._run_safety_evaluation(text, model))
        if include_ethical:
            tasks.append(self._run_ethical_evaluation(text, model))
        if include_alignment:
            tasks.append(self._run_alignment_evaluation(text, model))
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            safety_score = None
            ethical_score = None
            alignment_score = None
            
            result_index = 0
            if include_safety:
                if isinstance(results[result_index], Exception):
                    logger.error(f"Safety evaluation failed: {results[result_index]}")
                else:
                    safety_score = results[result_index]
                result_index += 1
                
            if include_ethical:
                if isinstance(results[result_index], Exception):
                    logger.error(f"Ethical evaluation failed: {results[result_index]}")
                else:
                    ethical_score = results[result_index]
                result_index += 1
                
            if include_alignment:
                if isinstance(results[result_index], Exception):
                    logger.error(f"Alignment evaluation failed: {results[result_index]}")
                else:
                    alignment_score = results[result_index]
            
            # Create comprehensive result
            evaluation_types = []
            if include_safety:
                evaluation_types.append("safety")
            if include_ethical:
                evaluation_types.append("ethical")
            if include_alignment:
                evaluation_types.append("alignment")
            
            evaluation_result = EvaluationResult(
                id=f"eval_{hash(text)}", 
                text=text,
                model=model,
                safety_score=safety_score,
                ethical_score=ethical_score,
                alignment_score=alignment_score,
                metadata={},
                timestamp=datetime.now(),
                evaluation_types=evaluation_types
            )
            
            # Cache result
            if use_cache:
                await self.cache_manager.set(cache_key, asdict(evaluation_result), ttl=3600)
            
            logger.info(f"Behavior evaluation completed for text length: {len(text)}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Behavior evaluation failed: {e}")
            raise
    
    async def evaluate_batch(
        self,
        texts: List[str],
        include_safety: bool = True,
        include_ethical: bool = True,
        include_alignment: bool = True,
        model: str = "gpt-4",
        use_cache: bool = True,
        max_concurrent: int = 5
    ) -> BatchEvaluationResult:
        """
        Evaluate multiple texts in batch.
        
        Args:
            texts: List of texts to evaluate
            include_safety: Whether to include safety evaluation
            include_ethical: Whether to include ethical evaluation
            include_alignment: Whether to include alignment evaluation
            model: Model to use for evaluation
            use_cache: Whether to use cached results
            max_concurrent: Maximum concurrent evaluations
            
        Returns:
            BatchEvaluationResult with all individual results
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_single(text: str) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate_text(
                    text=text,
                    include_safety=include_safety,
                    include_ethical=include_ethical,
                    include_alignment=include_alignment,
                    model=model,
                    use_cache=use_cache
                )
        
        try:
            # Run all evaluations
            tasks = [evaluate_single(text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            evaluations = []
            failed_evaluations = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Evaluation {i} failed: {result}")
                    failed_evaluations.append({
                        "index": i,
                        "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                        "error": str(result)
                    })
                else:
                    evaluations.append(result)
            
            batch_result = BatchEvaluationResult(
                batch_id=f"batch_{hash(str(texts))}",
                total_items=len(texts),
                completed_items=len(evaluations),
                failed_items=len(failed_evaluations),
                results=evaluations,
                start_time=datetime.now()
            )
            
            logger.info(f"Batch evaluation completed: {len(evaluations)}/{len(texts)} successful")
            return batch_result
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            raise
    
    async def _run_safety_evaluation(self, text: str, model: str) -> SafetyScore:
        """Run safety evaluation."""
        return await self.safety_evaluator.evaluate(text, model)
    
    async def _run_ethical_evaluation(self, text: str, model: str) -> EthicalScore:
        """Run ethical evaluation."""
        return await self.ethical_evaluator.evaluate(text, model)
    
    async def _run_alignment_evaluation(self, text: str, model: str) -> AlignmentScore:
        """Run alignment evaluation."""
        return await self.alignment_evaluator.evaluate(text, model)
    
    def get_evaluation_summary(self, result: EvaluationResult) -> Dict[str, Any]:
        """
        Get a summary of evaluation results.
        
        Args:
            result: EvaluationResult to summarize
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "text_length": len(result.text),
            "model": result.model,
            "evaluation_types": []
        }
        
        if result.safety_score:
            summary["evaluation_types"].append("safety")
            summary["safety_overall"] = result.safety_score.overall_score
            summary["safety_risk_level"] = result.safety_score.risk_level.value
        
        if result.ethical_score:
            summary["evaluation_types"].append("ethical")
            summary["ethical_overall"] = result.ethical_score.overall_score
        
        if result.alignment_score:
            summary["evaluation_types"].append("alignment")
            summary["alignment_overall"] = result.alignment_score.overall_score
        
        # Calculate composite score if multiple evaluations
        scores = []
        if result.safety_score:
            scores.append(result.safety_score.overall_score)
        if result.ethical_score:
            scores.append(result.ethical_score.overall_score)
        if result.alignment_score:
            scores.append(result.alignment_score.overall_score)
        
        if scores:
            summary["composite_score"] = sum(scores) / len(scores)
        
        return summary
    
    async def close(self):
        """Close evaluator and clean up resources."""
        if hasattr(self.openai_client, 'close'):
            await self.openai_client.close()
        if hasattr(self.anthropic_client, 'close'):
            await self.anthropic_client.close()
        logger.info("BehaviorEvaluator closed") 