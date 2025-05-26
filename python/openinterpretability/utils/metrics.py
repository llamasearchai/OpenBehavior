"""
Metrics collection utilities for OpenInterpretability platform.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime
from dataclasses import dataclass, field

from ..models.evaluation import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation."""
    evaluation_id: str
    processing_time: float
    model: str
    evaluation_types: List[str]
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsCollector:
    """
    Metrics collector for tracking evaluation performance and results.
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.evaluations: deque = deque()
        self.error_counts = defaultdict(int)
        self.model_usage = defaultdict(int)
        self.evaluation_type_usage = defaultdict(int)
        self.processing_times = defaultdict(list)
        
        # Real-time statistics
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        
        logger.info("MetricsCollector initialized")
    
    async def record_evaluation(self, result: EvaluationResult, processing_time: float = None):
        """
        Record metrics for an evaluation.
        
        Args:
            result: EvaluationResult to record metrics for
            processing_time: Optional processing time in seconds
        """
        metrics = EvaluationMetrics(
            evaluation_id=result.id,
            processing_time=processing_time or result.processing_time or 0.0,
            model=result.model,
            evaluation_types=result.evaluation_types,
            success=True
        )
        
        await self._record_metrics(metrics)
        
    async def record_error(self, error: str, model: str = None, evaluation_types: List[str] = None):
        """
        Record metrics for a failed evaluation.
        
        Args:
            error: Error message
            model: Model that failed
            evaluation_types: Types of evaluation attempted
        """
        metrics = EvaluationMetrics(
            evaluation_id=f"error_{int(time.time())}",
            processing_time=0.0,
            model=model or "unknown",
            evaluation_types=evaluation_types or [],
            success=False,
            error_message=error
        )
        
        await self._record_metrics(metrics)
        
    async def _record_metrics(self, metrics: EvaluationMetrics):
        """Internal method to record metrics."""
        # Add to deque with automatic cleanup
        self.evaluations.append(metrics)
        await self._cleanup_old_metrics()
        
        # Update counters
        self.total_evaluations += 1
        
        if metrics.success:
            self.successful_evaluations += 1
            
            # Record usage statistics
            self.model_usage[metrics.model] += 1
            for eval_type in metrics.evaluation_types:
                self.evaluation_type_usage[eval_type] += 1
            
            # Record processing time
            self.processing_times[metrics.model].append(metrics.processing_time)
            
        else:
            self.failed_evaluations += 1
            self.error_counts[metrics.error_message or "unknown"] += 1
    
    async def _cleanup_old_metrics(self):
        """Remove old metrics outside retention window."""
        cutoff_time = datetime.utcnow().timestamp() - (self.retention_hours * 3600)
        
        while self.evaluations and self.evaluations[0].timestamp.timestamp() < cutoff_time:
            self.evaluations.popleft()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        success_rate = (
            self.successful_evaluations / self.total_evaluations 
            if self.total_evaluations > 0 else 0.0
        )
        
        # Calculate average processing times
        avg_processing_times = {}
        for model, times in self.processing_times.items():
            if times:
                avg_processing_times[model] = sum(times) / len(times)
        
        return {
            "total_evaluations": self.total_evaluations,
            "successful_evaluations": self.successful_evaluations,
            "failed_evaluations": self.failed_evaluations,
            "success_rate": success_rate,
            "model_usage": dict(self.model_usage),
            "evaluation_type_usage": dict(self.evaluation_type_usage),
            "average_processing_times": avg_processing_times,
            "error_counts": dict(self.error_counts)
        }
    
    def get_recent_stats(self, hours: int = 1) -> Dict[str, Any]:
        """Get statistics for recent evaluations."""
        cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
        
        recent_evaluations = [
            eval_metric for eval_metric in self.evaluations
            if eval_metric.timestamp.timestamp() >= cutoff_time
        ]
        
        total_recent = len(recent_evaluations)
        successful_recent = sum(1 for e in recent_evaluations if e.success)
        failed_recent = total_recent - successful_recent
        
        recent_processing_times = [
            e.processing_time for e in recent_evaluations 
            if e.success and e.processing_time > 0
        ]
        
        avg_processing_time = (
            sum(recent_processing_times) / len(recent_processing_times)
            if recent_processing_times else 0.0
        )
        
        return {
            "period_hours": hours,
            "total_evaluations": total_recent,
            "successful_evaluations": successful_recent,
            "failed_evaluations": failed_recent,
            "success_rate": successful_recent / total_recent if total_recent > 0 else 0.0,
            "average_processing_time": avg_processing_time
        }
    
    def get_model_performance(self, model: str) -> Dict[str, Any]:
        """Get performance statistics for a specific model."""
        model_evaluations = [
            eval_metric for eval_metric in self.evaluations
            if eval_metric.model == model
        ]
        
        total_model = len(model_evaluations)
        successful_model = sum(1 for e in model_evaluations if e.success)
        failed_model = total_model - successful_model
        
        processing_times = [
            e.processing_time for e in model_evaluations
            if e.success and e.processing_time > 0
        ]
        
        return {
            "model": model,
            "total_evaluations": total_model,
            "successful_evaluations": successful_model,
            "failed_evaluations": failed_model,
            "success_rate": successful_model / total_model if total_model > 0 else 0.0,
            "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0.0,
            "min_processing_time": min(processing_times) if processing_times else 0.0,
            "max_processing_time": max(processing_times) if processing_times else 0.0
        }
    
    async def close(self):
        """Clean up metrics collector."""
        self.evaluations.clear()
        self.error_counts.clear()
        self.model_usage.clear()
        self.evaluation_type_usage.clear()
        self.processing_times.clear()
        logger.info("MetricsCollector closed") 