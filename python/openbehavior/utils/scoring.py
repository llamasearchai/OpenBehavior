"""
Scoring utilities for evaluation metrics and confidence intervals.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.stats as stats


@dataclass
class ScoreResult:
    """Result of a scoring operation."""
    score: float
    confidence: float
    metadata: Dict[str, float]


class WeightedScorer:
    """Weighted scoring system for multi-dimensional evaluations."""
    
    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        self.default_weights = default_weights or {
            "safety": 0.3,
            "ethical": 0.3,
            "alignment": 0.25,
            "quality": 0.15
        }
    
    def calculate_weighted_score(
        self,
        scores: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate weighted average score."""
        weights = weights or self.default_weights
        
        total_score = 0.0
        total_weight = 0.0
        
        for dimension, score in scores.items():
            weight = weights.get(dimension, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def calculate_confidence_weighted_score(
        self,
        scores: Dict[str, float],
        confidences: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """Calculate score weighted by both importance and confidence."""
        weights = weights or self.default_weights
        
        total_score = 0.0
        total_weight = 0.0
        avg_confidence = 0.0
        
        for dimension, score in scores.items():
            importance_weight = weights.get(dimension, 0.1)
            confidence = confidences.get(dimension, 0.5)
            
            # Combined weight considers both importance and confidence
            combined_weight = importance_weight * confidence
            
            total_score += score * combined_weight
            total_weight += combined_weight
            avg_confidence += confidence
        
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        final_confidence = avg_confidence / len(scores) if scores else 0.0
        
        return final_score, final_confidence


class ConfidenceInterval:
    """Calculate confidence intervals for evaluation scores."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def calculate_interval(
        self,
        scores: List[float],
        method: str = "bootstrap"
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a list of scores."""
        if not scores:
            return (0.0, 0.0)
        
        if method == "bootstrap":
            return self._bootstrap_interval(scores)
        elif method == "normal":
            return self._normal_interval(scores)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _bootstrap_interval(
        self,
        scores: List[float],
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        scores_array = np.array(scores)
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(
                scores_array, 
                size=len(scores_array), 
                replace=True
            )
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def _normal_interval(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate normal distribution confidence interval."""
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std_error = stats.sem(scores_array)
        
        alpha = 1 - self.confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, len(scores) - 1)
        
        margin_error = t_critical * std_error
        
        return (mean - margin_error, mean + margin_error)
    
    def assess_score_reliability(
        self,
        scores: List[float],
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """Assess reliability of scores based on confidence interval width."""
        if len(scores) < 2:
            return {"reliability": 0.0, "interval_width": 1.0}
        
        lower, upper = self.calculate_interval(scores)
        interval_width = upper - lower
        
        # Reliability is inversely related to interval width
        reliability = max(0.0, 1.0 - (interval_width / threshold))
        
        return {
            "reliability": reliability,
            "interval_width": interval_width,
            "lower_bound": lower,
            "upper_bound": upper,
            "mean": np.mean(scores),
            "std": np.std(scores)
        }


class MetricAggregator:
    """Aggregate multiple evaluation metrics into composite scores."""
    
    def __init__(self):
        self.scorer = WeightedScorer()
        self.confidence_calculator = ConfidenceInterval()
    
    def aggregate_evaluation_results(
        self,
        results: List[Dict[str, float]],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Aggregate multiple evaluation results."""
        if not results:
            return {}
        
        # Collect scores by dimension
        dimension_scores = {}
        for result in results:
            for dimension, score in result.items():
                if dimension not in dimension_scores:
                    dimension_scores[dimension] = []
                dimension_scores[dimension].append(score)
        
        # Calculate aggregated metrics
        aggregated = {}
        
        for dimension, scores in dimension_scores.items():
            aggregated[f"{dimension}_mean"] = np.mean(scores)
            aggregated[f"{dimension}_std"] = np.std(scores)
            aggregated[f"{dimension}_min"] = np.min(scores)
            aggregated[f"{dimension}_max"] = np.max(scores)
            
            # Confidence interval
            lower, upper = self.confidence_calculator.calculate_interval(scores)
            aggregated[f"{dimension}_ci_lower"] = lower
            aggregated[f"{dimension}_ci_upper"] = upper
        
        # Overall weighted score
        mean_scores = {
            dim: aggregated[f"{dim}_mean"] 
            for dim in dimension_scores.keys()
        }
        
        aggregated["overall_score"] = self.scorer.calculate_weighted_score(
            mean_scores, weights
        )
        
        return aggregated
    
    def calculate_trend_metrics(
        self,
        time_series_results: List[Tuple[str, Dict[str, float]]]
    ) -> Dict[str, float]:
        """Calculate trend metrics over time."""
        if len(time_series_results) < 2:
            return {}
        
        # Extract timestamps and scores
        timestamps = [result[0] for result in time_series_results]
        overall_scores = []
        
        for _, scores in time_series_results:
            overall_score = self.scorer.calculate_weighted_score(scores)
            overall_scores.append(overall_score)
        
        # Calculate trend
        x = np.arange(len(overall_scores))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, overall_scores)
        
        return {
            "trend_slope": slope,
            "trend_r_squared": r_value ** 2,
            "trend_p_value": p_value,
            "trend_direction": "improving" if slope > 0 else "declining" if slope < 0 else "stable",
            "volatility": np.std(overall_scores),
            "latest_score": overall_scores[-1],
            "score_change": overall_scores[-1] - overall_scores[0] if len(overall_scores) > 1 else 0
        } 