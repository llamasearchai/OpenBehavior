"""
Request models for OpenInterpretability API.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class EvaluationRequest:
    """Request for single text evaluation."""
    text: str
    evaluation_types: List[str] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.evaluation_types is None:
            self.evaluation_types = ["safety", "ethical", "alignment"]


@dataclass 
class BatchEvaluationRequest:
    """Request for batch text evaluation."""
    texts: List[str]
    evaluation_types: List[str] = None
    model: Optional[str] = None
    batch_size: int = 10
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.evaluation_types is None:
            self.evaluation_types = ["safety", "ethical", "alignment"]


@dataclass
class ModelAnalysisRequest:
    """Request for model behavior analysis."""
    model: str
    test_prompts: List[str]
    analysis_depth: str = "comprehensive"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelComparisonRequest:
    """Request for comparing two models."""
    model_a: str
    model_b: str
    test_prompts: List[str]
    comparison_dimensions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.comparison_dimensions is None:
            self.comparison_dimensions = ["safety", "ethical", "alignment"] 