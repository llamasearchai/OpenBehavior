"""
OpenInterpretability: Advanced LLM Behavior Analysis and Interpretability Platform

A comprehensive framework for evaluating, analyzing, and interpreting language model behavior
with focus on safety, ethics, alignment, and transparency.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

__version__ = "1.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

from .core.engine import InterpretabilityEngine
from .core.evaluator import BehaviorEvaluator
from .core.analyzer import ModelAnalyzer
from .api.client import OpenInterpretabilityClient

__all__ = [
    "InterpretabilityEngine",
    "BehaviorEvaluator", 
    "ModelAnalyzer",
    "OpenInterpretabilityClient"
] 