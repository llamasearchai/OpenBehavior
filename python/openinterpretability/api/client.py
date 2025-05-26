"""
OpenInterpretability Client - Python client for OpenInterpretability API

This module provides a convenient Python client for interacting with the
OpenInterpretability API server, enabling easy integration into other applications.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import json

from ..models.evaluation import EvaluationResult, BatchEvaluationResult
from ..core.analyzer import InterpretabilityReport, ModelComparison

logger = logging.getLogger(__name__)


class OpenInterpretabilityClient:
    """
    Python client for OpenInterpretability API.
    
    Provides convenient methods for evaluating text, analyzing models,
    and retrieving interpretability insights from the API server.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the OpenInterpretability client.
        
        Args:
            base_url: Base URL of the OpenInterpretability API server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Headers for requests
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "OpenInterpretability-Python-Client/1.0.0"
        }
        
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        logger.info(f"OpenInterpretability client initialized for {base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers
            )
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limited
                        if attempt < self.max_retries:
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    # Handle error responses
                    error_data = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=error_data
                    )
                    
            except aiohttp.ClientError as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed after {self.max_retries} retries: {e}")
                    raise
        
        raise Exception(f"Failed to make request after {self.max_retries} retries")
    
    async def evaluate_text(
        self,
        text: str,
        evaluation_types: Optional[List[str]] = None,
        model: str = "gpt-4",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single text for behavior analysis.
        
        Args:
            text: Text to evaluate
            evaluation_types: Types of evaluation to perform (safety, ethical, alignment)
            model: Model to use for evaluation
            metadata: Additional metadata
            
        Returns:
            Evaluation result as dictionary
        """
        if evaluation_types is None:
            evaluation_types = ["safety", "ethical", "alignment"]
            
        data = {
            "text": text,
            "evaluation_types": evaluation_types,
            "model": model,
            "metadata": metadata or {}
        }
        
        response = await self._make_request("POST", "/evaluate", data=data)
        return response
    
    async def batch_evaluate(
        self,
        texts: List[str],
        evaluation_types: Optional[List[str]] = None,
        model: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Start batch evaluation of multiple texts.
        
        Args:
            texts: List of texts to evaluate
            evaluation_types: Types of evaluation to perform
            model: Model to use for evaluation
            batch_size: Batch processing size
            
        Returns:
            Batch job information
        """
        if evaluation_types is None:
            evaluation_types = ["safety", "ethical", "alignment"]
            
        data = {
            "texts": texts,
            "evaluation_types": evaluation_types,
            "model": model,
            "batch_size": batch_size
        }
        
        response = await self._make_request("POST", "/evaluate/batch", data=data)
        return response
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get status of batch evaluation job.
        
        Args:
            batch_id: Batch job ID
            
        Returns:
            Batch job status
        """
        response = await self._make_request("GET", f"/evaluate/batch/{batch_id}")
        return response
    
    async def analyze_model(
        self,
        model: str,
        test_prompts: List[str],
        analysis_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Analyze model behavior patterns.
        
        Args:
            model: Model to analyze
            test_prompts: Test prompts for analysis
            analysis_depth: Analysis depth (basic, standard, comprehensive)
            
        Returns:
            Model analysis results
        """
        data = {
            "model": model,
            "test_prompts": test_prompts,
            "analysis_depth": analysis_depth
        }
        
        response = await self._make_request("POST", "/analyze/model", data=data)
        return response
    
    async def compare_models(
        self,
        model_a: str,
        model_b: str,
        test_prompts: List[str],
        comparison_dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare two models across test prompts.
        
        Args:
            model_a: First model to compare
            model_b: Second model to compare
            test_prompts: Test prompts for comparison
            comparison_dimensions: Dimensions to compare
            
        Returns:
            Model comparison results
        """
        data = {
            "model_a": model_a,
            "model_b": model_b,
            "test_prompts": test_prompts,
            "comparison_dimensions": comparison_dimensions
        }
        
        response = await self._make_request("POST", "/analyze/compare", data=data)
        return response
    
    async def get_supported_models(self) -> List[str]:
        """
        Get list of supported models.
        
        Returns:
            List of supported model names
        """
        response = await self._make_request("GET", "/models")
        return response.get("models", [])
    
    async def get_health(self) -> Dict[str, Any]:
        """
        Get API health status.
        
        Returns:
            Health status information
        """
        response = await self._make_request("GET", "/health")
        return response
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get API usage metrics.
        
        Returns:
            Usage metrics and statistics
        """
        response = await self._make_request("GET", "/metrics")
        return response
    
    async def generate_api_key(self, description: str = "Generated via client") -> str:
        """
        Generate a new API key (requires admin privileges).
        
        Args:
            description: Description for the API key
            
        Returns:
            Generated API key
        """
        data = {"description": description}
        response = await self._make_request("POST", "/auth/generate-key", data=data)
        return response.get("api_key", "")
    
    async def validate_api_key(self, api_key: Optional[str] = None) -> bool:
        """
        Validate API key.
        
        Args:
            api_key: API key to validate (uses client's key if not provided)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            response = await self._make_request("GET", "/health")
            return response.get("status") == "healthy"
        except Exception:
            return False
    
    async def close(self):
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("OpenInterpretability client session closed")


class SyncOpenInterpretabilityClient:
    """
    Synchronous wrapper for OpenInterpretabilityClient.
    
    Provides the same functionality as the async client but with
    synchronous method calls for easier integration in sync codebases.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the synchronous OpenInterpretability client.
        
        Args:
            base_url: Base URL of the OpenInterpretability API server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self._async_client = OpenInterpretabilityClient(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
    
    def evaluate_text(
        self,
        text: str,
        evaluation_types: Optional[List[str]] = None,
        model: str = "gpt-4",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous version of evaluate_text."""
        return asyncio.run(self._async_client.evaluate_text(
            text=text,
            evaluation_types=evaluation_types,
            model=model,
            metadata=metadata
        ))
    
    def batch_evaluate(
        self,
        texts: List[str],
        evaluation_types: Optional[List[str]] = None,
        model: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Synchronous version of batch_evaluate."""
        return asyncio.run(self._async_client.batch_evaluate(
            texts=texts,
            evaluation_types=evaluation_types,
            model=model,
            batch_size=batch_size
        ))
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Synchronous version of get_batch_status."""
        return asyncio.run(self._async_client.get_batch_status(batch_id))
    
    def analyze_model(
        self,
        model: str,
        test_prompts: List[str],
        analysis_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Synchronous version of analyze_model."""
        return asyncio.run(self._async_client.analyze_model(
            model=model,
            test_prompts=test_prompts,
            analysis_depth=analysis_depth
        ))
    
    def compare_models(
        self,
        model_a: str,
        model_b: str,
        test_prompts: List[str],
        comparison_dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Synchronous version of compare_models."""
        return asyncio.run(self._async_client.compare_models(
            model_a=model_a,
            model_b=model_b,
            test_prompts=test_prompts,
            comparison_dimensions=comparison_dimensions
        ))
    
    def get_supported_models(self) -> List[str]:
        """Synchronous version of get_supported_models."""
        return asyncio.run(self._async_client.get_supported_models())
    
    def get_health(self) -> Dict[str, Any]:
        """Synchronous version of get_health."""
        return asyncio.run(self._async_client.get_health())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Synchronous version of get_metrics."""
        return asyncio.run(self._async_client.get_metrics())
    
    def generate_api_key(self, description: str = "Generated via client") -> str:
        """Synchronous version of generate_api_key."""
        return asyncio.run(self._async_client.generate_api_key(description))
    
    def validate_api_key(self, api_key: Optional[str] = None) -> bool:
        """Synchronous version of validate_api_key."""
        return asyncio.run(self._async_client.validate_api_key(api_key))
    
    def close(self):
        """Close the client session."""
        asyncio.run(self._async_client.close())
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 