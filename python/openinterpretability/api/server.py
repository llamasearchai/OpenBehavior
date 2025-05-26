"""
FastAPI server for OpenInterpretability platform.
Provides comprehensive REST API for LLM behavior evaluation and analysis.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from ..core.engine import InterpretabilityEngine, EngineConfig
from ..models.evaluation import EvaluationResult, BatchEvaluationResult
from ..utils.config import get_config
from ..utils.auth import get_api_key, verify_api_key
from ..utils.rate_limit import RateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class EvaluateTextRequest(BaseModel):
    """Request model for text evaluation."""
    text: str = Field(..., min_length=1, max_length=50000, description="Text to evaluate")
    evaluation_types: List[str] = Field(default=["safety", "ethical", "alignment"], description="Types of evaluation to perform")
    model: Optional[str] = Field(default=None, description="Model to use for evaluation")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class BatchEvaluateRequest(BaseModel):
    """Request model for batch evaluation."""
    texts: List[str] = Field(..., min_length=1, max_length=1000, description="List of texts to evaluate")
    evaluation_types: List[str] = Field(default=["safety", "ethical", "alignment"], description="Types of evaluation to perform")
    model: Optional[str] = Field(default=None, description="Model to use for evaluation")
    batch_size: Optional[int] = Field(default=10, ge=1, le=50, description="Batch processing size")


class ModelAnalysisRequest(BaseModel):
    """Request model for model behavior analysis."""
    model: str = Field(..., description="Model to analyze")
    test_prompts: List[str] = Field(..., min_length=5, max_length=500, description="Test prompts for analysis")
    analysis_depth: str = Field(default="comprehensive", pattern="^(basic|standard|comprehensive)$", description="Analysis depth")


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    id: str
    text: str
    model: str
    safety_score: Optional[Dict[str, Any]]
    ethical_score: Optional[Dict[str, Any]]
    alignment_score: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str
    evaluation_types: List[str]
    processing_time: Optional[float]
    overall_risk_level: str
    is_acceptable: bool
    summary_scores: Dict[str, float]


class BatchJobResponse(BaseModel):
    """Response model for batch job status."""
    batch_id: str
    total_items: int
    completed_items: int
    failed_items: int
    success_rate: float
    is_completed: bool
    start_time: str
    end_time: Optional[str]
    errors: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Metrics response."""
    total_evaluations: int
    successful_evaluations: int
    failed_evaluations: int
    average_processing_time: float
    safety_score_distribution: Dict[str, int]
    ethical_score_distribution: Dict[str, int]
    alignment_score_distribution: Dict[str, int]


# Global state
engine: Optional[InterpretabilityEngine] = None
batch_jobs: Dict[str, BatchEvaluationResult] = {}
rate_limiter = RateLimiter()
start_time = time.time()


def get_engine() -> InterpretabilityEngine:
    """Get the global engine instance."""
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized"
        )
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global engine
    
    # Startup
    try:
        config_dict = get_config()
        engine_config = EngineConfig(
            openai_api_key=config_dict["openai"]["api_key"],
            anthropic_api_key=config_dict.get("anthropic", {}).get("api_key"),
            default_model=config_dict.get("model", {}).get("default", "gpt-4"),
            max_concurrent_evaluations=config_dict.get("engine", {}).get("max_concurrent", 10),
            enable_caching=config_dict.get("cache", {}).get("enabled", True),
            cache_ttl=config_dict.get("cache", {}).get("ttl", 3600),
            metrics_enabled=config_dict.get("metrics", {}).get("enabled", True)
        )
        
        engine = InterpretabilityEngine(config=engine_config)
        logger.info("OpenInterpretability engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise
    
    yield
    
    # Shutdown
    if engine:
        await engine.close()
        logger.info("OpenInterpretability engine closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="OpenInterpretability API",
        description="Advanced LLM Behavior Analysis and Interpretability Platform",
        version="1.0.0",
        contact={
            "name": "Nik Jois",
            "email": "nikjois@llamasearch.ai"
        },
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint providing basic information."""
        return {
            "name": "OpenInterpretability API",
            "version": "1.0.0",
            "description": "Advanced LLM Behavior Analysis and Interpretability Platform",
            "author": "Nik Jois <nikjois@llamasearch.ai>",
            "docs": "/docs",
            "health": "/health"
        }

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            uptime_seconds=time.time() - start_time
        )

    @app.post("/evaluate", response_model=EvaluationResponse)
    async def evaluate_text_endpoint(
        request: EvaluateTextRequest,
        background_tasks: BackgroundTasks,
        engine: InterpretabilityEngine = Depends(get_engine),
        api_key: str = Depends(get_api_key)
    ):
        """
        Evaluate text across specified dimensions.
        
        Performs comprehensive analysis including safety, ethical, and alignment evaluation.
        """
        # Rate limiting
        if not await rate_limiter.check_rate_limit(api_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        try:
            start_eval_time = time.time()
            
            # Perform evaluation
            result = await engine.evaluate_text(
                text=request.text,
                evaluation_types=request.evaluation_types,
                model=request.model,
                metadata=request.metadata
            )
            
            # Calculate processing time
            processing_time = time.time() - start_eval_time
            result.processing_time = processing_time
            
            # Convert to response model
            response_data = result.to_dict()
            response = EvaluationResponse(**response_data)
            
            logger.info(f"Evaluation completed successfully: {result.id}")
            return response
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Evaluation failed: {str(e)}"
            )

    @app.post("/evaluate/batch", response_model=BatchJobResponse)
    async def start_batch_evaluation(
        request: BatchEvaluateRequest,
        background_tasks: BackgroundTasks,
        engine: InterpretabilityEngine = Depends(get_engine),
        api_key: str = Depends(get_api_key)
    ):
        """
        Start batch evaluation of multiple texts.
        
        Returns immediately with batch job ID for tracking progress.
        """
        # Rate limiting
        if not await rate_limiter.check_rate_limit(api_key, weight=len(request.texts)):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        try:
            batch_id = str(uuid.uuid4())
            
            # Initialize batch job
            batch_job = BatchEvaluationResult(
                batch_id=batch_id,
                total_items=len(request.texts),
                completed_items=0,
                failed_items=0,
                results=[],
                start_time=datetime.utcnow()
            )
            
            batch_jobs[batch_id] = batch_job
            
            # Start batch processing in background
            background_tasks.add_task(
                process_batch_evaluation,
                batch_id,
                request.texts,
                request.evaluation_types,
                request.model,
                request.batch_size,
                engine
            )
            
            logger.info(f"Started batch evaluation: {batch_id}")
            
            return BatchJobResponse(
                batch_id=batch_id,
                total_items=batch_job.total_items,
                completed_items=batch_job.completed_items,
                failed_items=batch_job.failed_items,
                success_rate=batch_job.success_rate,
                is_completed=batch_job.is_completed,
                start_time=batch_job.start_time.isoformat(),
                end_time=None,
                errors=batch_job.errors
            )
            
        except Exception as e:
            logger.error(f"Failed to start batch evaluation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start batch evaluation: {str(e)}"
            )

    @app.get("/evaluate/batch/{batch_id}", response_model=BatchJobResponse)
    async def get_batch_status_endpoint(
        batch_id: str,
        api_key: str = Depends(get_api_key)
    ):
        """Get status of batch evaluation job."""
        if batch_id not in batch_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Batch job not found"
            )
        
        batch_job = batch_jobs[batch_id]
        
        return BatchJobResponse(
            batch_id=batch_id,
            total_items=batch_job.total_items,
            completed_items=batch_job.completed_items,
            failed_items=batch_job.failed_items,
            success_rate=batch_job.success_rate,
            is_completed=batch_job.is_completed,
            start_time=batch_job.start_time.isoformat(),
            end_time=batch_job.end_time.isoformat() if batch_job.end_time else None,
            errors=batch_job.errors
        )

    @app.get("/evaluate/batch/{batch_id}/results")
    async def get_batch_results(
        batch_id: str,
        offset: int = 0,
        limit: int = 100,
        api_key: str = Depends(get_api_key)
    ):
        """Get results from batch evaluation."""
        if batch_id not in batch_jobs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Batch job not found"
            )
        
        batch_job = batch_jobs[batch_id]
        results = batch_job.results[offset:offset + limit]
        
        return {
            "batch_id": batch_id,
            "total_results": len(batch_job.results),
            "offset": offset,
            "limit": limit,
            "results": [result.to_dict() for result in results]
        }

    @app.post("/analyze/model")
    async def analyze_model_behavior(
        request: ModelAnalysisRequest,
        background_tasks: BackgroundTasks,
        engine: InterpretabilityEngine = Depends(get_engine),
        api_key: str = Depends(get_api_key)
    ):
        """
        Analyze model behavior patterns across test prompts.
        
        Provides comprehensive behavioral analysis including consistency,
        bias detection, and risk pattern identification.
        """
        # Rate limiting
        if not await rate_limiter.check_rate_limit(api_key, weight=len(request.test_prompts)):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        try:
            analysis = await engine.analyze_model_behavior(
                model=request.model,
                test_prompts=request.test_prompts,
                analysis_depth=request.analysis_depth
            )
            
            logger.info(f"Model analysis completed for: {request.model}")
            return analysis
            
        except Exception as e:
            logger.error(f"Model analysis failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model analysis failed: {str(e)}"
            )

    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics(api_key: str = Depends(get_api_key)):
        """Get platform metrics and statistics."""
        try:
            # This would typically come from a metrics store
            # For now, return mock data
            return MetricsResponse(
                total_evaluations=1000,
                successful_evaluations=950,
                failed_evaluations=50,
                average_processing_time=2.5,
                safety_score_distribution={"0.0-0.2": 10, "0.2-0.4": 15, "0.4-0.6": 25, "0.6-0.8": 30, "0.8-1.0": 20},
                ethical_score_distribution={"0.0-0.2": 5, "0.2-0.4": 10, "0.4-0.6": 20, "0.6-0.8": 35, "0.8-1.0": 30},
                alignment_score_distribution={"0.0-0.2": 8, "0.2-0.4": 12, "0.4-0.6": 22, "0.6-0.8": 33, "0.8-1.0": 25}
            )
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get metrics: {str(e)}"
            )

    @app.get("/models")
    async def list_supported_models(api_key: str = Depends(get_api_key)):
        """List supported models for evaluation."""
        return {
            "supported_models": [
                {
                    "name": "gpt-4",
                    "provider": "openai",
                    "capabilities": ["safety", "ethical", "alignment"],
                    "description": "GPT-4 model with comprehensive evaluation capabilities"
                },
                {
                    "name": "gpt-3.5-turbo",
                    "provider": "openai",
                    "capabilities": ["safety", "ethical", "alignment"],
                    "description": "GPT-3.5 Turbo model for faster evaluation"
                },
                {
                    "name": "claude-3-opus",
                    "provider": "anthropic",
                    "capabilities": ["safety", "ethical", "alignment"],
                    "description": "Claude 3 Opus model with advanced reasoning"
                }
            ]
        }
    
    return app


# Create the app instance
app = create_app()


async def process_batch_evaluation(
    batch_id: str,
    texts: List[str],
    evaluation_types: List[str],
    model: Optional[str],
    batch_size: int,
    engine: InterpretabilityEngine
):
    """Process batch evaluation in background."""
    try:
        batch_job = batch_jobs[batch_id]
        
        # Process in batches
        results = await engine.batch_evaluate(
            texts=texts,
            evaluation_types=evaluation_types,
            model=model,
            batch_size=batch_size
        )
        
        # Update batch job
        batch_job.results = results
        batch_job.completed_items = len([r for r in results if r])
        batch_job.failed_items = len(texts) - batch_job.completed_items
        batch_job.end_time = datetime.utcnow()
        
        logger.info(f"Batch evaluation completed: {batch_id}")
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {batch_id}: {e}")
        batch_job = batch_jobs[batch_id]
        batch_job.errors.append(str(e))
        batch_job.failed_items = len(texts)
        batch_job.end_time = datetime.utcnow()


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "openinterpretability.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server() 