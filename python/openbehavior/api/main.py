"""
FastAPI server for OpenBehavior platform.
"""

import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import uuid
from datetime import datetime

from ..utils.config import Config
from ..utils.logging import get_logger
from ..models.interface import ModelFactory
from ..evaluation.ethical import EthicalEvaluator, EthicalEvalConfig
from ..evaluation.safety import SafetyEvaluator
from ..evaluation.alignment import AlignmentEvaluator
from ..prompts.engineer import PromptEngineer
from ..prompts.templates import PromptLibrary

logger = get_logger(__name__)

# Pydantic models for API
class EvaluationRequest(BaseModel):
    text: str
    text_id: Optional[str] = None
    evaluation_types: List[str] = Field(default=["ethical", "safety", "alignment"])
    evaluator_model: Optional[str] = None

class EvaluationResponse(BaseModel):
    text_id: str
    evaluation_results: Dict[str, Any]
    timestamp: datetime
    status: str

class PromptTestRequest(BaseModel):
    template: str
    variables: Dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024

class PromptTestResponse(BaseModel):
    response: str
    metadata: Dict[str, Any]

class BatchEvaluationRequest(BaseModel):
    texts: Dict[str, str]
    evaluation_types: List[str] = Field(default=["ethical", "safety", "alignment"])
    evaluator_model: Optional[str] = None

# Global state
app_state = {
    "config": None,
    "model_interface": None,
    "evaluators": {},
    "prompt_library": None,
    "prompt_engineer": None,
    "background_tasks": {}
}

def get_config() -> Config:
    """Get application configuration."""
    if app_state["config"] is None:
        raise HTTPException(status_code=500, detail="Configuration not initialized")
    return app_state["config"]

def get_model_interface():
    """Get model interface."""
    if app_state["model_interface"] is None:
        raise HTTPException(status_code=500, detail="Model interface not initialized")
    return app_state["model_interface"]

def get_evaluator(evaluation_type: str):
    """Get evaluator by type."""
    if evaluation_type not in app_state["evaluators"]:
        raise HTTPException(status_code=400, detail=f"Unknown evaluation type: {evaluation_type}")
    return app_state["evaluators"][evaluation_type]

def create_app(config: Config) -> FastAPI:
    """Create FastAPI application with configuration."""
    
    app = FastAPI(
        title="OpenBehavior API",
        description="API for evaluating and aligning LLM behavior",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize application state
    app_state["config"] = config
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize components on startup."""
        logger.info("Starting OpenBehavior API server")
        
        # Initialize model interface
        app_state["model_interface"] = ModelFactory.create(
            provider=config.model.provider,
            api_key=config.model.api_key,
            rate_limit=config.model.rate_limit,
            enable_cache=config.enable_cache,
            cache_ttl=config.cache_ttl
        )
        
        # Initialize evaluators
        app_state["evaluators"]["ethical"] = EthicalEvaluator(
            config=EthicalEvalConfig(
                evaluator_model=config.evaluation.evaluator_model,
                confidence_threshold=config.evaluation.confidence_threshold
            ),
            model_interface=app_state["model_interface"]
        )
        
        app_state["evaluators"]["safety"] = SafetyEvaluator(
            model_interface=app_state["model_interface"],
            evaluator_model=config.evaluation.evaluator_model
        )
        
        app_state["evaluators"]["alignment"] = AlignmentEvaluator(
            model_interface=app_state["model_interface"],
            evaluator_model=config.evaluation.evaluator_model
        )
        
        # Initialize prompt components
        app_state["prompt_library"] = PromptLibrary()
        app_state["prompt_library"].load_from_directory(config.data.template_dir)
        
        app_state["prompt_engineer"] = PromptEngineer(
            model_interface=app_state["model_interface"],
            prompt_library=app_state["prompt_library"]
        )
        
        logger.info("OpenBehavior API server started successfully")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "OpenBehavior API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "model_interface": app_state["model_interface"] is not None,
                "evaluators": len(app_state["evaluators"]) > 0,
                "prompt_library": app_state["prompt_library"] is not None
            }
        }
    
    @app.post("/evaluate", response_model=EvaluationResponse)
    async def evaluate_text(request: EvaluationRequest):
        """Evaluate text for ethical, safety, and alignment considerations."""
        text_id = request.text_id or str(uuid.uuid4())
        
        try:
            results = {}
            
            # Run requested evaluations
            for eval_type in request.evaluation_types:
                if eval_type in app_state["evaluators"]:
                    evaluator = app_state["evaluators"][eval_type]
                    result = await evaluator.evaluate(text_id, request.text)
                    results[eval_type] = result.to_dict()
                else:
                    logger.warning(f"Unknown evaluation type: {eval_type}")
            
            return EvaluationResponse(
                text_id=text_id,
                evaluation_results=results,
                timestamp=datetime.now(),
                status="completed"
            )
        
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/evaluate/batch")
    async def batch_evaluate(
        request: BatchEvaluationRequest,
        background_tasks: BackgroundTasks
    ):
        """Run batch evaluation in the background."""
        task_id = str(uuid.uuid4())
        
        async def run_batch_evaluation():
            """Background task for batch evaluation."""
            try:
                app_state["background_tasks"][task_id] = {
                    "status": "running",
                    "progress": 0,
                    "total": len(request.texts),
                    "results": {},
                    "started_at": datetime.now()
                }
                
                completed = 0
                
                for text_id, text in request.texts.items():
                    try:
                        results = {}
                        
                        for eval_type in request.evaluation_types:
                            if eval_type in app_state["evaluators"]:
                                evaluator = app_state["evaluators"][eval_type]
                                result = await evaluator.evaluate(text_id, text)
                                results[eval_type] = result.to_dict()
                        
                        app_state["background_tasks"][task_id]["results"][text_id] = results
                        completed += 1
                        app_state["background_tasks"][task_id]["progress"] = completed
                        
                    except Exception as e:
                        logger.error(f"Error evaluating {text_id}: {e}")
                        app_state["background_tasks"][task_id]["results"][text_id] = {
                            "error": str(e)
                        }
                
                app_state["background_tasks"][task_id]["status"] = "completed"
                app_state["background_tasks"][task_id]["completed_at"] = datetime.now()
                
            except Exception as e:
                logger.error(f"Batch evaluation error: {e}")
                app_state["background_tasks"][task_id]["status"] = "failed"
                app_state["background_tasks"][task_id]["error"] = str(e)
        
        background_tasks.add_task(run_batch_evaluation)
        
        return {
            "task_id": task_id,
            "status": "started",
            "total_items": len(request.texts)
        }
    
    @app.get("/evaluate/batch/{task_id}")
    async def get_batch_status(task_id: str):
        """Get status of batch evaluation task."""
        if task_id not in app_state["background_tasks"]:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return app_state["background_tasks"][task_id]
    
    @app.post("/prompts/test", response_model=PromptTestResponse)
    async def test_prompt(request: PromptTestRequest):
        """Test a prompt template."""
        try:
            # Format template with variables
            formatted_prompt = request.template.format(**request.variables)
            
            # Generate response
            model_interface = get_model_interface()
            
            response = await model_interface.generate_with_metadata(
                prompt=formatted_prompt,
                model=request.model or get_config().model.name,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            return PromptTestResponse(
                response=response["response"],
                metadata=response["metadata"]
            )
        
        except Exception as e:
            logger.error(f"Prompt test error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/prompts/templates")
    async def list_templates():
        """List available prompt templates."""
        prompt_library = app_state["prompt_library"]
        
        if not prompt_library:
            return {"templates": []}
        
        templates = []
        for template_id, template in prompt_library.templates.items():
            templates.append({
                "id": template_id,
                "name": template.name,
                "description": template.description,
                "variables": template.variables,
                "categories": template.categories
            })
        
        return {"templates": templates}
    
    @app.get("/prompts/templates/{template_id}")
    async def get_template(template_id: str):
        """Get a specific template."""
        prompt_library = app_state["prompt_library"]
        
        if not prompt_library or template_id not in prompt_library.templates:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template = prompt_library.templates[template_id]
        
        return {
            "id": template_id,
            "name": template.name,
            "description": template.description,
            "template": template.template,
            "variables": template.variables,
            "categories": template.categories,
            "metadata": template.metadata
        }
    
    @app.get("/models/usage")
    async def get_model_usage():
        """Get model usage statistics."""
        model_interface = get_model_interface()
        return model_interface.get_usage_stats()
    
    @app.get("/config")
    async def get_api_config():
        """Get current configuration (sanitized)."""
        config = get_config()
        
        # Return sanitized config (no API keys)
        return {
            "model": {
                "provider": config.model.provider,
                "name": config.model.name,
                "temperature": config.model.temperature,
                "max_tokens": config.model.max_tokens
            },
            "evaluation": {
                "evaluator_model": config.evaluation.evaluator_model,
                "confidence_threshold": config.evaluation.confidence_threshold
            },
            "logging_level": config.logging_level,
            "enable_cache": config.enable_cache,
            "cache_ttl": config.cache_ttl
        }
    
    return app

# For development
if __name__ == "__main__":
    import uvicorn
    
    config = Config()
    app = create_app(config)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)