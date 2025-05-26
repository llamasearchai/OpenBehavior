"""
Logging utilities for OpenBehavior platform.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """Setup logging configuration for OpenBehavior."""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Default format
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=[]
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    
    # Get root logger and add console handler
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)


class StructuredLogger:
    """Structured logger for JSON-formatted logs."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Create structured formatter
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
    
    def log_structured(
        self,
        level: str,
        message: str,
        **kwargs
    ):
        """Log a structured message with additional context."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            **kwargs
        }
        
        getattr(self.logger, level.lower())(json.dumps(log_data))
    
    def info(self, message: str, **kwargs):
        """Log info level structured message."""
        self.log_structured("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level structured message."""
        self.log_structured("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level structured message."""
        self.log_structured("error", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level structured message."""
        self.log_structured("debug", message, **kwargs)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logs."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class EvaluationLogger:
    """Specialized logger for evaluation events."""
    
    def __init__(self, name: str = "openbehavior.evaluation"):
        self.logger = get_logger(name)
    
    def log_evaluation_start(
        self,
        text_id: str,
        evaluation_type: str,
        model: str,
        **kwargs
    ):
        """Log the start of an evaluation."""
        self.logger.info(
            f"Starting {evaluation_type} evaluation for text {text_id} using {model}",
            extra={
                "event_type": "evaluation_start",
                "text_id": text_id,
                "evaluation_type": evaluation_type,
                "model": model,
                **kwargs
            }
        )
    
    def log_evaluation_complete(
        self,
        text_id: str,
        evaluation_type: str,
        score: float,
        duration: float,
        **kwargs
    ):
        """Log the completion of an evaluation."""
        self.logger.info(
            f"Completed {evaluation_type} evaluation for text {text_id}: score={score:.3f}, duration={duration:.2f}s",
            extra={
                "event_type": "evaluation_complete",
                "text_id": text_id,
                "evaluation_type": evaluation_type,
                "score": score,
                "duration": duration,
                **kwargs
            }
        )
    
    def log_evaluation_error(
        self,
        text_id: str,
        evaluation_type: str,
        error: str,
        **kwargs
    ):
        """Log an evaluation error."""
        self.logger.error(
            f"Error in {evaluation_type} evaluation for text {text_id}: {error}",
            extra={
                "event_type": "evaluation_error",
                "text_id": text_id,
                "evaluation_type": evaluation_type,
                "error": error,
                **kwargs
            }
        )
    
    def log_batch_evaluation(
        self,
        batch_size: int,
        evaluation_types: list,
        total_duration: float,
        success_count: int,
        error_count: int
    ):
        """Log batch evaluation summary."""
        self.logger.info(
            f"Batch evaluation complete: {batch_size} items, {success_count} successful, {error_count} errors, {total_duration:.2f}s total",
            extra={
                "event_type": "batch_evaluation_complete",
                "batch_size": batch_size,
                "evaluation_types": evaluation_types,
                "total_duration": total_duration,
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": success_count / batch_size if batch_size > 0 else 0
            }
        )


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self, name: str = "openbehavior.performance"):
        self.logger = get_logger(name)
    
    def log_api_call(
        self,
        provider: str,
        model: str,
        duration: float,
        tokens_used: int,
        success: bool,
        **kwargs
    ):
        """Log API call performance."""
        status = "success" if success else "failure"
        self.logger.info(
            f"API call to {provider}/{model}: {status}, {duration:.2f}s, {tokens_used} tokens",
            extra={
                "event_type": "api_call",
                "provider": provider,
                "model": model,
                "duration": duration,
                "tokens_used": tokens_used,
                "success": success,
                **kwargs
            }
        )
    
    def log_cache_hit(self, cache_key: str, cache_type: str = "default"):
        """Log cache hit."""
        self.logger.debug(
            f"Cache hit for key: {cache_key[:50]}...",
            extra={
                "event_type": "cache_hit",
                "cache_key": cache_key,
                "cache_type": cache_type
            }
        )
    
    def log_cache_miss(self, cache_key: str, cache_type: str = "default"):
        """Log cache miss."""
        self.logger.debug(
            f"Cache miss for key: {cache_key[:50]}...",
            extra={
                "event_type": "cache_miss",
                "cache_key": cache_key,
                "cache_type": cache_type
            }
        )
    
    def log_rate_limit(self, provider: str, retry_after: float):
        """Log rate limit hit."""
        self.logger.warning(
            f"Rate limit hit for {provider}, retry after {retry_after}s",
            extra={
                "event_type": "rate_limit",
                "provider": provider,
                "retry_after": retry_after
            }
        )


def configure_evaluation_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False
) -> Dict[str, logging.Logger]:
    """Configure logging for evaluation components."""
    
    # Setup base logging
    setup_logging(level=log_level, log_file=log_file)
    
    # Create specialized loggers
    loggers = {
        "evaluation": EvaluationLogger().logger,
        "performance": PerformanceLogger().logger,
        "main": get_logger("openbehavior")
    }
    
    if structured:
        # Add structured logging
        structured_logger = StructuredLogger("openbehavior.structured", log_level)
        loggers["structured"] = structured_logger.logger
    
    return loggers


class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# Convenience functions for common logging patterns
def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_async_function_call(func):
    """Decorator to log async function calls."""
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling async {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Async function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Async function {func.__name__} failed with error: {e}")
            raise
    
    return wrapper 