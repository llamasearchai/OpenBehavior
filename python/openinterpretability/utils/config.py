"""
Configuration management for OpenInterpretability platform.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration dataclass for type safety and validation."""
    
    # Model configuration
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-4"
    
    anthropic_api_key: Optional[str] = None
    anthropic_base_url: str = "https://api.anthropic.com"
    
    # Database configuration
    mongodb_url: str = "mongodb://localhost:27017/openinterpretability"
    redis_url: str = "redis://localhost:6379"
    
    # Engine configuration
    max_concurrent: int = 10
    timeout: int = 300
    
    # Cache configuration
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    # Metrics configuration
    metrics_enabled: bool = True
    
    # API configuration
    api_rate_limit: int = 100
    api_key_required: bool = False
    allowed_origins: list = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]
        
        # Load from environment
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # Validate required fields
        if not self.openai_api_key:
            logger.warning("OpenAI API key not set - some functionality may be limited")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(**{
            k: v for k, v in data.items() 
            if k in cls.__annotations__
        })


def get_config() -> Dict[str, Any]:
    """
    Load configuration from environment and config files.
    
    Returns:
        Configuration dictionary
    """
    config = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "default_model": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4")
        },
        "anthropic": {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        },
        "database": {
            "mongodb_url": os.getenv("MONGODB_URL", "mongodb://localhost:27017/openinterpretability"),
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379")
        },
        "engine": {
            "max_concurrent": int(os.getenv("MAX_CONCURRENT_EVALUATIONS", "10")),
            "timeout": int(os.getenv("EVALUATION_TIMEOUT", "300"))
        },
        "cache": {
            "enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
            "ttl": int(os.getenv("CACHE_TTL", "3600"))
        },
        "metrics": {
            "enabled": os.getenv("METRICS_ENABLED", "true").lower() == "true"
        },
        "api": {
            "rate_limit": int(os.getenv("API_RATE_LIMIT", "100")),
            "api_key_required": os.getenv("API_KEY_REQUIRED", "false").lower() == "true",
            "allowed_origins": os.getenv("ALLOWED_ORIGINS", "*").split(",")
        }
    }
    
    # Load from config file if specified
    config_path = os.getenv("OPENINTERPRETABILITY_CONFIG_PATH")
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required API keys
    if not config.get("openai", {}).get("api_key"):
        logger.warning("OpenAI API key not found - some functionality may be limited")
    
    # Validate numeric values
    try:
        max_concurrent = config.get("engine", {}).get("max_concurrent", 10)
        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
            
        timeout = config.get("engine", {}).get("timeout", 300)
        if not isinstance(timeout, int) or timeout < 30:
            raise ValueError("timeout must be at least 30 seconds")
            
        cache_ttl = config.get("cache", {}).get("ttl", 3600)
        if not isinstance(cache_ttl, int) or cache_ttl < 60:
            raise ValueError("cache_ttl must be at least 60 seconds")
            
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    logger.info("Configuration validation passed")


class ConfigManager:
    """Configuration manager with caching and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config = None
        self._config_obj = None
        
    def get_config(self) -> Dict[str, Any]:
        """Get configuration with caching."""
        if self._config is None:
            self._config = get_config()
            try:
                validate_config(self._config)
            except ValueError as e:
                logger.warning(f"Configuration validation warning: {e}")
        return self._config
    
    def get_config_object(self) -> Config:
        """Get configuration as Config object."""
        if self._config_obj is None:
            config_dict = self.get_config()
            # Flatten the config dict for the dataclass
            flat_config = {
                "openai_api_key": config_dict.get("openai", {}).get("api_key"),
                "openai_base_url": config_dict.get("openai", {}).get("base_url", "https://api.openai.com/v1"),
                "openai_default_model": config_dict.get("openai", {}).get("default_model", "gpt-4"),
                "anthropic_api_key": config_dict.get("anthropic", {}).get("api_key"),
                "anthropic_base_url": config_dict.get("anthropic", {}).get("base_url", "https://api.anthropic.com"),
                "mongodb_url": config_dict.get("database", {}).get("mongodb_url", "mongodb://localhost:27017/openinterpretability"),
                "redis_url": config_dict.get("database", {}).get("redis_url", "redis://localhost:6379"),
                "max_concurrent": config_dict.get("engine", {}).get("max_concurrent", 10),
                "timeout": config_dict.get("engine", {}).get("timeout", 300),
                "cache_enabled": config_dict.get("cache", {}).get("enabled", True),
                "cache_ttl": config_dict.get("cache", {}).get("ttl", 3600),
                "metrics_enabled": config_dict.get("metrics", {}).get("enabled", True),
                "api_rate_limit": config_dict.get("api", {}).get("rate_limit", 100),
                "api_key_required": config_dict.get("api", {}).get("api_key_required", False),
                "allowed_origins": config_dict.get("api", {}).get("allowed_origins", ["*"])
            }
            self._config_obj = Config(**flat_config)
        return self._config_obj
    
    def reload_config(self) -> None:
        """Reload configuration from sources."""
        self._config = None
        self._config_obj = None
        logger.info("Configuration reloaded")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        config = self.get_config()
        keys = key.split('.')
        
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


# Global config manager instance
config_manager = ConfigManager() 