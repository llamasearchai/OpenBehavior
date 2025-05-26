"""
Configuration management utilities.
"""

import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

@dataclass
class ModelConfig:
    """Model configuration."""
    provider: str = "openai"
    name: str = "gpt-4"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    rate_limit: Optional[int] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    ethical_dimensions: List[str] = field(default_factory=lambda: [
        "harm_prevention", "fairness", "honesty", "autonomy", "privacy"
    ])
    safety_categories: List[str] = field(default_factory=lambda: [
        "violence", "hate_speech", "self_harm", "dangerous_instructions"
    ])
    alignment_dimensions: List[str] = field(default_factory=lambda: [
        "helpfulness", "harmlessness", "honesty", "respect"
    ])
    evaluator_model: str = "gpt-4"
    confidence_threshold: float = 0.7
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

@dataclass
class DataConfig:
    """Data configuration."""
    template_dir: str = "./templates"
    data_dir: str = "./data"
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging_level: str = "INFO"
    enable_cache: bool = True
    cache_ttl: int = 3600
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'Config':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")
        
        # Create config objects
        config = cls()
        
        if 'model' in data:
            config.model = ModelConfig.from_dict(data['model'])
        
        if 'evaluation' in data:
            config.evaluation = EvaluationConfig.from_dict(data['evaluation'])
        
        if 'data' in data:
            config.data = DataConfig.from_dict(data['data'])
        
        # Set other fields
        for key in ['logging_level', 'enable_cache', 'cache_ttl']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        
        data = asdict(self)
        
        if filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.data.template_dir,
            self.data.data_dir,
            self.data.output_dir,
            self.data.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)