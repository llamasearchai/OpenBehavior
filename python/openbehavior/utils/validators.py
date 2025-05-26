"""
Validation utilities for prompts, responses, and content.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def validate_prompt(prompt: str) -> bool:
    """Validate that a prompt is well-formed and safe."""
    if not prompt or not isinstance(prompt, str):
        return False
    
    # Check minimum length
    if len(prompt.strip()) < 5:
        return False
    
    # Check for obvious injection attempts
    injection_patterns = [
        r"ignore\s+previous\s+instructions",
        r"system\s*:\s*you\s+are\s+now",
        r"<\s*script\s*>",
        r"javascript\s*:",
        r"eval\s*\(",
        r"exec\s*\("
    ]
    
    prompt_lower = prompt.lower()
    for pattern in injection_patterns:
        if re.search(pattern, prompt_lower):
            logger.warning(f"Potential injection pattern detected: {pattern}")
            return False
    
    return True


def validate_response(response: str) -> bool:
    """Validate that a response is appropriate and safe."""
    if not response or not isinstance(response, str):
        return False
    
    # Check for error indicators
    error_indicators = [
        "error occurred",
        "failed to generate",
        "timeout",
        "rate limit exceeded",
        "api error"
    ]
    
    response_lower = response.lower()
    for indicator in error_indicators:
        if indicator in response_lower:
            logger.warning(f"Error indicator found in response: {indicator}")
            return False
    
    # Check for potentially harmful content
    harmful_patterns = [
        r"how\s+to\s+(kill|murder|harm)",
        r"instructions\s+for\s+(bomb|explosive)",
        r"suicide\s+(methods|instructions)",
        r"self\s*-?\s*harm\s+guide"
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, response_lower):
            logger.warning(f"Potentially harmful content detected: {pattern}")
            return False
    
    return True


def validate_json_response(response: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Validate and parse JSON response."""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            return True, parsed
        else:
            return False, None
    except json.JSONDecodeError:
        return False, None


def validate_evaluation_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> bool:
    """Validate that an evaluation score is within expected range."""
    if not isinstance(score, (int, float)):
        return False
    
    return min_val <= score <= max_val


def validate_model_name(model_name: str) -> bool:
    """Validate model name format."""
    if not model_name or not isinstance(model_name, str):
        return False
    
    # Check for valid model name patterns
    valid_patterns = [
        r"^gpt-[34](\.\d+)?(-turbo)?$",
        r"^claude-[123](-\w+)*$",
        r"^text-davinci-\d+$",
        r"^[\w\-/]+$"  # General pattern for other models
    ]
    
    for pattern in valid_patterns:
        if re.match(pattern, model_name):
            return True
    
    return False


def validate_api_key(api_key: str, provider: str) -> bool:
    """Validate API key format for different providers."""
    if not api_key or not isinstance(api_key, str):
        return False
    
    provider_patterns = {
        "openai": r"^sk-[A-Za-z0-9]{48}$",
        "anthropic": r"^sk-ant-[A-Za-z0-9\-_]{95}$",
        "huggingface": r"^hf_[A-Za-z0-9]{37}$"
    }
    
    pattern = provider_patterns.get(provider.lower())
    if pattern:
        return bool(re.match(pattern, api_key))
    
    # Generic validation for unknown providers
    return len(api_key) >= 20 and api_key.isalnum()


def sanitize_text(text: str, max_length: int = 10000) -> str:
    """Sanitize text input by removing potentially harmful content."""
    if not text:
        return ""
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    # Remove potential script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove potential SQL injection patterns
    sql_patterns = [
        r";\s*drop\s+table",
        r";\s*delete\s+from",
        r"union\s+select",
        r"'\s*or\s+'1'\s*=\s*'1"
    ]
    
    for pattern in sql_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def validate_batch_size(batch_size: int, max_batch_size: int = 100) -> bool:
    """Validate batch size for processing."""
    return isinstance(batch_size, int) and 1 <= batch_size <= max_batch_size


def validate_temperature(temperature: float) -> bool:
    """Validate temperature parameter for model generation."""
    return isinstance(temperature, (int, float)) and 0.0 <= temperature <= 2.0


def validate_max_tokens(max_tokens: int) -> bool:
    """Validate max_tokens parameter."""
    return isinstance(max_tokens, int) and 1 <= max_tokens <= 32000


def validate_evaluation_config(config: Dict[str, Any]) -> List[str]:
    """Validate evaluation configuration and return list of errors."""
    errors = []
    
    required_fields = ["model", "temperature", "max_tokens"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    if "model" in config and not validate_model_name(config["model"]):
        errors.append(f"Invalid model name: {config['model']}")
    
    if "temperature" in config and not validate_temperature(config["temperature"]):
        errors.append(f"Invalid temperature: {config['temperature']}")
    
    if "max_tokens" in config and not validate_max_tokens(config["max_tokens"]):
        errors.append(f"Invalid max_tokens: {config['max_tokens']}")
    
    if "batch_size" in config and not validate_batch_size(config["batch_size"]):
        errors.append(f"Invalid batch_size: {config['batch_size']}")
    
    return errors


class ContentValidator:
    """Advanced content validation with configurable rules."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.blocked_patterns = []
        self.warning_patterns = []
        
        self._setup_default_patterns()
    
    def _setup_default_patterns(self):
        """Setup default validation patterns."""
        # Blocked patterns (will fail validation)
        self.blocked_patterns = [
            r"how\s+to\s+(kill|murder|harm)\s+",
            r"instructions\s+for\s+(bomb|explosive|weapon)",
            r"suicide\s+(methods|instructions|guide)",
            r"self\s*-?\s*harm\s+(guide|instructions)",
            r"illegal\s+(drugs|activities)\s+guide"
        ]
        
        # Warning patterns (will log warnings but pass validation)
        self.warning_patterns = [
            r"violence",
            r"weapon",
            r"dangerous",
            r"illegal"
        ]
    
    def add_blocked_pattern(self, pattern: str):
        """Add a new blocked pattern."""
        self.blocked_patterns.append(pattern)
    
    def add_warning_pattern(self, pattern: str):
        """Add a new warning pattern."""
        self.warning_patterns.append(pattern)
    
    def validate_content(self, content: str) -> Tuple[bool, List[str]]:
        """Validate content against all patterns."""
        if not content:
            return False, ["Empty content"]
        
        issues = []
        content_lower = content.lower()
        
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, content_lower):
                issues.append(f"Blocked pattern detected: {pattern}")
                if self.strict_mode:
                    return False, issues
        
        # Check warning patterns
        for pattern in self.warning_patterns:
            if re.search(pattern, content_lower):
                issues.append(f"Warning pattern detected: {pattern}")
                logger.warning(f"Warning pattern found: {pattern}")
        
        # In strict mode, any issues fail validation
        if self.strict_mode and issues:
            return False, issues
        
        # In normal mode, only blocked patterns fail validation
        blocked_issues = [issue for issue in issues if "Blocked pattern" in issue]
        return len(blocked_issues) == 0, issues 