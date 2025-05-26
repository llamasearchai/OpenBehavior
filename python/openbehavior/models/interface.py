"""
Enhanced model interface with comprehensive provider support and error handling.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging
import aiohttp
import backoff

from ..utils.logging import get_logger
from ..utils.rate_limiter import RateLimiter
from ..utils.cache import ModelCache

logger = get_logger(__name__)

class ModelInterface(ABC):
    """Enhanced interface for language model interactions."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None, cache: Optional[ModelCache] = None):
        self.rate_limiter = rate_limiter
        self.cache = cache
        self.request_count = 0
        self.total_tokens = 0
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def batch_generate(
        self,
        prompts: List[str],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts."""
        pass
    
    async def generate_with_metadata(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text and return with metadata."""
        start_time = time.time()
        
        response = await self.generate(prompt, model, **kwargs)
        
        return {
            "response": response,
            "metadata": {
                "model": model,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "generation_time": time.time() - start_time,
                "request_count": self.request_count,
                "total_tokens": self.total_tokens
            }
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens
        }

class OpenAIInterface(ModelInterface):
    """Enhanced OpenAI interface with advanced error handling and caching."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[ModelCache] = None
    ):
        super().__init__(rate_limiter, cache)
        
        try:
            import openai
            self.openai = openai
            
            # Configure client
            if api_key:
                self.openai.api_key = api_key
            else:
                import os
                self.openai.api_key = os.environ.get("OPENAI_API_KEY")
            
            if organization:
                self.openai.organization = organization
            
            if not self.openai.api_key:
                raise ValueError("OpenAI API key not provided")
                
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=60
    )
    async def generate(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text with comprehensive error handling and caching."""
        
        # Check cache first
        if self.cache:
            cache_key = self.cache.get_cache_key(
                prompt, model, temperature, max_tokens, top_p, 
                frequency_penalty, presence_penalty, stop
            )
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                logger.debug("Cache hit for prompt")
                return cached_response
        
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        try:
            # Determine API format based on model
            if model.startswith(("gpt-4", "gpt-3.5-turbo")):
                # Chat completion format
                response = await asyncio.to_thread(
                    self.openai.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop
                )
                
                result = response.choices[0].message.content
                
                # Update usage stats
                if hasattr(response, 'usage'):
                    self.total_tokens += response.usage.total_tokens
            else:
                # Legacy completion format
                response = await asyncio.to_thread(
                    self.openai.completions.create,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop
                )
                
                result = response.choices[0].text
                
                # Update usage stats
                if hasattr(response, 'usage'):
                    self.total_tokens += response.usage.total_tokens
            
            self.request_count += 1
            
            # Cache the result
            if self.cache:
                await self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def batch_generate(
        self,
        prompts: List[str],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        batch_size: int = 5,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts with batching."""
        responses = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Generate batch concurrently
            tasks = [
                self.generate(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                for prompt in batch
            ]
            
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for response in batch_responses:
                if isinstance(response, Exception):
                    logger.error(f"Batch generation error: {response}")
                    responses.append(f"ERROR: {str(response)}")
                else:
                    responses.append(response)
            
            # Brief pause between batches
            if i + batch_size < len(prompts):
                await asyncio.sleep(1)
        
        return responses

class AnthropicInterface(ModelInterface):
    """Interface for Anthropic Claude models."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[ModelCache] = None
    ):
        super().__init__(rate_limiter, cache)
        
        try:
            import anthropic
            self.anthropic = anthropic
            
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided")
            
            self.client = anthropic.Anthropic(api_key=api_key)
            
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with 'pip install anthropic'")
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=60
    )
    async def generate(
        self,
        prompt: str,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text using Anthropic Claude."""
        
        # Check cache
        if self.cache:
            cache_key = self.cache.get_cache_key(
                prompt, model, temperature, max_tokens, top_p
            )
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                return cached_response
        
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.content[0].text
            self.request_count += 1
            
            # Update token usage
            if hasattr(response, 'usage'):
                self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            
            # Cache result
            if self.cache:
                await self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def batch_generate(
        self,
        prompts: List[str],
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        tasks = [
            self.generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            for prompt in prompts
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)

class HuggingFaceInterface(ModelInterface):
    """Enhanced HuggingFace interface with local and API support."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_local: bool = False,
        device: str = "auto",
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[ModelCache] = None
    ):
        super().__init__(rate_limiter, cache)
        
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        self.use_local = use_local
        self.device = device
        
        if use_local:
            self._setup_local_models()
        
        if not self.api_key and not use_local:
            logger.warning("No HuggingFace API key provided, using local models only")
    
    def _setup_local_models(self):
        """Setup local model loading capabilities."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            self.transformers = {
                'AutoTokenizer': AutoTokenizer,
                'AutoModelForCausalLM': AutoModelForCausalLM,
                'pipeline': pipeline
            }
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.loaded_models = {}
            
        except ImportError:
            raise ImportError("Transformers package not installed. Install with 'pip install transformers torch'")
    
    async def _load_local_model(self, model_name: str):
        """Load a local model if not already loaded."""
        if model_name not in self.loaded_models:
            logger.info(f"Loading local model: {model_name}")
            
            tokenizer = self.transformers['AutoTokenizer'].from_pretrained(model_name)
            model = self.transformers['AutoModelForCausalLM'].from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else None
            )
            
            generator = self.transformers['pipeline'](
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            self.loaded_models[model_name] = generator
            logger.info(f"Model {model_name} loaded successfully")
    
    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        max_time=60
    )
    async def generate(
        self,
        prompt: str,
        model: str = "microsoft/DialoGPT-medium",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text using HuggingFace models."""
        
        # Check cache
        if self.cache:
            cache_key = self.cache.get_cache_key(
                prompt, model, temperature, max_tokens, top_p
            )
            cached_response = await self.cache.get(cache_key)
            if cached_response:
                return cached_response
        
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        
        try:
            if self.use_local:
                result = await self._generate_local(
                    prompt, model, temperature, max_tokens, top_p, **kwargs
                )
            else:
                result = await self._generate_api(
                    prompt, model, temperature, max_tokens, top_p, **kwargs
                )
            
            self.request_count += 1
            
            # Cache result
            if self.cache:
                await self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise
    
    async def _generate_local(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> str:
        """Generate using local model."""
        await self._load_local_model(model)
        generator = self.loaded_models[model]
        
        # Generate response
        def _generate():
            outputs = generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            return outputs[0]['generated_text'][len(prompt):]
        
        result = await asyncio.to_thread(_generate)
        return result.strip()
    
    async def _generate_api(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> str:
        """Generate using HuggingFace API."""
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "return_full_text": False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        if "generated_text" in result[0]:
                            return result[0]["generated_text"]
                        else:
                            return str(result[0])
                    elif isinstance(result, dict) and "generated_text" in result:
                        return result["generated_text"]
                    else:
                        return str(result)
                else:
                    error_text = await response.text()
                    raise Exception(f"HuggingFace API error: {response.status} - {error_text}")
    
    async def batch_generate(
        self,
        prompts: List[str],
        model: str = "microsoft/DialoGPT-medium",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        tasks = [
            self.generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            for prompt in prompts
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)

class ModelFactory:
    """Factory for creating model interfaces with advanced configuration."""
    
    @staticmethod
    def create(
        provider: str,
        api_key: Optional[str] = None,
        rate_limit: Optional[int] = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        **kwargs
    ) -> ModelInterface:
        """Create a model interface with optional rate limiting and caching."""
        
        # Setup rate limiter
        rate_limiter = None
        if rate_limit:
            rate_limiter = RateLimiter(rate_limit)
        
        # Setup cache
        cache = None
        if enable_cache:
            cache = ModelCache(ttl=cache_ttl)
        
        # Create interface based on provider
        if provider.lower() == "openai":
            return OpenAIInterface(
                api_key=api_key,
                rate_limiter=rate_limiter,
                cache=cache,
                **kwargs
            )
        elif provider.lower() == "anthropic":
            return AnthropicInterface(
                api_key=api_key,
                rate_limiter=rate_limiter,
                cache=cache,
                **kwargs
            )
        elif provider.lower() in ["huggingface", "hf"]:
            return HuggingFaceInterface(
                api_key=api_key,
                rate_limiter=rate_limiter,
                cache=cache,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model provider: {provider}")