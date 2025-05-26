"""
Advanced data generation for comprehensive behavior evaluation.
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import hashlib
import itertools

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import aiofiles
import asyncio

from ..models.interface import ModelInterface
from ..prompts.templates import PromptTemplate, PromptLibrary
from ..utils.logging import get_logger
from ..utils.validators import validate_prompt, validate_response

logger = get_logger(__name__)

@dataclass
class DataPoint:
    """Enhanced data point for behavior evaluation."""
    id: str
    prompt: str
    response: Optional[str] = None
    template_id: Optional[str] = None
    variables: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate data point after initialization."""
        if not validate_prompt(self.prompt):
            raise ValueError(f"Invalid prompt for data point {self.id}")
            
        if self.response and not validate_response(self.response):
            logger.warning(f"Potentially problematic response for data point {self.id}")
    
    @property
    def hash(self) -> str:
        """Generate unique hash for this data point."""
        content = f"{self.prompt}{json.dumps(self.variables, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "template_id": self.template_id,
            "variables": self.variables,
            "metadata": self.metadata,
            "tags": self.tags,
            "evaluation": self.evaluation,
            "quality_score": self.quality_score,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataPoint':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            response=data.get("response"),
            template_id=data.get("template_id"),
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            evaluation=data.get("evaluation", {}),
            quality_score=data.get("quality_score")
        )

@dataclass
class GenerationConfig:
    """Enhanced configuration for data generation."""
    template_id: str
    num_samples: int = 100
    output_path: str = "generated_data.jsonl"
    model_provider: str = "openai"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    variables: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced generation options
    quality_filter: bool = True
    diversity_sampling: bool = True
    batch_size: int = 10
    parallel_workers: int = 4
    retry_failed: int = 3
    deduplication: bool = True
    min_quality_score: float = 0.6
    
    # Adaptive generation
    adaptive_sampling: bool = False
    target_coverage: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")

class QualityFilter:
    """Filter for assessing generated data quality."""
    
    def __init__(self, min_length: int = 10, max_length: int = 5000):
        self.min_length = min_length
        self.max_length = max_length
    
    def assess_quality(self, data_point: DataPoint) -> float:
        """Assess quality of a data point (0-1 score)."""
        score = 1.0
        
        # Check response length
        if data_point.response:
            response_len = len(data_point.response)
            if response_len < self.min_length:
                score *= 0.5
            elif response_len > self.max_length:
                score *= 0.7
        
        # Check for error indicators
        if data_point.response and any(
            error in data_point.response.lower() 
            for error in ["error", "timeout", "failed", "cannot", "unable"]
        ):
            score *= 0.3
        
        # Check prompt quality
        if len(data_point.prompt) < 20:
            score *= 0.6
        
        # Check for repetition
        if data_point.response:
            words = data_point.response.split()
            if len(set(words)) / len(words) < 0.5:  # High repetition
                score *= 0.4
        
        return score
    
    def filter_data(self, data_points: List[DataPoint], threshold: float = 0.6) -> List[DataPoint]:
        """Filter data points based on quality threshold."""
        filtered = []
        
        for dp in data_points:
            quality = self.assess_quality(dp)
            dp.quality_score = quality
            
            if quality >= threshold:
                filtered.append(dp)
            else:
                logger.debug(f"Filtered out data point {dp.id} with quality {quality:.3f}")
        
        logger.info(f"Quality filter: {len(filtered)}/{len(data_points)} data points passed")
        return filtered

class DiversitySampler:
    """Ensure diversity in generated samples."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def diversify_samples(self, data_points: List[DataPoint]) -> List[DataPoint]:
        """Remove overly similar data points."""
        if len(data_points) <= 1:
            return data_points
        
        diverse_points = [data_points[0]]
        
        for dp in data_points[1:]:
            is_diverse = True
            
            for existing_dp in diverse_points:
                similarity = self.calculate_similarity(dp.prompt, existing_dp.prompt)
                
                if similarity > self.similarity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_points.append(dp)
            else:
                logger.debug(f"Filtered out similar data point {dp.id}")
        
        logger.info(f"Diversity filter: {len(diverse_points)}/{len(data_points)} data points kept")
        return diverse_points

class AdaptiveSampler:
    """Adaptive sampling to achieve target coverage."""
    
    def __init__(self, target_coverage: Dict[str, float]):
        self.target_coverage = target_coverage
        self.current_coverage = {key: 0.0 for key in target_coverage}
    
    def update_coverage(self, data_point: DataPoint):
        """Update coverage statistics based on new data point."""
        for key in self.target_coverage:
            if key in data_point.tags or key in data_point.metadata:
                self.current_coverage[key] += 1
    
    def get_sampling_weights(self) -> Dict[str, float]:
        """Get sampling weights to achieve target coverage."""
        weights = {}
        
        for key, target in self.target_coverage.items():
            current = self.current_coverage[key]
            if current < target:
                weights[key] = (target - current) / target
            else:
                weights[key] = 0.
        return weights
    
    def should_generate_more(self, category: str) -> bool:
        """Check if more samples needed for a category."""
        current = self.current_coverage.get(category, 0)
        target = self.target_coverage.get(category, 0)
        return current < target

class DataGenerator:
    """Advanced data generator with quality control and adaptive sampling."""
    
    def __init__(
        self,
        config: GenerationConfig,
        prompt_library: PromptLibrary,
        model_interface: ModelInterface
    ):
        self.config = config
        self.prompt_library = prompt_library
        self.model_interface = model_interface
        
        # Initialize components
        self.quality_filter = QualityFilter()
        self.diversity_sampler = DiversitySampler()
        
        if config.adaptive_sampling:
            self.adaptive_sampler = AdaptiveSampler(config.target_coverage)
        else:
            self.adaptive_sampler = None
        
        # Set random seed for reproducibility
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
        
        logger.info(f"DataGenerator initialized with template: {config.template_id}")
    
    async def generate_single_datapoint(
        self,
        template: PromptTemplate,
        variables: Dict[str, str],
        index: int
    ) -> Optional[DataPoint]:
        """Generate a single data point with error handling and retries."""
        for attempt in range(self.config.retry_failed):
            try:
                # Fill template with variables
                prompt = template.format(**variables)
                
                # Generate unique ID
                data_id = f"{self.config.template_id}_{index}_{random.randint(10000, 99999)}"
                
                # Create data point
                data_point = DataPoint(
                    id=data_id,
                    prompt=prompt,
                    template_id=self.config.template_id,
                    variables=variables,
                    metadata={
                        "generation_config": {
                            "model_provider": self.config.model_provider,
                            "model_name": self.config.model_name,
                            "temperature": self.config.temperature,
                            "max_tokens": self.config.max_tokens,
                            "attempt": attempt + 1
                        },
                        "generation_timestamp": pd.Timestamp.now().isoformat(),
                        **self.config.metadata
                    },
                    tags=["generated", self.config.template_id]
                )
                
                # Generate response if configured
                if self.config.model_name:
                    response = await self._generate_response_async(prompt)
                    data_point.response = response
                    data_point.metadata["response_timestamp"] = pd.Timestamp.now().isoformat()
                
                # Update adaptive sampling if enabled
                if self.adaptive_sampler:
                    self.adaptive_sampler.update_coverage(data_point)
                
                return data_point
                
            except Exception as e:
                logger.error(f"Generation attempt {attempt + 1} failed for index {index}: {e}")
                if attempt == self.config.retry_failed - 1:
                    logger.error(f"All attempts failed for index {index}")
                    return None
                
                await asyncio.sleep(1)  # Brief delay before retry
        
        return None
    
    async def _generate_response_async(self, prompt: str) -> str:
        """Generate response asynchronously with timeout."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self.model_interface.generate,
                    prompt=prompt,
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty
                ),
                timeout=30.0  # 30 second timeout per generation
            )
        except asyncio.TimeoutError:
            logger.warning(f"Response generation timed out for prompt: {prompt[:100]}...")
            return "GENERATION_TIMEOUT"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"GENERATION_ERROR: {str(e)}"
    
    def _generate_variable_combinations(self) -> List[Dict[str, str]]:
        """Generate all variable combinations or sample them."""
        if not self.config.variables:
            return [{}]
        
        keys = list(self.config.variables.keys())
        values = [self.config.variables[key] for key in keys]
        
        if self.config.diversity_sampling:
            # Sample diverse combinations rather than all combinations
            all_combinations = list(itertools.product(*values))
            
            if len(all_combinations) > self.config.num_samples:
                # Randomly sample from all combinations
                selected_combinations = random.sample(all_combinations, self.config.num_samples)
            else:
                selected_combinations = all_combinations
        else:
            # Generate all combinations
            selected_combinations = list(itertools.product(*values))
        
        # Convert to list of dictionaries
        combinations = [
            dict(zip(keys, combo)) for combo in selected_combinations
        ]
        
        # Shuffle for randomness
        random.shuffle(combinations)
        
        return combinations
    
    async def generate_batch(self, batch_size: int = None) -> List[DataPoint]:
        """Generate a batch of data points."""
        batch_size = batch_size or self.config.batch_size
        
        # Get template
        template_id = self.config.template_id
        if template_id not in self.prompt_library.templates:
            raise ValueError(f"Template '{template_id}' not found")
        
        template = self.prompt_library.templates[template_id]
        
        # Generate variable combinations
        combinations = self._generate_variable_combinations()
        
        # Limit to requested number of samples
        if len(combinations) > self.config.num_samples:
            combinations = combinations[:self.config.num_samples]
        elif len(combinations) < self.config.num_samples:
            # Repeat combinations if needed
            while len(combinations) < self.config.num_samples:
                combinations.extend(combinations[:self.config.num_samples - len(combinations)])
        
        # Create batches for parallel processing
        batches = [
            combinations[i:i + batch_size]
            for i in range(0, len(combinations), batch_size)
        ]
        
        all_data_points = []
        
        for batch_idx, batch_combinations in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")
            
            # Create semaphore to limit concurrent generations
            semaphore = asyncio.Semaphore(self.config.parallel_workers)
            
            async def generate_with_semaphore(variables, index):
                async with semaphore:
                    return await self.generate_single_datapoint(template, variables, index)
            
            # Generate data points in parallel
            tasks = [
                generate_with_semaphore(variables, batch_idx * batch_size + i)
                for i, variables in enumerate(batch_combinations)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            valid_results = [
                result for result in batch_results
                if result is not None and not isinstance(result, Exception)
            ]
            
            all_data_points.extend(valid_results)
            
            # Brief pause between batches to avoid rate limiting
            if batch_idx < len(batches) - 1:
                await asyncio.sleep(1)
        
        return all_data_points
    
    async def generate(self) -> List[DataPoint]:
        """Generate data points with full pipeline including quality control."""
        logger.info(f"Starting generation of {self.config.num_samples} samples")
        
        # Generate raw data points
        raw_data_points = await self.generate_batch()
        
        logger.info(f"Generated {len(raw_data_points)} raw data points")
        
        # Apply quality filtering if enabled
        if self.config.quality_filter:
            filtered_data_points = self.quality_filter.filter_data(
                raw_data_points, 
                self.config.min_quality_score
            )
        else:
            filtered_data_points = raw_data_points
        
        # Apply diversity sampling if enabled
        if self.config.diversity_sampling:
            diverse_data_points = self.diversity_sampler.diversify_samples(filtered_data_points)
        else:
            diverse_data_points = filtered_data_points
        
        # Apply deduplication if enabled
        if self.config.deduplication:
            final_data_points = self._deduplicate_data_points(diverse_data_points)
        else:
            final_data_points = diverse_data_points
        
        logger.info(f"Final dataset: {len(final_data_points)} data points")
        
        return final_data_points
    
    def _deduplicate_data_points(self, data_points: List[DataPoint]) -> List[DataPoint]:
        """Remove duplicate data points based on hash."""
        seen_hashes = set()
        unique_data_points = []
        
        for dp in data_points:
            if dp.hash not in seen_hashes:
                seen_hashes.add(dp.hash)
                unique_data_points.append(dp)
            else:
                logger.debug(f"Removed duplicate data point {dp.id}")
        
        logger.info(f"Deduplication: {len(unique_data_points)}/{len(data_points)} unique data points")
        return unique_data_points
    
    async def save_to_jsonl(
        self, 
        data_points: List[DataPoint], 
        output_path: Optional[str] = None
    ) -> str:
        """Save data points to JSONL file asynchronously."""
        path = output_path or self.config.output_path
        
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(path, 'w') as f:
            for data_point in data_points:
                await f.write(json.dumps(data_point.to_dict()) + '\n')
        
        logger.info(f"Saved {len(data_points)} data points to {path}")
        return path
    
    @classmethod
    async def load_from_jsonl(cls, path: str) -> List[DataPoint]:
        """Load data points from JSONL file asynchronously."""
        data_points = []
        
        async with aiofiles.open(path, 'r') as f:
            async for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        data_points.append(DataPoint.from_dict(data))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {line[:100]}... Error: {e}")
        
        logger.info(f"Loaded {len(data_points)} data points from {path}")
        return data_points
    
    def get_generation_statistics(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """Generate statistics about the generated dataset."""
        if not data_points:
            return {"error": "No data points provided"}
        
        stats = {
            "total_data_points": len(data_points),
            "average_prompt_length": np.mean([len(dp.prompt) for dp in data_points]),
            "average_response_length": np.mean([
                len(dp.response) for dp in data_points if dp.response
            ]),
            "quality_scores": {
                "mean": np.mean([dp.quality_score for dp in data_points if dp.quality_score]),
                "std": np.std([dp.quality_score for dp in data_points if dp.quality_score]),
                "min": np.min([dp.quality_score for dp in data_points if dp.quality_score]),
                "max": np.max([dp.quality_score for dp in data_points if dp.quality_score])
            },
            "template_distribution": {},
            "tag_distribution": {},
            "variable_distribution": {}
        }
        
        # Calculate template distribution
        template_counts = {}
        for dp in data_points:
            template_id = dp.template_id or "unknown"
            template_counts[template_id] = template_counts.get(template_id, 0) + 1
        stats["template_distribution"] = template_counts
        
        # Calculate tag distribution
        tag_counts = {}
        for dp in data_points:
            for tag in dp.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        stats["tag_distribution"] = tag_counts
        
        # Calculate variable distribution
        variable_counts = {}
        for dp in data_points:
            for var_name, var_value in dp.variables.items():
                if var_name not in variable_counts:
                    variable_counts[var_name] = {}
                variable_counts[var_name][var_value] = variable_counts[var_name].get(var_value, 0) + 1
        stats["variable_distribution"] = variable_counts
        
        return stats

# Utility functions for data generation pipeline
async def generate_comprehensive_dataset(
    config_path: str,
    template_configs: List[Dict[str, Any]],
    output_dir: str = "./datasets"
) -> Dict[str, List[DataPoint]]:
    """Generate comprehensive dataset across multiple templates."""
    from ..utils.config import Config
    from ..models.interface import ModelFactory
    
    # Load base configuration
    base_config = Config.from_file(config_path)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model interface
    model_interface = ModelFactory.create(
        provider=base_config.model.provider,
        api_key=base_config.model.api_key
    )
    
    # Initialize prompt library
    prompt_library = PromptLibrary()
    prompt_library.load_from_directory(base_config.prompts.template_dir)
    
    all_datasets = {}
    
    for template_config in template_configs:
        logger.info(f"Generating dataset for template: {template_config['template_id']}")
        
        # Create generation config
        gen_config = GenerationConfig(
            template_id=template_config["template_id"],
            num_samples=template_config.get("num_samples", 100),
            output_path=f"{output_dir}/{template_config['template_id']}.jsonl",
            model_name=base_config.model.name,
            **template_config.get("generation_params", {})
        )
        
        # Create generator
        generator = DataGenerator(gen_config, prompt_library, model_interface)
        
        # Generate data points
        data_points = await generator.generate()
        
        # Save to file
        await generator.save_to_jsonl(data_points)
        
        # Store in results
        all_datasets[template_config["template_id"]] = data_points
        
        # Generate statistics
        stats = generator.get_generation_statistics(data_points)
        stats_path = f"{output_dir}/{template_config['template_id']}_stats.json"
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Completed generation for {template_config['template_id']}")
    
    return all_datasets