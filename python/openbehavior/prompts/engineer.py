"""
Advanced prompt engineering with optimization and refinement capabilities.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import itertools
import hashlib

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from ..models.interface import ModelInterface
from ..utils.logging import get_logger
from ..utils.optimization import BayesianOptimizer
from .templates import PromptTemplate, PromptLibrary

logger = get_logger(__name__)

@dataclass
class PromptVariation:
    """Enhanced prompt variation with optimization tracking."""
    template_id: str
    template: str
    variables: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    parent_id: Optional[str] = None
    
    def format(self) -> str:
        """Format the template with variables."""
        try:
            return self.template.format(**self.variables)
        except KeyError as e:
            logger.warning(f"Missing variable {e} in template {self.template_id}")
            return self.template
    
    def hash(self) -> str:
        """Generate unique hash for this variation."""
        content = (
            self.template + 
            json.dumps(self.variables, sort_keys=True) +
            json.dumps(self.parameters, sort_keys=True)
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_id": self.template_id,
            "template": self.template,
            "variables": self.variables,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "version": self.version,
            "parent_id": self.parent_id,
            "hash": self.hash(),
            "formatted_prompt": self.format()
        }

@dataclass
class PromptTestResult:
    """Enhanced test results with comprehensive metrics."""
    variation: PromptVariation
    response: str
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        if not self.metrics:
            return 0.0
        
        # Default weights for common metrics
        default_weights = {
            "relevance": 0.3,
            "clarity": 0.2,
            "safety": 0.25,
            "helpfulness": 0.15,
            "coherence": 0.1
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in self.metrics.items():
            weight = default_weights.get(metric, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variation": self.variation.to_dict(),
            "response": self.response,
            "metrics": self.metrics,
            "overall_score": self.overall_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

class PromptOptimizer:
    """Bayesian optimization for prompt parameters."""
    
    def __init__(self, objective_metric: str = "overall_score"):
        self.objective_metric = objective_metric
        self.optimizer = BayesianOptimizer()
        self.history = []
    
    def suggest_parameters(
        self,
        parameter_space: Dict[str, Any],
        n_suggestions: int = 1
    ) -> List[Dict[str, Any]]:
        """Suggest next parameters to try."""
        if len(self.history) < 5:  # Random exploration initially
            suggestions = []
            for _ in range(n_suggestions):
                suggestion = {}
                for param, bounds in parameter_space.items():
                    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                        # Continuous parameter
                        suggestion[param] = np.random.uniform(bounds[0], bounds[1])
                    elif isinstance(bounds, list):
                        # Categorical parameter
                        suggestion[param] = np.random.choice(bounds)
                suggestions.append(suggestion)
            return suggestions
        
        # Use Bayesian optimization
        return self.optimizer.suggest(parameter_space, n_suggestions)
    
    def update_history(self, parameters: Dict[str, Any], result: PromptTestResult):
        """Update optimization history."""
        score = result.metrics.get(self.objective_metric, result.overall_score)
        
        self.history.append({
            "parameters": parameters,
            "score": score,
            "result": result,
            "timestamp": result.timestamp
        })
        
        # Update optimizer
        self.optimizer.update(parameters, score)
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get best parameters found so far."""
        if not self.history:
            return None
        
        best_entry = max(self.history, key=lambda x: x["score"])
        return best_entry["parameters"]

class PromptEvolutionEngine:
    """Evolutionary approach to prompt improvement."""
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation = 0
    
    def initialize_population(
        self,
        base_template: str,
        variable_options: Dict[str, List[str]]
    ) -> List[PromptVariation]:
        """Initialize random population of prompt variations."""
        population = []
        
        for i in range(self.population_size):
            variables = {}
            for var_name, options in variable_options.items():
                variables[var_name] = np.random.choice(options)
            
            variation = PromptVariation(
                template_id=f"evolved_gen0_{i}",
                template=base_template,
                variables=variables,
                metadata={"generation": 0, "individual": i}
            )
            
            population.append(variation)
        
        return population
    
    def mutate_variation(
        self,
        variation: PromptVariation,
        variable_options: Dict[str, List[str]]
    ) -> PromptVariation:
        """Mutate a prompt variation."""
        new_variables = variation.variables.copy()
        
        # Randomly mutate some variables
        for var_name in new_variables:
            if np.random.random() < self.mutation_rate:
                new_variables[var_name] = np.random.choice(variable_options[var_name])
        
        mutated = PromptVariation(
            template_id=f"evolved_gen{self.generation + 1}_mutated",
            template=variation.template,
            variables=new_variables,
            parent_id=variation.template_id,
            metadata={
                "generation": self.generation + 1,
                "operation": "mutation",
                "parent": variation.template_id
            }
        )
        
        return mutated
    
    def crossover_variations(
        self,
        parent1: PromptVariation,
        parent2: PromptVariation
    ) -> Tuple[PromptVariation, PromptVariation]:
        """Create offspring through crossover."""
        # Simple uniform crossover
        child1_vars = {}
        child2_vars = {}
        
        for var_name in parent1.variables:
            if np.random.random() < 0.5:
                child1_vars[var_name] = parent1.variables[var_name]
                child2_vars[var_name] = parent2.variables[var_name]
            else:
                child1_vars[var_name] = parent2.variables[var_name]
                child2_vars[var_name] = parent1.variables[var_name]
        
        child1 = PromptVariation(
            template_id=f"evolved_gen{self.generation + 1}_cross1",
            template=parent1.template,
            variables=child1_vars,
            parent_id=f"{parent1.template_id}+{parent2.template_id}",
            metadata={
                "generation": self.generation + 1,
                "operation": "crossover",
                "parents": [parent1.template_id, parent2.template_id]
            }
        )
        
        child2 = PromptVariation(
            template_id=f"evolved_gen{self.generation + 1}_cross2",
            template=parent1.template,
            variables=child2_vars,
            parent_id=f"{parent1.template_id}+{parent2.template_id}",
            metadata={
                "generation": self.generation + 1,
                "operation": "crossover",
                "parents": [parent1.template_id, parent2.template_id]
            }
        )
        
        return child1, child2
    
    def evolve_generation(
        self,
        population: List[PromptVariation],
        fitness_scores: List[float],
        variable_options: Dict[str, List[str]]
    ) -> List[PromptVariation]:
        """Evolve to next generation."""
        # Selection (tournament selection)
        selected = []
        for _ in range(self.population_size // 2):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(
                len(population), 
                size=min(tournament_size, len(population)), 
                replace=False
            )
            
            best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected.append(population[best_idx])
        
        # Crossover and mutation
        new_population = []
        
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
            
            if np.random.random() < self.crossover_rate:
                child1, child2 = self.crossover_variations(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])
        
        # Mutation
        mutated_population = []
        for variation in new_population:
            if np.random.random() < self.mutation_rate:
                mutated = self.mutate_variation(variation, variable_options)
                mutated_population.append(mutated)
            else:
                mutated_population.append(variation)
        
        # Ensure population size
        while len(mutated_population) < self.population_size:
            mutated_population.append(mutated_population[0])
        
        mutated_population = mutated_population[:self.population_size]
        
        self.generation += 1
        return mutated_population

class PromptEngineer:
    """Advanced prompt engineering with optimization and evolution."""
    
    def __init__(
        self,
        model_interface: ModelInterface,
        prompt_library: PromptLibrary,
        default_model: str = "gpt-4",
        output_dir: str = "prompt_experiments"
    ):
        self.model_interface = model_interface
        self.prompt_library = prompt_library
        self.default_model = default_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimization components
        self.optimizer = PromptOptimizer()
        self.evolution_engine = PromptEvolutionEngine()
        
        # Metrics collection
        self.evaluation_metrics = {}
        self._setup_default_metrics()
    
    def _setup_default_metrics(self):
        """Setup default evaluation metrics."""
        self.evaluation_metrics = {
            "length": self._calculate_length_score,
            "clarity": self._calculate_clarity_score,
            "coherence": self._calculate_coherence_score,
            "response_time": self._calculate_response_time,
        }
    
    def _calculate_length_score(self, prompt: str, response: str) -> float:
        """Calculate score based on response length appropriateness."""
        if not response:
            return 0.0
        
        # Ideal length range (characters)
        ideal_min, ideal_max = 100, 2000
        length = len(response)
        
        if ideal_min <= length <= ideal_max:
            return 1.0
        elif length < ideal_min:
            return length / ideal_min
        else:
            return max(0.1, 1.0 - (length - ideal_max) / ideal_max)
    
    def _calculate_clarity_score(self, prompt: str, response: str) -> float:
        """Calculate clarity score based on response structure."""
        if not response:
            return 0.0
        
        score = 0.0
        
        # Check for clear structure
        sentences = response.split('.')
        if len(sentences) > 1:
            score += 0.3
        
        # Check for appropriate punctuation
        punctuation_count = sum(1 for c in response if c in '.!?')
        if punctuation_count > 0:
            score += 0.3
        
        # Check for varied sentence length
        if sentences:
            lengths = [len(s.strip()) for s in sentences if s.strip()]
            if lengths and np.std(lengths) > 10:  # Some variation
                score += 0.4
        
        return min(1.0, score)
    
    def _calculate_coherence_score(self, prompt: str, response: str) -> float:
        """Calculate coherence score using simple heuristics."""
        if not response:
            return 0.0
        
        score = 0.0
        
        # Check for repetition (lower is better)
        words = response.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += min(1.0, unique_ratio * 1.2)
        
        # Check for common coherence indicators
        coherence_words = ['therefore', 'however', 'moreover', 'furthermore', 'in addition']
        found_indicators = sum(1 for word in coherence_words if word in response.lower())
        score += min(0.3, found_indicators * 0.1)
        
        return min(1.0, score)
    
    def _calculate_response_time(self, prompt: str, response: str) -> float:
        """Calculate response time score (to be set externally)."""
        # This would be set by the testing framework
        return 1.0
    
    async def test_variation(
        self,
        variation: PromptVariation,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        custom_metrics: Optional[List[Callable]] = None
    ) -> PromptTestResult:
        """Test a prompt variation comprehensively."""
        prompt = variation.format()
        model_name = model or self.default_model
        
        # Generate response with timing
        start_time = time.time()
        
        response = await asyncio.to_thread(
            self.model_interface.generate,
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {"response_time": response_time}
        
        # Apply built-in metrics
        for metric_name, metric_func in self.evaluation_metrics.items():
            try:
                if metric_name == "response_time":
                    metrics[metric_name] = 1.0 / (1.0 + response_time)  # Higher is better
                else:
                    metrics[metric_name] = metric_func(prompt, response)
            except Exception as e:
                logger.error(f"Error calculating metric {metric_name}: {e}")
                metrics[metric_name] = 0.0
        
        # Apply custom metrics
        if custom_metrics:
            for metric_func in custom_metrics:
                try:
                    metric_name = metric_func.__name__
                    metrics[metric_name] = metric_func(prompt, response)
                except Exception as e:
                    logger.error(f"Error calculating custom metric: {e}")
        
        # Create test result
        result = PromptTestResult(
            variation=variation,
            response=response,
            metrics=metrics,
            metadata={
                "model": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_length": len(prompt),
                "response_length": len(response)
            }
        )
        
        return result
    
    async def optimize_prompt(
        self,
        base_template: str,
        parameter_space: Dict[str, Any],
        test_prompts: List[str],
        max_iterations: int = 20,
        model: Optional[str] = None
    ) -> Tuple[PromptVariation, List[PromptTestResult]]:
        """Optimize prompt using Bayesian optimization."""
        all_results = []
        best_variation = None
        best_score = -float('inf')
        
        logger.info(f"Starting prompt optimization for {max_iterations} iterations")
        
        for iteration in range(max_iterations):
            # Get parameter suggestions
            suggestions = self.optimizer.suggest_parameters(parameter_space, n_suggestions=3)
            
            iteration_results = []
            
            for suggestion in suggestions:
                # Create variation with suggested parameters
                variation = PromptVariation(
                    template_id=f"opt_iter_{iteration}",
                    template=base_template,
                    parameters=suggestion,
                    metadata={"iteration": iteration}
                )
                
                # Test on all test prompts
                prompt_scores = []
                for test_prompt in test_prompts:
                    # Replace {prompt} placeholder in template
                    test_variation = PromptVariation(
                        template_id=variation.template_id,
                        template=base_template.replace("{input}", test_prompt),
                        parameters=suggestion,
                        metadata=variation.metadata
                    )
                    
                    result = await self.test_variation(test_variation, model=model)
                    prompt_scores.append(result.overall_score)
                    iteration_results.append(result)
                    
                    # Update optimizer with average score
                    avg_score = np.mean(prompt_scores)
                    self.optimizer.update_history(suggestion, result)
                    
                    # Track best variation
                    if avg_score > best_score:
                        best_score = avg_score
                        best_variation = test_variation
            
            all_results.extend(iteration_results)
            
            logger.info(f"Iteration {iteration + 1}/{max_iterations}, best score: {best_score:.3f}")
        
        logger.info(f"Optimization complete. Best score: {best_score:.3f}")
        return best_variation, all_results
    
    async def evolve_prompts(
        self,
        base_template: str,
        variable_options: Dict[str, List[str]],
        generations: int = 10,
        model: Optional[str] = None
    ) -> Tuple[List[PromptVariation], List[PromptTestResult]]:
        """Evolve prompts using genetic algorithm."""
        all_results = []
        
        # Initialize population
        population = self.evolution_engine.initialize_population(base_template, variable_options)
        
        logger.info(f"Starting prompt evolution for {generations} generations")
        
        for generation in range(generations):
            # Evaluate population
            generation_results = []
            fitness_scores = []
            
            for variation in population:
                result = await self.test_variation(variation, model=model)
                generation_results.append(result)
                fitness_scores.append(result.overall_score)
            
            all_results.extend(generation_results)
            
            # Log generation statistics
            avg_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            
            logger.info(f"Generation {generation + 1}/{generations}: "
                       f"avg={avg_fitness:.3f}, max={max_fitness:.3f}")
            
            # Evolve to next generation (except for last generation)
            if generation < generations - 1:
                population = self.evolution_engine.evolve_generation(
                    population, fitness_scores, variable_options
                )
        
        # Return final population and all results
        return population, all_results
    
    async def refine_prompt_iteratively(
        self,
        initial_prompt: str,
        feedback_examples: List[Dict[str, str]],
        max_iterations: int = 5,
        model: Optional[str] = None
    ) -> List[PromptTestResult]:
        """Iteratively refine prompt based on feedback examples."""
        current_prompt = initial_prompt
        all_results = []
        
        for iteration in range(max_iterations):
            # Test current prompt
            variation = PromptVariation(
                template_id=f"refined_iter_{iteration}",
                template=current_prompt,
                metadata={"iteration": iteration, "refinement_stage": True}
            )
            
            result = await self.test_variation(variation, model=model)
            all_results.append(result)
            
            # Generate refinement based on feedback
            if iteration < max_iterations - 1:
                current_prompt = await self._generate_refined_prompt(
                    current_prompt, result.response, feedback_examples, model
                )
        
        return all_results
    
    async def _generate_refined_prompt(
        self,
        current_prompt: str,
        current_response: str,
        feedback_examples: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> str:
        """Generate improved prompt based on feedback."""
        refinement_template = """
        Current prompt: {current_prompt}
        
        Current response: {current_response}
        
        Feedback examples:
        {feedback_examples}
        
        Please provide an improved version of the prompt that addresses the feedback and produces better responses. Return only the improved prompt.
        """
        
        feedback_text = "\n".join([
            f"Input: {ex.get('input', 'N/A')}\nExpected: {ex.get('expected', 'N/A')}\nIssue: {ex.get('issue', 'N/A')}"
            for ex in feedback_examples
        ])
        
        refinement_prompt = refinement_template.format(
            current_prompt=current_prompt,
            current_response=current_response,
            feedback_examples=feedback_text
        )
        
        refined_prompt = await asyncio.to_thread(
            self.model_interface.generate,
            prompt=refinement_prompt,
            model=model or self.default_model,
            temperature=0.3,
            max_tokens=1024
        )
        
        return refined_prompt.strip()
    
    def analyze_results(
        self,
        results: List[PromptTestResult],
        group_by: str = "template_id"
    ) -> pd.DataFrame:
        """Analyze test results and generate insights."""
        data = []
        
        for result in results:
            row = {
                "template_id": result.variation.template_id,
                "prompt": result.variation.format(),
                "response": result.response,
                "overall_score": result.overall_score,
                "timestamp": result.timestamp
            }
            
            # Add individual metrics
            for metric_name, score in result.metrics.items():
                row[f"metric_{metric_name}"] = score
            
            # Add metadata
            for key, value in result.metadata.items():
                row[f"meta_{key}"] = value
            
            # Add variation metadata
            for key, value in result.variation.metadata.items():
                row[f"var_{key}"] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by overall score
        df = df.sort_values('overall_score', ascending=False)
        
        return df
    
    async def save_experiment(
        self,
        experiment_name: str,
        results: List[PromptTestResult],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save experiment results."""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        experiment_data = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "metadata": metadata or {},
            "results": [result.to_dict() for result in results],
            "summary": {
                "total_tests": len(results),
                "avg_score": np.mean([r.overall_score for r in results]),
                "best_score": max([r.overall_score for r in results]) if results else 0,
                "worst_score": min([r.overall_score for r in results]) if results else 0
            }
        }
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(experiment_data, indent=2, default=str))
        
        logger.info(f"Experiment saved to: {filepath}")
        return filepath