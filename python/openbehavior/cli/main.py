"""
Command line interface for OpenBehavior platform.
"""

import asyncio
import click
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config import Config
from ..utils.logging import setup_logging, get_logger
from ..data.generator import DataGenerator, GenerationConfig
from ..prompts.templates import PromptLibrary
from ..prompts.engineer import PromptEngineer
from ..evaluation.ethical import EthicalEvaluator, EthicalEvalConfig
from ..evaluation.safety import SafetyEvaluator
from ..evaluation.alignment import AlignmentEvaluator
from ..models.interface import ModelFactory

logger = get_logger(__name__)

@click.group()
@click.option('--config', '-c', default='config.yaml', help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, config, log_level):
    """OpenBehavior: Platform for evaluating and aligning LLM behavior."""
    ctx.ensure_object(dict)
    
    # Setup logging
    setup_logging(level=log_level)
    
    # Load configuration
    try:
        ctx.obj['config'] = Config.from_file(config)
        ctx.obj['config'].create_directories()
    except FileNotFoundError:
        logger.warning(f"Configuration file {config} not found, using defaults")
        ctx.obj['config'] = Config()
        ctx.obj['config'].create_directories()
    
    logger.info("OpenBehavior CLI initialized")

@cli.group()
def data():
    """Data generation and management commands."""
    pass

@data.command()
@click.option('--template-id', required=True, help='Template ID to use for generation')
@click.option('--num-samples', default=100, help='Number of samples to generate')
@click.option('--output-path', help='Output file path')
@click.option('--model', help='Model to use for generation')
@click.option('--variables', help='Variables JSON file or string')
@click.pass_context
def generate(ctx, template_id, num_samples, output_path, model, variables):
    """Generate evaluation data from templates."""
    config = ctx.obj['config']
    
    # Parse variables
    variables_dict = {}
    if variables:
        try:
            if Path(variables).exists():
                with open(variables) as f:
                    variables_dict = json.load(f)
            else:
                variables_dict = json.loads(variables)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            click.echo(f"Error parsing variables: {e}", err=True)
            sys.exit(1)
    
    # Setup generation config
    gen_config = GenerationConfig(
        template_id=template_id,
        num_samples=num_samples,
        output_path=output_path or f"{config.data.output_dir}/{template_id}_{num_samples}.jsonl",
        model_name=model or config.model.name,
        variables=variables_dict
    )
    
    async def run_generation():
        # Load prompt library
        prompt_library = PromptLibrary()
        prompt_library.load_from_directory(config.data.template_dir)
        
        # Create model interface
        model_interface = ModelFactory.create(
            provider=config.model.provider,
            api_key=config.model.api_key,
            rate_limit=config.model.rate_limit,
            enable_cache=config.enable_cache
        )
        
        # Create generator
        generator = DataGenerator(gen_config, prompt_library, model_interface)
        
        # Generate data
        click.echo(f"Generating {num_samples} samples using template '{template_id}'...")
        data_points = await generator.generate()
        
        # Save results
        output_path = await generator.save_to_jsonl(data_points)
        
        # Generate statistics
        stats = generator.get_generation_statistics(data_points)
        stats_path = output_path.replace('.jsonl', '_stats.json')
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        click.echo(f"Generated {len(data_points)} data points")
        click.echo(f"Data saved to: {output_path}")
        click.echo(f"Statistics saved to: {stats_path}")
    
    asyncio.run(run_generation())

@cli.group()
def prompts():
    """Prompt engineering and testing commands."""
    pass

@prompts.command()
@click.option('--template-id', required=True, help='Template ID to optimize')
@click.option('--parameter-space', required=True, help='Parameter space JSON file')
@click.option('--test-prompts', required=True, help='Test prompts JSON file')
@click.option('--iterations', default=20, help='Number of optimization iterations')
@click.option('--model', help='Model to use for testing')
@click.pass_context
def optimize(ctx, template_id, parameter_space, test_prompts, iterations, model):
    """Optimize prompt parameters using Bayesian optimization."""
    config = ctx.obj['config']
    
    # Load parameter space
    try:
        with open(parameter_space) as f:
            param_space = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        click.echo(f"Error loading parameter space: {e}", err=True)
        sys.exit(1)
    
    # Load test prompts
    try:
        with open(test_prompts) as f:
            test_prompt_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        click.echo(f"Error loading test prompts: {e}", err=True)
        sys.exit(1)
    
    async def run_optimization():
        # Setup components
        prompt_library = PromptLibrary()
        prompt_library.load_from_directory(config.data.template_dir)
        
        model_interface = ModelFactory.create(
            provider=config.model.provider,
            api_key=config.model.api_key
        )
        
        engineer = PromptEngineer(model_interface, prompt_library)
        
        # Get base template
        if template_id not in prompt_library.templates:
            click.echo(f"Template '{template_id}' not found", err=True)
            sys.exit(1)
        
        base_template = prompt_library.templates[template_id].template
        
        # Run optimization
        click.echo(f"Optimizing prompt '{template_id}' for {iterations} iterations...")
        
        best_variation, results = await engineer.optimize_prompt(
            base_template=base_template,
            parameter_space=param_space,
            test_prompts=test_prompt_list,
            max_iterations=iterations,
            model=model or config.model.name
        )
        
        # Save results
        experiment_path = await engineer.save_experiment(
            f"optimize_{template_id}",
            results,
            metadata={
                "template_id": template_id,
                "iterations": iterations,
                "parameter_space": param_space,
                "best_score": best_variation.overall_score if best_variation else 0
            }
        )
        
        click.echo(f"Optimization complete!")
        click.echo(f"Best score: {best_variation.overall_score:.3f}")
        click.echo(f"Results saved to: {experiment_path}")
        
        if best_variation:
            click.echo(f"Best template: {best_variation.template}")
    
    asyncio.run(run_optimization())

@prompts.command()
@click.option('--template', required=True, help='Template to test')
@click.option('--variables', help='Variables JSON file or string')
@click.option('--model', help='Model to use for testing')
@click.option('--interactive', is_flag=True, help='Interactive testing mode')
@click.pass_context
def test(ctx, template, variables, model, interactive):
    """Test a prompt template."""
    config = ctx.obj['config']
    
    # Parse variables
    variables_dict = {}
    if variables:
        try:
            if Path(variables).exists():
                with open(variables) as f:
                    variables_dict = json.load(f)
            else:
                variables_dict = json.loads(variables)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            click.echo(f"Error parsing variables: {e}", err=True)
            sys.exit(1)
    
    async def run_test():
        # Create model interface
        model_interface = ModelFactory.create(
            provider=config.model.provider,
            api_key=config.model.api_key
        )
        
        if interactive:
            # Interactive mode
            click.echo("Interactive prompt testing mode. Type 'quit' to exit.")
            
            while True:
                user_input = click.prompt("Enter test input (or 'quit')")
                if user_input.lower() == 'quit':
                    break
                
                # Format template with user input
                formatted_template = template.format(input=user_input, **variables_dict)
                
                click.echo(f"Formatted prompt: {formatted_template}")
                click.echo("Generating response...")
                
                try:
                    response = await model_interface.generate(
                        prompt=formatted_template,
                        model=model or config.model.name,
                        temperature=config.model.temperature,
                        max_tokens=config.model.max_tokens
                    )
                    
                    click.echo(f"Response: {response}")
                    click.echo("-" * 50)
                
                except Exception as e:
                    click.echo(f"Error: {e}", err=True)
        else:
            # Single test
            formatted_template = template.format(**variables_dict)
            
            click.echo(f"Testing template: {formatted_template}")
            
            try:
                response = await model_interface.generate(
                    prompt=formatted_template,
                    model=model or config.model.name,
                    temperature=config.model.temperature,
                    max_tokens=config.model.max_tokens
                )
                
                click.echo(f"Response: {response}")
            
            except Exception as e:
                click.echo(f"Error: {e}", err=True)
    
    asyncio.run(run_test())

@cli.group()
def evaluate():
    """Evaluation commands."""
    pass

@evaluate.command()
@click.option('--data-path', required=True, help='Path to data file (JSONL)')
@click.option('--output-path', help='Output path for results')
@click.option('--dimensions', help='Comma-separated list of ethical dimensions')
@click.option('--evaluator-model', help='Model to use for evaluation')
@click.pass_context
def ethical(ctx, data_path, output_path, dimensions, evaluator_model):
    """Run ethical evaluation on data."""
    config = ctx.obj['config']
    
    # Parse dimensions
    eval_dimensions = dimensions.split(',') if dimensions else config.evaluation.ethical_dimensions
    
    async def run_ethical_evaluation():
        # Load data
        try:
            data_points = []
            with open(data_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    data_points.append(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            click.echo(f"Error loading data: {e}", err=True)
            sys.exit(1)
        
        # Create model interface
        model_interface = ModelFactory.create(
            provider=config.model.provider,
            api_key=config.model.api_key
        )
        
        # Create evaluator
        eval_config = EthicalEvalConfig(
            evaluator_model=evaluator_model or config.evaluation.evaluator_model,
            confidence_threshold=config.evaluation.confidence_threshold
        )
        
        evaluator = EthicalEvaluator(eval_config, model_interface)
        
        # Run evaluation
        click.echo(f"Running ethical evaluation on {len(data_points)} items...")
        
        results = []
        with click.progressbar(data_points) as bar:
            for i, data_point in enumerate(bar):
                try:
                    result = await evaluator.evaluate(
                        text_id=data_point.get('id', str(i)),
                        text=data_point.get('response', data_point.get('text', ''))
                    )
                    results.append(result)
                
                except Exception as e:
                    logger.error(f"Error evaluating item {i}: {e}")
                    continue
        
        # Save results
        output_file = output_path or f"{config.data.output_dir}/ethical_evaluation_{len(results)}.json"
        
        with open(output_file, 'w') as f:
            json.dump([result.to_dict() for result in results], f, indent=2, default=str)
        
        # Generate summary
        summary = {
            "total_evaluated": len(results),
            "average_score": sum(r.overall_score for r in results) / len(results) if results else 0,
            "dimension_averages": {}
        }
        
        # Calculate dimension averages
        from collections import defaultdict
        dimension_scores = defaultdict(list)
        
        for result in results:
            for dimension, score in result.dimension_scores.items():
                dimension_scores[dimension.value].append(score)
        
        for dimension, scores in dimension_scores.items():
            summary["dimension_averages"][dimension] = sum(scores) / len(scores)
        
        click.echo(f"Evaluation complete!")
        click.echo(f"Results saved to: {output_file}")
        click.echo(f"Average score: {summary['average_score']:.3f}")
        
        for dimension, avg_score in summary["dimension_averages"].items():
            click.echo(f"  {dimension}: {avg_score:.3f}")
    
    asyncio.run(run_ethical_evaluation())

@evaluate.command()
@click.option('--data-path', required=True, help='Path to data file (JSONL)')
@click.option('--output-path', help='Output path for results')
@click.option('--evaluator-model', help='Model to use for evaluation')
@click.pass_context
def safety(ctx, data_path, output_path, evaluator_model):
    """Run safety evaluation on data."""
    config = ctx.obj['config']
    
    async def run_safety_evaluation():
        # Load data
        try:
            data_points = []
            with open(data_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    data_points.append(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            click.echo(f"Error loading data: {e}", err=True)
            sys.exit(1)
        
        # Create model interface
        model_interface = ModelFactory.create(
            provider=config.model.provider,
            api_key=config.model.api_key
        )
        
        # Create evaluator
        evaluator = SafetyEvaluator(
            model_interface=model_interface,
            evaluator_model=evaluator_model or config.evaluation.evaluator_model
        )
        
        # Run evaluation
        click.echo(f"Running safety evaluation on {len(data_points)} items...")
        
        results = []
        with click.progressbar(data_points) as bar:
            for i, data_point in enumerate(bar):
                try:
                    result = await evaluator.evaluate(
                        text_id=data_point.get('id', str(i)),
                        text=data_point.get('response', data_point.get('text', ''))
                    )
                    results.append(result)
                
                except Exception as e:
                    logger.error(f"Error evaluating item {i}: {e}")
                    continue
        
        # Save results
        output_file = output_path or f"{config.data.output_dir}/safety_evaluation_{len(results)}.json"
        
        with open(output_file, 'w') as f:
            json.dump([result.to_dict() for result in results], f, indent=2, default=str)
        
        # Generate summary
        risk_levels = {}
        for result in results:
            risk_level = result.risk_level
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
        
        click.echo(f"Safety evaluation complete!")
        click.echo(f"Results saved to: {output_file}")
        click.echo(f"Risk level distribution:")
        
        for level, count in risk_levels.items():
            percentage = (count / len(results)) * 100
            click.echo(f"  {level}: {count} ({percentage:.1f}%)")
    
    asyncio.run(run_safety_evaluation())

@evaluate.command()
@click.option('--data-path', required=True, help='Path to data file (JSONL)')
@click.option('--output-path', help='Output path for results')
@click.option('--evaluator-model', help='Model to use for evaluation')
@click.pass_context
def alignment(ctx, data_path, output_path, evaluator_model):
    """Run alignment evaluation on data."""
    config = ctx.obj['config']
    
    async def run_alignment_evaluation():
        # Load data
        try:
            data_points = []
            with open(data_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    data_points.append(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            click.echo(f"Error loading data: {e}", err=True)
            sys.exit(1)
        
        # Create model interface
        model_interface = ModelFactory.create(
            provider=config.model.provider,
            api_key=config.model.api_key
        )
        
        # Create evaluator
        evaluator = AlignmentEvaluator(
            model_interface=model_interface,
            evaluator_model=evaluator_model or config.evaluation.evaluator_model
        )
        
        # Run evaluation
        click.echo(f"Running alignment evaluation on {len(data_points)} items...")
        
        results = []
        with click.progressbar(data_points) as bar:
            for i, data_point in enumerate(bar):
                try:
                    result = await evaluator.evaluate(
                        text_id=data_point.get('id', str(i)),
                        text=data_point.get('response', data_point.get('text', ''))
                    )
                    results.append(result)
                
                except Exception as e:
                    logger.error(f"Error evaluating item {i}: {e}")
                    continue
        
        # Save results
        output_file = output_path or f"{config.data.output_dir}/alignment_evaluation_{len(results)}.json"
        
        with open(output_file, 'w') as f:
            json.dump([result.to_dict() for result in results], f, indent=2, default=str)
        
        # Generate summary
        avg_score = sum(r.overall_score for r in results) / len(results) if results else 0
        
        click.echo(f"Alignment evaluation complete!")
        click.echo(f"Results saved to: {output_file}")
        click.echo(f"Average alignment score: {avg_score:.3f}")
        
        # Show recommendations summary
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        from collections import Counter
        common_recommendations = Counter(all_recommendations).most_common(5)
        
        if common_recommendations:
            click.echo("Most common recommendations:")
            for rec, count in common_recommendations:
                click.echo(f"  {rec} ({count} times)")
    
    asyncio.run(run_alignment_evaluation())

@cli.command()
@click.option('--host', default='localhost', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.pass_context
def server(ctx, host, port, reload):
    """Start the OpenBehavior API server."""
    try:
        import uvicorn
        from ..api.main import create_app
    except ImportError:
        click.echo("FastAPI dependencies not installed. Install with 'pip install fastapi uvicorn'", err=True)
        sys.exit(1)
    
    config = ctx.obj['config']
    app = create_app(config)
    
    click.echo(f"Starting OpenBehavior API server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=config.logging_level.lower()
    )

@cli.command()
@click.pass_context
def dashboard(ctx):
    """Launch the OpenBehavior dashboard."""
    import subprocess
    import os
    
    # Get the dashboard directory
    dashboard_dir = Path(__file__).parent.parent.parent / "dashboard"
    
    if not dashboard_dir.exists():
        click.echo("Dashboard not found. Please install the dashboard components.", err=True)
        sys.exit(1)
    
    try:
        # Check if npm is available
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        
        # Start the dashboard
        click.echo("Starting OpenBehavior dashboard...")
        os.chdir(dashboard_dir)
        subprocess.run(["npm", "run", "dev"], check=True)
        
    except subprocess.CalledProcessError:
        click.echo("npm not found. Please install Node.js and npm.", err=True)
        sys.exit(1)
    except FileNotFoundError:
        click.echo("npm not found. Please install Node.js and npm.", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli()