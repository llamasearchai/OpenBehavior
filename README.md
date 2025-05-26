# OpenBehavior

**Advanced LLM Behavior Analysis and Evaluation Platform**

OpenBehavior is a comprehensive framework for evaluating, analyzing, and understanding the behavior of Large Language Models (LLMs). It provides tools for safety assessment, ethical evaluation, alignment analysis, and behavioral pattern detection.

## Features

### Core Capabilities

* **Multi-dimensional Evaluation**: Safety, ethical, and alignment assessments
* **Behavioral Analysis**: Pattern detection and behavior insights
* **Model Comparison**: Side-by-side analysis of different models
* **Batch Processing**: Efficient evaluation of multiple texts
* **Real-time API**: RESTful API for integration with other systems
* **CLI Interface**: Command-line tools for easy interaction

### Evaluation Dimensions

* **Safety**: Harm prevention, toxicity detection, safety guidelines compliance
* **Ethics**: Fairness, honesty, transparency assessment
* **Alignment**: Helpfulness, relevance, instruction following

### Technical Features

* **Async Processing**: High-performance concurrent evaluations
* **Caching System**: Intelligent caching with TTL support
* **Database Integration**: SQLite-based storage for results
* **Extensible Architecture**: Plugin-based evaluation system
* **Comprehensive Testing**: Full test coverage for core functionality

## Installation

### Prerequisites

* Python 3.8+
* OpenAI API key (optional)
* Anthropic API key (optional)

### Install from Source

```bash
git clone https://github.com/llamasearchai/OpenBehavior.git
cd OpenBehavior
pip install -r requirements.txt
pip install -e .
```

### Configuration

Create a configuration file:

```bash
python -m openbehavior.cli config init --output config.yaml
```

Edit the configuration file with your API keys:

```yaml
openai:
  api_key: "your-openai-api-key"
  default_model: "gpt-4"

anthropic:
  api_key: "your-anthropic-api-key"

model:
  default: "gpt-4"

engine:
  max_concurrent: 10
  timeout: 30

cache:
  enabled: true
  ttl: 3600

database:
  url: "sqlite:///evaluations.db"

api:
  host: "localhost"
  port: 8000
```

## Usage

### Command Line Interface

#### Basic Text Evaluation

```bash
python -m openbehavior.cli evaluate \
  --text "Your text to evaluate" \
  --types safety,ethical,alignment
```

#### Batch Evaluation

```bash
python -m openbehavior.cli batch-evaluate \
  --input prompts.txt \
  --types safety \
  --output-file results.json
```

#### Model Analysis

```bash
python -m openbehavior.cli analyze \
  --model gpt-4 \
  --prompts "Test prompt 1" "Test prompt 2" \
  --depth standard
```

#### Model Comparison

```bash
python -m openbehavior.cli compare \
  --model-a gpt-4 \
  --model-b gpt-3.5-turbo \
  --prompts "Compare these models"
```

#### Interactive Mode

```bash
python -m openbehavior.cli interactive
```

### Python API

#### Basic Usage

```python
import asyncio
from openbehavior.core.behavior_engine import BehaviorEngine, EngineConfig

async def main():
    # Configure the engine
    config = EngineConfig(
        openai_api_key="your-api-key",
        default_model="gpt-4"
    )
    
    # Create engine
    engine = BehaviorEngine(config=config)
    
    # Evaluate text
    result = await engine.evaluate_text(
        text="Hello, how can I help you today?",
        evaluation_types=["safety", "ethical", "alignment"]
    )
    
    print(f"Overall Score: {result.overall_score}")
    print(f"Safety Score: {result.safety_score.overall_score}")
    print(f"Ethical Score: {result.ethical_score.overall_score}")
    
    await engine.close()

asyncio.run(main())
```

#### Batch Processing

```python
async def batch_example():
    engine = BehaviorEngine(config=config)
    
    texts = [
        "First text to evaluate",
        "Second text to evaluate",
        "Third text to evaluate"
    ]
    
    results = await engine.evaluate_batch(
        texts=texts,
        evaluation_types=["safety"],
        max_concurrent=5
    )
    
    for result in results.results:
        print(f"Text: {result.prompt}")
        print(f"Score: {result.overall_score}")
    
    await engine.close()
```

#### Model Analysis

```python
async def analysis_example():
    engine = BehaviorEngine(config=config)
    
    analysis = await engine.analyze_model_behavior(
        model="gpt-4",
        test_prompts=["Test prompt 1", "Test prompt 2"],
        analysis_depth="comprehensive"
    )
    
    print("Behavior Analysis:", analysis["behavior_analysis"])
    print("Statistics:", analysis["statistics"])
    
    await engine.close()
```

### REST API

#### Start the API Server

```bash
python -m openbehavior.api.main
```

#### API Endpoints

**Evaluate Text**

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here",
    "evaluation_types": ["safety", "ethical"]
  }'
```

**Batch Evaluation**

```bash
curl -X POST "http://localhost:8000/evaluate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Text 1", "Text 2"],
    "evaluation_types": ["safety"]
  }'
```

**Model Analysis**

```bash
curl -X POST "http://localhost:8000/analyze/model" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "test_prompts": ["Test prompt"]
  }'
```

## Architecture

### Core Components

* **BehaviorEngine**: Multi-dimensional text evaluation and analysis
* **SafetyEvaluator**: Safety assessment and harm detection
* **EthicalEvaluator**: Ethical evaluation and bias detection
* **AlignmentEvaluator**: Alignment and instruction following assessment
* **PromptEngineer**: Advanced prompt engineering and optimization
* **DataGenerator**: Synthetic data generation for testing

### Evaluation Pipeline

1. **Text Preprocessing**: Input validation and normalization
2. **Multi-dimensional Analysis**: Parallel evaluation across safety, ethical, and alignment dimensions
3. **Score Aggregation**: Weighted combination of dimension scores
4. **Result Storage**: Optional database persistence
5. **Insight Generation**: Behavioral pattern detection and recommendations

## Testing

Run the test suite:

```bash
# Run all core tests
python -m pytest python/tests/openbehavior/ -v

# Run with coverage
python -m pytest --cov=openbehavior --cov-report=html
```

## Evaluation Metrics

### Safety Scores

* **Harm Prevention**: Assessment of potential physical, psychological, or social harm
* **Toxicity Detection**: Identification of toxic, offensive, or inappropriate content
* **Safety Guidelines**: Compliance with established safety protocols

### Ethical Scores

* **Fairness**: Evaluation of bias and equitable treatment
* **Honesty**: Assessment of truthfulness and accuracy
* **Transparency**: Clarity and openness in communication

### Alignment Scores

* **Helpfulness**: Utility and value of the response
* **Relevance**: Appropriateness to the given context
* **Instruction Following**: Adherence to specific instructions or guidelines

## Extensibility

### Custom Evaluators

```python
from openbehavior.evaluation.safety import SafetyEvaluator

class CustomEvaluator(SafetyEvaluator):
    async def evaluate_custom_dimension(self, text: str) -> CustomScore:
        # Implement your custom evaluation logic
        pass
```

### Plugin System

```python
from openbehavior.core.behavior_engine import BehaviorEngine

# Register custom evaluator
engine = BehaviorEngine()
engine.register_evaluator("custom", CustomEvaluator())
```

## Performance

* **Concurrent Processing**: Up to 10 concurrent evaluations by default
* **Caching**: Intelligent result caching with configurable TTL
* **Batch Optimization**: Efficient processing of multiple texts
* **Memory Management**: Optimized for large-scale evaluations

## Contributing

We welcome contributions! Please see our Contributing Guidelines for details.

### Development Setup

```bash
git clone https://github.com/llamasearchai/OpenBehavior.git
cd OpenBehavior
pip install -r requirements.txt
pip install -e .
```

### Running Tests

```bash
python -m pytest python/tests/ -v
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* OpenAI for GPT models and API
* Anthropic for Claude models and API
* The open-source community for various dependencies

## Support

* **Issues**: GitHub Issues
* **Discussions**: GitHub Discussions
* **Email**: support@llamasearch.ai

## Roadmap

* Additional model provider integrations
* Advanced visualization dashboard
* Real-time monitoring and alerting
* Custom evaluation template system
* Integration with popular ML frameworks
* Enhanced behavioral pattern detection
* Multi-language support

---

**OpenBehavior** - Advanced LLM Behavior Analysis and Evaluation Platform