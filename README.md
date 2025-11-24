# Thesis Evolution Benchmark

A domain-agnostic evolutionary benchmark platform for optimizing prompts and pipeline structures for thesis (VKR) generation.

## Features

- **Mode 1: Prompt Evolution** - Evolve prompts for fixed pipeline structures
- **Mode 2: Pipeline Evolution** - Evolve pipeline structures with optimized prompts
- Meta-prompt driven evolution with mutation, crossover, and selection
- Comprehensive evaluation system (rule-based, LLM-judge, consistency)
- Full reporting and generation tracking

## Installation

```bash
pip install -e .
```

## Configuration

1. Copy `.env.example` to `.env`
2. Set your `LLM_API_KEY` and `LLM_BASE_URL` for DeepSeek API

## Usage

### Run Prompt Evolution

```bash
python -m thesis_evo_bench.cli.main run-prompt-evolution \
    --config configs/evolution_prompt_config.yaml \
    --topics configs/topics_example.yaml \
    --pipeline configs/pipeline_fixed.yaml
```

### Run Pipeline Evolution

```bash
python -m thesis_evo_bench.cli.main run-pipeline-evolution \
    --config configs/evolution_pipeline_config.yaml \
    --topics configs/topics_example.yaml \
    --initial-population configs/pipelines_population_example.yaml
```

### Run Single Pipeline

```bash
python -m thesis_evo_bench.cli.main run-single-pipeline \
    --pipeline configs/pipeline_fixed.yaml \
    --topic "Your topic here"
```

## Project Structure

```
thesis-evo-bench/
├── thesis_evo_bench/      # Main package
│   ├── config/           # Configuration management
│   ├── llm/              # LLM client wrapper
│   ├── core/             # Core evolution logic
│   └── cli/              # Command-line interface
├── configs/              # YAML configuration files
├── data/                 # Output data (runs, generations, outputs, evaluations, reports)
└── README.md
```

## Requirements

- Python 3.10+
- DeepSeek API access (uses `deepseek-reasoner` model by default)

