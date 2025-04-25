# Episodic Memory LLM Benchmarking System

This document provides an overview of the benchmarking system for the Episodic Memory LLM project. The benchmarking system allows you to evaluate the performance improvements that episodic memory capabilities bring to large language models (LLMs).

## Overview

The benchmarking system compares:
1. **Traditional LLM Performance**: Standard language model responses without memory
2. **Episodic Memory-Enhanced Performance**: The same model augmented with episodic memory

It measures improvements across multiple metrics, including:
- Response quality and relevance
- Conversation coherence
- Context-awareness
- Factual accuracy
- Entity recall and memory usage

## Running Benchmarks

### Basic Usage

```bash
# Run a benchmark with the default model
python -m src.run_benchmarks

# Run a benchmark with a specific model
python -m src.run_benchmarks --model llama-3.1-8b-instant

# Run benchmarks on all supported models
python -m src.run_benchmarks --all-models
```

### Command Line Options

The benchmarking script supports the following options:

| Option | Description |
|--------|-------------|
| `--model MODEL_NAME` | Specify a single model to benchmark |
| `--all-models` | Run benchmarks on all supported models |
| `--output-dir DIR` | Directory to save visualization outputs (default: "visualizations") |
| `--simple` | Run with simplified test scenarios for quick testing |
| `--advanced` | Run with advanced test scenarios (more comprehensive) |
| `--text-output FILE` | Custom path for the detailed results text file |
| `--no-browser` | Don't open browser with results when complete |
| `--verbose` | Display detailed progress information |

### Supported Models

The benchmarking system currently supports these models via the Groq API:

- qwen-2.5-32b
- mixtral-8x7b-32768
- mistral-saba-24b
- llama-3.1-8b-instant
- llama-3.2-3b-preview
- llama-3.3-70b-versatile

## Output and Visualizations

The benchmarking system generates:

1. **Per-Model Visualizations**: Charts showing performance with and without episodic memory
2. **Multi-Model Comparison**: When multiple models are benchmarked, comparative visualizations are generated
3. **Detailed Text Report**: A comprehensive text file with metric values, improvement percentages, and test scenario details

### Visualization Types

- **Accuracy Comparison**: Compares factual accuracy with/without episodic memory
- **Memory Usage**: Shows how efficiently memory is utilized
- **Response Quality**: Evaluates response quality metrics
- **Performance Radar**: Multi-dimensional view of various capabilities
- **Conversation Coherence**: Measures conversation flow improvements

### Combined Model Comparison

When benchmarking multiple models, the system generates:
- Overall model performance comparison
- Improvement percentages across key metrics
- Capabilities radar chart comparing all models
- Best model by metric analysis

## Interpreting Results

The text report provides:
1. **Per-Model Performance**: Detailed metrics for each model
2. **Improvement Percentages**: How much episodic memory improved each metric
3. **Model Ranking**: Overall ranking of models based on memory enhancement benefits
4. **Test Scenario Details**: Information about the test scenarios used

## Example

To run a quick benchmark on a single model:

```bash
python -m src.run_benchmarks --model llama-3.1-8b-instant --simple
```

To run a comprehensive benchmark across all models:

```bash
python -m src.run_benchmarks --all-models --advanced --output-dir results
```

## Requirements

- Python 3.8+
- A valid Groq API key (set in `.env` file)
- Required Python packages (listed in requirements.txt)

## Configuration

Benchmarking configuration is stored in `.env`:
- `GROQ_API_KEY`: Your Groq API key
- `MODEL_NAME`: Default model to use if not specified
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Neo4j connection details for knowledge graph
- `EMBEDDING_MODEL`: Model to use for embeddings (default: all-MiniLM-L6-v2) 