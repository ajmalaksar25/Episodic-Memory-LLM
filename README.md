# Episodic Memory LLM

A memory-enhanced language model system that provides episodic memory capabilities to large language models.

## Project Structure

```
.
├── main.py                  # Main entry point
├── run_benchmark.bat        # Batch file to run benchmarks
├── run_analyze.bat          # Batch file to analyze benchmark results
├── run_chat.bat             # Batch file to run the chat interface
├── run_web.bat              # Batch file to run the web interface
├── run_tests.bat            # Batch file to run tests
├── requirements.txt         # Project dependencies
├── src/                     # Source code
│   ├── __init__.py          # Package initialization
│   ├── episodic_memory.py   # Core episodic memory module
│   ├── chatbot.py           # Smart chatbot with memory
│   ├── benchmarking.py      # Benchmarking system
│   ├── test_scenarios.py    # Test scenarios for benchmarking
│   ├── run_benchmarks.py    # Benchmark runner
│   ├── example_usage.py     # Example usage of the memory module
│   ├── interface.py         # Command-line interface
│   ├── streamlit_interface.py # Streamlit web interface
│   ├── memory_bot_interface.py # Memory bot interface
│   ├── groq_conversational_bot.py # Groq conversational bot
│   ├── llm_providers/       # LLM provider implementations
│   │   ├── __init__.py      # Package initialization
│   │   ├── base.py          # Base LLM provider class
│   │   ├── groq_provider.py # Groq provider implementation
│   │   └── openai_provider.py # OpenAI provider implementation
│   ├── utils/               # Utility functions
│   │   ├── __init__.py      # Package initialization
│   │   ├── create_index.py  # Create index.html for visualizations
│   │   └── analyze_benchmarks.py # Analyze benchmark results
│   ├── tests/               # Test modules
│   │   ├── __init__.py      # Package initialization
│   │   ├── test_episodic_memory.py # Test episodic memory module
│   │   ├── test_memory_comparison.py # Test memory comparison
│   │   └── test_memory_enhanced_chatbot.py # Test memory-enhanced chatbot
│   └── visualizations/      # Visualization modules
│       └── __init__.py      # Package initialization
└── visualizations/          # Visualization outputs
    └── ...                  # Generated visualization files
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Episodic-Memory-LLM.git
   cd Episodic-Memory-LLM
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your-groq-api-key
   MODEL_NAME=llama3-8b-8192
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-neo4j-password
   ```

## Usage

### Running the Chat Interface

```
python main.py chat
```

Or use the batch file:

```
run_chat.bat
```

### Running Benchmarks

```
python main.py benchmark --model llama3-8b-8192 --output-dir visualizations --simple
```

Or use the batch file:

```
run_benchmark.bat --simple
```

### Analyzing Benchmark Results

```
python main.py analyze --model llama3-8b-8192 --output-dir visualizations --generate-report
```

Or use the batch file:

```
run_analyze.bat --model llama3-8b-8192
```

### Starting the Web Interface

```
python main.py web --port 8501
```

Or use the batch file:

```
run_web.bat
```

### Running Tests

```
python main.py test
```

Or use the batch file:

```
run_tests.bat
```

To run a specific test file:

```
python main.py test --test-file test_episodic_memory
```

## Features

- Episodic memory module for LLMs
- Memory prioritization and decay
- Knowledge graph integration with Neo4j
- Benchmarking system for memory performance
- Visualization of memory performance
- Web interface for interacting with the memory-enhanced chatbot

## License

This project is licensed under the MIT License - see the LICENSE file for details. 