import asyncio
import os
import argparse
import webbrowser
from dotenv import load_dotenv
from benchmarking import BenchmarkingSystem
from test_scenarios import create_model_specific_scenarios, get_simple_test_scenarios, get_advanced_test_scenarios
from episodic_memory import EpisodicMemoryModule, MemoryConfig
from llm_providers.groq_provider import GroqProvider
from create_index import create_index_html

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run benchmarks for Episodic Memory Module')
    parser.add_argument('--model', type=str, default=os.environ.get('MODEL_NAME', 'llama3-8b-8192'), 
                        help='Model name to use for benchmarking')
    parser.add_argument('--output-dir', type=str, default=os.environ.get('BENCHMARK_OUTPUT_DIR', 'visualizations'),
                        help='Directory to save visualization outputs')
    parser.add_argument('--compare-models', action='store_true',
                        help='Compare performance across multiple models')
    parser.add_argument('--simple', action='store_true',
                        help='Run with simplified test scenarios for quick testing')
    parser.add_argument('--advanced', action='store_true',
                        help='Run with advanced test scenarios for comprehensive testing')
    args = parser.parse_args()
    
    # Create visualizations directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load environment variables from .env file
    print("Loading environment variables from .env file...")
    load_dotenv()
    
    # Get Groq API key from environment
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        print("Error: GROQ_API_KEY environment variable is not set")
        return
    
    # Get Neo4j connection details
    neo4j_uri = os.environ.get('NEO4J_URI')
    if not neo4j_uri:
        print("Warning: NEO4J_URI not found in .env file")
        print("Using default Neo4j URI: bolt://localhost:7687")
        neo4j_uri = "bolt://localhost:7687"
    
    neo4j_user = os.environ.get('NEO4J_USER')
    if not neo4j_user:
        print("Warning: NEO4J_USER not found in .env file")
        print("Using default Neo4j user: neo4j")
        neo4j_user = "neo4j"
    
    neo4j_password = os.environ.get('NEO4J_PASSWORD')
    if not neo4j_password:
        print("Warning: NEO4J_PASSWORD not found in .env file")
        print("Using default Neo4j password: password")
        neo4j_password = "password"
    
    print("\n===================================================")
    print(f"Starting benchmark for {args.model}")
    print("===================================================")
    
    # Get embedding model from environment
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    print(f"Using embedding model: {embedding_model}")
    print(f"Using Neo4j at: {neo4j_uri}")
    print(f"Output directory: {args.output_dir}")
    
    if args.compare_models:
        # Run benchmarks for multiple models
        await run_model_comparison(
            groq_api_key, 
            args.output_dir, 
            args.simple,
            args.advanced,
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            embedding_model
        )
    else:
        # Run benchmark for a single model
        await run_single_model_benchmark(
            groq_api_key, 
            args.model, 
            args.output_dir, 
            args.simple,
            args.advanced,
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            embedding_model
        )
    
    # Create index.html file
    print("Creating index.html...")
    create_index_html(args.output_dir)
    
    print("\n===================================================")
    print("Benchmarking complete")
    print("===================================================")
    
    # Open the results in the default browser
    index_path = os.path.join(args.output_dir, "index.html")
    print(f"\nResults are available at: {index_path}")
    print("Opening results in your default browser...")
    webbrowser.open(f"file://{os.path.abspath(index_path)}")

async def run_single_model_benchmark(
    api_key: str, 
    model_name: str, 
    output_dir: str, 
    simple: bool = False,
    advanced: bool = False,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """Run benchmark for a single model"""
    print("\nChecking required packages...")
    print(f"Running benchmarks for model: {model_name}")
    
    # Initialize LLM provider using the existing GroqProvider
    llm_provider = GroqProvider(
        api_key=api_key, 
        model_name=model_name,
        temperature=0.7,
        max_tokens=1000
    )
    
    # Get test scenarios based on flags
    if simple:
        print("Using simplified test scenarios for quick testing")
        test_scenarios = get_simple_test_scenarios()
    elif advanced:
        print("Using advanced test scenarios for comprehensive testing")
        test_scenarios = get_advanced_test_scenarios()
    else:
        # Default to model-specific scenarios
        test_scenarios = create_model_specific_scenarios(model_name)
    
    # Initialize benchmarking system
    benchmark_system = BenchmarkingSystem(
        llm_provider=llm_provider,
        test_scenarios=test_scenarios,
        model_name=model_name
    )
    
    # Create memory configuration
    memory_config = MemoryConfig(
        max_context_items=10,
        memory_decay_factor=0.95,
        importance_threshold=0.5,
        min_references_to_keep=2,
        decay_interval=999999,  # Disable automatic decay for benchmarking
        cleanup_interval=999999,  # Disable automatic cleanup for benchmarking
        decay_check_interval=999999  # Disable automatic decay checks for benchmarking
    )
    
    try:
        # Initialize Episodic Memory Module with the correct parameters
        memory_module = EpisodicMemoryModule(
            llm_provider=llm_provider,
            collection_name=f"benchmark_{model_name}",
            embedding_model=embedding_model,
            config=memory_config,
            # Add Neo4j connection parameters
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
        
        print("Running traditional LLM benchmark...")
        await benchmark_system.run_traditional_benchmark()
        
        print("Running Episodic Memory benchmark...")
        await benchmark_system.run_episodic_benchmark(memory_module)
        
        print("Generating visualizations...")
        model_output_dir = f"{output_dir}/{model_name}"
        os.makedirs(model_output_dir, exist_ok=True)
        benchmark_system.generate_visualizations(model_output_dir)
        
        print(f"Benchmarking complete! Visualizations have been saved to '{model_output_dir}'")
        
        # Clean up after benchmarking
        duplicates, orphaned = await memory_module.cleanup_memories()
        print(f"\nCleaned up {duplicates} duplicate memories")
        print(f"Cleaned up {orphaned} orphaned entities")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

async def run_model_comparison(
    api_key: str, 
    output_dir: str, 
    simple: bool = False,
    advanced: bool = False,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    embedding_model: str = "all-MiniLM-L6-v2"
):
    """Run benchmarks for multiple models and compare results"""
    # Define models to compare (one smaller, one larger)
    models = ['llama3-8b-8192', 'llama3-70b-8192']
    
    # Run benchmarks for each model
    for model in models:
        await run_single_model_benchmark(
            api_key, 
            model, 
            output_dir, 
            simple,
            advanced,
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            embedding_model
        )
    
    # Generate cross-model comparison visualizations
    await generate_model_comparison_visualizations(models, output_dir)

async def generate_model_comparison_visualizations(models: list, output_dir: str):
    """Generate visualizations comparing performance across models"""
    # This would load the results from each model's benchmark and create comparison visualizations
    # For simplicity, we'll just create a placeholder HTML file with instructions
    
    comparison_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cross-Model Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .model-links {{ margin: 20px 0; }}
            .model-link {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Cross-Model Comparison</h1>
        <p>View individual model results:</p>
        <div class="model-links">
    """
    
    for model in models:
        comparison_html += f'        <div class="model-link"><a href="./{model}/comprehensive_comparison.html">{model} - Comprehensive Comparison</a></div>\n'
        comparison_html += f'        <div class="model-link"><a href="./{model}/capabilities_radar.html">{model} - Capabilities Radar</a></div>\n'
    
    comparison_html += """
        </div>
        <p>To generate a true cross-model comparison, run the analysis script on the individual model results.</p>
    </body>
    </html>
    """
    
    with open(f"{output_dir}/model_comparison.html", "w") as f:
        f.write(comparison_html)
    
    print(f"Model comparison page created at '{output_dir}/model_comparison.html'")

if __name__ == "__main__":
    asyncio.run(main()) 