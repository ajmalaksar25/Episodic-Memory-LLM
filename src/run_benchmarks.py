import os
import sys
import argparse
import asyncio
import time
import json
import webbrowser
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add src directory to path for proper imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Handle imports for both package usage and direct script execution
try:
    # When running as a module within the package
    from src.benchmarking import BenchmarkingSystem
    from src.test_scenarios import get_simple_test_scenarios, get_advanced_test_scenarios
    from src.episodic_memory import EpisodicMemoryModule, MemoryConfig
    from src.llm_providers.groq_provider import GroqProvider
    from src.utils.create_index import create_index_html
except ImportError:
    # When running directly as a script
    from benchmarking import BenchmarkingSystem
    from test_scenarios import get_simple_test_scenarios, get_advanced_test_scenarios
    from episodic_memory import EpisodicMemoryModule, MemoryConfig
    from llm_providers.groq_provider import GroqProvider
    from utils.create_index import create_index_html

# Define all models to benchmark
BENCHMARK_MODELS = [
    "qwen-2.5-32b",
    "mixtral-8x7b-32768",
    "mistral-saba-24b", 
    "llama-3.1-8b-instant",
    "llama-3.2-3b-preview",
    "llama-3.3-70b-versatile"
]

async def main():
    """Main entry point for benchmarking"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run benchmarks for memory-enhanced LLMs")
    
    parser.add_argument("--model", type=str, help="Model name to use")
    parser.add_argument("--all-models", action="store_true", help="Benchmark all models")
    parser.add_argument("--output-dir", type=str, default="visualizations", 
                        help="Directory to save visualization outputs")
    parser.add_argument("--simple", action="store_true", 
                        help="Run with simplified test scenarios for quick testing")
    parser.add_argument("--advanced", action="store_true", 
                        help="Run with advanced test scenarios for comprehensive testing")
    parser.add_argument("--text-output", type=str, default=None,
                        help="Custom path for the text output file")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser with results when complete")
    parser.add_argument("--verbose", action="store_true",
                        help="Display detailed progress information")
    parser.add_argument("--delay", type=int, default=5,
                        help="Delay in seconds between benchmarking models (default: 5)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries per model on failure (default: 3)")
    parser.add_argument("--cross-model", action="store_true",
                        help="Generate cross-model comparison visualizations")
                        
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which models to benchmark
    models_to_benchmark = []
    if args.all_models:
        models_to_benchmark = BENCHMARK_MODELS
        print(f"Benchmarking all models: {', '.join(BENCHMARK_MODELS)}")
    elif args.model:
        if args.model in BENCHMARK_MODELS:
            models_to_benchmark = [args.model]
        else:
            print(f"Warning: Model {args.model} not in known models list. Running anyway.")
            models_to_benchmark = [args.model]
    else:
        # Default to using the model from the environment variable
        default_model = os.getenv("MODEL_NAME", "llama3-8b-8192")
        models_to_benchmark = [default_model]
        print(f"No model specified, using default model: {default_model}")
    
    # Create detailed results text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.text_output:
        results_text_file = args.text_output
    else:
        results_text_file = os.path.join(args.output_dir, f"benchmark_detailed_results_{timestamp}.txt")
    
    # Initialize the text file with header
    with open(results_text_file, "w") as f:
        f.write("EPISODIC MEMORY LLM - BENCHMARK DETAILED RESULTS\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models tested: {', '.join(models_to_benchmark)}\n")
        f.write(f"{'='*80}\n\n")
    
    # Show benchmark configuration
    print("\n=== BENCHMARK CONFIGURATION ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Detailed results file: {results_text_file}")
    print(f"Number of models to benchmark: {len(models_to_benchmark)}")
    print(f"Simplified scenarios: {'Yes' if args.simple else 'No'}")
    print(f"Advanced scenarios: {'Yes' if args.advanced else 'No'}")
    print(f"Delay between models: {args.delay} seconds")
    print(f"Max retries per model: {args.max_retries}")
    print("=============================\n")
    
    # Run benchmarks for each model
    all_results = {}
    start_time_all = time.time()
    
    for i, model in enumerate(models_to_benchmark):
        # Add a delay between models to avoid rate limiting
        if i > 0 and args.delay > 0:
            print(f"Waiting {args.delay} seconds before benchmarking next model...")
            await asyncio.sleep(args.delay)
            
        # Try benchmarking with retries
        retry_count = 0
        success = False
        
        while retry_count < args.max_retries and not success:
            try:
                if retry_count > 0:
                    print(f"Retry attempt {retry_count}/{args.max_retries} for model {model}")
                    # Add exponential backoff for retries
                    backoff_time = min(30, 2 ** retry_count)
                    print(f"Waiting {backoff_time} seconds before retrying...")
                    await asyncio.sleep(backoff_time)
                
                print(f"\n{'='*30} BENCHMARKING {model} {'='*30}")
                start_time = time.time()
                results = await run_benchmark_for_model(
                    model_name=model,
                    output_dir=args.output_dir,
                    simplified=args.simple,
                    advanced=args.advanced,
                    results_text_file=results_text_file
                )
                all_results[model] = results
                elapsed_time = time.time() - start_time
                print(f"Benchmark for {model} completed in {elapsed_time:.2f} seconds")
                print(f"{'='*80}")
                success = True
            except Exception as e:
                retry_count += 1
                print(f"Error running benchmark for {model}: {e}")
                
                if "rate limit" in str(e).lower() or "429" in str(e):
                    print(f"Hit rate limit for {model}, will retry after longer delay")
                    # Add a longer delay for rate limit errors
                    await asyncio.sleep(min(60, 5 * retry_count))
                    
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                    
                if retry_count >= args.max_retries:
                    print(f"Maximum retries reached for {model}, skipping to next model")
    
    total_time = time.time() - start_time_all
    print(f"\nTotal benchmarking time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Generate combined visualization if multiple models were benchmarked
    result_path = None
    if len(all_results) > 1:
        try:
            print("\nGenerating combined model comparison visualization...")
            result_path = await generate_combined_comparison(args.output_dir, all_results)
            if result_path:
                print(f"Combined visualization saved to: {result_path}")
            
            # Add combined results summary to the text file
            write_comparison_summary(results_text_file, all_results)
            
        except Exception as e:
            print(f"Error generating combined visualization: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    elif len(all_results) == 1:
        # For single model, use its visualization path
        model = list(all_results.keys())[0]
        result_path = os.path.join(args.output_dir, model, "index.html")
    
    print("\nAll benchmarks completed!")
    print(f"Detailed results saved to: {results_text_file}")
    
    # Open browser with results if not disabled
    if result_path and not args.no_browser and os.path.exists(result_path):
        print(f"Opening results in browser: {result_path}")
        webbrowser.open(f"file://{os.path.abspath(result_path)}")

    # If all models are benchmarked, generate cross-model comparison
    if (args.all_models or (len(models_to_benchmark) > 1)) and not args.no_browser:
        await generate_cross_model_comparison(args.output_dir, all_results)
        
        # Open the cross-model comparison in browser
        cross_model_index = os.path.join(args.output_dir, "all_models", "index.html")
        if os.path.exists(cross_model_index) and not args.no_browser:
            webbrowser.open(f"file://{os.path.abspath(cross_model_index)}")
            
    # If only cross-model comparison is requested without running benchmarks
    elif args.cross_model:
        await generate_cross_model_comparison(args.output_dir)
        
        # Open the cross-model comparison in browser
        cross_model_index = os.path.join(args.output_dir, "all_models", "index.html")
        if os.path.exists(cross_model_index) and not args.no_browser:
            webbrowser.open(f"file://{os.path.abspath(cross_model_index)}")

async def run_benchmark_for_model(
    model_name: str, 
    output_dir: str, 
    simplified: bool = False, 
    advanced: bool = False,
    results_text_file: str = None
) -> Dict[str, Any]:
    """Run benchmark for a specific model"""
    print(f"\n{'='*50}")
    print(f"Running benchmark for model: {model_name}")
    print(f"{'='*50}")
    
    # Get API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment variables")
        return {}
    
    # Get test scenarios
    if simplified:
        print("Using simplified test scenarios...")
        test_scenarios = get_simple_test_scenarios()
    elif advanced:
        print("Using advanced test scenarios...")
        test_scenarios = get_advanced_test_scenarios() + get_simple_test_scenarios()
    else:
        print("Using standard test scenarios...")
        test_scenarios = get_advanced_test_scenarios()
    
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', "bolt://localhost:7687")
    neo4j_user = os.getenv('NEO4J_USER', "neo4j")
    neo4j_password = os.getenv('NEO4J_PASSWORD', "password")
    embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    
    # Create a unique collection name for this benchmark run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_name = f"benchmark_{model_name}_{timestamp}"
    
    # Resources to clean up
    memory_module = None
    llm_provider = None
    
    # Results dictionary that will be populated during benchmarking
    results = {
        "baseline": {},
        "episodic": {},
        "improvements": {}
    }
    
    try:
        # Initialize LLM provider with retry mechanism
        llm_provider = GroqProvider(
            api_key=groq_api_key, 
            model_name=model_name,
            temperature=0.7,
            max_tokens=1000,
            max_retries=3,
            min_delay=2.0,
            max_delay=30.0
        )
        
        # Initialize benchmarking system
        benchmark_system = BenchmarkingSystem(
            llm_provider=llm_provider,
            test_scenarios=test_scenarios,
            model_name=model_name
        )
        
        # Create memory configuration with maintenance disabled
        memory_config = MemoryConfig(
            max_context_items=10,
            memory_decay_factor=0.95,
            importance_threshold=0.5,
            min_references_to_keep=2,
            decay_interval=999999,  # Disable automatic decay for benchmarking
            cleanup_interval=999999,  # Disable automatic cleanup for benchmarking
            decay_check_interval=999999  # Disable automatic decay checks for benchmarking
        )
        
        # Initialize Episodic Memory Module
        print(f"Initializing memory module with collection: {collection_name}")
        try:
            memory_module = EpisodicMemoryModule(
                llm_provider=llm_provider,
                collection_name=collection_name,
                embedding_model=embedding_model,
                config=memory_config,
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password
            )
        except Exception as e:
            print(f"Error initializing memory module: {e}")
            print("Trying alternative connection...")
            # Try with default connection as fallback
            memory_module = EpisodicMemoryModule(
                llm_provider=llm_provider,
                collection_name=collection_name,
                embedding_model=embedding_model,
                config=memory_config
            )
        
        # Run benchmarks
        print("Running traditional LLM benchmark...")
        await benchmark_system.run_traditional_benchmark()
        
        print("Running Episodic Memory benchmark...")
        await benchmark_system.run_episodic_benchmark(memory_module)
        
        # Create output directory for this model
        model_output_dir = f"{output_dir}/{model_name}"
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Generate visualizations
        print("Generating visualizations...")
        visualization_success = False
        try:
            vis_files = benchmark_system.generate_visualizations(model_output_dir)
            visualization_success = True
            print(f"Successfully generated {len(vis_files)} visualizations")
        except Exception as vis_error:
            print(f"Error generating visualizations: {vis_error}")
            import traceback
            traceback.print_exc()
        
        # Get results and save to file
        baseline_metrics = benchmark_system.baseline_metrics.get_metrics_dict()
        episodic_metrics = benchmark_system.episodic_metrics.get_metrics_dict()
        
        # Update results dictionary
        results["baseline"] = baseline_metrics
        results["episodic"] = episodic_metrics
        
        # Calculate improvements
        improvements = {}
        for key in baseline_metrics:
            if key in episodic_metrics and isinstance(baseline_metrics[key], (int, float)) and baseline_metrics[key] != 0:
                improvements[key] = ((episodic_metrics[key] - baseline_metrics[key]) / baseline_metrics[key]) * 100
        
        results["improvements"] = improvements
        
        # Always create index.html even if some visualizations failed
        print("Creating index.html for model results...")
        create_model_index_html(model_output_dir, model_name, results)
        
        # Save detailed results to text file if provided
        if results_text_file:
            save_detailed_results_to_text(
                results_text_file, 
                model_name, 
                baseline_metrics, 
                episodic_metrics, 
                improvements,
                test_scenarios
            )
        
        # Clean up memory data
        try:
            print("\nCleaning up memory resources...")
            duplicates, orphaned = await memory_module.cleanup_memories()
            print(f"Cleaned up {duplicates} duplicate memories")
            print(f"Cleaned up {orphaned} orphaned entities")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        
        # Return results for combined visualization
        return results
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        
        # If there was an error during benchmarking, still try to create an index.html
        # with whatever partial results we got
        try:
            model_output_dir = f"{output_dir}/{model_name}"
            os.makedirs(model_output_dir, exist_ok=True)
            create_model_index_html(model_output_dir, model_name, results)
        except Exception as idx_error:
            print(f"Error creating index.html: {idx_error}")
            
        return results
    
    finally:
        # Ensure proper cleanup regardless of success or failure
        if memory_module is not None:
            try:
                print("Performing final cleanup of memory resources...")
                # Force cleanup of memory module resources
                await memory_module.cleanup_memories()
                # Explicitly close any Neo4j connections
                if hasattr(memory_module, "_close_connections"):
                    await memory_module._close_connections()
                elif hasattr(memory_module, "__del__"):
                    memory_module.__del__()
            except Exception as final_cleanup_error:
                print(f"Error during final cleanup: {final_cleanup_error}")
                
        # For good measure, suggest garbage collection
        try:
            import gc
            gc.collect()
        except:
            pass

def save_detailed_results_to_text(
    filename: str, 
    model_name: str, 
    baseline_metrics: Dict[str, Any],
    episodic_metrics: Dict[str, Any],
    improvements: Dict[str, float],
    test_scenarios: List[Dict[str, Any]]
) -> None:
    """
    Save detailed benchmark results to a text file.
    
    Args:
        filename: Path to the text file
        model_name: Name of the model
        baseline_metrics: Metrics from the baseline (traditional) benchmark
        episodic_metrics: Metrics from the episodic memory benchmark
        improvements: Dictionary of improvement percentages
        test_scenarios: List of test scenarios used in the benchmark
    """
    with open(filename, "a") as f:
        f.write(f"\n\n{'='*40}\n")
        f.write(f"MODEL: {model_name}\n")
        f.write(f"{'='*40}\n\n")
        
        # Write metrics summary
        f.write("PERFORMANCE METRICS SUMMARY:\n")
        f.write("--------------------------\n")
        f.write(f"{'Metric':<30} {'Baseline':<15} {'Episodic':<15} {'Improvement':<15}\n")
        f.write(f"{'-'*75}\n")
        
        # Sort metrics by improvement (descending)
        sorted_metrics = sorted(
            [(k, v) for k, v in improvements.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        for metric, improvement in sorted_metrics:
            if metric in baseline_metrics and metric in episodic_metrics:
                baseline_val = baseline_metrics[metric]
                episodic_val = episodic_metrics[metric]
                
                # Format values based on type
                if isinstance(baseline_val, float):
                    baseline_str = f"{baseline_val:.4f}"
                    episodic_str = f"{episodic_val:.4f}"
                else:
                    baseline_str = str(baseline_val)
                    episodic_str = str(episodic_val)
                
                f.write(f"{metric:<30} {baseline_str:<15} {episodic_str:<15} {improvement:+.2f}%\n")
        
        # Write overall improvement
        overall_improvement = sum(improvements.values()) / len(improvements) if improvements else 0
        f.write(f"\nOVERALL IMPROVEMENT: {overall_improvement:+.2f}%\n\n")
        
        # Write scenario details
        f.write("TEST SCENARIOS SUMMARY:\n")
        f.write("---------------------\n")
        f.write(f"Total number of test scenarios: {len(test_scenarios)}\n\n")
        
        for i, scenario in enumerate(test_scenarios):
            f.write(f"Scenario {i+1}:\n")
            f.write(f"  Type: {scenario.get('type', 'General')}\n")
            f.write(f"  Description: {scenario.get('description', 'N/A')}\n")
            f.write(f"  Complexity: {scenario.get('complexity', 'Medium')}\n")
            f.write("\n")
        
        f.write(f"\n{'='*75}\n")
        f.write(f"End of results for {model_name}\n")
        f.write(f"{'='*75}\n\n")

async def generate_combined_comparison(output_dir: str, model_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate combined visualizations comparing multiple models.
    
    Args:
        output_dir: Directory to save the visualizations
        model_results: Dictionary of model results, keyed by model name
        
    Returns:
        Path to the generated visualization file
    """
    if not model_results:
        print("No results to visualize")
        return None
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract common metrics across all models
    common_metrics = set()
    for model, results in model_results.items():
        if "improvements" in results:
            common_metrics.update(results["improvements"].keys())
    
    common_metrics = list(common_metrics)
    
    # Filter to include only meaningful metrics (accuracy, coherence, etc.)
    key_metrics = [
        m for m in common_metrics 
        if any(term in m.lower() for term in 
               ["accuracy", "coherence", "relevance", "factual", "context", "memory"])
    ]
    
    if not key_metrics:
        key_metrics = common_metrics[:5]  # Use top 5 metrics if no key metrics found
    
    # 1. Create improvement comparison bar chart
    plt.figure(figsize=(14, 8))
    
    models = list(model_results.keys())
    x = np.arange(len(models))
    width = 0.8 / len(key_metrics)
    
    for i, metric in enumerate(key_metrics):
        values = [
            model_results[model].get("improvements", {}).get(metric, 0)
            for model in models
        ]
        plt.bar(x + i * width, values, width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Improvement (%)')
    plt.title('Memory Enhancement Improvement by Model')
    plt.xticks(x + width * (len(key_metrics) - 1) / 2, models, rotation=45)
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, metric in enumerate(key_metrics):
        for j, model in enumerate(models):
            value = model_results[model].get("improvements", {}).get(metric, 0)
            plt.text(
                j + i * width, 
                value + 0.5, 
                f"{value:.1f}%", 
                ha='center', 
                va='bottom', 
                fontsize=8,
                rotation=90
            )
    
    improvement_chart_path = os.path.join(comparison_dir, "improvement_comparison.png")
    plt.savefig(improvement_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Create radar chart for overall capabilities
    metrics_for_radar = key_metrics[:5]  # Use top 5 metrics for radar chart
    
    # Create radar chart
    num_metrics = len(metrics_for_radar)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for model in models:
        values = [
            model_results[model].get("episodic", {}).get(metric, 0)
            for metric in metrics_for_radar
        ]
        
        # Normalize values to 0-1 range for better visualization
        max_values = []
        for i, metric in enumerate(metrics_for_radar):
            max_val = max(
                [model_results[m].get("episodic", {}).get(metric, 0) for m in models]
            )
            max_val = max(max_val, 1)  # Avoid division by zero
            max_values.append(max_val)
        
        normalized_values = [values[i] / max_values[i] for i in range(len(values))]
        normalized_values += normalized_values[:1]  # Close the loop
        
        ax.plot(angles, normalized_values, linewidth=2, label=model)
        ax.fill(angles, normalized_values, alpha=0.1)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_for_radar)
    plt.title('Model Capabilities Comparison (Episodic Memory Mode)', size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    radar_chart_path = os.path.join(comparison_dir, "capabilities_radar.png")
    plt.savefig(radar_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create overall score comparison
    plt.figure(figsize=(12, 6))
    
    overall_scores = []
    for model in models:
        # Calculate average improvement across all metrics
        improvements = model_results[model].get("improvements", {})
        avg_improvement = sum(improvements.values()) / len(improvements) if improvements else 0
        overall_scores.append(avg_improvement)
    
    # Sort models by overall score
    sorted_indices = np.argsort(overall_scores)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = [overall_scores[i] for i in sorted_indices]
    
    plt.barh(sorted_models, sorted_scores, color='skyblue')
    plt.xlabel('Average Improvement (%)')
    plt.title('Overall Model Performance Comparison')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, score in enumerate(sorted_scores):
        plt.text(score + 0.5, i, f"{score:.2f}%", va='center')
    
    overall_chart_path = os.path.join(comparison_dir, "overall_comparison.png")
    plt.savefig(overall_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create index.html file for easy viewing
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multi-Model Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2 {{ color: #333; }}
            .chart-container {{ margin: 30px 0; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .model-link {{ margin: 10px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>Multi-Model Comparison</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="chart-container">
            <h2>Overall Model Performance</h2>
            <img src="overall_comparison.png" alt="Overall Model Comparison">
        </div>
        
        <div class="chart-container">
            <h2>Improvement by Metric</h2>
            <img src="improvement_comparison.png" alt="Improvement by Metric">
        </div>
        
        <div class="chart-container">
            <h2>Model Capabilities Radar</h2>
            <img src="capabilities_radar.png" alt="Model Capabilities Radar">
        </div>
        
        <h2>Individual Model Results</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Overall Improvement</th>
                <th>Links</th>
            </tr>
    """
    
    for model in models:
        improvements = model_results[model].get("improvements", {})
        avg_improvement = sum(improvements.values()) / len(improvements) if improvements else 0
        
        html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{avg_improvement:.2f}%</td>
                <td><a href="../{model}/index.html">Detailed Results</a></td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    index_path = os.path.join(comparison_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(html_content)
    
    print(f"Multi-model comparison visualizations saved to: {comparison_dir}")
    return index_path

def write_comparison_summary(filename: str, model_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Write a summary of the comparison between multiple models to the text file.
    
    Args:
        filename: Path to the text file
        model_results: Dictionary of model results, keyed by model name
    """
    if not model_results:
        return
    
    with open(filename, "a") as f:
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("MULTI-MODEL COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Calculate overall improvement for each model
        overall_improvements = {}
        for model, results in model_results.items():
            improvements = results.get("improvements", {})
            if improvements:
                avg_improvement = sum(improvements.values()) / len(improvements)
                overall_improvements[model] = avg_improvement
        
        # Sort models by overall improvement (descending)
        sorted_models = sorted(
            overall_improvements.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Write ranking table
        f.write("MODEL RANKING BY OVERALL IMPROVEMENT:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Rank':<5} {'Model':<30} {'Overall Improvement':<20}\n")
        f.write("-" * 60 + "\n")
        
        for i, (model, improvement) in enumerate(sorted_models):
            f.write(f"{i+1:<5} {model:<30} {improvement:+.2f}%\n")
        
        f.write("\n")
        
        # Find the best model for each key metric
        common_metrics = set()
        for model, results in model_results.items():
            if "improvements" in results:
                common_metrics.update(results["improvements"].keys())
        
        key_metrics = [
            m for m in common_metrics 
            if any(term in m.lower() for term in 
                ["accuracy", "coherence", "relevance", "factual", "context", "memory"])
        ]
        
        if not key_metrics and common_metrics:
            key_metrics = list(common_metrics)[:5]  # Use top 5 if no key metrics found
        
        f.write("BEST MODEL BY METRIC:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Metric':<30} {'Best Model':<20} {'Improvement':<15}\n")
        f.write("-" * 65 + "\n")
        
        for metric in key_metrics:
            best_model = ""
            best_improvement = float("-inf")
            
            for model, results in model_results.items():
                improvement = results.get("improvements", {}).get(metric)
                if improvement is not None and improvement > best_improvement:
                    best_model = model
                    best_improvement = improvement
            
            if best_model:
                f.write(f"{metric:<30} {best_model:<20} {best_improvement:+.2f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Multi-Model Comparison Summary\n")
        f.write("=" * 80 + "\n\n")

def create_model_index_html(output_dir: str, model_name: str, results: Dict[str, Any]) -> str:
    """
    Create an index.html file for a single model's visualizations.
    This is particularly useful when some visualizations might have failed.
    
    Args:
        output_dir: Directory to save the index file
        model_name: Name of the model
        results: Dictionary of benchmark results
        
    Returns:
        Path to the created index.html file
    """
    # Get available visualization files
    vis_files = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.html') and filename != 'index.html':
            vis_files.append(filename)
    
    # Calculate overall improvement
    overall_improvement = 0
    if "improvements" in results and results["improvements"]:
        improvements = results["improvements"]
        overall_improvement = sum(improvements.values()) / len(improvements)
    # If no improvements dict but we have both baseline and episodic metrics, calculate directly
    elif "baseline" in results and "episodic" in results and results["baseline"] and results["episodic"]:
        common_metrics = set(results["baseline"].keys()).intersection(set(results["episodic"].keys()))
        improvement_values = []
        
        for metric in common_metrics:
            baseline_val = results["baseline"].get(metric, 0)
            episodic_val = results["episodic"].get(metric, 0)
            
            if isinstance(baseline_val, (int, float)) and isinstance(episodic_val, (int, float)) and baseline_val > 0:
                improvement = ((episodic_val - baseline_val) / baseline_val) * 100
                improvement_values.append(improvement)
        
        if improvement_values:
            overall_improvement = sum(improvement_values) / len(improvement_values)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{model_name} Benchmark Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2 {{ color: #333; }}
            .summary {{ margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
            .vis-container {{ margin: 20px 0; }}
            .metric {{ margin: 10px 0; }}
            .improvement {{ color: green; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>{model_name} Benchmark Results</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Overall Improvement:</strong> <span class="improvement">{overall_improvement:.2f}%</span></p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Available Visualizations</h2>
    """
    
    if vis_files:
        for vis_file in vis_files:
            name = vis_file.replace('.html', '').replace('_', ' ').title()
            html_content += f"""
            <div class="vis-container">
                <h3>{name}</h3>
                <p><a href="{vis_file}" target="_blank">Open {name} Visualization</a></p>
            </div>
            """
    else:
        html_content += """
        <p>No visualizations found. This could be due to errors during the benchmark process.</p>
        """
    
    # Add metrics section if available
    if "baseline" in results and "episodic" in results:
        html_content += """
        <h2>Metrics Summary</h2>
        <table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Metric</th>
                <th>Baseline</th>
                <th>Episodic</th>
                <th>Improvement</th>
            </tr>
        """
        
        # Only show metrics that are in improvements
        if "improvements" in results:
            for metric, improvement in results["improvements"].items():
                baseline_val = results["baseline"].get(metric, "N/A")
                episodic_val = results["episodic"].get(metric, "N/A")
                
                if isinstance(baseline_val, (int, float)) and isinstance(episodic_val, (int, float)):
                    html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{baseline_val:.4f}</td>
                        <td>{episodic_val:.4f}</td>
                        <td>{improvement:+.2f}%</td>
                    </tr>
                    """
        # If no improvements data but we have baseline and episodic metrics, display those
        elif results["baseline"] and results["episodic"]:
            # Combine all metrics from both baseline and episodic
            all_metrics = set(results["baseline"].keys()).union(set(results["episodic"].keys()))
            
            for metric in all_metrics:
                baseline_val = results["baseline"].get(metric, "N/A")
                episodic_val = results["episodic"].get(metric, "N/A")
                
                if isinstance(baseline_val, (int, float)) and isinstance(episodic_val, (int, float)):
                    # Calculate improvement directly
                    if baseline_val > 0:
                        improvement = ((episodic_val - baseline_val) / baseline_val) * 100
                    else:
                        improvement = 0 if episodic_val == 0 else 100
                    
                    html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{baseline_val:.4f}</td>
                        <td>{episodic_val:.4f}</td>
                        <td>{improvement:+.2f}%</td>
                    </tr>
                    """
                else:
                    # Handle case where one of the metrics is not a number
                    html_content += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{baseline_val}</td>
                        <td>{episodic_val}</td>
                        <td>N/A</td>
                    </tr>
                    """
        
        html_content += """
        </table>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Save the HTML file
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(html_content)
    
    return index_path

async def generate_cross_model_comparison(output_dir: str, model_results: Dict[str, Dict[str, Any]] = None) -> str:
    """
    Generate cross-model comparison visualizations
    
    Args:
        output_dir: Directory to save visualizations
        model_results: Dictionary of model results (optional)
        
    Returns:
        Path to the cross-model comparison index file
    """
    print("\nGenerating cross-model comparison visualizations...")
    
    # Create all_models directory if it doesn't exist
    all_models_dir = os.path.join(output_dir, "all_models")
    os.makedirs(all_models_dir, exist_ok=True)
    
    # Create images directory
    images_dir = os.path.join(all_models_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # If model_results is not provided, get all model directories
    if not model_results:
        model_results = {}
        model_dirs = []
        
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path) and item != "all_models":
                model_dirs.append(item)
        
        # Load the result files for each model
        for model in model_dirs:
            model_dir = os.path.join(output_dir, model)
            if os.path.exists(os.path.join(model_dir, "index.html")):
                model_results[model] = {
                    "model": model,
                    "metrics": extract_metrics_from_html(os.path.join(model_dir, "index.html"))
                }
    
    # Generate comparison metrics for all metric types
    all_metrics = generate_metrics_comparison(model_results, all_models_dir)
    
    # Generate overall improvement comparison
    improvement_data = generate_improvement_comparison(model_results, all_models_dir)
    
    # Create index.html file
    index_html = create_cross_model_index(all_models_dir, model_results.keys(), all_metrics)
    
    print(f"Cross-model comparison generated at: {os.path.join(output_dir, 'all_models', 'index.html')}")
    
    return os.path.join(output_dir, "all_models", "index.html")

def extract_metrics_from_html(index_file: str) -> Dict[str, Dict[str, float]]:
    """
    Extract metrics from a model's index.html file
    
    Args:
        index_file: Path to the index.html file
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "overall_improvement": 0,
        "metrics_table": {}
    }
    
    try:
        from bs4 import BeautifulSoup
        
        with open(index_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract overall improvement
        improvement_span = soup.select('.improvement')
        if improvement_span:
            overall_improvement = improvement_span[0].text.strip()
            # Remove % sign and handle both positive and negative values
            overall_improvement = overall_improvement.strip('%').strip()
            if overall_improvement.startswith('+'):
                overall_improvement = overall_improvement[1:]
            try:
                metrics["overall_improvement"] = float(overall_improvement)
            except ValueError:
                pass
        
        # Extract metrics from table
        table_rows = soup.select('table tr')[1:]  # Skip header row
        
        for row in table_rows:
            cells = row.select('td')
            if len(cells) >= 4:
                metric_name = cells[0].text.strip()
                baseline_value = cells[1].text.strip()
                episodic_value = cells[2].text.strip()
                improvement = cells[3].text.strip()
                
                # Remove % sign and + sign if present
                if '%' in improvement:
                    improvement = improvement.strip('%').strip()
                if improvement.startswith('+'):
                    improvement = improvement[1:]
                
                try:
                    metrics["metrics_table"][metric_name] = {
                        "baseline": float(baseline_value),
                        "episodic": float(episodic_value),
                        "improvement": float(improvement)
                    }
                except ValueError:
                    continue
    
    except Exception as e:
        print(f"Error extracting metrics from {index_file}: {e}")
    
    return metrics

def generate_metrics_comparison(model_results: Dict[str, Dict[str, Any]], output_dir: str) -> List[str]:
    """
    Generate comparison visualizations for each metric across all models
    
    Args:
        model_results: Dictionary of model results
        output_dir: Directory to save visualizations
        
    Returns:
        List of metric names
    """
    # Collect all unique metrics across all models
    all_metrics = set()
    for model, results in model_results.items():
        if "metrics" in results and "metrics_table" in results["metrics"]:
            all_metrics.update(results["metrics"]["metrics_table"].keys())
    
    all_metrics = list(all_metrics)
    
    # Create comparison plot for each metric
    for metric in all_metrics:
        models = []
        baseline_values = []
        episodic_values = []
        
        for model, results in model_results.items():
            if "metrics" in results and "metrics_table" in results["metrics"] and metric in results["metrics"]["metrics_table"]:
                models.append(model)
                baseline_values.append(results["metrics"]["metrics_table"][metric]["baseline"])
                episodic_values.append(results["metrics"]["metrics_table"][metric]["episodic"])
        
        # If we have data for this metric, create a plot
        if models:
            plt.figure(figsize=(14, 8))
            plt.style.use('ggplot')
            
            # Set up the bar positions
            x = np.arange(len(models))
            width = 0.35
            
            # Create the bars
            fig, ax = plt.subplots(figsize=(14, 8))
            rects1 = ax.bar(x - width/2, baseline_values, width, label='Traditional LLM', color='#636EFA')
            rects2 = ax.bar(x + width/2, episodic_values, width, label='Episodic Memory', color='#EF553B')
            
            # Format the metric name for display
            metric_display = metric.replace('_', ' ').title()
            
            # Add labels and title
            ax.set_xlabel('Models', fontweight='bold')
            ax.set_ylabel(metric_display, fontweight='bold')
            ax.set_title(f'{metric_display} Comparison Across Models', fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            
            # Add data labels
            def add_labels(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontweight='bold')
            
            add_labels(rects1)
            add_labels(rects2)
            
            # Add improvement percentages
            for i in range(len(models)):
                if baseline_values[i] > 0:
                    improvement = (((episodic_values[i] - baseline_values[i]) / baseline_values[i]) * 100)
                    color = 'green' if improvement > 0 else 'red'
                    sign = '+' if improvement > 0 else ''
                    ax.annotate(f'{sign}{improvement:.1f}%',
                                xy=(x[i] + width/2, episodic_values[i]),
                                xytext=(0, 10),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                color=color, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the figure to images directory
            metric_filename = metric.lower().replace(' ', '_')
            img_path = os.path.join(output_dir, "images", f"{metric_filename}_comparison.png")
            plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create an HTML file showing all metric comparisons
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Metrics Comparison Across Models</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2, h3 { color: #333; }
            .summary { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
            .visualization { margin: 30px 0; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Metrics Comparison Across Models</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>This page shows how different models perform on various metrics, comparing traditional LLMs with episodic memory-enhanced versions.</p>
            <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
    """
    
    for metric in all_metrics:
        metric_display = metric.replace('_', ' ').title()
        metric_filename = metric.lower().replace(' ', '_')
        
        # Only include metrics for which we created visualizations
        img_path = os.path.join(output_dir, "images", f"{metric_filename}_comparison.png")
        if os.path.exists(img_path):
            html_content += f"""
            <div class="visualization">
                <h2>{metric_display} Comparison</h2>
                <img src="images/{metric_filename}_comparison.png" alt="{metric_display} Comparison">
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML file
    metrics_html_path = os.path.join(output_dir, "metrics_comparison.html")
    with open(metrics_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return all_metrics

def generate_improvement_comparison(model_results: Dict[str, Dict[str, Any]], output_dir: str) -> Dict[str, float]:
    """
    Generate overall improvement comparison across all models
    
    Args:
        model_results: Dictionary of model results
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of improvement percentages by model
    """
    models = []
    improvements = []
    improvement_data = {}
    
    for model, results in model_results.items():
        if "metrics" in results and "overall_improvement" in results["metrics"]:
            models.append(model)
            improvements.append(results["metrics"]["overall_improvement"])
            improvement_data[model] = results["metrics"]["overall_improvement"]
    
    if not models:
        return improvement_data
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    plt.style.use('ggplot')
    
    # Create a colorful bar chart with different colors based on value
    colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in improvements]
    bars = plt.bar(models, improvements, color=colors, edgecolor='#333333', alpha=0.8)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        color = 'green' if height >= 0 else 'red'
        sign = '+' if height >= 0 else ''
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{sign}{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', color=color)
    
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Overall Improvement (%)', fontweight='bold')
    plt.title('Overall Improvement Comparison Across Models', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='#7f8c8d', linestyle='-', alpha=0.3)
    
    # Save the figure
    img_path = os.path.join(output_dir, "images", "overall_improvement_comparison.png")
    plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create HTML file
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Overall Improvement Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2 { color: #333; }
            .summary { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
            .visualization { margin: 30px 0; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .positive { color: green; font-weight: bold; }
            .negative { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Overall Improvement Comparison</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>This visualization shows the overall improvement percentage for each model when using episodic memory compared to traditional LLM.</p>
            <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="visualization">
            <img src="images/overall_improvement_comparison.png" alt="Overall Improvement Comparison">
        </div>
        
        <h2>Model Rankings</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Overall Improvement</th>
            </tr>
    """
    
    # Sort models by improvement for ranking
    ranked_models = sorted(zip(models, improvements), key=lambda x: x[1], reverse=True)
    
    for i, (model, improvement) in enumerate(ranked_models, 1):
        sign = '+' if improvement >= 0 else ''
        css_class = 'positive' if improvement >= 0 else 'negative'
        
        html_content += f"""
            <tr>
                <td>{i}</td>
                <td>{model}</td>
                <td class="{css_class}">{sign}{improvement:.2f}%</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write the HTML file
    improvement_html_path = os.path.join(output_dir, "overall_improvement_comparison.html")
    with open(improvement_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return improvement_data

def create_cross_model_index(output_dir: str, models: List[str], metrics: List[str]) -> str:
    """
    Create an index.html file for cross-model comparisons
    
    Args:
        output_dir: Directory to save the index file
        models: List of model names
        metrics: List of metric names
        
    Returns:
        Path to the index file
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cross-Model Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2 { color: #333; }
            .summary { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
            .section { margin: 30px 0; }
            .visualization { margin: 20px 0; }
            .card-container { display: flex; flex-wrap: wrap; gap: 20px; }
            .card { 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 15px; 
                width: 300px;
                transition: transform 0.3s;
            }
            .card:hover { 
                transform: translateY(-5px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            }
            .card h3 { margin-top: 0; }
            .nav-links { margin: 20px 0; }
            .nav-links a { 
                display: inline-block;
                padding: 10px 15px;
                margin-right: 10px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            .nav-links a:hover { background-color: #2980b9; }
            .model-links { 
                display: flex; 
                flex-wrap: wrap; 
                gap: 10px; 
                margin: 20px 0;
            }
            .model-links a { 
                text-decoration: none;
                padding: 8px 12px;
                background-color: #f5f5f5;
                color: #333;
                border-radius: 4px;
                transition: background-color 0.3s;
            }
            .model-links a:hover { background-color: #e0e0e0; }
        </style>
    </head>
    <body>
        <h1>Cross-Model Benchmark Comparison</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>This page provides comparative analysis across all benchmarked models, showing how different models perform with episodic memory enhancement.</p>
            <p><strong>Models Analyzed:</strong> """ + ', '.join(models) + """</p>
            <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="nav-links">
            <a href="overall_improvement_comparison.html">Overall Improvement Comparison</a>
            <a href="metrics_comparison.html">Detailed Metrics Comparison</a>
        </div>
        
        <div class="section">
            <h2>Visualization Highlights</h2>
            <div class="visualization">
                <h3>Overall Improvement Across Models</h3>
                <img src="images/overall_improvement_comparison.png" alt="Overall Improvement Comparison" style="max-width: 100%; height: auto;">
            </div>
    """
    
    # Add a highlight metric visualization if available
    highlight_metrics = ["response_time_avg", "context_score_avg", "relevance_score_avg"]
    for metric in highlight_metrics:
        if metric in metrics:
            metric_display = metric.replace('_', ' ').title()
            metric_filename = metric.lower().replace(' ', '_')
            img_path = os.path.join(output_dir, "images", f"{metric_filename}_comparison.png")
            
            if os.path.exists(img_path):
                html_content += f"""
                <div class="visualization">
                    <h3>{metric_display} Comparison</h3>
                    <img src="images/{metric_filename}_comparison.png" alt="{metric_display} Comparison" style="max-width: 100%; height: auto;">
                </div>
                """
                break
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Individual Model Results</h2>
            <p>Click on a model name to view its detailed benchmark results:</p>
            <div class="model-links">
    """
    
    for model in models:
        html_content += f"""
                <a href="../{model}/index.html">{model}</a>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return index_path

if __name__ == "__main__":
    asyncio.run(main()) 