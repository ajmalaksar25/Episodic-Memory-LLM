#!/usr/bin/env python3
"""
Episodic Memory LLM - Main Entry Point

This script serves as the main entry point for the Episodic Memory LLM project.
It provides a command-line interface to access different functionalities.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point for the Episodic Memory LLM project."""
    # Load environment variables
    load_dotenv()
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Episodic Memory LLM - A memory-enhanced language model system',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Chat interface command
    chat_parser = subparsers.add_parser('chat', help='Start the chat interface')
    chat_parser.add_argument('--model', type=str, default=os.environ.get('MODEL_NAME', 'llama3-8b-8192'),
                            help='Model name to use for chat')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--model', type=str, default=os.environ.get('MODEL_NAME', 'llama3-8b-8192'),
                                help='Model name to use for benchmarking')
    benchmark_parser.add_argument('--output-dir', type=str, default='visualizations',
                                help='Directory to save visualization outputs')
    benchmark_parser.add_argument('--simple', action='store_true',
                                help='Run with simplified test scenarios for quick testing')
    benchmark_parser.add_argument('--advanced', action='store_true',
                                help='Run with advanced test scenarios for comprehensive testing')
    benchmark_parser.add_argument('--compare-models', action='store_true',
                                help='Compare performance across multiple models')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze benchmark results')
    analyze_parser.add_argument('--model', type=str, default=os.environ.get('MODEL_NAME', 'llama3-8b-8192'),
                              help='Model name to analyze')
    analyze_parser.add_argument('--output-dir', type=str, default='visualizations',
                              help='Directory containing visualization outputs')
    analyze_parser.add_argument('--generate-report', action='store_true',
                              help='Generate a comprehensive report')
    
    # Web interface command
    web_parser = subparsers.add_parser('web', help='Start the web interface')
    web_parser.add_argument('--port', type=int, default=8501,
                          help='Port to run the web interface on')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--test-file', type=str, default=None,
                           help='Specific test file to run')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'chat':
        from src.groq_conversational_bot import main as chat_main
        chat_main()
    elif args.command == 'benchmark':
        import asyncio
        from src.run_benchmarks import main as benchmark_main
        asyncio.run(benchmark_main())
    elif args.command == 'analyze':
        from src.utils.analyze_benchmarks import main as analyze_main
        analyze_main(args.model, args.output_dir, args.generate_report)
    elif args.command == 'web':
        import subprocess
        subprocess.run(['streamlit', 'run', 'src/streamlit_interface.py', '--server.port', str(args.port)])
    elif args.command == 'test':
        import unittest
        if args.test_file:
            unittest.main(module=f'src.tests.{args.test_file}')
        else:
            test_loader = unittest.TestLoader()
            test_suite = test_loader.discover('src/tests')
            test_runner = unittest.TextTestRunner()
            test_runner.run(test_suite)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 