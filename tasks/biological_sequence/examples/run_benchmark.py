"""
Example script for running the Biological Sequence Manipulation Benchmark.

Usage:
    python examples/run_benchmark.py --config configs/biological_sequence/model_config.json --models gpt-3.5-turbo gpt-4
    python examples/run_benchmark.py --quick-test
    python examples/run_benchmark.py --generate-data-only
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add parent directory to path to import benchmark modules
sys.path.append(str(Path(__file__).parent.parent))

from src.benchmark import BiologicalSequenceBenchmark
from src.model_interface import ModelFactory
from configs.locator import resolve_config_path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Biological Sequence Manipulation Benchmark"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="model_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specific models to test (use model keys from config)"
    )
    
    parser.add_argument(
        "--tasks", 
        nargs="+", 
        choices=["dna_complement", "rna_complement", "protein_three_to_one", "protein_one_to_three"],
        help="Specific tasks to run"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Run a quick test with minimal cases"
    )
    
    parser.add_argument(
        "--generate-data-only", 
        action="store_true",
        help="Only generate test data without running models"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=True,
        help="Show detailed progress"
    )
    
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save results to files"
    )
    
    return parser.parse_args()


def create_quick_test_config():
    """Create a minimal configuration for quick testing."""
    return {
        "benchmark_settings": {
            "results_dir": "results",
            "save_raw_responses": True,
            "save_detailed_results": True
        },
        "tasks": {
            "dna_complement": {
                "enabled": True,
                "num_cases": 5,
                "sequence_length_range": [5, 8]
            },
            "rna_complement": {
                "enabled": True,
                "num_cases": 5,
                "sequence_length_range": [5, 8]
            },
            "protein_three_to_one": {
                "enabled": True,
                "num_cases": 5,
                "sequence_length_range": [3, 5]
            },
            "protein_one_to_three": {
                "enabled": True,
                "num_cases": 5,
                "sequence_length_range": [3, 5]
            }
        }
    }


def validate_environment():
    """Check if required environment variables are set."""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI models",
        "DEEPSEEK_API_KEY": "DeepSeek models"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} (required for {description})")
    
    if missing_vars:
        print("Warning: Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("Some models may not work without proper API keys.")
        print()


def main():
    """Main function to run the benchmark."""
    args = parse_arguments()
    
    # Validate environment
    validate_environment()
    
    # Initialize benchmark
    if args.quick_test:
        print("Running quick test mode...")
        config = create_quick_test_config()
        benchmark = BiologicalSequenceBenchmark()
        benchmark.config = config
    else:
        config_path = args.config
        if not Path(config_path).is_absolute():
            config_path = resolve_config_path("biological_sequence", config_path)
        benchmark = BiologicalSequenceBenchmark(config_path)
    
    # Update output directory if specified
    if args.output_dir:
        benchmark.config["benchmark_settings"]["results_dir"] = args.output_dir
    
    # Filter tasks if specified
    if args.tasks:
        for task_type in list(benchmark.config["tasks"].keys()):
            if task_type not in args.tasks:
                benchmark.config["tasks"][task_type]["enabled"] = False
    
    # Generate test data only
    if args.generate_data_only:
        print("Generating test data...")
        test_data = benchmark.generate_test_data()
        
        # Save test data
        os.makedirs(benchmark.config["benchmark_settings"]["results_dir"], exist_ok=True)
        data_path = os.path.join(
            benchmark.config["benchmark_settings"]["results_dir"], 
            "test_data.json"
        )
        
        # Convert test data to JSON-serializable format
        serializable_data = {}
        for task_type, cases in test_data.items():
            serializable_data[task_type] = cases
        
        with open(data_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Test data saved to: {data_path}")
        
        # Show summary
        total_cases = sum(len(cases) for cases in test_data.values())
        print(f"\nGenerated {total_cases} test cases:")
        for task_type, cases in test_data.items():
            print(f"  {task_type}: {len(cases)} cases")
        
        return
    
    # Filter models if specified
    model_configs = benchmark.config.get("models", {})
    if args.models:
        filtered_models = {k: v for k, v in model_configs.items() if k in args.models}
        if not filtered_models:
            print(f"Error: None of the specified models {args.models} found in config")
            print(f"Available models: {list(model_configs.keys())}")
            return
        model_configs = filtered_models
    
    if not model_configs:
        print("Error: No models configured for testing")
        return
    
    # Show what will be tested
    print("Benchmark Configuration:")
    print(f"Models to test: {list(model_configs.keys())}")
    enabled_tasks = [k for k, v in benchmark.config["tasks"].items() if v.get("enabled")]
    print(f"Tasks to run: {enabled_tasks}")
    
    total_cases = sum(
        benchmark.config["tasks"][task]["num_cases"] 
        for task in enabled_tasks
    )
    print(f"Total test cases: {total_cases}")
    print(f"Results will be saved to: {benchmark.config['benchmark_settings']['results_dir']}")
    print()
    
    try:
        # Run the benchmark
        results = benchmark.run_full_benchmark(
            model_configs=model_configs,
            save_results=not args.no_save,
            verbose=args.verbose
        )
        
        # Show summary of results
        print("\n" + "="*60)
        print("BENCHMARK COMPLETED")
        print("="*60)
        
        for model_name, model_results in results["results"].items():
            if "error" in model_results:
                print(f"\n{model_name}: FAILED - {model_results['error']}")
                continue
            
            # Calculate overall accuracy for this model
            all_model_results = []
            for task_results in model_results.values():
                all_model_results.extend(task_results)
            
            if all_model_results:
                accuracy = sum(r.is_correct for r in all_model_results) / len(all_model_results)
                print(f"\n{model_name}: {accuracy:.2%} overall accuracy")
                
                # Show per-task breakdown
                for task_type, task_results in model_results.items():
                    task_accuracy = sum(r.is_correct for r in task_results) / len(task_results)
                    print(f"  {task_type}: {task_accuracy:.2%} ({sum(r.is_correct for r in task_results)}/{len(task_results)})")
        
        if not args.no_save:
            print(f"\nDetailed results saved to: {benchmark.config['benchmark_settings']['results_dir']}")
    
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
