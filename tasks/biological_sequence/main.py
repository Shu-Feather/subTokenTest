"""
Main entry point for running the Biological Sequence Manipulation Benchmark.

Usage:
    python main.py --config configs/biological_sequence/model_config.json --models gpt-3.5-turbo gpt-4
    python main.py --quick-test --verbose
    python main.py --generate-data-only
"""

import os
import sys
import argparse
import json
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent


def find_project_root() -> Path:
    for parent in TASK_ROOT.parents:
        if (parent / "cli.py").exists():
            return parent
    return TASK_ROOT.parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import BiologicalSequenceBenchmark
from src.model_interface import ModelFactory
from configs.locator import resolve_config_path


def resolve_path(path_str: str) -> str:
    """Resolve a file path relative to CWD or repository root (with dataset fallback)."""
    path = Path(path_str)

    candidates = []
    if path.is_absolute():
        candidates.append(path)
        relative = Path(*path.parts[1:]) if len(path.parts) > 1 else Path(path.name)
    else:
        relative = path

    candidates.extend([
        TASK_ROOT / relative,
        TASK_ROOT / "datasets" / relative.name,
        PROJECT_ROOT / relative,
        PROJECT_ROOT / "test" / "datasets" / relative.name,
        relative,
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str(path)


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
        "--load-data",
        type=str,
        default=None,
        help="Load test data from existing JSON file instead of generating new data"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed progress including prompts and model responses"
    )
    
    parser.add_argument(
        "--no-save", 
        action="store_true",
        help="Don't save results to files"
    )

    parser.add_argument(
        "--restricted-reasoning",
        action="store_true",
        help="Use restricted thinking prompts that encourage minimal reasoning"
    )
    
    return parser.parse_args()

def load_test_data_from_file(filepath: str, verbose: bool = False) -> dict:
    """
    Load test data from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing test data
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary mapping task types to test cases
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    filepath = resolve_path(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Test data file not found: {filepath}")
    
    if verbose:
        print(f"Loading test data from: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support different JSON formats
        test_data = {}
        
        # Format 1: {"datasets": {...}} - from generate_datasets.py
        if 'datasets' in data:
            test_data = data['datasets']
            if verbose and 'metadata' in data:
                print("Dataset metadata found:")
                metadata = data['metadata']
                if 'generated_at' in metadata:
                    print(f"  Generated at: {metadata['generated_at']}")
                if 'parameters' in metadata:
                    params = metadata['parameters']
                    print(f"  Parameters: {params}")
        
        # Format 2: {"data": [...], "metadata": {...}} - single task file
        elif 'data' in data:
            # This is a single task file, need to determine task type
            if data['data'] and len(data['data']) > 0:
                task_type = data['data'][0].get('task_type', 'unknown')
                test_data = {task_type: data['data']}
            else:
                raise ValueError("Empty data array in JSON file")
        
        # Format 3: Direct task mapping {"dna_complement": [...], ...}
        elif any(key in data for key in ['dna_complement', 'rna_complement', 
                                          'protein_three_to_one', 'protein_one_to_three']):
            test_data = {k: v for k, v in data.items() 
                        if k in ['dna_complement', 'rna_complement', 
                                'protein_three_to_one', 'protein_one_to_three']}
        
        else:
            raise ValueError(
                "Unrecognized JSON format. Expected format with 'datasets', 'data', "
                "or direct task type keys (dna_complement, etc.)"
            )
        
        # Validate test data
        if not test_data:
            raise ValueError("No valid test data found in JSON file")
        
        # Verify data structure
        for task_type, cases in test_data.items():
            if not isinstance(cases, list):
                raise ValueError(f"Task '{task_type}' data must be a list")
            if not cases:
                raise ValueError(f"Task '{task_type}' has no test cases")
            
            # Verify required fields in first case
            required_fields = ['task_type', 'input', 'expected_output']
            first_case = cases[0]
            missing_fields = [f for f in required_fields if f not in first_case]
            if missing_fields:
                raise ValueError(
                    f"Task '{task_type}' cases missing required fields: {missing_fields}"
                )
        
        if verbose:
            print(f"✓ Loaded {len(test_data)} task(s):")
            for task_type, cases in test_data.items():
                print(f"  - {task_type}: {len(cases)} cases")
        
        return test_data
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading test data: {e}")

def main():
    """Main function to run the benchmark."""
    args = parse_arguments()

    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = resolve_config_path("biological_sequence", args.config)
    
    # Print verbose mode status
    if args.verbose:
        print("\n" + "="*80)
        print("VERBOSE MODE ENABLED - Detailed interaction logs will be displayed")
        print("="*80 + "\n")
    
    # Initialize benchmark
    effective_config = config_path if os.path.exists(config_path) else None
    benchmark = BiologicalSequenceBenchmark(effective_config)
    
    # Set verbose mode in benchmark
    benchmark.verbose = args.verbose

    # Apply restricted reasoning flag
    benchmark.config.setdefault("benchmark_settings", {})
    if args.restricted_reasoning:
        benchmark.config["benchmark_settings"]["restricted_reasoning"] = True
    # Sync runtime flag with updated config
    benchmark.restricted_reasoning = benchmark.config["benchmark_settings"].get(
        "restricted_reasoning", False
    )
    
    # Update output directory if specified
    if args.output_dir:
        benchmark.config["benchmark_settings"]["results_dir"] = args.output_dir
    
    # Filter tasks if specified
    if args.tasks:
        for task_type in list(benchmark.config["tasks"].keys()):
            if task_type not in args.tasks:
                benchmark.config["tasks"][task_type]["enabled"] = False
    
    # Load test data from file or generate new data
    test_data = None
    if args.load_data:
        try:
            test_data = load_test_data_from_file(resolve_path(args.load_data), verbose=args.verbose)
            
            # Update benchmark config to match loaded data
            for task_type in benchmark.config["tasks"].keys():
                if task_type not in test_data:
                    benchmark.config["tasks"][task_type]["enabled"] = False
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading test data: {e}")
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
    
    if test_data:
        total_cases = sum(len(cases) for cases in test_data.values())
        print(f"Total test cases: {total_cases} (loaded from file)")
    else:
        total_cases = sum(
            benchmark.config["tasks"][task]["num_cases"] 
            for task in enabled_tasks
        )
        print(f"Total test cases: {total_cases} (will be generated)")
    
    print(f"Results will be saved to: {benchmark.config['benchmark_settings']['results_dir']}")
    print(f"Verbose mode: {'ON' if args.verbose else 'OFF'}")
    print()
    
    try:
        # Run the benchmark with pre-loaded data if available
        if test_data:
            if args.verbose:
                print("Using pre-loaded test data from file\n")
            
            # Run benchmark on all models with pre-loaded data
            results = benchmark.run_multiple_models(
                model_configs, 
                test_data=test_data, 
                verbose=True
            )
            
            # Save results if requested
            if not args.no_save:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                benchmark.save_results(results, timestamp)
            
            # Wrap results in expected format
            results = {
                "results": results,
                "test_data": test_data,
                "config": benchmark.config,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Generate data and run benchmark
            results = benchmark.run_full_benchmark(
                model_configs=model_configs,
                save_results=not args.no_save,
                verbose=True  # Always show progress bar
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
