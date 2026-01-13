"""
Main entry point for the Gomoku Benchmark.
Location: /main.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from configs.locator import resolve_config_path

TASK_ROOT = Path(__file__).resolve().parent

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
CONFIG_PATH = PROJECT_ROOT / "configs" / "gomoku"
for candidate in [CONFIG_PATH, PROJECT_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from configs.gomoku.config import ModelConfig, BenchmarkConfig, DEFAULT_MODELS
from src.benchmark_runner import BenchmarkRunner
from src.evaluator import print_evaluation_summary


def resolve_path(path_str: str) -> str:
    """Resolve path relative to CWD or repository root (with dataset fallback)."""
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Gomoku (Five-in-a-Row) Benchmark for Large Language Models",
    )
    
    parser.add_argument(
        '--models', 
        nargs='+', 
        choices=list(DEFAULT_MODELS.keys()) + ['all'],
        help='Models to test (default: all available models)'
    )
    
    parser.add_argument(
        '--board-sizes', 
        type=int, 
        nargs='+', 
        help='Board sizes to test'
    )
    
    parser.add_argument(
        '--test-counts', 
        type=int, 
        nargs='+', 
        help='Number of test cases per configuration'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        default='example_config.json',
        help='JSON configuration file path (default: configs/gomoku/example_config.json)'
    )
    
    parser.add_argument(
        '--test-file',
        type=str,
        default=None,
        help='Load test cases from JSON file instead of generating'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='datasets',
        help='Directory for test data (default: datasets)'
    )
    
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate test data, do not run models'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--save-response',
        type=str,
        default=None,
        help='Save detailed response logs to specified JSON file'
    )
    
    parser.add_argument(
        '--restricted-reasoning',
        action='store_true',
        help='Use restricted thinking prompts that encourage minimal reasoning'
    )
    
    return parser.parse_args()

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)

def load_test_file(test_file_path: str) -> List[Dict[str, Any]]:
    """Load test cases from JSON file"""
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test cases from {test_file_path}")
        return test_data
    except Exception as e:
        print(f"Error loading test file {test_file_path}: {e}")
        sys.exit(1)

def generate_test_data_only(board_sizes: List[int], test_counts: List[int], data_dir: str):
    """Generate test data without running models"""
    from src.board_generator import generate_test_cases
    import os
    import json
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating test data...")
    
    for board_size in board_sizes:
        for num_cases in test_counts:
            print(f"Generating {num_cases} cases for {board_size}x{board_size} board...")
            
            test_cases = generate_test_cases(board_size, num_cases)
            
            data_filename = f"test_data_{board_size}x{board_size}_{num_cases}.json"
            data_path = os.path.join(data_dir, data_filename)
            
            data_to_save = [
                {
                    'board': board,
                    'expected': expected,
                    'board_size': board_size
                }
                for board, expected in test_cases
            ]
            
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            print(f"Saved to {data_path}")
    
    print("Test data generation completed!")

def main():
    """Main function"""
    args = parse_arguments()
    if args.config_file:
        if not Path(args.config_file).is_absolute():
            args.config_file = resolve_config_path("gomoku", args.config_file)
        else:
            args.config_file = resolve_path(args.config_file)
    if args.test_file:
        args.test_file = resolve_path(args.test_file)
    
    # Load configuration from file if specified
    if args.config_file:
        config_data = load_config_file(args.config_file)
        # Override with command line arguments if provided
        models = args.models or config_data.get('models', ['gpt-4'])
        board_sizes = config_data.get('board_sizes', args.board_sizes)
        test_counts = config_data.get('test_counts', args.test_counts)
        output_dir = config_data.get('output_dir', args.output_dir)
        data_dir = config_data.get('data_dir', args.data_dir)
        test_file = args.test_file or config_data.get('test_file')
        save_response = args.save_response or config_data.get('save_response')
    else:
        models = args.models
        board_sizes = args.board_sizes
        test_counts = args.test_counts
        output_dir = args.output_dir
        data_dir = args.data_dir
        test_file = args.test_file
        save_response = args.save_response
    
    # Default to all models if none specified
    if not models:
        models = list(DEFAULT_MODELS.keys())
    elif 'all' in models:
        models = list(DEFAULT_MODELS.keys())
    
    # Validate models
    invalid_models = [m for m in models if m not in DEFAULT_MODELS]
    if invalid_models:
        print(f"Warning: Unknown models will be skipped: {invalid_models}")
        models = [m for m in models if m in DEFAULT_MODELS]
    
    if not models:
        print("Error: No valid models specified")
        sys.exit(1)
    
    # Generate test data only
    if args.generate_only:
        generate_test_data_only(board_sizes, test_counts, data_dir)
        return

    # Apply restricted reasoning flag to selected models
    if args.restricted_reasoning:
        for model_name in models:
            DEFAULT_MODELS[model_name].restricted_reasoning = True
    
    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(
        board_sizes=board_sizes,
        test_counts=test_counts,
        output_dir=output_dir,
        data_dir=data_dir,
        verbose=args.verbose,
        test_file=test_file,
        save_response=save_response  # Add save_response to config
    )
    
    # Get model configurations
    model_configs = [DEFAULT_MODELS[model_name] for model_name in models]
    
    print(f"Starting benchmark with:")
    print(f"  Models: {models}")
    print(f"  Board sizes: {board_sizes}")
    print(f"  Test counts: {test_counts}")
    if test_file:
        print(f"  Test file: {test_file}")
    if save_response:
        print(f"  Response log file: {save_response}")
    print(f"  Output directory: {output_dir}")
    print(f"  Data directory: {data_dir}\n")
    
    # Create and run benchmark
    runner = BenchmarkRunner(benchmark_config)
    
    try:
        results = runner.run_benchmark(model_configs, board_sizes, test_counts)
        
        # Print summary
        print("\n" + "="*50)
        print("BENCHMARK COMPLETED")
        print("="*50)
        
        summary = results.get('summary', {})
        if summary:
            print(f"Total models tested: {summary.get('total_models_tested', 0)}")
            print(f"Total test cases: {summary.get('total_test_cases', 0)}")
            print(f"Best performing model: {summary.get('best_model', 'N/A')}")
            
            print(f"\nModel Performance Summary:")
            print("-" * 80)
            print(f"{'Model':<20} {'Avg Accuracy':<15} {'Total Cases':<12} {'Success Rate':<12}")
            print("-" * 80)
            
            for model_name, model_summary in summary.get('model_summaries', {}).items():
                avg_acc = model_summary.get('average_accuracy', 0)
                total_cases = model_summary.get('total_cases', 0)
                success_cases = model_summary.get('successful_cases', 0)
                success_rate = success_cases / total_cases if total_cases > 0 else 0
                
                print(f"{model_name:<20} {avg_acc:<15.2%} {total_cases:<12} {success_rate:<12.2%}")
        
        print(f"\nDetailed results saved in: {output_dir}/")
        if save_response:
            print(f"Response logs saved to: {save_response}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
