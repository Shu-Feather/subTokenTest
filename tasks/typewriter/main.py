"""
Typewriter Effect Simulation Benchmark
Main entry point for running evaluations
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

TASK_ROOT = Path(__file__).resolve().parent
PARENT_OF_TASK = TASK_ROOT.parent


def find_project_root() -> Path:
    for parent in TASK_ROOT.parents:
        if (parent / "cli.py").exists():
            return parent
    return TASK_ROOT.parents[-1]


PROJECT_ROOT = find_project_root()

# Ensure task-local modules are importable before shared ones
for candidate in [TASK_ROOT, PARENT_OF_TASK, PROJECT_ROOT / "configs" / "typewriter", PROJECT_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Load environment variables
load_dotenv()

from typewriter.evaluation.evaluator import TypewriterBenchmarkEvaluator
from typewriter.data.test_cases import TestCaseLoader, load_test_cases, load_test_cases_from_dir
from typewriter.utils.helpers import save_results, create_summary_report, setup_environment
from configs.typewriter.model_config import list_available_models
from typewriter.models import create_model, VLLM_AVAILABLE


def resolve_path(path_str: str) -> str:
    """Resolve path relative to CWD or repository root."""
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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Typewriter Effect Simulation Benchmark",
    )
    
    # Model selection
    parser.add_argument(
        '--models', 
        nargs='+', 
        help=f'Models to evaluate. Available: {", ".join(list_available_models())}',
        default=None
    )
    
    # Prompt types
    parser.add_argument(
        '--prompt-types',
        nargs='+',
        choices=['system', 'few_shot'],
        default=['system'],
        help='Types of prompts to use (default: system)'
    )
    
    # Test cases loading
    parser.add_argument(
        '--test-file',
        type=str,
        help='JSON file containing test cases'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default='datasets',
        help='Directory containing test cases (default: datasets)'
    )
    
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['easy', 'medium', 'hard', 'all'],
        default='all',
        help='Difficulty level of test cases (default: all)'
    )
    
    # Number of samples
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of test samples to use per task (default: all available samples)'
    )

    # vLLM options
    parser.add_argument(
        '--use-vllm',
        action='store_true',
        help='Use vLLM for local models (faster inference)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Use batch generation (only works with vLLM)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # List models
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    parser.add_argument(
        '--restricted-reasoning',
        action='store_true',
        help='Use restricted thinking prompts that encourage minimal reasoning'
    )
    
    args = parser.parse_args()
    if args.test_file:
        args.test_file = resolve_path(args.test_file)
    args.test_dir = resolve_path(args.test_dir)
    args.output_dir = resolve_path(args.output_dir)
    
    # Set up environment
    setup_environment()
    
    # Handle special cases
    if args.list_models:
        print("Available models:")
        for model in list_available_models():
            print(f"  - {model}")
        if VLLM_AVAILABLE:
            print("\n✓ vLLM is available for accelerated local model inference")
        else:
            print("\n✗ vLLM is not available. Install with: pip install vllm")
        return
    
    if not args.models:
        print("Error: No models specified. Use --models to specify models or --list-models to see available models.")
        sys.exit(1)
    
    # Warn if vLLM requested but not available
    if args.use_vllm and not VLLM_AVAILABLE:
        print("Warning: vLLM requested but not installed. Falling back to standard inference.")
        print("Install vLLM with: pip install vllm")
    
    # Load test cases
    test_cases = None
    if args.test_file:
        if not os.path.exists(args.test_file):
            print(f"Error: Test file {args.test_file} not found.")
            sys.exit(1)
        
        print(f"Loading test cases from file: {args.test_file}")
        test_cases = load_test_cases(args.test_file, args.difficulty)

    elif os.path.exists(args.test_dir):
        print(f"Loading test cases from directory: {args.test_dir} (difficulty: {args.difficulty})")
        try:
            test_cases = load_test_cases_from_dir(args.test_dir, args.difficulty)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Using legacy test case generator...")
            test_cases = None
    
    if test_cases:
        print(f"Loaded {len(test_cases['task1'])} test cases for Task 1")
        print(f"Loaded {len(test_cases['task2'])} test cases for Task 2")

        # Apply num_samples limit if specified
        if args.num_samples is not None:
            if args.num_samples <= 0:
                print(f"Error: --num-samples must be a positive integer, got {args.num_samples}")
                sys.exit(1)
            
            original_task1_count = len(test_cases['task1'])
            original_task2_count = len(test_cases['task2'])
            
            test_cases['task1'] = test_cases['task1'][:args.num_samples]
            test_cases['task2'] = test_cases['task2'][:args.num_samples]
            
            print(f"Limiting to {args.num_samples} samples per task:")
            print(f"  Task 1: {original_task1_count} -> {len(test_cases['task1'])} samples")
            print(f"  Task 2: {original_task2_count} -> {len(test_cases['task2'])} samples")
            
    # Run evaluation
    evaluator = TypewriterBenchmarkEvaluator(restricted_reasoning=args.restricted_reasoning)
    
    try:
        if len(args.models) == 1:
            # Single model evaluation
            model_name = args.models[0]
            print(f"Evaluating single model: {model_name}")
            if args.use_vllm:
                print("Using vLLM for inference")
            if args.batch:
                print("Using batch generation")
            
            model = create_model(model_name, use_vllm=args.use_vllm)
            results = {}
            
            for prompt_type in args.prompt_types:
                print(f"Using prompt type: {prompt_type}")
                result = evaluator.evaluate_model(
                    model, 
                    test_cases, 
                    prompt_type,
                    use_batch=args.batch
                )
                results[prompt_type] = result
            
            # Save results
            if len(args.prompt_types) == 1:
                final_results = results[args.prompt_types[0]]
            else:
                final_results = {model_name: results}
            
        else:
            # Multiple model evaluation
            print(f"Evaluating multiple models: {', '.join(args.models)}")
            if args.use_vllm:
                print("Using vLLM for local models")
            if args.batch:
                print("Using batch generation where supported")
            
            final_results = evaluator.evaluate_multiple_models(
                args.models, 
                test_cases, 
                args.prompt_types,
                use_vllm=args.use_vllm,
                use_batch=args.batch
            )
        
        # Save results
        output_path = save_results(final_results, args.output_dir)
        
        # Generate and print summary
        summary = create_summary_report(final_results)
        print(summary)
        
        # Save summary to file
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
