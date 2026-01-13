"""
Main script to run the Context-Aware Redaction benchmark
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

TASK_ROOT = Path(__file__).resolve().parent


def find_project_root() -> Path:
    for parent in TASK_ROOT.parents:
        if (parent / "cli.py").exists():
            return parent
    return TASK_ROOT.parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_json
from src.models import VLLMModel, APIModel
from src.benchmark import ContextAwareRedactionBenchmark
from configs.locator import resolve_config_path


def resolve_path(path_str: str) -> str:
    """Resolve file path relative to CWD or repository root (with dataset fallback)."""
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


def create_model(model_type: str, model_name: str, config: dict, 
                 api_key: str = None, base_url: str = None, verbose: bool = False):
    """
    Create model instance based on type
    
    Args:
        model_type: Type of model ('vllm' or 'api')
        model_name: Model name or path
        config: Configuration dictionary
        api_key: API key (for API models)
        base_url: Base URL (for API models)
        verbose: Verbose output
        
    Returns:
        Model instance
    """
    if model_type == 'vllm':
        return VLLMModel(model_name, config, verbose=verbose)
    elif model_type == 'api':
        if not api_key:
            raise ValueError("API key required for API models")
        return APIModel(model_name, config, api_key, base_url, verbose=verbose)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Context-Aware Redaction Benchmark'
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    
    # Model settings
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['vllm', 'api'],
                        help='Type of model to use (vllm for local models, api for API-based models)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name or path (e.g., "meta-llama/Llama-2-7b-chat-hf", "gpt-4", "deepseek-chat")')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key (required for API models)')
    parser.add_argument('--base_url', type=str, default=None,
                        help='Base URL for API (optional, for custom endpoints like DeepSeek)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=None,
                        help='GPU memory fraction for vLLM models (0-1, overrides config)')
    parser.add_argument('--tensor_parallel_size', type=int, default=None,
                        help='Tensor parallel size for vLLM models (>=1, overrides config)')
    parser.add_argument('--enforce_eager', dest='enforce_eager', action='store_true',
                        help='Force vLLM eager mode (overrides config)')
    parser.add_argument('--no_enforce_eager', dest='enforce_eager', action='store_false',
                        help='Disable vLLM eager mode (overrides config)')
    parser.set_defaults(enforce_eager=None)
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to test dataset JSON file')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to test (default: all)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output including prompts and model responses')
    parser.add_argument('--restricted-reasoning', action='store_true',
                        help='Use restricted thinking prompts that encourage minimal reasoning')
    
    args = parser.parse_args()

    if not Path(args.config).is_absolute():
        args.config = resolve_config_path("context_aware_redaction", args.config)
    args.dataset = resolve_path(args.dataset)
    
    # Load configuration
    config = load_config(args.config)
    prompt_cfg = config.setdefault('prompt', {})
    if args.restricted_reasoning:
        prompt_cfg['restricted_reasoning'] = True
    else:
        prompt_cfg.setdefault('restricted_reasoning', False)
    
    # Override vLLM settings if provided
    vllm_cfg = config.setdefault('models', {}).setdefault('vllm', {})
    if args.gpu_memory_utilization is not None:
        if not 0 < args.gpu_memory_utilization <= 1:
            raise ValueError("--gpu_memory_utilization must be between 0 and 1")
        vllm_cfg['gpu_memory_utilization'] = args.gpu_memory_utilization
    if args.tensor_parallel_size is not None:
        if args.tensor_parallel_size < 1:
            raise ValueError("--tensor_parallel_size must be >= 1")
        vllm_cfg['tensor_parallel_size'] = args.tensor_parallel_size
    if args.enforce_eager is not None:
        vllm_cfg['enforce_eager'] = args.enforce_eager
    
    # Update num_samples from config if not specified
    if args.num_samples is None:
        args.num_samples = config['test'].get('num_samples')
    
    print("="*80)
    print("Context-Aware Redaction Benchmark")
    print("="*80)
    print(f"Model Type: {args.model_type}")
    print(f"Model Name: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Num Samples: {args.num_samples if args.num_samples else 'All'}")
    print(f"Verbose: {args.verbose}")
    print("="*80)
    
    # Create model
    print("\nInitializing model...")
    model = create_model(
        model_type=args.model_type,
        model_name=args.model_name,
        config=config,
        api_key=args.api_key,
        base_url=args.base_url,
        verbose=args.verbose
    )
    
    # Create benchmark
    print("Initializing benchmark...")
    benchmark = ContextAwareRedactionBenchmark(
        config=config,
        model=model,
        verbose=args.verbose
    )
    
    # Run benchmark
    print("\nRunning benchmark...")
    results = benchmark.run(
        dataset_path=args.dataset,
        num_samples=args.num_samples
    )
    
    # Print summary
    benchmark.print_summary(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_name.replace('/', '_').replace('\\', '_')
    output_file = os.path.join(
        args.output_dir,
        f"results_{model_name_safe}_{timestamp}.json"
    )
    
    save_json(results, output_file)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Save summary
    summary = {
        'model': results['model'],
        'timestamp': timestamp,
        'dataset': args.dataset,
        'total_samples': results['total_samples'],
        'overall_metrics': results['overall_metrics'],
        'difficulty_metrics': results['difficulty_metrics']
    }
    
    summary_file = os.path.join(
        args.output_dir,
        f"summary_{model_name_safe}_{timestamp}.json"
    )
    save_json(summary, summary_file)
    print(f"Summary saved to: {summary_file}")
    
    print("\nBenchmark completed successfully!")


if __name__ == '__main__':
    main()
