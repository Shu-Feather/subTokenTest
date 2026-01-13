"""
Main entry point for Adversarial Prompt Canonicalization Benchmark.
"""

import argparse
import yaml
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

TASK_ROOT = Path(__file__).resolve().parent


def find_project_root() -> Path:
    for parent in TASK_ROOT.parents:
        if (parent / "cli.py").exists():
            return parent
    return TASK_ROOT.parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import BenchmarkDataGenerator
from src.models import VLLMModel, OpenAIModel, DeepSeekModel
from src.evaluator import Evaluator
from configs.locator import resolve_config_path


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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def aggregate_token_usage(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate token usage statistics from all results.
    
    Args:
        results: List of evaluation results, each containing usage_info
        
    Returns:
        Dict with aggregated token usage statistics
    """
    total_stats = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "output_tokens": 0,
        "num_requests": 0,
    }
    
    by_category = {}
    by_difficulty = {}
    by_perturbation = {}
    
    for result in results:
        usage_info = result.get('usage_info')
        if not usage_info:
            continue
        
        # Update total stats
        total_stats["num_requests"] += 1
        total_stats["total_tokens"] += usage_info.get("total_tokens", 0)
        total_stats["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
        total_stats["completion_tokens"] += usage_info.get("completion_tokens", 0)
        total_stats["reasoning_tokens"] += usage_info.get("reasoning_tokens", 0)
        total_stats["output_tokens"] += usage_info.get("output_tokens", 0)
        
        # Aggregate by category
        category = result.get('category', 'unknown')
        if category not in by_category:
            by_category[category] = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "count": 0,
            }
        by_category[category]["count"] += 1
        by_category[category]["total_tokens"] += usage_info.get("total_tokens", 0)
        by_category[category]["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
        by_category[category]["completion_tokens"] += usage_info.get("completion_tokens", 0)
        by_category[category]["reasoning_tokens"] += usage_info.get("reasoning_tokens", 0)
        by_category[category]["output_tokens"] += usage_info.get("output_tokens", 0)
        
        # Aggregate by difficulty
        difficulty = result.get('difficulty', 'unknown')
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "count": 0,
            }
        by_difficulty[difficulty]["count"] += 1
        by_difficulty[difficulty]["total_tokens"] += usage_info.get("total_tokens", 0)
        by_difficulty[difficulty]["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
        by_difficulty[difficulty]["completion_tokens"] += usage_info.get("completion_tokens", 0)
        by_difficulty[difficulty]["reasoning_tokens"] += usage_info.get("reasoning_tokens", 0)
        by_difficulty[difficulty]["output_tokens"] += usage_info.get("output_tokens", 0)
        
        # Aggregate by perturbation type
        perturbation = result.get('perturbation_type', 'unknown')
        if perturbation not in by_perturbation:
            by_perturbation[perturbation] = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "count": 0,
            }
        by_perturbation[perturbation]["count"] += 1
        by_perturbation[perturbation]["total_tokens"] += usage_info.get("total_tokens", 0)
        by_perturbation[perturbation]["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
        by_perturbation[perturbation]["completion_tokens"] += usage_info.get("completion_tokens", 0)
        by_perturbation[perturbation]["reasoning_tokens"] += usage_info.get("reasoning_tokens", 0)
        by_perturbation[perturbation]["output_tokens"] += usage_info.get("output_tokens", 0)
    
    # Calculate averages for categories
    for category, stats in by_category.items():
        if stats["count"] > 0:
            stats["avg_total_tokens"] = stats["total_tokens"] / stats["count"]
            stats["avg_prompt_tokens"] = stats["prompt_tokens"] / stats["count"]
            stats["avg_completion_tokens"] = stats["completion_tokens"] / stats["count"]
            stats["avg_reasoning_tokens"] = stats["reasoning_tokens"] / stats["count"]
            stats["avg_output_tokens"] = stats["output_tokens"] / stats["count"]
            if stats["total_tokens"] > 0:
                stats["reasoning_ratio"] = stats["reasoning_tokens"] / stats["total_tokens"]
    
    # Calculate averages for difficulties
    for difficulty, stats in by_difficulty.items():
        if stats["count"] > 0:
            stats["avg_total_tokens"] = stats["total_tokens"] / stats["count"]
            stats["avg_prompt_tokens"] = stats["prompt_tokens"] / stats["count"]
            stats["avg_completion_tokens"] = stats["completion_tokens"] / stats["count"]
            stats["avg_reasoning_tokens"] = stats["reasoning_tokens"] / stats["count"]
            stats["avg_output_tokens"] = stats["output_tokens"] / stats["count"]
            if stats["total_tokens"] > 0:
                stats["reasoning_ratio"] = stats["reasoning_tokens"] / stats["total_tokens"]
    
    # Calculate averages for perturbations
    for perturbation, stats in by_perturbation.items():
        if stats["count"] > 0:
            stats["avg_total_tokens"] = stats["total_tokens"] / stats["count"]
            stats["avg_prompt_tokens"] = stats["prompt_tokens"] / stats["count"]
            stats["avg_completion_tokens"] = stats["completion_tokens"] / stats["count"]
            stats["avg_reasoning_tokens"] = stats["reasoning_tokens"] / stats["count"]
            stats["avg_output_tokens"] = stats["output_tokens"] / stats["count"]
            if stats["total_tokens"] > 0:
                stats["reasoning_ratio"] = stats["reasoning_tokens"] / stats["total_tokens"]
    
    # Calculate overall averages
    if total_stats["num_requests"] > 0:
        total_stats["avg_total_tokens"] = total_stats["total_tokens"] / total_stats["num_requests"]
        total_stats["avg_prompt_tokens"] = total_stats["prompt_tokens"] / total_stats["num_requests"]
        total_stats["avg_completion_tokens"] = total_stats["completion_tokens"] / total_stats["num_requests"]
        total_stats["avg_reasoning_tokens"] = total_stats["reasoning_tokens"] / total_stats["num_requests"]
        total_stats["avg_output_tokens"] = total_stats["output_tokens"] / total_stats["num_requests"]
        if total_stats["total_tokens"] > 0:
            total_stats["reasoning_ratio"] = total_stats["reasoning_tokens"] / total_stats["total_tokens"]
    
    return {
        "total": total_stats,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "by_perturbation": by_perturbation,
    }


def save_results(results: list, metrics: dict, output_dir: str, model_name: str):
    """Save evaluation results and metrics to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"{model_name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Aggregate token usage statistics
    token_usage = aggregate_token_usage(results)
    
    # Save detailed results
    results_file = output_path / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save token usage statistics
    token_usage_file = output_path / "token_usage.json"
    with open(token_usage_file, 'w') as f:
        json.dump(token_usage, f, indent=2)
    
    # Save human-readable report
    report_file = output_path / "report.txt"
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ADVERSARIAL PROMPT CANONICALIZATION BENCHMARK REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Samples: {metrics['total_samples']}\n\n")
        
        # Performance metrics
        f.write("-"*70 + "\n")
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Exact Match Rate: {metrics['exact_match_rate']:.2%}\n")
        f.write(f"Average Similarity Score: {metrics['avg_similarity_score']:.4f}\n")
        f.write(f"Average Levenshtein Distance: {metrics['avg_levenshtein_distance']:.2f}\n")
        f.write(f"Answer Tags Found Rate: {metrics['tags_found_rate']:.2%}\n\n")
        
        # Token usage statistics
        f.write("-"*70 + "\n")
        f.write("OVERALL TOKEN USAGE STATISTICS\n")
        f.write("-"*70 + "\n")
        total_usage = token_usage["total"]
        f.write(f"Total Requests: {total_usage['num_requests']}\n")
        f.write(f"Total Tokens: {total_usage['total_tokens']:,}\n")
        f.write(f"Total Prompt Tokens: {total_usage['prompt_tokens']:,}\n")
        f.write(f"Total Completion Tokens: {total_usage['completion_tokens']:,}\n")
        f.write(f"Total Reasoning Tokens: {total_usage['reasoning_tokens']:,}\n")
        f.write(f"Total Output Tokens: {total_usage['output_tokens']:,}\n\n")
        f.write(f"Avg Tokens per Request: {total_usage.get('avg_total_tokens', 0):.2f}\n")
        f.write(f"Avg Prompt Tokens: {total_usage.get('avg_prompt_tokens', 0):.2f}\n")
        f.write(f"Avg Completion Tokens: {total_usage.get('avg_completion_tokens', 0):.2f}\n")
        f.write(f"Avg Reasoning Tokens: {total_usage.get('avg_reasoning_tokens', 0):.2f}\n")
        f.write(f"Avg Output Tokens: {total_usage.get('avg_output_tokens', 0):.2f}\n")
        if total_usage.get('reasoning_ratio'):
            f.write(f"Reasoning Token Ratio: {total_usage['reasoning_ratio']:.2%}\n")
        f.write("\n")
        
        # Metrics by difficulty
        f.write("-"*70 + "\n")
        f.write("METRICS BY DIFFICULTY\n")
        f.write("-"*70 + "\n")
        for diff, diff_metrics in metrics['by_difficulty'].items():
            f.write(f"\n{diff}:\n")
            f.write(f"  Performance:\n")
            f.write(f"    Count: {diff_metrics['count']}\n")
            f.write(f"    Exact Match Rate: {diff_metrics['exact_match_rate']:.2%}\n")
            f.write(f"    Avg Similarity: {diff_metrics['avg_similarity']:.4f}\n")
            f.write(f"    Tags Found Rate: {diff_metrics['tags_found_rate']:.2%}\n")
            
            if diff in token_usage['by_difficulty']:
                usage = token_usage['by_difficulty'][diff]
                f.write(f"  Token Usage:\n")
                f.write(f"    Avg Total: {usage.get('avg_total_tokens', 0):.2f}\n")
                f.write(f"    Avg Prompt: {usage.get('avg_prompt_tokens', 0):.2f}\n")
                f.write(f"    Avg Completion: {usage.get('avg_completion_tokens', 0):.2f}\n")
                f.write(f"    Avg Reasoning: {usage.get('avg_reasoning_tokens', 0):.2f}\n")
                if usage.get('reasoning_ratio'):
                    f.write(f"    Reasoning Ratio: {usage['reasoning_ratio']:.2%}\n")
        
        # Metrics by category
        f.write("\n" + "-"*70 + "\n")
        f.write("METRICS BY CATEGORY\n")
        f.write("-"*70 + "\n")
        for cat, cat_metrics in metrics['by_category'].items():
            f.write(f"\n{cat}:\n")
            f.write(f"  Performance:\n")
            f.write(f"    Count: {cat_metrics['count']}\n")
            f.write(f"    Exact Match Rate: {cat_metrics['exact_match_rate']:.2%}\n")
            f.write(f"    Avg Similarity: {cat_metrics['avg_similarity']:.4f}\n")
            f.write(f"    Tags Found Rate: {cat_metrics['tags_found_rate']:.2%}\n")
            
            if cat in token_usage['by_category']:
                usage = token_usage['by_category'][cat]
                f.write(f"  Token Usage:\n")
                f.write(f"    Avg Total: {usage.get('avg_total_tokens', 0):.2f}\n")
                f.write(f"    Avg Prompt: {usage.get('avg_prompt_tokens', 0):.2f}\n")
                f.write(f"    Avg Completion: {usage.get('avg_completion_tokens', 0):.2f}\n")
                f.write(f"    Avg Reasoning: {usage.get('avg_reasoning_tokens', 0):.2f}\n")
                if usage.get('reasoning_ratio'):
                    f.write(f"    Reasoning Ratio: {usage['reasoning_ratio']:.2%}\n")
        
        # Metrics by perturbation type
        f.write("\n" + "-"*70 + "\n")
        f.write("METRICS BY PERTURBATION TYPE\n")
        f.write("-"*70 + "\n")
        for pert, pert_metrics in metrics['by_perturbation'].items():
            f.write(f"\n{pert}:\n")
            f.write(f"  Performance:\n")
            f.write(f"    Count: {pert_metrics['count']}\n")
            f.write(f"    Exact Match Rate: {pert_metrics['exact_match_rate']:.2%}\n")
            f.write(f"    Avg Similarity: {pert_metrics['avg_similarity']:.4f}\n")
            f.write(f"    Tags Found Rate: {pert_metrics['tags_found_rate']:.2%}\n")
            
            if pert in token_usage['by_perturbation']:
                usage = token_usage['by_perturbation'][pert]
                f.write(f"  Token Usage:\n")
                f.write(f"    Avg Total: {usage.get('avg_total_tokens', 0):.2f}\n")
                f.write(f"    Avg Prompt: {usage.get('avg_prompt_tokens', 0):.2f}\n")
                f.write(f"    Avg Completion: {usage.get('avg_completion_tokens', 0):.2f}\n")
                f.write(f"    Avg Reasoning: {usage.get('avg_reasoning_tokens', 0):.2f}\n")
                if usage.get('reasoning_ratio'):
                    f.write(f"    Reasoning Ratio: {usage['reasoning_ratio']:.2%}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\nResults saved to: {output_path}")
    print(f"  - Detailed results: {results_file}")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Token usage: {token_usage_file}")
    print(f"  - Report: {report_file}")
    
    return output_path


def print_metrics_summary(metrics: dict, token_usage: dict):
    """Print metrics summary to console."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Total Samples: {metrics['total_samples']}")
    print(f"  Exact Match Rate: {metrics['exact_match_rate']:.2%}")
    print(f"  Average Similarity Score: {metrics['avg_similarity_score']:.4f}")
    print(f"  Average Levenshtein Distance: {metrics['avg_levenshtein_distance']:.2f}")
    print(f"  Answer Tags Found Rate: {metrics['tags_found_rate']:.2%}")
    
    # Token usage summary
    print(f"\nToken Usage Summary:")
    total_usage = token_usage["total"]
    print(f"  Total Requests: {total_usage['num_requests']}")
    print(f"  Total Tokens: {total_usage['total_tokens']:,}")
    print(f"  Total Prompt Tokens: {total_usage['prompt_tokens']:,}")
    print(f"  Total Completion Tokens: {total_usage['completion_tokens']:,}")
    print(f"  Total Reasoning Tokens: {total_usage['reasoning_tokens']:,}")
    print(f"  Total Output Tokens: {total_usage['output_tokens']:,}")
    print(f"\n  Avg Tokens per Request: {total_usage.get('avg_total_tokens', 0):.2f}")
    print(f"  Avg Prompt Tokens: {total_usage.get('avg_prompt_tokens', 0):.2f}")
    print(f"  Avg Completion Tokens: {total_usage.get('avg_completion_tokens', 0):.2f}")
    print(f"  Avg Reasoning Tokens: {total_usage.get('avg_reasoning_tokens', 0):.2f}")
    if total_usage.get('reasoning_ratio'):
        print(f"  Reasoning Token Ratio: {total_usage['reasoning_ratio']:.2%}")
    
    print("\n" + "-"*70)
    print("By Difficulty:")
    print("-"*70)
    for diff, diff_metrics in metrics['by_difficulty'].items():
        perf_str = (f"  {diff}: Match={diff_metrics['exact_match_rate']:.2%}, "
                   f"Similarity={diff_metrics['avg_similarity']:.4f}, "
                   f"Tags={diff_metrics['tags_found_rate']:.2%}")
        
        if diff in token_usage['by_difficulty']:
            usage = token_usage['by_difficulty'][diff]
            perf_str += (f"\n         Tokens: Avg={usage.get('avg_total_tokens', 0):.1f}, "
                        f"Reasoning={usage.get('avg_reasoning_tokens', 0):.1f}")
            if usage.get('reasoning_ratio'):
                perf_str += f" ({usage['reasoning_ratio']:.1%})"
        
        print(perf_str)
    
    print("\n" + "-"*70)
    print("By Category:")
    print("-"*70)
    for cat, cat_metrics in metrics['by_category'].items():
        perf_str = (f"  {cat}: Match={cat_metrics['exact_match_rate']:.2%}, "
                   f"Similarity={cat_metrics['avg_similarity']:.4f}")
        
        if cat in token_usage['by_category']:
            usage = token_usage['by_category'][cat]
            perf_str += (f"\n         Tokens: Avg={usage.get('avg_total_tokens', 0):.1f}, "
                        f"Reasoning={usage.get('avg_reasoning_tokens', 0):.1f}")
            if usage.get('reasoning_ratio'):
                perf_str += f" ({usage['reasoning_ratio']:.1%})"
        
        print(perf_str)
    
    print("\n" + "-"*70)
    print("By Perturbation Type:")
    print("-"*70)
    for pert, pert_metrics in metrics['by_perturbation'].items():
        perf_str = (f"  {pert}: Match={pert_metrics['exact_match_rate']:.2%}, "
                   f"Similarity={pert_metrics['avg_similarity']:.4f}")
        
        if pert in token_usage['by_perturbation']:
            usage = token_usage['by_perturbation'][pert]
            perf_str += (f"\n         Tokens: Avg={usage.get('avg_total_tokens', 0):.1f}, "
                        f"Reasoning={usage.get('avg_reasoning_tokens', 0):.1f}")
            if usage.get('reasoning_ratio'):
                perf_str += f" ({usage['reasoning_ratio']:.1%})"
        
        print(perf_str)
    
    print("="*70 + "\n")


def initialize_model(args, config, verbose: bool):
    """Initialize the appropriate model based on arguments."""
    model_type = args.model_type.lower()
    models_cfg = config.get("models", {})
    
    if model_type == "vllm":
        if not args.model_path:
            raise ValueError("--model_path is required for VLLM models")
        
        print(f"\nInitializing VLLM model from: {args.model_path}")
        vllm_cfg = models_cfg.get("vllm", {})
        model = VLLMModel(
            model=args.model_path,
            config=vllm_cfg,
            verbose=verbose
        )
        return model, Path(args.model_path).name
    
    elif model_type == "openai":
        print(f"\nInitializing OpenAI model")
        openai_cfg = models_cfg.get("openai", {})
        model_name = openai_cfg.get("model_name", "gpt-4")
        model = OpenAIModel(
            model_name=model_name,
            config=openai_cfg,
            verbose=verbose
        )
        return model, model_name
    
    elif model_type == "deepseek":
        print(f"\nInitializing DeepSeek model")
        deepseek_cfg = models_cfg.get("deepseek", {})
        model_name = deepseek_cfg.get("model_name", "deepseek-chat")
        model = DeepSeekModel(
            model_name=model_name,
            config=deepseek_cfg,
            verbose=verbose
        )
        return model, model_name
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: vllm, openai, deepseek")


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Prompt Detection & Canonicalization Benchmark"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.yaml",
        help="Path to benchmark configuration file"
    )
    
    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["vllm", "openai", "deepseek"],
        help="Type of model to evaluate"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model (required for VLLM, local path or HuggingFace ID)"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="GPU memory fraction to allocate when loading VLLM models (0-1, overrides config)"
    )
    
    parser.add_argument(
        "--tensor-parallel-size",
        dest="tensor_parallel_size",
        type=int,
        default=None,
        help="Tensor parallel size for VLLM (int >=1, overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=None,
        help="Batch size for vLLM generation (overrides config)"
    )
    
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        help="Force vLLM to run in eager mode (overrides config)"
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Disable eager mode for vLLM (overrides config)"
    )
    parser.set_defaults(enforce_eager=None)
    
    # Benchmark settings
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to generate (overrides config)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (overrides config)"
    )
    
    # Context generation
    parser.add_argument(
        "--use_generated_contexts",
        action="store_true",
        help="Use GPT-generated contexts instead of templates"
    )
    
    parser.add_argument(
        "--contexts_file",
        type=str,
        default=None,
        help="Path to generated contexts JSON file"
    )
    
    # Verbose mode
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed LLM interactions"
    )
    
    # Data generation only
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Only generate and save test data without evaluation"
    )

    parser.add_argument(
        "--restricted_reasoning",
        action="store_true",
        help="Use restricted thinking prompts that encourage minimal reasoning"
    )
    
    parser.add_argument(
        "--load_data",
        type=str,
        default=None,
        help="Load test data from existing JSON file"
    )
    
    args = parser.parse_args()

    if args.contexts_file:
        args.contexts_file = resolve_path(args.contexts_file)
    if args.load_data:
        args.load_data = resolve_path(args.load_data)

    # Resolve centralized config location
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = resolve_config_path("adversarial_prompt", args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Override config with command line arguments
    if args.num_samples:
        config['benchmark']['num_samples'] = args.num_samples
    if args.seed:
        config['benchmark']['seed'] = args.seed
    if args.output_dir:
        config['benchmark']['output_dir'] = args.output_dir
    if args.use_generated_contexts:
        config['benchmark']['use_generated_contexts'] = True
    if args.contexts_file:
        config['benchmark']['generated_contexts_file'] = args.contexts_file
    if args.gpu_memory_utilization is not None:
        if not 0 < args.gpu_memory_utilization <= 1:
            raise ValueError("--gpu_memory_utilization must be between 0 and 1")
        config.setdefault('models', {}).setdefault('vllm', {})['gpu_memory_utilization'] = args.gpu_memory_utilization
    if args.tensor_parallel_size is not None:
        if args.tensor_parallel_size < 1:
            raise ValueError("--tensor-parallel-size must be an integer >= 1")
        config.setdefault('models', {}).setdefault('vllm', {})['tensor_parallel_size'] = args.tensor_parallel_size
    if args.enforce_eager is not None:
        config.setdefault('models', {}).setdefault('vllm', {})['enforce_eager'] = args.enforce_eager
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be >= 1")
        config.setdefault('models', {}).setdefault('vllm', {})['batch_size'] = args.batch_size
    # Prompt restrictions
    prompt_cfg = config.setdefault('prompt', {})
    if args.restricted_reasoning:
        prompt_cfg['restricted_reasoning'] = True
    else:
        prompt_cfg.setdefault('restricted_reasoning', False)
    
    # Create output directory
    output_dir = config['benchmark']['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate or load test data
    if args.load_data:
        print(f"\nLoading test data from: {args.load_data}")
        with open(args.load_data, 'r') as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} samples")
    else:
        print("\nGenerating test data...")
        generator = BenchmarkDataGenerator(
            config=config,
            seed=config['benchmark']['seed']
        )
        samples = generator.generate_samples(config['benchmark']['num_samples'])
        
        # Print statistics
        stats = generator.get_statistics(samples)
        print(f"\nGenerated {stats['total_samples']} samples:")
        print("  By category:", stats['by_category'])
        print("  By perturbation:", stats['by_perturbation'])
        print("  By difficulty:", stats['by_difficulty'])
        print("  By source:", stats['by_source'])
        
        # Save generated data
        data_file = Path(output_dir) / f"test_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(data_file, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"\nTest data saved to: {data_file}")
    
    # Exit if generate_only mode
    if args.generate_only:
        print("\nGeneration complete (--generate_only mode). Exiting.")
        return
    
    # Initialize model
    model, model_name = initialize_model(args, config, args.verbose)
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = Evaluator(model=model, config=config, verbose=args.verbose)
    
    # Run evaluation
    print("\n" + "="*70)
    print("STARTING EVALUATION")
    print("="*70)
    
    results, metrics = evaluator.evaluate_batch(samples)
    
    # Aggregate token usage
    token_usage = aggregate_token_usage(results)
    
    # Print summary
    print_metrics_summary(metrics, token_usage)
    
    # Save results
    output_path = save_results(
        results=results,
        metrics=metrics,
        output_dir=output_dir,
        model_name=model_name.replace('/', '_')
    )
    
    print("\nBenchmark complete!")

    # Gracefully shut down vLLM / distributed resources
    try:
        if hasattr(model, "shutdown"):
            model.shutdown()
        else:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
