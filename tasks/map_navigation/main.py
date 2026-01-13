import argparse
import json
import os
import sys
import importlib
import importlib.machinery
import importlib.util
import json as json_lib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TASK_ROOT = Path(__file__).resolve().parent

from configs.locator import resolve_config_path
from prompts import get_sokoban_prompt, get_frozenlake_prompt
from prompts.sokoban_prompt import get_sokoban_system_prompt, get_sokoban_user_prompt
from prompts.frozenlake_prompt import get_frozenlake_system_prompt, get_frozenlake_user_prompt
from evaluators import ExactMatchEvaluator
from utils import parse_answer, setup_logger, log_interaction, save_interaction


def _load_shared_models():
    """Load the repo-level models module while avoiding the local map_navigation.models package."""
    original_sys_path = list(sys.path)
    try:
        task_dir = str(Path(__file__).resolve().parent)
        sys.path = [str(PROJECT_ROOT)] + [p for p in sys.path if p != task_dir]
        return importlib.import_module("models")
    finally:
        sys.path = original_sys_path


_shared_models = _load_shared_models()
VLLMModel = _shared_models.VLLMModel
APIModel = _shared_models.APIModel

# Populated at runtime from a config file
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {}
VLLM_SUPPORTED_MODELS: List[str] = []


def load_model_configs(config_path: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Load MODEL_CONFIGS and VLLM_SUPPORTED_MODELS from a config file (py/json/yaml)."""
    path = Path(config_path)
    if path.suffix == ".py":
        loader = importlib.machinery.SourceFileLoader("map_nav_model_config", str(path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load config module from {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_configs = getattr(module, "MODEL_CONFIGS", {})
        vllm_supported = getattr(module, "VLLM_SUPPORTED_MODELS", [])
    else:
        if path.suffix in {".json"}:
            data = json_lib.loads(path.read_text())
        elif path.suffix in {".yml", ".yaml"}:
            import yaml  # type: ignore
            data = yaml.safe_load(path.read_text())
        else:
            raise ValueError(f"Unsupported config file type: {path.suffix}")
        model_configs = data.get("MODEL_CONFIGS", {})
        vllm_supported = data.get("VLLM_SUPPORTED_MODELS", [])

    if not isinstance(model_configs, dict):
        raise ValueError("MODEL_CONFIGS must be a mapping in the config file")
    if not isinstance(vllm_supported, list):
        vllm_supported = list(vllm_supported)
    return model_configs, vllm_supported


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

    # Fallback: place under project root
    return str((PROJECT_ROOT / relative).resolve())


def load_dataset(data_path: str) -> Dict[str, Any]:
    """Load dataset from JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)


def create_model(model_name: str, model_type: str = None, verbose: bool = False, **kwargs):
    """
    Create model instance based on model name and type.
    
    Args:
        model_name: Name of the model
        model_type: Type of model interface ('vllm' or 'api')
        verbose: Whether to enable verbose output
        **kwargs: Additional model parameters
        
    Returns:
        Model instance
    """
    # Auto-detect model type if not specified
    if model_type is None:
        if model_name in MODEL_CONFIGS:
            model_type = 'api'
        elif os.path.exists(model_name):
            # Local path -> assume vLLM
            model_type = 'vllm'
        elif model_name in VLLM_SUPPORTED_MODELS:
            model_type = 'vllm'
        elif any(model_name.startswith(prefix) for prefix in ['meta-llama/', 'Qwen/', 'mistralai/']):
            model_type = 'vllm'
        else:
            raise ValueError(f"Cannot auto-detect model type for {model_name}. Please specify --model-type")
    
    if model_type == 'vllm':
        return VLLMModel(model_name, verbose=verbose, **kwargs)
    
    elif model_type == 'api':
        # Get configuration
        config = MODEL_CONFIGS.get(model_name, {})
        
        # Determine API key environment variable
        api_key_env = config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env) or kwargs.get('api_key')
        
        # Get API base URL
        api_base = config.get('api_base')
        
        # Merge config with kwargs (kwargs take precedence)
        model_params = {
            'api_key': api_key,
            'api_base': api_base,
            'verbose': verbose,
            'temperature': config.get('temperature', 0.0),
            'max_tokens': config.get('max_tokens', 2048),
            'reasoning_effort': config.get('reasoning_effort', 'medium'),
            **kwargs  # Override with command-line arguments
        }
        
        # Use model_name from config if available
        actual_model_name = config.get('model_name', model_name)
        
        return APIModel(
            model_name=actual_model_name,
            **model_params
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_prompt(env_type: str, map_str: str, question: str, use_system_prompt: bool = True,
               restricted_reasoning: bool = False):
    """
    Get appropriate prompt for the environment type.
    
    Args:
        env_type: Type of environment ('sokoban' or 'frozenlake')
        map_str: String representation of the map
        question: Question to ask
        use_system_prompt: Whether to return separate system and user prompts
        
    Returns:
        Either full prompt string or tuple of (system_prompt, user_prompt)
    """
    if env_type == 'sokoban':
        if use_system_prompt:
            return get_sokoban_system_prompt(), get_sokoban_user_prompt(map_str, question)
        else:
            return get_sokoban_prompt(map_str, question, restricted_reasoning=restricted_reasoning)
    elif env_type == 'frozenlake':
        if use_system_prompt:
            return (
                get_frozenlake_system_prompt(restricted_reasoning=restricted_reasoning),
                get_frozenlake_user_prompt(map_str, question, restricted_reasoning=restricted_reasoning)
            )
        else:
            return get_frozenlake_prompt(map_str, question, restricted_reasoning=restricted_reasoning)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def run_evaluation(
    model,
    dataset: Dict[str, Any],
    evaluator: ExactMatchEvaluator,
    logger,
    verbose: bool = False,
    save_response_path: str = None,
    restricted_reasoning: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation on the dataset.
    
    Args:
        model: Model instance
        dataset: Dataset dictionary
        evaluator: Evaluator instance
        logger: Logger instance
        verbose: Whether to log verbose output
        save_response_path: Path to save detailed responses
        
    Returns:
        Evaluation results
    """
    data = dataset['data']
    interactions = [] if save_response_path else None
    token_stats = {
        'total_tokens': 0,
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'reasoning_tokens': 0,
        'output_tokens': 0,
        'samples_with_usage': 0,
    }
    
    # Determine if model supports system prompts
    use_system_prompt = hasattr(model, 'generate_with_system')
    supports_batch = hasattr(model, 'generate_batch')
    supports_batch_system = hasattr(model, 'generate_batch_with_system')
    batch_size = max(1, int(getattr(model, 'batch_size', 1) or 1))
    
    logger.info(f"Starting evaluation on {len(data)} tasks...")
    
    task_id = 0
    for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
        batch = data[start:start + batch_size]
        prompts_for_logging: List[str] = []
        system_prompts: List[str] = []
        user_prompts: List[str] = []
        for task in batch:
            if use_system_prompt:
                sys_prompt, usr_prompt = get_prompt(
                    task['env_type'], task['map'], task['question'],
                    use_system_prompt=True, restricted_reasoning=restricted_reasoning
                )
                system_prompts.append(sys_prompt)
                user_prompts.append(usr_prompt)
                prompts_for_logging.append(sys_prompt + "\n\n" + usr_prompt)
            else:
                prompt = get_prompt(
                    task['env_type'], task['map'], task['question'],
                    use_system_prompt=False, restricted_reasoning=restricted_reasoning
                )
                system_prompts.append("")
                user_prompts.append(prompt)
                prompts_for_logging.append(prompt)

        responses_with_usage: List[Tuple[str, Dict[str, int]]] = []
        try:
            if use_system_prompt and supports_batch_system and batch_size > 1:
                responses_with_usage = model.generate_batch_with_system(system_prompts, user_prompts)
            elif supports_batch and batch_size > 1:
                responses_with_usage = model.generate_batch(prompts_for_logging)
            else:
                for sys_prompt, usr_prompt, full_prompt in zip(system_prompts, user_prompts, prompts_for_logging):
                    if use_system_prompt and hasattr(model, 'generate_with_system'):
                        responses_with_usage.append(model.generate_with_system(sys_prompt, usr_prompt))
                    else:
                        responses_with_usage.append(model.generate(full_prompt))
        except Exception as e:
            logger.error(f"Error during batch generation: {str(e)}")
            responses_with_usage = [("", {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}) for _ in batch]

        for task, full_prompt, (response, token_usage) in zip(batch, prompts_for_logging, responses_with_usage):
            ground_truth = task['answer']
            parsed_answer = parse_answer(response)
            
            is_correct = evaluator.evaluate_single(parsed_answer, ground_truth, task)
            evaluator.add_result(task_id, task, parsed_answer, ground_truth, is_correct)
            
            if verbose:
                log_interaction(logger, task_id, full_prompt, response, token_usage, verbose=True)
                logger.info(f"Parsed Answer: {parsed_answer}")
                logger.info(f"Ground Truth: {ground_truth}")
                logger.info(f"Correct: {is_correct}\n")
            
            if save_response_path:
                save_interaction(
                    interactions, task_id, task, full_prompt, response,
                    parsed_answer, is_correct, token_usage
                )

            if token_usage:
                token_stats['samples_with_usage'] += 1
                token_stats['total_tokens'] += token_usage.get('total_tokens', 0) or 0
                token_stats['prompt_tokens'] += token_usage.get('prompt_tokens', 0) or 0
                token_stats['completion_tokens'] += token_usage.get('completion_tokens', 0) or 0
                token_stats['reasoning_tokens'] += token_usage.get('reasoning_tokens', 0) or 0
                token_stats['output_tokens'] += token_usage.get('output_tokens', 0) or 0

            task_id += 1
    
    # Compute metrics
    metrics = evaluator.compute_metrics()
    total_samples = len(data) if data else 0
    def _avg(value: int) -> float:
        return value / total_samples if total_samples > 0 else 0.0
    reasoning_ratio = (
        token_stats['reasoning_tokens'] / token_stats['total_tokens']
        if token_stats['total_tokens'] > 0 else 0.0
    )
    metrics['token_usage'] = {
        'total_tokens': token_stats['total_tokens'],
        'avg_tokens': _avg(token_stats['total_tokens']),
        'prompt_tokens': token_stats['prompt_tokens'],
        'avg_prompt_tokens': _avg(token_stats['prompt_tokens']),
        'completion_tokens': token_stats['completion_tokens'],
        'avg_completion_tokens': _avg(token_stats['completion_tokens']),
        'reasoning_tokens': token_stats['reasoning_tokens'],
        'avg_reasoning_tokens': _avg(token_stats['reasoning_tokens']),
        'output_tokens': token_stats['output_tokens'],
        'avg_output_tokens': _avg(token_stats['output_tokens']),
        'reasoning_token_ratio': reasoning_ratio,
        'samples_with_usage': token_stats['samples_with_usage'],
        'total_samples': total_samples,
    }
    
    # Save detailed responses if requested
    if save_response_path:
        os.makedirs(os.path.dirname(save_response_path), exist_ok=True)
        with open(save_response_path, 'w') as f:
            json.dump({
                'interactions': interactions,
                'metrics': metrics
            }, f, indent=2)
        logger.info(f"Detailed responses saved to {save_response_path}")
    
    return metrics


def print_results(metrics: Dict[str, Any], logger):
    """Print evaluation results in a formatted way."""
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['correct_tasks']}/{metrics['total_tasks']})")
    
    logger.info(f"\nAccuracy by Environment:")
    for env, acc in metrics['accuracy_by_env'].items():
        logger.info(f"  {env}: {acc:.4f}")
    
    logger.info(f"\nAccuracy by Task Type:")
    for task_type, acc in sorted(metrics['accuracy_by_task_type'].items()):
        logger.info(f"  {task_type}: {acc:.4f}")
    
    if metrics['accuracy_by_env_and_task']:
        logger.info(f"\nAccuracy by Environment and Task Type:")
        for key, acc in sorted(metrics['accuracy_by_env_and_task'].items()):
            logger.info(f"  {key}: {acc:.4f}")

    # Token usage summary
    token_usage = metrics.get('token_usage')
    if token_usage:
        logger.info(f"\nToken Usage Summary:")
        logger.info(f"  Total tokens: {token_usage['total_tokens']:,}")
        logger.info(f"  Avg tokens: {token_usage['avg_tokens']:.2f}")
        logger.info(f"  Total prompt tokens: {token_usage['prompt_tokens']:,}")
        logger.info(f"  Avg prompt tokens: {token_usage['avg_prompt_tokens']:.2f}")
        logger.info(f"  Total completion tokens: {token_usage['completion_tokens']:,}")
        logger.info(f"  Avg completion tokens: {token_usage['avg_completion_tokens']:.2f}")
        logger.info(f"  Total reasoning tokens: {token_usage['reasoning_tokens']:,}")
        logger.info(f"  Avg reasoning tokens: {token_usage['avg_reasoning_tokens']:.2f}")
        logger.info(f"  Total output tokens: {token_usage['output_tokens']:,}")
        logger.info(f"  Avg output tokens: {token_usage['avg_output_tokens']:.2f}")
        logger.info(f"  Reasoning token ratio: {token_usage['reasoning_token_ratio']:.4f}")
    
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='2D Map Navigation Benchmark')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--model-type', type=str, choices=['vllm', 'api'],
                       help='Type of model interface (auto-detected if not specified)')
    parser.add_argument('--api-key', type=str,
                       help='API key for API models')
    parser.add_argument('--reasoning-effort', type=str, 
                       choices=['low', 'medium', 'high'], default='medium',
                       help='Reasoning effort for o-series models (default: medium)')
    parser.add_argument('--config', type=str, default='model_config.py',
                        help='Path to model configuration file (default: configs/map_navigation/model_config.py)')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data JSON file')
    
    # Output arguments
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save evaluation results')
    parser.add_argument('--save-response', type=str,
                       help='Path to save detailed responses')
    
    # Logging arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--restricted-reasoning', action='store_true',
                       help='Use restricted thinking prompts that encourage minimal reasoning')
    
    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Top-p nucleus sampling for generation')
    parser.add_argument('--max-tokens', type=int, default=None,
                       help='Maximum tokens to generate (override config; default: use config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for vLLM generation')
    
    # vLLM specific arguments
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                       help='Tensor parallel size for vLLM')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization for vLLM')
    
    args = parser.parse_args()

    # Resolve config path to centralized /configs directory
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = resolve_config_path("map_navigation", config_path)
    global MODEL_CONFIGS, VLLM_SUPPORTED_MODELS
    MODEL_CONFIGS, VLLM_SUPPORTED_MODELS = load_model_configs(config_path)

    # Resolve file paths relative to repo root when necessary
    args.data = resolve_path(args.data)
    args.output = resolve_path(args.output)
    if args.save_response:
        args.save_response = resolve_path(args.save_response)

    # Normalize output path: avoid filesystem root, ensure parent exists
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    if out_path.parent == Path("/"):
        out_path = (PROJECT_ROOT / "results" / out_path.name).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(out_path)

    if args.save_response:
        resp_path = Path(args.save_response)
        if not resp_path.is_absolute():
            resp_path = (Path.cwd() / resp_path).resolve()
        if resp_path.parent == Path("/"):
            resp_path = (PROJECT_ROOT / "results" / resp_path.name).resolve()
        resp_path.parent.mkdir(parents=True, exist_ok=True)
        args.save_response = str(resp_path)
    
    # Setup logger
    logger = setup_logger(verbose=args.verbose)
    
    logger.info(f"Loading dataset from {args.data}...")
    dataset = load_dataset(args.data)
    logger.info(f"Loaded {len(dataset['data'])} tasks")
    logger.info(f"Dataset metadata: {dataset.get('metadata', {})}")
    
    # Create model
    logger.info(f"Initializing model: {args.model} (type: {args.model_type or 'auto'})")
    model_kwargs = {'verbose': args.verbose}
    if args.temperature is not None:
        model_kwargs['temperature'] = args.temperature
    if args.max_tokens is not None:
        model_kwargs['max_tokens'] = args.max_tokens
    if args.top_p is not None:
        model_kwargs['top_p'] = args.top_p
    if args.batch_size is not None:
        model_kwargs['batch_size'] = args.batch_size
    
    # Add reasoning effort for o-series models
    if 'o1' in args.model.lower() or 'o3' in args.model.lower() or 'o4' in args.model.lower():
        model_kwargs['reasoning_effort'] = args.reasoning_effort
    
    if args.model_type == 'vllm' or (args.model_type is None and args.model in VLLM_SUPPORTED_MODELS):
        model_kwargs.update({
            'tensor_parallel_size': args.tensor_parallel_size,
            'gpu_memory_utilization': args.gpu_memory_utilization,
        })
    
    if args.api_key:
        model_kwargs['api_key'] = args.api_key
    
    model = create_model(args.model, args.model_type, **model_kwargs)
    logger.info("Model initialized successfully")
    
    # Create evaluator
    evaluator = ExactMatchEvaluator()
    
    # Run evaluation
    metrics = run_evaluation(
        model=model,
        dataset=dataset,
        evaluator=evaluator,
        logger=logger,
        verbose=args.verbose,
        save_response_path=args.save_response,
        restricted_reasoning=args.restricted_reasoning
    )
    
    # Print results
    print_results(metrics, logger)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        'model': args.model,
        'dataset': args.data,
        'metrics': metrics,
        'config': {
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
