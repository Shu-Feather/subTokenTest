"""
Main script for RSA-Difference Benchmark
Path: main.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

TASK_ROOT = Path(__file__).resolve().parent

def find_project_root() -> Path:
    for parent in TASK_ROOT.parents:
        if (parent / "cli.py").exists():
            return parent
    return TASK_ROOT.parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import RSAPatternGenerator
from src.model_interface import create_model_interface
from src.evaluator import Evaluator
from src.utils import load_config, create_prompt, save_response, format_results, save_results
from configs.locator import resolve_config_path


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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='RSA-Difference Benchmark'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['vllm', 'openai', 'deepseek'],
        help='Type of model to use'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to local model (for vllm)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name of the model (for API-based models)'
    )
    
    # Data generation arguments
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to generate/evaluate'
    )
    parser.add_argument(
        '--num-differences',
        type=int,
        default=5,
        help='Number of differences per sample'
    )
    parser.add_argument(
        '--pattern-width',
        type=int,
        default=19,
        help='Width of RSA pattern (excluding borders)'
    )
    parser.add_argument(
        '--pattern-height',
        type=int,
        default=9,
        help='Height of RSA pattern (excluding borders)'
    )
    parser.add_argument(
        '--key-size',
        type=int,
        default=2048,
        help='RSA key size to display'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--coordinate-weight',
        type=float,
        default=0.5,
        help='Weight for coordinate accuracy in overall score'
    )
    parser.add_argument(
        '--replacement-weight',
        type=float,
        default=0.5,
        help='Weight for replacement accuracy in overall score'
    )
    
    # Output arguments
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed conversation logs'
    )
    parser.add_argument(
        '--save-response',
        type=str,
        help='Path to save model responses'
    )
    parser.add_argument(
        '--output-results',
        type=str,
        default='outputs/results.json',
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--restricted-reasoning',
        action='store_true',
        help='Use restricted thinking prompts that ask the model to answer directly without heavy reasoning'
    )
    parser.add_argument(
        '--reasoning-effort',
        type=str,
        choices=['low', 'medium', 'high'],
        help='Reasoning effort for o-series OpenAI models (Responses API)'
    )
    
    # Data management arguments
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate test data without evaluation'
    )
    parser.add_argument(
        '--output-data',
        type=str,
        help='Path to save generated data'
    )
    parser.add_argument(
        '--input-data',
        type=str,
        help='Path to load pre-generated data'
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    return parser.parse_args()


def generate_data(args, generator: RSAPatternGenerator) -> List[Dict]:
    """
    Generate test data
    
    Args:
        args: Command line arguments
        generator: RSA pattern generator
        
    Returns:
        List of generated samples
    """
    print(f"Generating {args.num_samples} samples...")
    samples = generator.generate_batch(
        num_samples=args.num_samples,
        num_differences=args.num_differences,
        key_size=args.key_size
    )
    
    if args.output_data:
        print(f"Saving generated data to {args.output_data}")
        generator.save_samples(samples, args.output_data)
    
    return samples


def run_evaluation(args, samples: List[Dict], config: Dict, restricted_reasoning: bool):
    """
    Run evaluation on samples
    
    Args:
        args: Command line arguments
        samples: List of test samples
        config: Configuration dictionary
    """
    # Create model interface
    print(f"Initializing {args.model_type} model...")
    model_kwargs = {}
    if args.reasoning_effort:
        model_kwargs["reasoning_effort"] = args.reasoning_effort
    model = create_model_interface(
        model_type=args.model_type,
        model_path=args.model_path,
        model_name=args.model_name,
        config=config,
        **model_kwargs
    )
    
    # Create evaluator
    evaluator = Evaluator(
        coordinate_weight=args.coordinate_weight,
        replacement_weight=args.replacement_weight
    )
    
    # Prepare for response saving
    responses = []
    predictions = []
    ground_truths = []
    
    # Token usage statistics
    total_usage = {
        'total_tokens': 0,
        'input_tokens': 0,
        'output_tokens': 0,
        'reasoning_tokens': 0,
    }
    
    # Process samples (batched if supported)
    print(f"Evaluating {len(samples)} samples...")
    batch_size = max(1, int(getattr(model, 'batch_size', 1) or 1))
    sample_id = 0
    for start in tqdm(range(0, len(samples), batch_size)):
        batch = samples[start:start + batch_size]
        prompts = [
            create_prompt(sample['pattern1'], sample['pattern2'], restricted_reasoning=restricted_reasoning)
            for sample in batch
        ]
        
        if args.verbose:
            print("\n" + "="*60)
            print(f"Batch {start + 1}-{start + len(batch)}")
            print("="*60)
        
        try:
            if hasattr(model, 'generate_batch') and batch_size > 1:
                batch_outputs = model.generate_batch(prompts)
            else:
                batch_outputs = [model.generate(p) for p in prompts]
        except Exception as e:
            print(f"Error generating batch starting at {start + 1}: {e}")
            batch_outputs = [("", {'total_tokens': None, 'input_tokens': None, 'output_tokens': None, 'reasoning_tokens': None}) for _ in batch]
        
        for sample, prompt, (response, usage_info) in zip(batch, prompts, batch_outputs):
            try:
                if usage_info and usage_info.get('total_tokens') is not None:
                    total_usage['total_tokens'] += usage_info.get('total_tokens', 0)
                    total_usage['input_tokens'] += usage_info.get('input_tokens', 0)
                    total_usage['output_tokens'] += usage_info.get('output_tokens', 0)
                    total_usage['reasoning_tokens'] += usage_info.get('reasoning_tokens', 0)
                
                if args.verbose:
                    print("\nPROMPT:")
                    print(prompt)
                    print("\nMODEL RESPONSE:")
                    print(response)
                    if usage_info:
                        print("\nTOKEN USAGE:")
                        print(f"  Total Tokens: {usage_info.get('total_tokens')}")
                        print(f"  Input Tokens: {usage_info.get('input_tokens')}")
                        print(f"  Output Tokens: {usage_info.get('output_tokens')}")
                        print(f"  Reasoning Tokens: {usage_info.get('reasoning_tokens')}")
                    print("="*60)
            except Exception:
                pass
            
            prediction = evaluator.parse_prediction(response)
            
            if args.verbose:
                print("\nPARSED PREDICTION:")
                for diff in prediction:
                    print(f"  ({diff['x']}, {diff['y']}): {diff['original']} -> {diff['modified']}")
                print("\nGROUND TRUTH:")
                for diff in sample['ground_truth']:
                    print(f"  ({diff['x']}, {diff['y']}): {diff['original']} -> {diff['modified']}")
                
                sample_result = evaluator.evaluate_sample(prediction, sample['ground_truth'])
                print(f"\nSample Score: {sample_result['overall_score']:.4f}")
                print(f"  Coordinate F1: {sample_result['coordinate_f1']:.4f}")
                print(f"  Replacement Accuracy: {sample_result['replacement_accuracy']:.4f}")
            
            predictions.append(prediction)
            ground_truths.append(sample['ground_truth'])
            
            if args.save_response:
                responses.append({
                    'sample_id': sample_id,
                    'prompt': prompt,
                    'response': response,
                    'prediction': prediction,
                    'ground_truth': sample['ground_truth'],
                    'token_usage': {
                        'total_tokens': usage_info.get('total_tokens') if usage_info else None,
                        'input_tokens': usage_info.get('input_tokens') if usage_info else None,
                        'output_tokens': usage_info.get('output_tokens') if usage_info else None,
                        'reasoning_tokens': usage_info.get('reasoning_tokens') if usage_info else None,
                    }
                })
            sample_id += 1
    
    # Save responses
    if args.save_response:
        print(f"\nSaving responses to {args.save_response}")
        save_response(responses, args.save_response)
    
    # Evaluate all predictions
    print("\nEvaluating predictions...")
    results = evaluator.evaluate_batch(predictions, ground_truths)
    
    # Print results
    print("\n" + format_results(results))
    
    # Print token usage statistics
    print("\n" + "="*60)
    print("TOKEN USAGE STATISTICS")
    print("="*60)
    print(f"Total Tokens:      {total_usage['total_tokens']:,}")
    print(f"Input Tokens:      {total_usage['input_tokens']:,}")
    print(f"Output Tokens:     {total_usage['output_tokens']:,}")
    print(f"Reasoning Tokens:  {total_usage['reasoning_tokens']:,}")
    print(f"Average per Sample:")
    print(f"  Total:     {total_usage['total_tokens'] / len(samples):.1f}")
    print(f"  Input:     {total_usage['input_tokens'] / len(samples):.1f}")
    print(f"  Output:    {total_usage['output_tokens'] / len(samples):.1f}")
    print(f"  Reasoning: {total_usage['reasoning_tokens'] / len(samples):.1f}")
    print("="*60)
    
    # Save results
    print(f"\nSaving results to {args.output_results}")
    
    # Add metadata to results
    results['metadata'] = {
        'model_type': args.model_type,
        'model_path': args.model_path,
        'model_name': args.model_name,
        'num_samples': args.num_samples,
        'num_differences': args.num_differences,
        'pattern_width': args.pattern_width,
        'pattern_height': args.pattern_height,
        'key_size': args.key_size,
        'coordinate_weight': args.coordinate_weight,
        'replacement_weight': args.replacement_weight
    }
    
    # Add token usage statistics to results
    results['token_usage'] = {
        'total': total_usage,
        'average_per_sample': {
            'total_tokens': total_usage['total_tokens'] / len(samples) if len(samples) > 0 else 0,
            'input_tokens': total_usage['input_tokens'] / len(samples) if len(samples) > 0 else 0,
            'output_tokens': total_usage['output_tokens'] / len(samples) if len(samples) > 0 else 0,
            'reasoning_tokens': total_usage['reasoning_tokens'] / len(samples) if len(samples) > 0 else 0,
        }
    }
    
    save_results(results, args.output_results)
    
    # Cleanup
    model.cleanup()

def main():
    """Main function"""
    args = parse_args()

    if args.input_data:
        args.input_data = resolve_path(args.input_data)
    if args.output_data:
        args.output_data = resolve_path(args.output_data)
    args.output_results = resolve_path(args.output_results)

    if not Path(args.config).is_absolute():
        args.config = resolve_config_path("rsa_randomart", args.config)
    
    # Load configuration
    config = load_config(args.config)
    restricted_reasoning = args.restricted_reasoning or config.get("prompt", {}).get("restricted_reasoning", False)
    
    # Update config with data generation defaults if present
    if 'data_generation' in config:
        data_config = config['data_generation']
        if args.pattern_width == 19 and 'default_width' in data_config:
            args.pattern_width = data_config['default_width']
        if args.pattern_height == 9 and 'default_height' in data_config:
            args.pattern_height = data_config['default_height']
        if args.num_differences == 5 and 'default_num_differences' in data_config:
            args.num_differences = data_config['default_num_differences']
    
    # Create data generator
    available_elements = None
    if 'data_generation' in config and 'available_elements' in config['data_generation']:
        available_elements = config['data_generation']['available_elements']
    
    generator = RSAPatternGenerator(
        width=args.pattern_width,
        height=args.pattern_height,
        available_elements=available_elements
    )
    
    # Load or generate data
    if args.input_data:
        print(f"Loading data from {args.input_data}")
        all_samples = generator.load_samples(args.input_data)

        # Limit samples based on --num-samples
        if args.num_samples is not None and args.num_samples < len(all_samples):
            print(f"Using {args.num_samples} samples out of {len(all_samples)} available")
            samples = all_samples[:args.num_samples]
        else:
            print(f"Using all {len(all_samples)} samples from the file")
            samples = all_samples
            
    else:
        samples = generate_data(args, generator)
    
    # If generate-only mode, exit here
    if args.generate_only:
        print("Data generation complete. Exiting (--generate-only mode).")
        return
    
    # Validate model arguments
    if args.model_type is None:
        raise ValueError("--model-type is required for evaluation")
    
    if args.model_type == 'vllm' and args.model_path is None:
        raise ValueError("--model-path is required for vllm model type")
    
    # Run evaluation
    run_evaluation(args, samples, config, restricted_reasoning)
    
    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()
