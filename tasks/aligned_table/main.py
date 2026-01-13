"""
Main script for running the Aligned-Table Benchmark
Location: main.py
"""

import os
import json
import argparse
import yaml
import random
import numpy as np
import sys
from pathlib import Path

from tqdm import tqdm
from typing import Dict, List, Tuple

TASK_ROOT = Path(__file__).resolve().parent


def find_project_root() -> Path:
    for parent in TASK_ROOT.parents:
        if (parent / "cli.py").exists():
            return parent
    return TASK_ROOT.parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generator import DataGenerator
from src.llm_interface import LLMInterface
from src.prompt_builder import PromptBuilder
from src.evaluator import Evaluator
from src.utils import format_table
from configs.locator import resolve_config_path


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def resolve_path(path_str: str) -> str:
    """Resolve a file path, trying CWD then repo root (with dataset fallback)."""
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


def normalize_test_cases(test_cases: List[Dict], config: Dict, 
                         format_filter: str = None) -> List[Dict]:
    """
    Normalize test cases to ensure all required fields are present.
    This handles both benchmark-generated and GPT-generated contexts.
    
    Args:
        test_cases: List of test case dictionaries
        config: Configuration dictionary
        format_filter: Optional format filter
        
    Returns:
        Normalized list of test cases
    """
    normalized_cases = []
    table_formats = config['test'].get('table_formats', ['latex', 'markdown', 'text'])
    
    # Check if format distribution is specified
    format_dist = config['test'].get('format_distribution', None)
    
    if format_dist:
        # Use weighted random selection
        formats = list(format_dist.keys())
        weights = list(format_dist.values())
        
        # Normalize weights to sum to 1.0
        total = sum(weights)
        weights = [w / total for w in weights]
    else:
        # Use uniform distribution
        formats = table_formats
        weights = None

    for idx, test_case in enumerate(test_cases):
        # Check if this is a GPT-generated context (missing benchmark fields)
        if 'table_format' not in test_case:
            # This is from generate_contexts.py
            # Assign random or specified format
            if format_filter:
                table_format = format_filter.lower()
            else:
                if weights:
                    table_format = np.random.choice(formats, p=weights)
                else:
                    # Uniform random choice
                    table_format = random.choice(formats)
            
            # Build normalized test case
            normalized_case = {
                'id': test_case.get('id', idx),
                'entity_type': 'custom',  # GPT-generated doesn't have entity type
                'table_data': test_case.get('table_data', []),
                'context': test_case.get('context', ''),
                'table_format': table_format,
                'num_rows': test_case.get('num_rows', len(test_case.get('table_data', []))),
                'num_cols': test_case.get('num_cols', len(test_case.get('table_data', [[]])[0]) if test_case.get('table_data') else 0),
                'description': test_case.get('description', '')
            }
        else:
            # This is already a properly formatted test case
            normalized_case = test_case
            # Ensure it has an ID
            if 'id' not in normalized_case:
                normalized_case['id'] = idx
        
        normalized_cases.append(normalized_case)
    
    return normalized_cases


def print_verbose_info(test_case: Dict, prompt: str, response: str, 
                      eval_result: Dict, usage_info: Dict, verbose: bool):
    """Print detailed information if verbose mode is enabled."""
    if not verbose:
        return
    
    print(f"\n{'='*80}")
    print(f"TEST CASE #{test_case['id']}")
    print(f"{'='*80}")
    print(f"Entity Type: {test_case['entity_type']}")
    print(f"Table Format: {test_case['table_format']}")
    print(f"Dimensions: {test_case['num_rows']} rows × {test_case['num_cols']} columns")
    print(f"\n{'-'*80}")
    print(f"GROUND TRUTH TABLE:")
    print(f"{'-'*80}")
    ground_truth_table = format_table(test_case['table_data'], test_case['table_format'])
    print(ground_truth_table)
    print(f"\n{'-'*80}")
    print(f"PROMPT SENT TO LLM:")
    print(f"{'-'*80}")
    print(prompt)
    print(f"\n{'-'*80}")
    print(f"LLM RESPONSE:")
    print(f"{'-'*80}")
    print(response)
    print(f"\n{'-'*80}")
    print(f"TOKEN USAGE:")
    print(f"{'-'*80}")
    print(f"Total Tokens: {usage_info.get('total_tokens', 0):,}")
    print(f"Prompt/Input Tokens: {usage_info.get('prompt_tokens', 0):,}")
    print(f"Completion Tokens: {usage_info.get('completion_tokens', 0):,}")
    print(f"Reasoning Tokens: {usage_info.get('reasoning_tokens', 0):,}")
    print(f"Visible Output Tokens: {usage_info.get('output_tokens', 0):,}")
    
    # Calculate and display thinking ratio if applicable
    if usage_info.get('total_tokens', 0) > 0 and usage_info.get('reasoning_tokens', 0) > 0:
        thinking_ratio = usage_info['reasoning_tokens'] / usage_info['total_tokens']
        print(f"Thinking Ratio: {thinking_ratio:.2%}")
    
    print(f"\n{'-'*80}")
    print(f"EVALUATION RESULTS:")
    print(f"{'-'*80}")
    print(f"Content Score: {eval_result['content_score']:.4f}")
    print(f"Alignment Score: {eval_result['alignment_score']:.4f}")
    print(f"Total Score: {eval_result['total_score']:.4f}")
    print(f"Is Aligned: {eval_result['is_aligned']}")
    if 'content_details' in eval_result:
        details = eval_result['content_details']
        print(f"\nContent Details:")
        print(f"  - Cell Accuracy: {details.get('cell_accuracy', 0):.4f}")
        print(f"  - Correct Cells: {details.get('correct_cells', 0)}/{details.get('total_cells', 0)}")
    print(f"{'='*80}\n")


def compute_token_statistics(all_usage_info: List[Dict]) -> Dict:
    """
    Compute summary statistics for token usage.
    
    Args:
        all_usage_info: List of usage info dictionaries
        
    Returns:
        Dictionary with token usage statistics
    """
    if not all_usage_info:
        return {
            'total_tokens': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_reasoning_tokens': 0,
            'total_output_tokens': 0,
            'avg_tokens_per_case': 0.0,
            'avg_prompt_tokens_per_case': 0.0,
            'avg_completion_tokens_per_case': 0.0,
            'avg_reasoning_tokens_per_case': 0.0,
            'avg_output_tokens_per_case': 0.0,
        }
    
    total_tokens = sum(info.get('total_tokens', 0) for info in all_usage_info)
    total_prompt_tokens = sum(info.get('prompt_tokens', 0) for info in all_usage_info)
    total_completion_tokens = sum(info.get('completion_tokens', 0) for info in all_usage_info)
    total_reasoning_tokens = sum(info.get('reasoning_tokens', 0) for info in all_usage_info)
    total_output_tokens = sum(info.get('output_tokens', 0) for info in all_usage_info)
    
    num_cases = len(all_usage_info)
    
    return {
        'total_tokens': total_tokens,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'total_reasoning_tokens': total_reasoning_tokens,
        'total_output_tokens': total_output_tokens,
        'avg_tokens_per_case': total_tokens / num_cases if num_cases > 0 else 0.0,
        'avg_prompt_tokens_per_case': total_prompt_tokens / num_cases if num_cases > 0 else 0.0,
        'avg_completion_tokens_per_case': total_completion_tokens / num_cases if num_cases > 0 else 0.0,
        'avg_reasoning_tokens_per_case': total_reasoning_tokens / num_cases if num_cases > 0 else 0.0,
        'avg_output_tokens_per_case': total_output_tokens / num_cases if num_cases > 0 else 0.0,
    }


def run_benchmark(args):
    """Run the aligned-table benchmark."""
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    if args.num_samples:
        config['test']['num_samples'] = args.num_samples
    prompt_cfg = config.setdefault('prompt', {})
    if args.restricted_reasoning:
        prompt_cfg['restricted_reasoning'] = True
    else:
        prompt_cfg.setdefault('restricted_reasoning', False)
    
    print("Aligned-Table Benchmark")
    print("=" * 80)
    
    # Initialize components
    print("Initializing components...")
    data_generator = DataGenerator(config)
    prompt_builder = PromptBuilder(
        restricted_reasoning=prompt_cfg.get('restricted_reasoning', False)
    )
    evaluator = Evaluator(config)
    
    # Generate or load test cases
    if args.test_file:
        args.test_file = resolve_path(args.test_file)
        print(f"Loading test cases from {args.test_file}...")
        test_cases = data_generator.load_test_cases(args.test_file)

        # Normalize test cases (add missing fields if needed)
        print("Normalizing test cases...")
        test_cases = normalize_test_cases(test_cases, config, args.format)

    else:
        print(f"Generating {config['test']['num_samples']} test cases...")
        test_cases = data_generator.generate_test_cases()
        
        # Save generated test cases
        if args.save_test_cases:
            test_case_file = args.save_test_cases
            data_generator.save_test_cases(test_cases, test_case_file)
            print(f"Saved test cases to {test_case_file}")
    
    # Filter test cases by format if specified
    if args.format:
        original_count = len(test_cases)
        test_cases = [tc for tc in test_cases if tc['table_format'] == args.format.lower()]
        print(f"Filtered to {len(test_cases)} test cases with format: {args.format} (from {original_count} total)")
    
    if len(test_cases) == 0:
        print("ERROR: No test cases to evaluate!")
        return None, None, None, None
    
    # Initialize LLM interface
    print(f"Initializing LLM interface (type: {args.model_type}, model: {args.model_name})...")
    llm = LLMInterface.create(
        model_type=args.model_type,
        model_name=args.model_name,
        config=config,
        api_key=args.api_key
    )
    
    # Run evaluation
    print(f"\nRunning evaluation on {len(test_cases)} test cases...")
    print("=" * 80)
    
    all_results = []
    all_responses = []
    all_usage_info = []
    
    # Process test cases
    for test_case in tqdm(test_cases, desc="Evaluating", disable=args.verbose):
        # Build prompt
        prompt = prompt_builder.build_prompt(test_case)
        
        # Get LLM response with usage information
        try:
            response, usage_info = llm.generate_with_usage(prompt)
        except Exception as e:
            print(f"\nError generating response for test case {test_case['id']}: {e}")
            response = ""
            usage_info = {
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'reasoning_tokens': 0,
                'output_tokens': 0,
            }
            
        all_responses.append(response)
        all_usage_info.append(usage_info)
        
        # Evaluate response
        eval_result = evaluator.evaluate(
            response=response,
            ground_truth=test_case['table_data'],
            table_format=test_case['table_format'],
            verbose=args.verbose
        )
        
        # Add test case info and usage info to result
        eval_result['test_id'] = test_case['id']
        eval_result['table_format'] = test_case['table_format']
        eval_result['entity_type'] = test_case['entity_type']
        eval_result['usage_info'] = usage_info
        
        all_results.append(eval_result)
        
        # Print verbose info if enabled
        print_verbose_info(test_case, prompt, response, eval_result, usage_info, args.verbose)
    
    # Compute summary statistics
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    overall_stats = evaluator.compute_summary_statistics(all_results)
    
    print(f"\nOverall Statistics:")
    print(f"  Total Cases: {overall_stats['total_cases']}")
    print(f"  Valid Cases: {overall_stats['valid_cases']}")
    print(f"  Average Content Score: {overall_stats['avg_content_score']:.4f}")
    print(f"  Average Alignment Score: {overall_stats['avg_alignment_score']:.4f}")
    print(f"  Average Total Score: {overall_stats['avg_total_score']:.4f}")
    print(f"  Alignment Rate: {overall_stats['alignment_rate']:.4f}")
    print(f"  Perfect Content Rate: {overall_stats['perfect_content_rate']:.4f}")
    print(f"  Perfect Alignment Rate: {overall_stats['perfect_alignment_rate']:.4f}")
    print(f"  Perfect Total Rate: {overall_stats['perfect_total_rate']:.4f}")
    
    # Compute token usage statistics
    token_stats = compute_token_statistics(all_usage_info)
    
    print(f"\nToken Usage Statistics:")
    print(f"  Total Tokens: {token_stats['total_tokens']:,}")
    print(f"  Total Prompt Tokens: {token_stats['total_prompt_tokens']:,}")
    print(f"  Total Completion Tokens: {token_stats['total_completion_tokens']:,}")
    print(f"  Total Reasoning Tokens: {token_stats['total_reasoning_tokens']:,}")
    print(f"  Total Visible Output Tokens: {token_stats['total_output_tokens']:,}")
    print(f"  Average Tokens per Case: {token_stats['avg_tokens_per_case']:.2f}")
    print(f"  Average Prompt Tokens per Case: {token_stats['avg_prompt_tokens_per_case']:.2f}")
    print(f"  Average Completion Tokens per Case: {token_stats['avg_completion_tokens_per_case']:.2f}")
    print(f"  Average Reasoning Tokens per Case: {token_stats['avg_reasoning_tokens_per_case']:.2f}")
    print(f"  Average Visible Output Tokens per Case: {token_stats['avg_output_tokens_per_case']:.2f}")
    
    # Calculate and display overall thinking ratio if applicable
    if token_stats['total_tokens'] > 0 and token_stats['total_reasoning_tokens'] > 0:
        overall_thinking_ratio = token_stats['total_reasoning_tokens'] / token_stats['total_tokens']
        print(f"  Overall Thinking Ratio: {overall_thinking_ratio:.2%}")
    
    # Compute format-specific statistics
    format_stats = evaluator.compute_format_statistics(all_results, test_cases)
    
    print(f"\nFormat-Specific Statistics:")
    for fmt, stats in format_stats.items():
        if not stats or 'error' in stats:
            print(f"\n  {fmt.upper()}:")
            print(f"    No valid results for this format")
            continue

        print(f"\n  {fmt.upper()}:")
        print(f"    Cases: {stats.get('total_cases', 0)}")
        print(f"    Avg Content Score: {stats.get('avg_content_score', 0.0):.4f}")
        print(f"    Avg Alignment Score: {stats.get('avg_alignment_score', 0.0):.4f}")
        print(f"    Avg Total Score: {stats.get('avg_total_score', 0.0):.4f}")
        print(f"    Alignment Rate: {stats.get('alignment_rate', 0.0):.4f}")
    
    # Compute format-specific token usage statistics
    format_token_stats = {}
    for fmt in ['latex', 'markdown', 'text']:
        fmt_usage_info = [
            all_usage_info[i] for i, tc in enumerate(test_cases)
            if tc['table_format'] == fmt
        ]
        if fmt_usage_info:
            format_token_stats[fmt] = compute_token_statistics(fmt_usage_info)
    
    if format_token_stats:
        print(f"\nFormat-Specific Token Usage:")
        for fmt, stats in format_token_stats.items():
            print(f"\n  {fmt.upper()}:")
            print(f"    Total Tokens: {stats['total_tokens']:,}")
            print(f"    Total Prompt Tokens: {stats['total_prompt_tokens']:,}")
            print(f"    Total Completion Tokens: {stats['total_completion_tokens']:,}")
            print(f"    Total Reasoning Tokens: {stats['total_reasoning_tokens']:,}")
            print(f"    Total Visible Output Tokens: {stats['total_output_tokens']:,}")
            print(f"    Avg Tokens per Case: {stats['avg_tokens_per_case']:.2f}")
            print(f"    Avg Prompt Tokens per Case: {stats['avg_prompt_tokens_per_case']:.2f}")
            print(f"    Avg Completion Tokens per Case: {stats['avg_completion_tokens_per_case']:.2f}")
            print(f"    Avg Reasoning Tokens per Case: {stats['avg_reasoning_tokens_per_case']:.2f}")
            print(f"    Avg Visible Output Tokens per Case: {stats['avg_output_tokens_per_case']:.2f}")
            
            # Calculate thinking ratio for this format
            if stats['total_tokens'] > 0 and stats['total_reasoning_tokens'] > 0:
                thinking_ratio = stats['total_reasoning_tokens'] / stats['total_tokens']
                print(f"    Thinking Ratio: {thinking_ratio:.2%}")
    
    # Save results
    if args.output:
        print(f"\nSaving results to {args.output}...")
        
        output_data = {
            'config': {
                'model_type': args.model_type,
                'model_name': args.model_name,
                'num_test_cases': len(test_cases)
            },
            'overall_statistics': overall_stats,
            'token_statistics': token_stats,
            'format_statistics': format_stats,
            'format_token_statistics': format_token_stats,
            'detailed_results': all_results
        }
        
        # Optionally save responses
        if args.save_responses:
            output_data['responses'] = [
                {
                    'test_id': tc['id'],
                    'prompt': prompt_builder.build_prompt(tc),
                    'response': resp,
                    'usage_info': usage
                }
                for tc, resp, usage in zip(test_cases, all_responses, all_usage_info)
            ]
        
        save_results(output_data, args.output)
        print(f"Results saved successfully!")
    
    # Save detailed log if requested
    if args.save_log:
        print(f"\nSaving detailed log to {args.save_log}...")
        
        log_data = {
            'config': {
                'model_type': args.model_type,
                'model_name': args.model_name,
                'num_test_cases': len(test_cases),
                'timestamp': __import__('datetime').datetime.now().isoformat()
            },
            'test_cases': []
        }
        
        for tc, prompt, resp, usage, result in zip(
            test_cases, 
            [prompt_builder.build_prompt(tc) for tc in test_cases],
            all_responses, 
            all_usage_info, 
            all_results
        ):
            log_entry = {
                'test_id': tc['id'],
                'entity_type': tc['entity_type'],
                'table_format': tc['table_format'],
                'dimensions': {
                    'rows': tc['num_rows'],
                    'cols': tc['num_cols']
                },
                'ground_truth_table': tc['table_data'],
                'context': tc.get('context', ''),
                'prompt': prompt,
                'response': resp,
                'usage_info': usage,
                'evaluation': {
                    'content_score': result['content_score'],
                    'alignment_score': result['alignment_score'],
                    'total_score': result['total_score'],
                    'is_aligned': result['is_aligned'],
                    'content_details': result.get('content_details', {})
                }
            }
            log_data['test_cases'].append(log_entry)
        
        # Add summary statistics
        log_data['summary'] = {
            'overall_statistics': overall_stats,
            'token_statistics': token_stats,
            'format_statistics': format_stats,
            'format_token_statistics': format_token_stats
        }
        
        save_results(log_data, args.save_log)
        print(f"Detailed log saved successfully!")
    
    print("\n" + "=" * 80)
    print("Benchmark completed!")
    print("=" * 80)
    
    return overall_stats, format_stats, token_stats, all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Aligned-Table Benchmark for evaluating LLM table formatting capabilities'
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    # Model settings
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['vllm', 'openai', 'deepseek'],
                       help='Type of model to use')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key for cloud models (optional, can use env variable)')
    
    # Test settings
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of test samples to generate (overrides config)')
    parser.add_argument('--test_file', type=str, default=None,
                       help='Load test cases from JSON file instead of generating')
    parser.add_argument('--save_test_cases', type=str, default=None,
                       help='Save generated test cases to JSON file')
    parser.add_argument('--format', type=str, default=None,
                       choices=['latex', 'markdown', 'text'],
                       help='Filter test cases by specific format')
    
    # Output settings
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output file for results')
    parser.add_argument('--save_responses', action='store_true',
                       help='Save LLM responses in output file')
    parser.add_argument('--save_log', type=str, default=None,
                       help='Save detailed log with prompts, responses, and usage info')
    
    # Verbose mode
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information during evaluation')

    parser.add_argument(
        '--restricted-reasoning',
        action='store_true',
        help='Use restricted thinking prompts that ask the model to answer directly without heavy reasoning'
    )
    
    args = parser.parse_args()

    # Normalize config path to centralized configs directory
    if not Path(args.config).is_absolute():
        args.config = resolve_config_path("aligned_table", args.config)
    
    # Run benchmark
    run_benchmark(args)


if __name__ == '__main__':
    main()
