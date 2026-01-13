"""
Analyze token usage from evaluation results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from pathlib import Path
import pandas as pd

def analyze_usage_from_file(filepath: str):
    """Analyze token usage from a single results file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print("="*70)
    print(f"Usage Analysis: {filepath}")
    print("="*70)
    
    if 'model_info' in results:
        # Single model result
        model_name = results['model_info']['name']
        print(f"\nModel: {model_name}")
        print(f"Prompt Type: {results.get('prompt_type', 'N/A')}")
        
        # Check if usage tracking is supported
        if not results.get('supports_usage_tracking', False):
            print("\n⚠️  This model does not support usage tracking")
            return
        
        # Overall usage
        total_usage = results.get('total_usage', {})
        if total_usage.get('samples_with_usage', 0) > 0:
            print("\n--- Overall Usage ---")
            print(f"Total Tokens: {total_usage.get('total_tokens', 0):,}")
            print(f"Input Tokens: {total_usage.get('input_tokens', 0):,}")
            print(f"Output Tokens: {total_usage.get('output_tokens', 0):,}")
            if total_usage.get('reasoning_tokens', 0) > 0:
                print(f"Reasoning Tokens: {total_usage.get('reasoning_tokens', 0):,}")
            print(f"\nAverage per Sample: {total_usage.get('avg_tokens_per_sample', 0):.1f} tokens")
        
        # Per-task breakdown
        print("\n--- Task Breakdown ---")
        for task_name in ['task1', 'task2']:
            task_results = results.get(f'{task_name}_results', [])
            task_usage = _calculate_task_usage(task_results)
            
            task_display = "Task 1 (Typewriter)" if task_name == 'task1' else "Task 2 (Backspace)"
            print(f"\n{task_display}:")
            print(f"  Samples: {task_usage['count']}")
            if task_usage['count'] > 0:
                print(f"  Total Tokens: {task_usage['total']:,}")
                print(f"  Avg Tokens: {task_usage['avg']:.1f}")
        
        # Detailed per-sample analysis
        print("\n--- Per-Sample Usage (Top 5 highest) ---")
        _print_top_usage_samples(results)
    
    print("\n" + "="*70)

def _calculate_task_usage(task_results: list) -> dict:
    """Calculate usage for a specific task"""
    usage = {
        'count': 0,
        'total': 0,
        'avg': 0.0
    }
    
    for result in task_results:
        usage_info = result.get('usage_info')
        if usage_info:
            usage['count'] += 1
            usage['total'] += usage_info.get('total_tokens', 0)
    
    if usage['count'] > 0:
        usage['avg'] = usage['total'] / usage['count']
    
    return usage

def _print_top_usage_samples(results: dict):
    """Print samples with highest token usage"""
    all_samples = []
    
    # Collect from both tasks
    for task_name in ['task1_results', 'task2_results']:
        task_results = results.get(task_name, [])
        for result in task_results:
            usage_info = result.get('usage_info')
            if usage_info:
                all_samples.append({
                    'input': result.get('input', ''),
                    'total_tokens': usage_info.get('total_tokens', 0),
                    'input_tokens': usage_info.get('input_tokens', 0),
                    'output_tokens': usage_info.get('output_tokens', 0),
                    'task': 'Task 1' if task_name == 'task1_results' else 'Task 2'
                })
    
    # Sort by total tokens
    all_samples.sort(key=lambda x: x['total_tokens'], reverse=True)
    
    # Print top 5
    for i, sample in enumerate(all_samples[:5], 1):
        print(f"\n{i}. [{sample['task']}] Input: {sample['input'][:50]}...")
        print(f"   Total: {sample['total_tokens']}, "
              f"Input: {sample['input_tokens']}, "
              f"Output: {sample['output_tokens']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze token usage from results")
    parser.add_argument('results_file', help='Path to results JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: File not found: {args.results_file}")
        sys.exit(1)
    
    analyze_usage_from_file(args.results_file)

if __name__ == "__main__":
    main()