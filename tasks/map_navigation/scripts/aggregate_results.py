"""
Aggregate results from multiple saved response files.
Computes accuracy by task type, overall accuracy, and average token usage.
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import pandas as pd


def load_response_file(file_path: str) -> Dict[str, Any]:
    """
    Load a saved response JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing interactions and metrics
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_model_name(file_path: str) -> str:
    """
    Extract model name from file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Model name extracted from filename
    """
    # Get filename without extension
    filename = Path(file_path).stem
    
    # Remove common suffixes like '_detailed', '_sokoban', '_frozenlake'
    suffixes_to_remove = ['_detailed', '_sokoban', '_frozenlake', '_8x8', '_12x12']
    for suffix in suffixes_to_remove:
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
    
    return filename


def compute_task_type_accuracy(interactions: List[Dict]) -> Dict[str, float]:
    """
    Compute accuracy for each task type.
    
    Args:
        interactions: List of interaction dictionaries
        
    Returns:
        Dictionary mapping task_type to accuracy
    """
    task_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for interaction in interactions:
        task_type = interaction.get('task_type')
        is_correct = interaction.get('is_correct', False)
        
        if task_type is not None:
            task_type_stats[task_type]['total'] += 1
            if is_correct:
                task_type_stats[task_type]['correct'] += 1
    
    # Compute accuracy
    task_type_accuracy = {}
    for task_type, stats in task_type_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            task_type_accuracy[f'type_{task_type}'] = accuracy
        else:
            task_type_accuracy[f'type_{task_type}'] = 0.0
    
    return task_type_accuracy


def compute_overall_accuracy(interactions: List[Dict]) -> float:
    """
    Compute overall accuracy across all tasks.
    
    Args:
        interactions: List of interaction dictionaries
        
    Returns:
        Overall accuracy as a float
    """
    if not interactions:
        return 0.0
    
    correct = sum(1 for i in interactions if i.get('is_correct', False))
    total = len(interactions)
    
    return correct / total if total > 0 else 0.0


def compute_average_tokens(interactions: List[Dict]) -> Dict[str, float]:
    """
    Compute average token usage across all interactions.
    
    Args:
        interactions: List of interaction dictionaries
        
    Returns:
        Dictionary with average token statistics
    """
    if not interactions:
        return {
            'avg_total_tokens': 0.0,
            'avg_prompt_tokens': 0.0,
            'avg_completion_tokens': 0.0,
            'avg_reasoning_tokens': 0.0,
            'avg_output_tokens': 0.0,
        }
    
    total_tokens_sum = 0
    prompt_tokens_sum = 0
    completion_tokens_sum = 0
    reasoning_tokens_sum = 0
    output_tokens_sum = 0
    
    for interaction in interactions:
        token_usage = interaction.get('token_usage', {})
        total_tokens_sum += token_usage.get('total_tokens', 0)
        prompt_tokens_sum += token_usage.get('prompt_tokens', 0)
        completion_tokens_sum += token_usage.get('completion_tokens', 0)
        reasoning_tokens_sum += token_usage.get('reasoning_tokens', 0)
        output_tokens_sum += token_usage.get('output_tokens', 0)
    
    count = len(interactions)
    
    return {
        'avg_total_tokens': total_tokens_sum / count,
        'avg_prompt_tokens': prompt_tokens_sum / count,
        'avg_completion_tokens': completion_tokens_sum / count,
        'avg_reasoning_tokens': reasoning_tokens_sum / count,
        'avg_output_tokens': output_tokens_sum / count,
    }


def compute_environment_accuracy(interactions: List[Dict]) -> Dict[str, float]:
    """
    Compute accuracy by environment type.
    
    Args:
        interactions: List of interaction dictionaries
        
    Returns:
        Dictionary mapping environment type to accuracy
    """
    env_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for interaction in interactions:
        env_type = interaction.get('env_type', 'unknown')
        is_correct = interaction.get('is_correct', False)
        
        env_stats[env_type]['total'] += 1
        if is_correct:
            env_stats[env_type]['correct'] += 1
    
    # Compute accuracy
    env_accuracy = {}
    for env_type, stats in env_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            env_accuracy[env_type] = accuracy
        else:
            env_accuracy[env_type] = 0.0
    
    return env_accuracy


def aggregate_single_file(file_path: str) -> Dict[str, Any]:
    """
    Aggregate statistics from a single response file.
    
    Args:
        file_path: Path to the response JSON file
        
    Returns:
        Dictionary with aggregated statistics
    """
    print(f"Processing: {file_path}")
    
    # Load data
    data = load_response_file(file_path)
    interactions = data.get('interactions', [])
    
    if not interactions:
        print(f"  Warning: No interactions found in {file_path}")
        return None
    
    # Extract model name
    model_name = extract_model_name(file_path)
    
    # Compute statistics
    task_type_acc = compute_task_type_accuracy(interactions)
    overall_acc = compute_overall_accuracy(interactions)
    env_acc = compute_environment_accuracy(interactions)
    token_stats = compute_average_tokens(interactions)
    
    # Compile results
    results = {
        'model_name': model_name,
        'file_path': file_path,
        'total_tasks': len(interactions),
        'overall_accuracy': overall_acc,
        'env_accuracy': env_acc,
        'task_type_accuracy': task_type_acc,
        'token_stats': token_stats,
    }
    
    print(f"  Model: {model_name}")
    print(f"  Tasks: {len(interactions)}")
    print(f"  Overall Accuracy: {overall_acc:.4f}")
    print(f"  Avg Completion Tokens: {token_stats['avg_completion_tokens']:.2f}")
    
    return results


def aggregate_directory(directory: str, pattern: str = "*.json") -> List[Dict[str, Any]]:
    """
    Aggregate statistics from all JSON files in a directory.
    
    Args:
        directory: Path to directory containing response files
        pattern: Glob pattern to match files (default: "*.json")
        
    Returns:
        List of aggregated results
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Find all matching files
    json_files = list(directory_path.glob(pattern))
    
    if not json_files:
        print(f"No files matching pattern '{pattern}' found in {directory}")
        return []
    
    print(f"Found {len(json_files)} files to process\n")
    
    # Aggregate each file
    all_results = []
    for file_path in sorted(json_files):
        try:
            result = aggregate_single_file(str(file_path))
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
        print()  # Empty line between files
    
    return all_results


def create_summary_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary table from aggregated results.
    
    Args:
        results: List of aggregated result dictionaries
        
    Returns:
        DataFrame with summary statistics
    """
    if not results:
        return pd.DataFrame()
    
    # Collect all unique task types
    all_task_types = set()
    for result in results:
        all_task_types.update(result['task_type_accuracy'].keys())
    
    all_task_types = sorted(all_task_types)
    
    # Build table data
    table_data = []
    
    for result in results:
        row = {
            'Model': result['model_name'],
            'Total Tasks': result['total_tasks'],
            'Overall Acc': result['overall_accuracy'],
            'Avg Completion Tokens': result['token_stats']['avg_completion_tokens'],
            'Avg Reasoning Tokens': result['token_stats']['avg_reasoning_tokens'],
        }
        
        # Add environment accuracy
        for env, acc in result['env_accuracy'].items():
            row[f'{env.capitalize()} Acc'] = acc
        
        # Add task type accuracy
        for task_type in all_task_types:
            acc = result['task_type_accuracy'].get(task_type, 0.0)
            row[task_type] = acc
        
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Sort by overall accuracy (descending)
    df = df.sort_values('Overall Acc', ascending=False)
    
    return df


def export_to_csv(df: pd.DataFrame, output_path: str):
    """
    Export DataFrame to CSV file.
    
    Args:
        df: DataFrame to export
        output_path: Path to output CSV file
    """
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"Results exported to: {output_path}")


def export_to_markdown(df: pd.DataFrame, output_path: str):
    """
    Export DataFrame to Markdown file.
    
    Args:
        df: DataFrame to export
        output_path: Path to output Markdown file
    """
    with open(output_path, 'w') as f:
        f.write("# Aggregated Results\n\n")
        f.write(df.to_markdown(index=False, floatfmt='.4f'))
        f.write("\n")
    
    print(f"Markdown table exported to: {output_path}")


def print_summary(results: List[Dict[str, Any]]):
    """
    Print a summary of all results.
    
    Args:
        results: List of aggregated result dictionaries
    """
    if not results:
        print("No results to summarize.")
        return
    
    print("\n" + "="*80)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*80)
    
    # Create and display summary table
    df = create_summary_table(results)
    
    # Display with better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print("\n" + str(df))
    print("\n" + "="*80)
    
    # Print best performing model
    best_model = df.iloc[0]
    print(f"\nBest Overall Accuracy: {best_model['Model']} ({best_model['Overall Acc']:.4f})")
    
    # Print model with lowest token usage
    df_sorted_tokens = df.sort_values('Avg Completion Tokens')
    most_efficient = df_sorted_tokens.iloc[0]
    print(f"Most Token Efficient: {most_efficient['Model']} ({most_efficient['Avg Completion Tokens']:.2f} tokens)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate results from saved response JSON files'
    )
    parser.add_argument(
        '--directory', '-d',
        type=str,
        required=True,
        help='Directory containing response JSON files'
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='*_detailed.json',
        help='Glob pattern to match files (default: *_detailed.json)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        help='Path to save results as CSV'
    )
    parser.add_argument(
        '--output-md',
        type=str,
        help='Path to save results as Markdown'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Path to save detailed results as JSON'
    )
    
    args = parser.parse_args()
    
    # Aggregate results
    results = aggregate_directory(args.directory, args.pattern)
    
    if not results:
        print("No valid results found.")
        return
    
    # Print summary
    print_summary(results)
    
    # Create summary table
    df = create_summary_table(results)
    
    # Export to CSV if requested
    if args.output_csv:
        export_to_csv(df, args.output_csv)
    
    # Export to Markdown if requested
    if args.output_md:
        export_to_markdown(df, args.output_md)
    
    # Export detailed JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results exported to: {args.output_json}")


if __name__ == '__main__':
    main()