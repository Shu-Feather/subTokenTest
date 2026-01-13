"""
Utility functions for RSA-Difference Benchmark
Path: src/utils.py
"""

import yaml
import json
import os
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

from configs.locator import resolve_config_path


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            config_path = resolve_config_path("rsa_randomart", config_path)
    
    if not config_path or not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def create_prompt(pattern1: list, pattern2: list, restricted_reasoning: bool = False) -> str:
    """
    Create prompt for the model
    
    Args:
        pattern1: First RSA pattern (list of strings)
        pattern2: Second RSA pattern (list of strings)
        
    Returns:
        Formatted prompt string
    """
    pattern1_str = '\n'.join(pattern1)
    pattern2_str = '\n'.join(pattern2)
    
    prompt = f"""You are given two RSA key fingerprint patterns. Your task is to find all the differences between them.

The coordinate system is defined as follows:
- Origin (0, 0) is at the top-left '+' character
- X-axis increases horizontally to the right
- Y-axis increases vertically downward
- Each character position has a length of 1

For each difference you find, report:
1. The coordinate (x, y)
2. What element was replaced: original_element -> new_element

Use the format: (x, y): A -> B
- If the original position was empty (space), represent it as a space
- If the new element is empty (space), represent it as a space

Example format:
<answer>
(7, 3):   -> o
(8, 3): o ->  
(3, 5): o -> .
</answer>

Following are the two RSA key fingerprint patterns for comparison:

Pattern 1:
{pattern1_str}

Pattern 2:
{pattern2_str}

Please identify all differences between Pattern 1 and Pattern 2.
Put your final answer between <answer> and </answer> tags.
"""
    if restricted_reasoning:
        prompt += (
            "\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
        )
    
    return prompt


def save_response(
    responses: list,
    output_path: str
):
    """
    Save model responses to file
    
    Args:
        responses: List of response dictionaries
        output_path: Path to save responses
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)


def format_results(results: Dict) -> str:
    """
    Format evaluation results for display
    
    Args:
        results: Results dictionary from evaluator
        
    Returns:
        Formatted string
    """
    output = []
    output.append("=" * 60)
    output.append("RSA-Difference Benchmark Results")
    output.append("=" * 60)
    output.append(f"Number of samples: {results['num_samples']}")
    output.append("")
    output.append("Coordinate Metrics:")
    output.append(f"  Precision: {results['avg_coordinate_precision']:.4f}")
    output.append(f"  Recall:    {results['avg_coordinate_recall']:.4f}")
    output.append(f"  F1 Score:  {results['avg_coordinate_f1']:.4f}")
    output.append("")
    output.append("Replacement Metrics:")
    output.append(f"  Accuracy:  {results['avg_replacement_accuracy']:.4f}")
    output.append("")
    output.append("Overall Score:")
    output.append(f"  {results['avg_overall_score']:.4f}")
    output.append("")
    output.append("Statistics:")
    output.append(f"  Total predictions: {results['total_predictions']}")
    output.append(f"  Total ground truth: {results['total_ground_truth']}")
    output.append(f"  Correct coordinates: {results['total_correct_coords']}")
    output.append(f"  Correct replacements: {results['total_correct_replacements']}")
    
    # Add token usage if available
    if 'token_usage' in results:
        output.append("")
        output.append("Token Usage:")
        total = results['token_usage']['total']
        avg = results['token_usage']['average_per_sample']
        output.append(f"  Total tokens:      {total['total_tokens']:,}")
        output.append(f"  Input tokens:      {total['input_tokens']:,}")
        output.append(f"  Output tokens:     {total['output_tokens']:,}")
        output.append(f"  Reasoning tokens:  {total['reasoning_tokens']:,}")
        output.append("")
        output.append("  Average per sample:")
        output.append(f"    Total:     {avg['total_tokens']:.1f}")
        output.append(f"    Input:     {avg['input_tokens']:.1f}")
        output.append(f"    Output:    {avg['output_tokens']:.1f}")
        output.append(f"    Reasoning: {avg['reasoning_tokens']:.1f}")
    
    output.append("=" * 60)
    
    return '\n'.join(output)


def save_results(
    results: Dict,
    output_path: str
):
    """
    Save evaluation results to file
    
    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
