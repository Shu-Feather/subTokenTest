"""
Script to analyze and compare benchmark results
"""

import json
import argparse
from typing import List, Dict
import matplotlib.pyplot as plt


def load_results(filepath: str) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_models(result_files: List[str]):
    """Compare results from multiple models."""
    
    all_results = {}
    
    for filepath in result_files:
        results = load_results(filepath)
        model_name = results['config']['model_name']
        all_results[model_name] = results
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\n{'Model':<20} {'Content':<12} {'Alignment':<12} {'Total':<12} {'Align Rate':<12}")
    print("-"*80)
    
    for model_name, results in all_results.items():
        stats = results['overall_statistics']
        print(f"{model_name:<20} "
              f"{stats['avg_content_score']:<12.4f} "
              f"{stats['avg_alignment_score']:<12.4f} "
              f"{stats['avg_total_score']:<12.4f} "
              f"{stats['alignment_rate']:<12.4f}")
    
    # Plot comparison
    plot_comparison(all_results)


def plot_comparison(all_results: Dict):
    """Create comparison plots."""
    
    models = list(all_results.keys())
    content_scores = [r['overall_statistics']['avg_content_score'] for r in all_results.values()]
    alignment_scores = [r['overall_statistics']['avg_alignment_score'] for r in all_results.values()]
    total_scores = [r['overall_statistics']['avg_total_score'] for r in all_results.values()]
    
    x = range(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar([i - width for i in x], content_scores, width, label='Content Score')
    ax.bar(x, alignment_scores, width, label='Alignment Score')
    ax.bar([i + width for i in x], total_scores, width, label='Total Score')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Aligned-Table Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print(f"\nComparison plot saved to: model_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('result_files', nargs='+', help='Result JSON files to compare')
    
    args = parser.parse_args()
    
    compare_models(args.result_files)


if __name__ == '__main__':
    main()