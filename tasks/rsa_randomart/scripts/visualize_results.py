"""
Visualization script for benchmark results
Path: scripts/visualize_results.py
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(result_path):
    """Load results from JSON file"""
    with open(result_path, 'r') as f:
        return json.load(f)


def plot_metrics_comparison(results_dict, output_path='outputs/metrics_comparison.png'):
    """
    Plot comparison of different metrics across models
    
    Args:
        results_dict: Dictionary mapping model names to their results
        output_path: Path to save the plot
    """
    models = list(results_dict.keys())
    
    metrics = {
        'Coordinate\nPrecision': [r['avg_coordinate_precision'] for r in results_dict.values()],
        'Coordinate\nRecall': [r['avg_coordinate_recall'] for r in results_dict.values()],
        'Coordinate\nF1': [r['avg_coordinate_f1'] for r in results_dict.values()],
        'Replacement\nAccuracy': [r['avg_replacement_accuracy'] for r in results_dict.values()],
        'Overall\nScore': [r['avg_overall_score'] for r in results_dict.values()]
    }
    
    x = np.arange(len(models))
    width = 0.15
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for attribute, measurement in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=8)
        multiplier += 1
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison Across Metrics')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left', ncols=5)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to {output_path}")
    plt.close()


def plot_individual_performance(results, output_path='outputs/individual_performance.png'):
    """
    Plot performance on individual samples
    
    Args:
        results: Results dictionary
        output_path: Path to save the plot
    """
    individual_results = results['individual_results']
    
    sample_ids = range(1, len(individual_results) + 1)
    coord_f1 = [r['coordinate_f1'] for r in individual_results]
    replacement_acc = [r['replacement_accuracy'] for r in individual_results]
    overall_scores = [r['overall_score'] for r in individual_results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: F1 and Replacement Accuracy
    ax1.plot(sample_ids, coord_f1, marker='o', label='Coordinate F1', alpha=0.7)
    ax1.plot(sample_ids, replacement_acc, marker='s', label='Replacement Accuracy', alpha=0.7)
    ax1.axhline(y=np.mean(coord_f1), color='blue', linestyle='--', alpha=0.5, label=f'Avg F1: {np.mean(coord_f1):.3f}')
    ax1.axhline(y=np.mean(replacement_acc), color='orange', linestyle='--', alpha=0.5, label=f'Avg Repl: {np.mean(replacement_acc):.3f}')
    ax1.set_xlabel('Sample ID')
    ax1.set_ylabel('Score')
    ax1.set_title('Coordinate F1 and Replacement Accuracy per Sample')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Overall Score
    ax2.bar(sample_ids, overall_scores, alpha=0.7, color='green')
    ax2.axhline(y=np.mean(overall_scores), color='red', linestyle='--', label=f'Average: {np.mean(overall_scores):.3f}')
    ax2.set_xlabel('Sample ID')
    ax2.set_ylabel('Overall Score')
    ax2.set_title('Overall Score per Sample')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved individual performance plot to {output_path}")
    plt.close()


def plot_confusion_matrix_style(results, output_path='outputs/prediction_analysis.png'):
    """
    Plot prediction statistics
    
    Args:
        results: Results dictionary
        output_path: Path to save the plot
    """
    individual_results = results['individual_results']
    
    # Collect statistics
    total_pred = [r['num_predictions'] for r in individual_results]
    total_gt = [r['num_ground_truth'] for r in individual_results]
    correct_coords = [r['num_correct_coords'] for r in individual_results]
    correct_repl = [r['num_correct_replacements'] for r in individual_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Predictions vs Ground Truth
    axes[0, 0].scatter(total_gt, total_pred, alpha=0.6)
    axes[0, 0].plot([0, max(total_gt)], [0, max(total_gt)], 'r--', label='Perfect Match')
    axes[0, 0].set_xlabel('Ground Truth Count')
    axes[0, 0].set_ylabel('Prediction Count')
    axes[0, 0].set_title('Predictions vs Ground Truth')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Correct Coordinates Distribution
    axes[0, 1].hist(correct_coords, bins=15, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(x=np.mean(correct_coords), color='red', linestyle='--', label=f'Mean: {np.mean(correct_coords):.1f}')
    axes[0, 1].set_xlabel('Number of Correct Coordinates')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Correct Coordinates')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Correct Replacements Distribution
    axes[1, 0].hist(correct_repl, bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(x=np.mean(correct_repl), color='red', linestyle='--', label=f'Mean: {np.mean(correct_repl):.1f}')
    axes[1, 0].set_xlabel('Number of Correct Replacements')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Correct Replacements')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Success Rate
    success_rates = [r['num_correct_replacements'] / r['num_correct_coords'] 
                     if r['num_correct_coords'] > 0 else 0 
                     for r in individual_results]
    axes[1, 1].hist(success_rates, bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(x=np.mean(success_rates), color='red', linestyle='--', label=f'Mean: {np.mean(success_rates):.3f}')
    axes[1, 1].set_xlabel('Replacement Success Rate')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Replacement Accuracy Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved prediction analysis to {output_path}")
    plt.close()


def generate_report(results, output_path='outputs/report.txt'):
    """
    Generate text report
    
    Args:
        results: Results dictionary
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RSA-DIFFERENCE BENCHMARK DETAILED REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Metadata
        if 'metadata' in results:
            f.write("CONFIGURATION\n")
            f.write("-"*70 + "\n")
            for key, value in results['metadata'].items():
                f.write(f"{key:.<30} {value}\n")
            f.write("\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of Samples:................ {results['num_samples']}\n")
        f.write(f"Total Predictions:................ {results['total_predictions']}\n")
        f.write(f"Total Ground Truth:............... {results['total_ground_truth']}\n")
        f.write(f"Correct Coordinates:.............. {results['total_correct_coords']}\n")
        f.write(f"Correct Replacements:............. {results['total_correct_replacements']}\n")
        f.write("\n")
        
        # Metrics
        f.write("METRICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Coordinate Precision:............. {results['avg_coordinate_precision']:.4f}\n")
        f.write(f"Coordinate Recall:................ {results['avg_coordinate_recall']:.4f}\n")
        f.write(f"Coordinate F1:.................... {results['avg_coordinate_f1']:.4f}\n")
        f.write(f"Replacement Accuracy:............. {results['avg_replacement_accuracy']:.4f}\n")
        f.write(f"Overall Score:.................... {results['avg_overall_score']:.4f}\n")
        f.write("\n")
        
        # Individual Results Summary
        f.write("INDIVIDUAL RESULTS SUMMARY\n")
        f.write("-"*70 + "\n")
        individual = results['individual_results']
        
        overall_scores = [r['overall_score'] for r in individual]
        f.write(f"Best Score:....................... {max(overall_scores):.4f}\n")
        f.write(f"Worst Score:...................... {min(overall_scores):.4f}\n")
        f.write(f"Median Score:..................... {np.median(overall_scores):.4f}\n")
        f.write(f"Standard Deviation:............... {np.std(overall_scores):.4f}\n")
        f.write("\n")
        
        # Top 5 and Bottom 5
        sorted_results = sorted(enumerate(individual), key=lambda x: x[1]['overall_score'], reverse=True)
        
        f.write("TOP 5 PERFORMING SAMPLES\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Sample':>8} {'Overall':>10} {'Coord F1':>10} {'Repl Acc':>10}\n")
        for idx, result in sorted_results[:5]:
            f.write(f"{idx+1:>8} {result['overall_score']:>10.4f} "
                   f"{result['coordinate_f1']:>10.4f} {result['replacement_accuracy']:>10.4f}\n")
        f.write("\n")
        
        f.write("BOTTOM 5 PERFORMING SAMPLES\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Sample':>8} {'Overall':>10} {'Coord F1':>10} {'Repl Acc':>10}\n")
        for idx, result in sorted_results[-5:]:
            f.write(f"{idx+1:>8} {result['overall_score']:>10.4f} "
                   f"{result['coordinate_f1']:>10.4f} {result['replacement_accuracy']:>10.4f}\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
        f.write(f"Report generated: {results.get('timestamp', 'N/A')}\n")
        f.write("="*70 + "\n")
    
    print(f"Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize RSA-Difference Benchmark results'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to result JSON file(s)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        help='Labels for each result file (for comparison plots)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/visualizations',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Create comparison plots (requires multiple result files)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_list = [load_results(path) for path in args.results]
    
    if args.labels:
        if len(args.labels) != len(results_list):
            raise ValueError("Number of labels must match number of result files")
        labels = args.labels
    else:
        labels = [f"Model {i+1}" for i in range(len(results_list))]
    
    # Generate visualizations
    if args.compare and len(results_list) > 1:
        print("Generating comparison plots...")
        results_dict = {label: results for label, results in zip(labels, results_list)}
        plot_metrics_comparison(
            results_dict,
            output_path=f"{args.output_dir}/metrics_comparison.png"
        )
    
    # Generate individual plots for each result
    for i, (label, results) in enumerate(zip(labels, results_list)):
        print(f"\nGenerating plots for {label}...")
        
        prefix = f"{args.output_dir}/{label.replace(' ', '_')}"
        
        plot_individual_performance(
            results,
            output_path=f"{prefix}_individual_performance.png"
        )
        
        plot_confusion_matrix_style(
            results,
            output_path=f"{prefix}_prediction_analysis.png"
        )

        generate_report(
            results,
            output_path=f"{prefix}_report.txt"
        )
    
    print(f"\n✓ All visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    main()