#!/usr/bin/env python3
"""
Analysis and visualization script for Aligned-Table Benchmark results
Location: scripts/analysis_scores.py
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultsAnalyzer:
    """Analyze benchmark results and generate visualizations."""
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer with results file.
        
        Args:
            results_file: Path to results JSON file
        """
        self.results_file = results_file
        self.data = self.load_results()
        self.model_name = self.data['config'].get('model_name', 'Unknown')
        
    def load_results(self) -> Dict:
        """Load results from JSON file."""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def print_summary_stats(self):
        """Print summary statistics to console."""
        print("=" * 80)
        print(f"ANALYSIS SUMMARY: {self.model_name}")
        print("=" * 80)
        
        overall = self.data['overall_statistics']
        
        print("\n📊 Overall Performance:")
        print(f"  Total Test Cases: {overall['total_cases']}")
        print(f"  Valid Cases: {overall['valid_cases']}")
        print(f"  Average Content Score: {overall['avg_content_score']:.4f}")
        print(f"  Average Alignment Score: {overall['avg_alignment_score']:.4f}")
        print(f"  Average Total Score: {overall['avg_total_score']:.4f}")
        print(f"  Alignment Rate: {overall['alignment_rate']:.2%}")
        
        print("\n🎯 Perfect Scores:")
        print(f"  Perfect Content: {overall['perfect_content_count']}/{overall['total_cases']} ({overall['perfect_content_rate']:.2%})")
        print(f"  Perfect Alignment: {overall['perfect_alignment_count']}/{overall['total_cases']} ({overall['perfect_alignment_rate']:.2%})")
        print(f"  Perfect Total: {overall['perfect_total_count']}/{overall['total_cases']} ({overall['perfect_total_rate']:.2%})")
        
        print("\n📝 Format-Specific Performance:")
        format_stats = self.data['format_statistics']
        
        for fmt in ['latex', 'markdown', 'text']:
            if fmt in format_stats:
                stats = format_stats[fmt]
                print(f"\n  {fmt.upper()}:")
                print(f"    Cases: {stats['total_cases']}")
                print(f"    Avg Content Score: {stats['avg_content_score']:.4f}")
                print(f"    Avg Alignment Score: {stats['avg_alignment_score']:.4f}")
                print(f"    Avg Total Score: {stats['avg_total_score']:.4f}")
                print(f"    Alignment Rate: {stats['alignment_rate']:.2%}")
                print(f"    Perfect Total: {stats['perfect_total_count']}/{stats['total_cases']}")
        
        print("\n" + "=" * 80)
    
    def analyze_score_distribution(self) -> Tuple[List, List, List]:
        """Analyze score distributions."""
        content_scores = []
        alignment_scores = []
        total_scores = []
        
        for result in self.data['detailed_results']:
            content_scores.append(result['content_score'])
            alignment_scores.append(result['alignment_score'])
            total_scores.append(result['total_score'])
        
        return content_scores, alignment_scores, total_scores
    
    def analyze_by_format(self) -> Dict:
        """Analyze scores grouped by format."""
        format_data = defaultdict(lambda: {
            'content': [],
            'alignment': [],
            'total': [],
            'is_aligned': []
        })
        
        for result in self.data['detailed_results']:
            fmt = result['table_format']
            format_data[fmt]['content'].append(result['content_score'])
            format_data[fmt]['alignment'].append(result['alignment_score'])
            format_data[fmt]['total'].append(result['total_score'])
            format_data[fmt]['is_aligned'].append(result['is_aligned'])
        
        return dict(format_data)
    
    def analyze_failure_modes(self) -> Dict:
        """Analyze common failure patterns."""
        failures = {
            'low_content': [],      # content < 0.5
            'low_alignment': [],    # alignment < 0.5
            'both_low': [],         # both < 0.5
            'misaligned': [],       # is_aligned = False
            'wrong_dimensions': []  # row/col mismatch
        }
        
        for result in self.data['detailed_results']:
            test_id = result['test_id']
            content = result['content_score']
            alignment = result['alignment_score']
            details = result.get('content_details', {})
            
            if content < 0.5:
                failures['low_content'].append(test_id)
            
            if alignment < 0.5:
                failures['low_alignment'].append(test_id)
            
            if content < 0.5 and alignment < 0.5:
                failures['both_low'].append(test_id)
            
            if not result['is_aligned']:
                failures['misaligned'].append(test_id)
            
            if details.get('row_score', 1.0) < 1.0 or details.get('col_score', 1.0) < 1.0:
                failures['wrong_dimensions'].append(test_id)
        
        return failures
    
    def plot_score_distributions(self, output_dir: str):
        """Plot score distribution histograms."""
        content, alignment, total = self.analyze_score_distribution()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Score Distributions - {self.model_name}', fontsize=16, fontweight='bold')
        
        # Content scores
        axes[0].hist(content, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(content), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(content):.3f}')
        axes[0].set_xlabel('Content Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Content Score Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Alignment scores
        axes[1].hist(alignment, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(alignment), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(alignment):.3f}')
        axes[1].set_xlabel('Alignment Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Alignment Score Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Total scores
        axes[2].hist(total, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[2].axvline(np.mean(total), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total):.3f}')
        axes[2].set_xlabel('Total Score', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].set_title('Total Score Distribution', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'{Path(self.results_file).stem}_score_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_format_comparison(self, output_dir: str):
        """Plot comparison across formats."""
        format_data = self.analyze_by_format()
        
        formats = list(format_data.keys())
        metrics = ['content', 'alignment', 'total']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Performance by Format - {self.model_name}', fontsize=16, fontweight='bold')
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        for idx, metric in enumerate(metrics):
            data_to_plot = [format_data[fmt][metric] for fmt in formats]
            
            bp = axes[idx].boxplot(data_to_plot, labels=[f.upper() for f in formats],
                                   patch_artist=True, showmeans=True)
            
            # Customize boxplot colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[idx].set_ylabel(f'{metric.capitalize()} Score', fontsize=12)
            axes[idx].set_title(f'{metric.capitalize()} Score by Format', fontsize=14, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].set_ylim([0, 1])
            
            # Add mean values as text
            for i, fmt in enumerate(formats):
                mean_val = np.mean(format_data[fmt][metric])
                axes[idx].text(i+1, mean_val + 0.05, f'{mean_val:.3f}', 
                             ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'{Path(self.results_file).stem}_format_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_scatter_content_vs_alignment(self, output_dir: str):
        """Plot scatter plot of content vs alignment scores."""
        content_scores = []
        alignment_scores = []
        formats = []
        
        for result in self.data['detailed_results']:
            content_scores.append(result['content_score'])
            alignment_scores.append(result['alignment_score'])
            formats.append(result['table_format'])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Color map for formats
        format_colors = {'latex': '#e74c3c', 'markdown': '#3498db', 'text': '#2ecc71'}
        
        for fmt in set(formats):
            mask = [f == fmt for f in formats]
            x = [c for c, m in zip(content_scores, mask) if m]
            y = [a for a, m in zip(alignment_scores, mask) if m]
            ax.scatter(x, y, label=fmt.upper(), alpha=0.6, s=100, 
                      color=format_colors.get(fmt, '#95a5a6'), edgecolors='black', linewidth=1)
        
        # Add diagonal line (perfect correlation)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Perfect Correlation')
        
        # Add quadrant lines
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Content Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Alignment Score', fontsize=14, fontweight='bold')
        ax.set_title(f'Content vs Alignment Scores - {self.model_name}', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add correlation coefficient
        corr = np.corrcoef(content_scores, alignment_scores)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
               transform=ax.transAxes, fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'{Path(self.results_file).stem}_scatter.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_failure_analysis(self, output_dir: str):
        """Plot failure mode analysis."""
        failures = self.analyze_failure_modes()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Failure Analysis - {self.model_name}', fontsize=16, fontweight='bold')
        
        # Failure types bar chart
        failure_types = ['Low Content\n(<0.5)', 'Low Alignment\n(<0.5)', 
                        'Both Low', 'Misaligned', 'Wrong\nDimensions']
        failure_counts = [
            len(failures['low_content']),
            len(failures['low_alignment']),
            len(failures['both_low']),
            len(failures['misaligned']),
            len(failures['wrong_dimensions'])
        ]
        
        colors = ['#e74c3c', '#f39c12', '#c0392b', '#9b59b6', '#34495e']
        bars = ax1.bar(failure_types, failure_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax1.set_title('Failure Modes Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Alignment rate by format
        format_data = self.analyze_by_format()
        formats = list(format_data.keys())
        alignment_rates = [sum(format_data[fmt]['is_aligned']) / len(format_data[fmt]['is_aligned']) 
                          for fmt in formats]
        
        colors_fmt = ['#e74c3c', '#3498db', '#2ecc71']
        bars = ax2.bar([f.upper() for f in formats], alignment_rates, 
                      color=colors_fmt[:len(formats)], alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('Alignment Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Perfect Alignment Rate by Format', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'{Path(self.results_file).stem}_failure_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_detailed_metrics_heatmap(self, output_dir: str):
        """Plot heatmap of detailed metrics by test case."""
        metrics_data = []
        test_ids = []
        
        for result in self.data['detailed_results']:
            test_ids.append(result['test_id'])
            details = result.get('content_details', {})
            metrics_data.append([
                result['content_score'],
                result['alignment_score'],
                details.get('row_score', 0),
                details.get('col_score', 0),
                details.get('cell_accuracy', 0)
            ])
        
        metrics_data = np.array(metrics_data)
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(test_ids) * 0.3)))
        
        im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(range(5))
        ax.set_xticklabels(['Content', 'Alignment', 'Row', 'Column', 'Cell Acc.'], 
                          rotation=45, ha='right')
        ax.set_yticks(range(len(test_ids)))
        ax.set_yticklabels([f'Test {i}' for i in test_ids])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=20, fontsize=12)
        
        # Add values as text
        for i in range(len(test_ids)):
            for j in range(5):
                text = ax.text(j, i, f'{metrics_data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f'Detailed Metrics Heatmap - {self.model_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'{Path(self.results_file).stem}_metrics_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_score_progression(self, output_dir: str):
        """Plot score progression across test cases."""
        test_ids = []
        content_scores = []
        alignment_scores = []
        total_scores = []
        
        for result in sorted(self.data['detailed_results'], key=lambda x: x['test_id']):
            test_ids.append(result['test_id'])
            content_scores.append(result['content_score'])
            alignment_scores.append(result['alignment_score'])
            total_scores.append(result['total_score'])
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(test_ids, content_scores, marker='o', label='Content Score', 
               linewidth=2, markersize=6, color='#3498db')
        ax.plot(test_ids, alignment_scores, marker='s', label='Alignment Score', 
               linewidth=2, markersize=6, color='#2ecc71')
        ax.plot(test_ids, total_scores, marker='^', label='Total Score', 
               linewidth=2, markersize=6, color='#e74c3c')
        
        # Add average lines
        ax.axhline(np.mean(content_scores), color='#3498db', linestyle='--', 
                  alpha=0.5, label=f'Avg Content: {np.mean(content_scores):.3f}')
        ax.axhline(np.mean(alignment_scores), color='#2ecc71', linestyle='--', 
                  alpha=0.5, label=f'Avg Alignment: {np.mean(alignment_scores):.3f}')
        ax.axhline(np.mean(total_scores), color='#e74c3c', linestyle='--', 
                  alpha=0.5, label=f'Avg Total: {np.mean(total_scores):.3f}')
        
        ax.set_xlabel('Test Case ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Score Progression - {self.model_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'{Path(self.results_file).stem}_progression.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_cell_accuracy_analysis(self, output_dir: str):
        """Plot cell accuracy analysis."""
        correct_cells = []
        total_cells = []
        accuracy_rates = []
        formats = []
        
        for result in self.data['detailed_results']:
            details = result.get('content_details', {})
            correct = details.get('correct_cells', 0)
            total = details.get('total_cells', 1)
            
            correct_cells.append(correct)
            total_cells.append(total)
            accuracy_rates.append(correct / total if total > 0 else 0)
            formats.append(result['table_format'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Cell Accuracy Analysis - {self.model_name}', 
                    fontsize=16, fontweight='bold')
        
        # Stacked bar chart
        test_ids = range(len(correct_cells))
        incorrect_cells = [t - c for c, t in zip(correct_cells, total_cells)]
        
        ax1.bar(test_ids, correct_cells, label='Correct', color='#2ecc71', alpha=0.7)
        ax1.bar(test_ids, incorrect_cells, bottom=correct_cells, 
               label='Incorrect', color='#e74c3c', alpha=0.7)
        
        ax1.set_xlabel('Test Case ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
        ax1.set_title('Correct vs Incorrect Cells', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Accuracy rate by format
        format_accuracy = defaultdict(list)
        for fmt, acc in zip(formats, accuracy_rates):
            format_accuracy[fmt].append(acc)
        
        format_names = list(format_accuracy.keys())
        format_avg_acc = [np.mean(format_accuracy[fmt]) for fmt in format_names]
        
        colors_fmt = ['#e74c3c', '#3498db', '#2ecc71']
        bars = ax2.bar([f.upper() for f in format_names], format_avg_acc,
                      color=colors_fmt[:len(format_names)], alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.set_ylabel('Average Cell Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Cell Accuracy by Format', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'{Path(self.results_file).stem}_cell_accuracy.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def export_detailed_report(self, output_dir: str):
        """Export detailed text report."""
        output_path = Path(output_dir) / f'{Path(self.results_file).stem}_detailed_report.txt'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DETAILED ANALYSIS REPORT: {self.model_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            overall = self.data['overall_statistics']
            for key, value in overall.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            # Format-specific statistics
            f.write("\n" + "=" * 80 + "\n")
            f.write("FORMAT-SPECIFIC STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            for fmt, stats in self.data['format_statistics'].items():
                f.write(f"\n{fmt.upper()}\n")
                f.write("-" * 40 + "\n")
                for key, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            # Failure analysis
            f.write("\n" + "=" * 80 + "\n")
            f.write("FAILURE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            failures = self.analyze_failure_modes()
            for failure_type, test_ids in failures.items():
                f.write(f"\n{failure_type.replace('_', ' ').title()}:\n")
                f.write(f"  Count: {len(test_ids)}\n")
                f.write(f"  Test IDs: {test_ids}\n")
            
            # Individual test case details
            f.write("\n" + "=" * 80 + "\n")
            f.write("INDIVIDUAL TEST CASE DETAILS\n")
            f.write("=" * 80 + "\n\n")
            
            for result in sorted(self.data['detailed_results'], key=lambda x: x['test_id']):
                f.write(f"\nTest Case #{result['test_id']} ({result['table_format'].upper()})\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Content Score: {result['content_score']:.4f}\n")
                f.write(f"  Alignment Score: {result['alignment_score']:.4f}\n")
                f.write(f"  Total Score: {result['total_score']:.4f}\n")
                f.write(f"  Is Aligned: {result['is_aligned']}\n")
                
                if 'content_details' in result:
                    details = result['content_details']
                    f.write(f"  Row Score: {details.get('row_score', 0):.4f}\n")
                    f.write(f"  Column Score: {details.get('col_score', 0):.4f}\n")
                    f.write(f"  Cell Accuracy: {details.get('cell_accuracy', 0):.4f}\n")
                    f.write(f"  Correct Cells: {details.get('correct_cells', 0)}/{details.get('total_cells', 0)}\n")
                    f.write(f"  Dimensions: {details.get('extracted_rows', 0)}x{details.get('extracted_cols', 0)} ")
                    f.write(f"(expected: {details.get('expected_rows', 0)}x{details.get('expected_cols', 0)})\n")
        
        print(f"✓ Saved: {output_path}")
    
    def generate_all_visualizations(self, output_dir: str = 'analysis_output'):
        """Generate all visualizations and reports."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS AND REPORTS")
        print("=" * 80 + "\n")
        
        # Print summary to console
        self.print_summary_stats()
        
        # Generate all plots
        print("\n📊 Generating plots...")
        self.plot_score_distributions(output_dir)
        self.plot_format_comparison(output_dir)
        self.plot_scatter_content_vs_alignment(output_dir)
        self.plot_failure_analysis(output_dir)
        self.plot_detailed_metrics_heatmap(output_dir)
        self.plot_score_progression(output_dir)
        self.plot_cell_accuracy_analysis(output_dir)
        
        # Export detailed report
        print("\n📝 Generating detailed report...")
        self.export_detailed_report(output_dir)
        
        print("\n" + "=" * 80)
        print(f"✓ All visualizations saved to: {output_dir}")
        print("=" * 80)


def compare_multiple_models(results_files: List[str], output_dir: str = 'comparison_output'):
    """Compare results from multiple models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("COMPARING MULTIPLE MODELS")
    print("=" * 80 + "\n")
    
    # Load all results
    all_data = []
    model_names = []
    
    for file in results_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
            model_names.append(data['config'].get('model_name', Path(file).stem))
    
    # Compare overall scores
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison', fontsize=18, fontweight='bold')
    
    # 1. Overall scores comparison
    metrics = ['avg_content_score', 'avg_alignment_score', 'avg_total_score']
    metric_labels = ['Content Score', 'Alignment Score', 'Total Score']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        scores = [data['overall_statistics'][metric] for data in all_data]
        axes[0, 0].bar(x + i * width, scores, width, label=label, color=color, alpha=0.7)
        
        # Add value labels
        for j, score in enumerate(scores):
            axes[0, 0].text(x[j] + i * width, score + 0.02, f'{score:.3f}',
                          ha='center', va='bottom', fontsize=9)
    
    axes[0, 0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Overall Performance Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x + width)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Alignment rate comparison
    alignment_rates = [data['overall_statistics']['alignment_rate'] for data in all_data]
    bars = axes[0, 1].bar(model_names, alignment_rates, color='#9b59b6', alpha=0.7, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[0, 1].set_ylabel('Alignment Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Perfect Alignment Rate', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Perfect score rates comparison
    perfect_metrics = ['perfect_content_rate', 'perfect_alignment_rate', 'perfect_total_rate']
    perfect_labels = ['Perfect Content', 'Perfect Alignment', 'Perfect Total']
    
    for i, (metric, label, color) in enumerate(zip(perfect_metrics, perfect_labels, colors)):
        rates = [data['overall_statistics'][metric] for data in all_data]
        axes[1, 0].bar(x + i * width, rates, width, label=label, color=color, alpha=0.7)
    
    axes[1, 0].set_ylabel('Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Perfect Score Rates', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Format-specific comparison (using best format for each model)
    format_scores = defaultdict(list)
    
    for data in all_data:
        for fmt in ['latex', 'markdown', 'text']:
            if fmt in data['format_statistics']:
                format_scores[fmt].append(data['format_statistics'][fmt]['avg_total_score'])
            else:
                format_scores[fmt].append(0)
    
    x_fmt = np.arange(len(model_names))
    width_fmt = 0.25
    
    for i, (fmt, scores) in enumerate(format_scores.items()):
        axes[1, 1].bar(x_fmt + i * width_fmt, scores, width_fmt, 
                      label=fmt.upper(), alpha=0.7)
    
    axes[1, 1].set_ylabel('Average Total Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Performance by Format', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x_fmt + width_fmt)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_path / 'model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Create comparison table
    comparison_table_path = output_path / 'comparison_table.txt'
    with open(comparison_table_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("MODEL COMPARISON TABLE\n")
        f.write("=" * 100 + "\n\n")
        
        # Header
        f.write(f"{'Model':<30} {'Content':<12} {'Alignment':<12} {'Total':<12} {'Align%':<12}\n")
        f.write("-" * 100 + "\n")
        
        # Data rows
        for name, data in zip(model_names, all_data):
            stats = data['overall_statistics']
            f.write(f"{name:<30} ")
            f.write(f"{stats['avg_content_score']:<12.4f} ")
            f.write(f"{stats['avg_alignment_score']:<12.4f} ")
            f.write(f"{stats['avg_total_score']:<12.4f} ")
            f.write(f"{stats['alignment_rate']:<12.2%}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("FORMAT-SPECIFIC COMPARISON\n")
        f.write("=" * 100 + "\n\n")
        
        for fmt in ['latex', 'markdown', 'text']:
            f.write(f"\n{fmt.upper()}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Model':<30} {'Content':<12} {'Alignment':<12} {'Total':<12} {'Cases':<12}\n")
            f.write("-" * 100 + "\n")
            
            for name, data in zip(model_names, all_data):
                if fmt in data['format_statistics']:
                    stats = data['format_statistics'][fmt]
                    f.write(f"{name:<30} ")
                    f.write(f"{stats['avg_content_score']:<12.4f} ")
                    f.write(f"{stats['avg_alignment_score']:<12.4f} ")
                    f.write(f"{stats['avg_total_score']:<12.4f} ")
                    f.write(f"{stats['total_cases']:<12}\n")
                else:
                    f.write(f"{name:<30} N/A\n")
    
    print(f"✓ Saved: {comparison_table_path}")
    
    print("\n" + "=" * 80)
    print(f"✓ All comparison visualizations saved to: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize Aligned-Table Benchmark results'
    )
    
    parser.add_argument('results_file', type=str, nargs='+',
                       help='Path to results JSON file(s)')
    parser.add_argument('--output', type=str, default='analysis_output',
                       help='Output directory for visualizations')
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple models (requires multiple input files)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots (only generate text reports)')
    
    args = parser.parse_args()
    
    try:
        if args.compare and len(args.results_file) > 1:
            # Compare multiple models
            compare_multiple_models(args.results_file, args.output)
        elif len(args.results_file) == 1:
            # Analyze single model
            analyzer = ResultsAnalyzer(args.results_file[0])
            
            if args.no_plots:
                analyzer.print_summary_stats()
                analyzer.export_detailed_report(args.output)
            else:
                analyzer.generate_all_visualizations(args.output)
        else:
            print("Error: Please provide at least one results file.")
            print("Use --compare flag to compare multiple models.")
            sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()