"""
Comprehensive analysis and visualization script for benchmark results.
Analyzes LLM performance on adversarial prompt canonicalization task.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd


class ResultsAnalyzer:
    """Analyzer for benchmark results with visualization capabilities."""
    
    def __init__(self, results_path: str, output_dir: str = None):
        """
        Initialize results analyzer.
        
        Args:
            results_path: Path to results.json file
            output_dir: Directory to save visualizations (default: same as results)
        """
        self.results_path = Path(results_path)
        self.results = self._load_results()
        
        # Filter out empty responses
        self.valid_results = [r for r in self.results if r.get('raw_response', '').strip()]
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.results_path.parent / "analysis"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def _load_results(self) -> List[Dict]:
        """Load results from JSON file."""
        with open(self.results_path, 'r') as f:
            return json.load(f)
    
    def print_summary_statistics(self):
        """Print comprehensive summary statistics."""
        total = len(self.results)
        valid = len(self.valid_results)
        empty = total - valid
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS ANALYSIS")
        print("="*80)
        print(f"Results file: {self.results_path}")
        print(f"Total samples: {total}")
        print(f"Valid responses (non-empty): {valid}")
        print(f"Empty responses: {empty} ({empty/total*100:.1f}%)")
        print("="*80 + "\n")
        
        if valid == 0:
            print("⚠️  No valid results to analyze!")
            return
        
        # Basic metrics
        exact_matches = sum(1 for r in self.valid_results if r['exact_match'])
        tags_found = sum(1 for r in self.valid_results if r.get('tags_found', False))
        
        levenshtein_distances = [r['levenshtein_distance'] for r in self.valid_results]
        similarity_scores = [r['similarity_score'] for r in self.valid_results]
        
        print("📊 OVERALL METRICS")
        print("-" * 80)
        print(f"Exact Match Rate: {exact_matches}/{valid} ({exact_matches/valid*100:.2f}%)")
        print(f"Answer Tags Usage: {tags_found}/{valid} ({tags_found/valid*100:.2f}%)")
        print(f"\nLevenshtein Distance:")
        print(f"  Mean: {np.mean(levenshtein_distances):.2f}")
        print(f"  Median: {np.median(levenshtein_distances):.2f}")
        print(f"  Std: {np.std(levenshtein_distances):.2f}")
        print(f"  Min: {np.min(levenshtein_distances):.0f}")
        print(f"  Max: {np.max(levenshtein_distances):.0f}")
        print(f"\nSimilarity Score:")
        print(f"  Mean: {np.mean(similarity_scores):.4f}")
        print(f"  Median: {np.median(similarity_scores):.4f}")
        print(f"  Std: {np.std(similarity_scores):.4f}")
        print(f"  Min: {np.min(similarity_scores):.4f}")
        print(f"  Max: {np.max(similarity_scores):.4f}")
        
        # Breakdown by category
        print("\n📁 BY CATEGORY")
        print("-" * 80)
        by_category = self._group_by_field('category')
        for category, results in sorted(by_category.items()):
            cat_total = len(results)
            cat_exact = sum(1 for r in results if r['exact_match'])
            cat_tags = sum(1 for r in results if r.get('tags_found', False))
            cat_similarity = np.mean([r['similarity_score'] for r in results])
            
            print(f"{category}:")
            print(f"  Samples: {cat_total}")
            print(f"  Exact Match: {cat_exact}/{cat_total} ({cat_exact/cat_total*100:.1f}%)")
            print(f"  Tags Used: {cat_tags}/{cat_total} ({cat_tags/cat_total*100:.1f}%)")
            print(f"  Avg Similarity: {cat_similarity:.4f}")
        
        # Breakdown by difficulty
        print("\n📈 BY DIFFICULTY LEVEL")
        print("-" * 80)
        by_difficulty = self._group_by_field('difficulty')
        for difficulty in ['easy', 'medium', 'hard', 'unknown']:
            if difficulty not in by_difficulty:
                continue
            results = by_difficulty[difficulty]
            diff_total = len(results)
            diff_exact = sum(1 for r in results if r['exact_match'])
            diff_tags = sum(1 for r in results if r.get('tags_found', False))
            diff_similarity = np.mean([r['similarity_score'] for r in results])
            
            print(f"{difficulty.upper()}:")
            print(f"  Samples: {diff_total}")
            print(f"  Exact Match: {diff_exact}/{diff_total} ({diff_exact/diff_total*100:.1f}%)")
            print(f"  Tags Used: {diff_tags}/{diff_total} ({diff_tags/diff_total*100:.1f}%)")
            print(f"  Avg Similarity: {diff_similarity:.4f}")
        
        # Breakdown by perturbation type
        print("\n🔀 BY PERTURBATION TYPE")
        print("-" * 80)
        by_perturbation = self._group_by_field('perturbation_type')
        for pert_type, results in sorted(by_perturbation.items()):
            pert_total = len(results)
            pert_exact = sum(1 for r in results if r['exact_match'])
            pert_tags = sum(1 for r in results if r.get('tags_found', False))
            pert_similarity = np.mean([r['similarity_score'] for r in results])
            
            print(f"{pert_type}:")
            print(f"  Samples: {pert_total}")
            print(f"  Exact Match: {pert_exact}/{pert_total} ({pert_exact/pert_total*100:.1f}%)")
            print(f"  Tags Used: {pert_tags}/{pert_total} ({pert_tags/pert_total*100:.1f}%)")
            print(f"  Avg Similarity: {pert_similarity:.4f}")
        
        # Tag usage correlation with accuracy
        print("\n🏷️  TAG USAGE CORRELATION")
        print("-" * 80)
        with_tags = [r for r in self.valid_results if r.get('tags_found', False)]
        without_tags = [r for r in self.valid_results if not r.get('tags_found', False)]
        
        if with_tags:
            with_tags_exact = sum(1 for r in with_tags if r['exact_match'])
            with_tags_similarity = np.mean([r['similarity_score'] for r in with_tags])
            print(f"With tags ({len(with_tags)} samples):")
            print(f"  Exact Match: {with_tags_exact}/{len(with_tags)} ({with_tags_exact/len(with_tags)*100:.1f}%)")
            print(f"  Avg Similarity: {with_tags_similarity:.4f}")
        
        if without_tags:
            without_tags_exact = sum(1 for r in without_tags if r['exact_match'])
            without_tags_similarity = np.mean([r['similarity_score'] for r in without_tags])
            print(f"Without tags ({len(without_tags)} samples):")
            print(f"  Exact Match: {without_tags_exact}/{len(without_tags)} ({without_tags_exact/len(without_tags)*100:.1f}%)")
            print(f"  Avg Similarity: {without_tags_similarity:.4f}")
        
        print("\n" + "="*80 + "\n")
    
    def _group_by_field(self, field: str) -> Dict[str, List[Dict]]:
        """Group results by a specific field."""
        grouped = defaultdict(list)
        for result in self.valid_results:
            key = result.get(field, 'unknown')
            grouped[key].append(result)
        return dict(grouped)
    
    def plot_levenshtein_distance_histogram(self, save: bool = True):
        """Plot histogram of Levenshtein distances."""
        distances = [r['levenshtein_distance'] for r in self.valid_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        ax1.hist(distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(distances):.2f}')
        ax1.axvline(np.median(distances), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(distances):.2f}')
        ax1.set_xlabel('Levenshtein Distance', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Levenshtein Distance', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by difficulty
        by_difficulty = self._group_by_field('difficulty')
        difficulty_order = ['easy', 'medium', 'hard', 'unknown']
        data_for_box = []
        labels_for_box = []
        
        for diff in difficulty_order:
            if diff in by_difficulty:
                data_for_box.append([r['levenshtein_distance'] for r in by_difficulty[diff]])
                labels_for_box.append(f"{diff}\n(n={len(by_difficulty[diff])})")
        
        bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_ylabel('Levenshtein Distance', fontsize=12)
        ax2.set_title('Levenshtein Distance by Difficulty Level', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'levenshtein_distance_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        plt.show()
    
    def plot_similarity_score_histogram(self, save: bool = True):
        """Plot histogram of similarity scores."""
        scores = [r['similarity_score'] for r in self.valid_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        ax1.hist(scores, bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax1.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(scores):.4f}')
        ax1.axvline(np.median(scores), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(scores):.4f}')
        ax1.set_xlabel('Similarity Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Similarity Scores', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by perturbation type
        by_perturbation = self._group_by_field('perturbation_type')
        data_for_box = []
        labels_for_box = []
        
        for pert_type in sorted(by_perturbation.keys()):
            data_for_box.append([r['similarity_score'] for r in by_perturbation[pert_type]])
            labels_for_box.append(f"{pert_type}\n(n={len(by_perturbation[pert_type])})")
        
        bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightyellow')
        ax2.set_ylabel('Similarity Score', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.set_title('Similarity Score by Perturbation Type', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'similarity_score_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        plt.show()
    
    def plot_performance_by_category(self, save: bool = True):
        """Plot performance metrics by category."""
        by_category = self._group_by_field('category')
        
        categories = sorted(by_category.keys())
        exact_match_rates = []
        tag_usage_rates = []
        avg_similarities = []
        sample_counts = []
        
        for cat in categories:
            results = by_category[cat]
            total = len(results)
            exact = sum(1 for r in results if r['exact_match'])
            tags = sum(1 for r in results if r.get('tags_found', False))
            similarity = np.mean([r['similarity_score'] for r in results])
            
            exact_match_rates.append(exact / total * 100)
            tag_usage_rates.append(tags / total * 100)
            avg_similarities.append(similarity * 100)
            sample_counts.append(total)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Plot 1: Exact Match and Tag Usage
        bars1 = ax1.bar(x - width/2, exact_match_rates, width, label='Exact Match Rate',
                       color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, tag_usage_rates, width, label='Tag Usage Rate',
                       color='coral', alpha=0.8)
        
        ax1.set_ylabel('Percentage (%)', fontsize=12)
        ax1.set_title('Exact Match and Tag Usage Rate by Category', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{cat}\n(n={sample_counts[i]})" 
                             for i, cat in enumerate(categories)], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 105)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Average Similarity Score
        bars3 = ax2.bar(x, avg_similarities, color='mediumseagreen', alpha=0.8)
        ax2.set_ylabel('Average Similarity Score (%)', fontsize=12)
        ax2.set_title('Average Similarity Score by Category', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{cat}\n(n={sample_counts[i]})" 
                             for i, cat in enumerate(categories)], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'performance_by_category.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        plt.show()
    
    def plot_performance_by_difficulty(self, save: bool = True):
        """Plot performance progression across difficulty levels."""
        by_difficulty = self._group_by_field('difficulty')
        
        difficulty_order = ['easy', 'medium', 'hard']
        difficulties = [d for d in difficulty_order if d in by_difficulty]
        
        if not difficulties:
            print("⚠️  No difficulty level information available")
            return
        
        exact_match_rates = []
        avg_similarities = []
        avg_levenshtein = []
        sample_counts = []
        
        for diff in difficulties:
            results = by_difficulty[diff]
            total = len(results)
            exact = sum(1 for r in results if r['exact_match'])
            similarity = np.mean([r['similarity_score'] for r in results])
            levenshtein = np.mean([r['levenshtein_distance'] for r in results])
            
            exact_match_rates.append(exact / total * 100)
            avg_similarities.append(similarity * 100)
            avg_levenshtein.append(levenshtein)
            sample_counts.append(total)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        x = np.arange(len(difficulties))
        
        # Plot 1: Performance metrics
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(x, exact_match_rates, marker='o', linewidth=2, markersize=10,
                        label='Exact Match Rate', color='steelblue')
        line2 = ax1.plot(x, avg_similarities, marker='s', linewidth=2, markersize=10,
                        label='Avg Similarity Score', color='coral')
        line3 = ax1_twin.plot(x, avg_levenshtein, marker='^', linewidth=2, markersize=10,
                             label='Avg Levenshtein Distance', color='green')
        
        ax1.set_xlabel('Difficulty Level', fontsize=12)
        ax1.set_ylabel('Percentage (%)', fontsize=12, color='black')
        ax1_twin.set_ylabel('Levenshtein Distance', fontsize=12, color='green')
        ax1.set_title('Performance Progression by Difficulty', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{diff.upper()}\n(n={sample_counts[i]})" 
                             for i, diff in enumerate(difficulties)])
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        
        # Add value annotations
        for i, (em, sim) in enumerate(zip(exact_match_rates, avg_similarities)):
            ax1.annotate(f'{em:.1f}%', (x[i], em), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9, color='steelblue')
            ax1.annotate(f'{sim:.1f}%', (x[i], sim), textcoords="offset points",
                        xytext=(0,-15), ha='center', fontsize=9, color='coral')
        
        # Plot 2: Distribution comparison
        data_for_violin = []
        labels_for_violin = []
        
        for diff in difficulties:
            data_for_violin.append([r['similarity_score'] for r in by_difficulty[diff]])
            labels_for_violin.append(f"{diff.upper()}\n(n={len(by_difficulty[diff])})")
        
        parts = ax2.violinplot(data_for_violin, positions=x, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        ax2.set_xlabel('Difficulty Level', fontsize=12)
        ax2.set_ylabel('Similarity Score', fontsize=12)
        ax2.set_title('Similarity Score Distribution by Difficulty', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_for_violin)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'performance_by_difficulty.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        plt.show()
    
    def plot_scatter_analysis(self, save: bool = True):
        """Plot scatter analysis of metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        levenshtein = [r['levenshtein_distance'] for r in self.valid_results]
        similarity = [r['similarity_score'] for r in self.valid_results]
        original_lengths = [len(r['original']) for r in self.valid_results]
        perturbed_lengths = [len(r['perturbed']) for r in self.valid_results]
        
        # Plot 1: Levenshtein vs Similarity (inverse relationship)
        colors_exact = ['green' if r['exact_match'] else 'red' for r in self.valid_results]
        ax1.scatter(levenshtein, similarity, c=colors_exact, alpha=0.6, s=50)
        ax1.set_xlabel('Levenshtein Distance', fontsize=11)
        ax1.set_ylabel('Similarity Score', fontsize=11)
        ax1.set_title('Levenshtein Distance vs Similarity Score', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Exact Match'),
                          Patch(facecolor='red', label='Not Exact Match')]
        ax1.legend(handles=legend_elements, loc='best')
        
        # Plot 2: Original length vs Performance
        ax2.scatter(original_lengths, similarity, alpha=0.6, s=50, c='steelblue')
        ax2.set_xlabel('Original Text Length (characters)', fontsize=11)
        ax2.set_ylabel('Similarity Score', fontsize=11)
        ax2.set_title('Text Length vs Performance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(original_lengths, similarity, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(original_lengths), p(sorted(original_lengths)), 
                "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
        ax2.legend()
        
        # Plot 3: Perturbation intensity (length increase) vs Performance
        length_increase = [p - o for p, o in zip(perturbed_lengths, original_lengths)]
        ax3.scatter(length_increase, similarity, alpha=0.6, s=50, c='coral')
        ax3.set_xlabel('Perturbation Length Increase (characters)', fontsize=11)
        ax3.set_ylabel('Similarity Score', fontsize=11)
        ax3.set_title('Perturbation Intensity vs Performance', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Tag usage impact
        with_tags_sim = [r['similarity_score'] for r in self.valid_results if r.get('tags_found', False)]
        without_tags_sim = [r['similarity_score'] for r in self.valid_results if not r.get('tags_found', False)]
        
        data_for_box = [with_tags_sim, without_tags_sim]
        labels_for_box = [f'With Tags\n(n={len(with_tags_sim)})', 
                         f'Without Tags\n(n={len(without_tags_sim)})']
        
        bp = ax4.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax4.set_ylabel('Similarity Score', fontsize=11)
        ax4.set_title('Tag Usage Impact on Performance', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add mean lines
        ax4.axhline(np.mean(with_tags_sim), color='green', linestyle='--', alpha=0.7,
                   label=f'Mean with tags: {np.mean(with_tags_sim):.3f}')
        ax4.axhline(np.mean(without_tags_sim), color='red', linestyle='--', alpha=0.7,
                   label=f'Mean without tags: {np.mean(without_tags_sim):.3f}')
        ax4.legend(fontsize=9)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'scatter_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        plt.show()
    
    def plot_confusion_heatmap(self, save: bool = True):
        """Plot heatmap showing performance across category × difficulty."""
        by_category = self._group_by_field('category')
        
        # Prepare data for heatmap
        categories = sorted(by_category.keys())
        difficulties = ['easy', 'medium', 'hard']
        
        # Create matrices for different metrics
        exact_match_matrix = np.zeros((len(categories), len(difficulties)))
        similarity_matrix = np.zeros((len(categories), len(difficulties)))
        count_matrix = np.zeros((len(categories), len(difficulties)))
        
        for i, cat in enumerate(categories):
            cat_results = by_category[cat]
            by_diff = defaultdict(list)
            for r in cat_results:
                diff = r.get('difficulty', 'unknown')
                if diff in difficulties:
                    by_diff[diff].append(r)
            
            for j, diff in enumerate(difficulties):
                if diff in by_diff and len(by_diff[diff]) > 0:
                    results = by_diff[diff]
                    count_matrix[i, j] = len(results)
                    exact_match_matrix[i, j] = sum(1 for r in results if r['exact_match']) / len(results) * 100
                    similarity_matrix[i, j] = np.mean([r['similarity_score'] for r in results]) * 100
                else:
                    exact_match_matrix[i, j] = np.nan
                    similarity_matrix[i, j] = np.nan
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Heatmap 1: Exact Match Rate
        sns.heatmap(exact_match_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                   xticklabels=[d.upper() for d in difficulties],
                   yticklabels=categories, ax=ax1, cbar_kws={'label': 'Exact Match Rate (%)'})
        ax1.set_title('Exact Match Rate (%) by Category × Difficulty', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Difficulty Level', fontsize=12)
        ax1.set_ylabel('Category', fontsize=12)
        
        # Heatmap 2: Average Similarity Score
        sns.heatmap(similarity_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=[d.upper() for d in difficulties],
                   yticklabels=categories, ax=ax2, cbar_kws={'label': 'Avg Similarity (%)'})
        ax2.set_title('Average Similarity Score (%) by Category × Difficulty',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Difficulty Level', fontsize=12)
        ax2.set_ylabel('Category', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'performance_heatmap.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        plt.show()
    
    def generate_failure_report(self, threshold: float = 0.9, save: bool = True):
        """Generate detailed report of failed cases."""
        failures = [r for r in self.valid_results if r['similarity_score'] < threshold]
        
        print("\n" + "="*80)
        print(f"FAILURE ANALYSIS (Similarity < {threshold})")
        print("="*80)
        print(f"Total failures: {len(failures)} / {len(self.valid_results)} "
              f"({len(failures)/len(self.valid_results)*100:.1f}%)\n")
        
        if len(failures) == 0:
            print("✅ No failures found!")
            return
        
        # Analyze failure patterns
        by_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        by_perturbation = defaultdict(list)
        
        for f in failures:
            by_category[f['category']].append(f)
            by_difficulty[f.get('difficulty', 'unknown')].append(f)
            by_perturbation[f['perturbation_type']].append(f)
        
        print("📊 Failures by Category:")
        for cat in sorted(by_category.keys()):
            print(f"  {cat}: {len(by_category[cat])}")
        
        print("\n📊 Failures by Difficulty:")
        for diff in ['easy', 'medium', 'hard', 'unknown']:
            if diff in by_difficulty:
                print(f"  {diff}: {len(by_difficulty[diff])}")
        
        print("\n📊 Failures by Perturbation Type:")
        for pert in sorted(by_perturbation.keys()):
            print(f"  {pert}: {len(by_perturbation[pert])}")
        
        # Tag usage in failures
        failures_with_tags = sum(1 for f in failures if f.get('tags_found', False))
        print(f"\n🏷️  Failures with tags: {failures_with_tags}/{len(failures)} "
              f"({failures_with_tags/len(failures)*100:.1f}%)")
        
        # Top 10 worst cases
        print("\n" + "="*80)
        print("TOP 10 WORST PREDICTIONS")
        print("="*80)
        
        sorted_failures = sorted(failures, key=lambda x: x['similarity_score'])[:10]
        
        for i, failure in enumerate(sorted_failures, 1):
            print(f"\n{i}. Sample ID: {failure['sample_id']}")
            print(f"   Similarity: {failure['similarity_score']:.4f}")
            print(f"   Levenshtein Distance: {failure['levenshtein_distance']}")
            print(f"   Category: {failure['category']}")
            print(f"   Difficulty: {failure.get('difficulty', 'unknown')}")
            print(f"   Perturbation: {failure['perturbation_type']}")
            print(f"   Tags Found: {failure.get('tags_found', False)}")
            print(f"   Original:   {failure['original'][:100]}...")
            print(f"   Perturbed:  {failure['perturbed'][:100]}...")
            print(f"   Prediction: {failure['prediction'][:100]}...")
        
        # Save detailed failure report
        if save:
            report_path = self.output_dir / 'failure_report.txt'
            with open(report_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"DETAILED FAILURE REPORT (Similarity < {threshold})\n")
                f.write("="*80 + "\n\n")
                f.write(f"Total failures: {len(failures)} / {len(self.valid_results)}\n")
                f.write(f"Failure rate: {len(failures)/len(self.valid_results)*100:.2f}%\n\n")
                
                for i, failure in enumerate(sorted_failures, 1):
                    f.write("-"*80 + "\n")
                    f.write(f"Failure #{i}\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Sample ID: {failure['sample_id']}\n")
                    f.write(f"Similarity Score: {failure['similarity_score']:.4f}\n")
                    f.write(f"Levenshtein Distance: {failure['levenshtein_distance']}\n")
                    f.write(f"Category: {failure['category']}\n")
                    f.write(f"Difficulty: {failure.get('difficulty', 'unknown')}\n")
                    f.write(f"Perturbation Type: {failure['perturbation_type']}\n")
                    f.write(f"Tags Found: {failure.get('tags_found', False)}\n\n")
                    f.write(f"Original Text:\n{failure['original']}\n\n")
                    f.write(f"Perturbed Text:\n{failure['perturbed']}\n\n")
                    f.write(f"Prediction:\n{failure['prediction']}\n\n")
                    f.write(f"Raw Response:\n{failure['raw_response']}\n\n")
            
            print(f"\n📄 Detailed failure report saved to: {report_path}")
        
        print("\n" + "="*80 + "\n")
    
    def export_summary_csv(self):
        """Export summary statistics to CSV."""
        # Overall summary
        summary_data = {
            'Metric': [],
            'Value': []
        }
        
        total = len(self.results)
        valid = len(self.valid_results)
        
        summary_data['Metric'].extend([
            'Total Samples',
            'Valid Responses',
            'Empty Responses',
            'Exact Match Rate (%)',
            'Tag Usage Rate (%)',
            'Mean Similarity Score',
            'Median Similarity Score',
            'Mean Levenshtein Distance',
            'Median Levenshtein Distance'
        ])
        
        exact_matches = sum(1 for r in self.valid_results if r['exact_match'])
        tags_found = sum(1 for r in self.valid_results if r.get('tags_found', False))
        similarities = [r['similarity_score'] for r in self.valid_results]
        levenshtein_dists = [r['levenshtein_distance'] for r in self.valid_results]
        
        summary_data['Value'].extend([
            total,
            valid,
            total - valid,
            f"{exact_matches/valid*100:.2f}" if valid > 0 else "0.00",
            f"{tags_found/valid*100:.2f}" if valid > 0 else "0.00",
            f"{np.mean(similarities):.4f}",
            f"{np.median(similarities):.4f}",
            f"{np.mean(levenshtein_dists):.2f}",
            f"{np.median(levenshtein_dists):.2f}"
        ])
        
        df_summary = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'summary_statistics.csv'
        df_summary.to_csv(summary_path, index=False)
        print(f"📊 Summary statistics saved to: {summary_path}")
        
        # Per-sample details
        details_data = []
        for r in self.valid_results:
            details_data.append({
                'sample_id': r['sample_id'],
                'category': r['category'],
                'difficulty': r.get('difficulty', 'unknown'),
                'perturbation_type': r['perturbation_type'],
                'exact_match': r['exact_match'],
                'tags_found': r.get('tags_found', False),
                'similarity_score': r['similarity_score'],
                'levenshtein_distance': r['levenshtein_distance'],
                'original_length': len(r['original']),
                'perturbed_length': len(r['perturbed']),
                'prediction_length': len(r['prediction'])
            })
        
        df_details = pd.DataFrame(details_data)
        details_path = self.output_dir / 'detailed_results.csv'
        df_details.to_csv(details_path, index=False)
        print(f"📊 Detailed results saved to: {details_path}")
    
    def run_full_analysis(self, failure_threshold: float = 0.9):
        """Run complete analysis pipeline."""
        print("\n" + "🔬 " + "="*76 + " 🔬")
        print("   RUNNING FULL BENCHMARK ANALYSIS")
        print("🔬 " + "="*76 + " 🔬\n")
        
        # Print summary statistics
        self.print_summary_statistics()
        
        # Generate all visualizations
        print("\n📊 Generating visualizations...\n")
        
        self.plot_levenshtein_distance_histogram()
        self.plot_similarity_score_histogram()
        self.plot_performance_by_category()
        self.plot_performance_by_difficulty()
        self.plot_scatter_analysis()
        self.plot_confusion_heatmap()
        
        # Generate failure report
        self.generate_failure_report(threshold=failure_threshold)
        
        # Export CSV summaries
        print("\n📊 Exporting summary data...\n")
        self.export_summary_csv()
        
        print("\n✅ " + "="*76 + " ✅")
        print(f"   ANALYSIS COMPLETE - All outputs saved to: {self.output_dir}")
        print("✅ " + "="*76 + " ✅\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize benchmark results",
    )
    
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to results.json file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for analysis (default: same directory as results)'
    )
    
    parser.add_argument(
        '--failure_threshold',
        type=float,
        default=0.9,
        help='Similarity threshold for failure analysis (default: 0.9)'
    )
    
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip generating plots (only print statistics)'
    )
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results).exists():
        print(f"❌ Error: Results file not found: {args.results}")
        return
    
    # Initialize analyzer
    analyzer = ResultsAnalyzer(args.results, args.output)
    
    # Run analysis
    if args.no_plots:
        analyzer.print_summary_statistics()
        analyzer.generate_failure_report(threshold=args.failure_threshold)
        analyzer.export_summary_csv()
    else:
        analyzer.run_full_analysis(failure_threshold=args.failure_threshold)


if __name__ == "__main__":
    main()