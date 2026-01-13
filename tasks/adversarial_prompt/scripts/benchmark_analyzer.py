import json
import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class BenchmarkAnalyzer:
    """
    A comprehensive analyzer for benchmark results from multiple model runs.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the analyzer with a base directory containing result folders.
        
        Args:
            base_dir: Path to directory containing result folders
        """
        self.base_dir = Path(base_dir)
        self.results_data = {}
        self.metrics_data = {}
        self.token_usage_data = {}
        
    def load_all_results(self):
        """
        Load all results from subdirectories in base_dir.
        """
        print("Loading benchmark results...")
        
        for folder in self.base_dir.iterdir():
            if not folder.is_dir():
                continue
                
            model_name = folder.name
            results_file = folder / "results.json"
            metrics_file = folder / "metrics.json"
            token_usage_file = folder / "token_usage.json"
            
            # Load results.json
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    # Filter out invalid samples
                    valid_results = [
                        r for r in results 
                        if r.get('tags_found', False) and r.get('raw_response', '').strip()
                    ]
                    self.results_data[model_name] = valid_results
                    print(f"  Loaded {len(valid_results)}/{len(results)} valid samples for {model_name}")
            
            # Load metrics.json
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    self.metrics_data[model_name] = json.load(f)
            
            # Load token_usage.json
            if token_usage_file.exists():
                with open(token_usage_file, 'r', encoding='utf-8') as f:
                    self.token_usage_data[model_name] = json.load(f)
        
        print(f"\nLoaded data for {len(self.results_data)} models\n")
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table comparing all models.
        """
        summary_data = []
        
        for model_name in self.results_data.keys():
            metrics = self.metrics_data.get(model_name, {})
            token_usage = self.token_usage_data.get(model_name, {})
            
            row = {
                'Model': model_name,
                'Valid Samples': len(self.results_data[model_name]),
                'Exact Match Rate': metrics.get('exact_match_rate', 0),
                'Avg Similarity': metrics.get('avg_similarity_score', 0),
                'Avg Levenshtein Dist': metrics.get('avg_levenshtein_distance', 0),
                'Tags Found Rate': metrics.get('tags_found_rate', 0),
                'Avg Total Tokens': token_usage.get('total', {}).get('avg_total_tokens', 0),
                'Avg Reasoning Tokens': token_usage.get('total', {}).get('avg_reasoning_tokens', 0),
                'Reasoning Ratio': token_usage.get('total', {}).get('reasoning_ratio', 0),
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Exact Match Rate', ascending=False)
        return df
    
    def plot_overall_performance(self, save_path: str = None):
        """
        Plot overall performance comparison across models.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Overall Performance Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.metrics_data.keys())
        
        # 1. Exact Match Rate
        exact_match_rates = [
            self.metrics_data[m].get('exact_match_rate', 0) for m in models
        ]
        axes[0, 0].barh(models, exact_match_rates, color='steelblue')
        axes[0, 0].set_xlabel('Exact Match Rate')
        axes[0, 0].set_title('Exact Match Rate by Model')
        axes[0, 0].set_xlim(0, 1)
        for i, v in enumerate(exact_match_rates):
            axes[0, 0].text(v + 0.01, i, f'{v:.2%}', va='center')
        
        # 2. Average Similarity Score
        avg_similarity = [
            self.metrics_data[m].get('avg_similarity_score', 0) for m in models
        ]
        axes[0, 1].barh(models, avg_similarity, color='seagreen')
        axes[0, 1].set_xlabel('Average Similarity Score')
        axes[0, 1].set_title('Average Similarity Score by Model')
        axes[0, 1].set_xlim(0.0, 1.0)
        for i, v in enumerate(avg_similarity):
            axes[0, 1].text(v + 0.001, i, f'{v:.4f}', va='center')
        
        # 3. Average Levenshtein Distance
        avg_levenshtein = [
            self.metrics_data[m].get('avg_levenshtein_distance', 0) for m in models
        ]
        axes[1, 0].barh(models, avg_levenshtein, color='coral')
        axes[1, 0].set_xlabel('Average Levenshtein Distance')
        axes[1, 0].set_title('Average Levenshtein Distance by Model (Lower is Better)')
        for i, v in enumerate(avg_levenshtein):
            axes[1, 0].text(v + 0.1, i, f'{v:.2f}', va='center')
        
        # 4. Tags Found Rate
        tags_found_rate = [
            self.metrics_data[m].get('tags_found_rate', 0) for m in models
        ]
        axes[1, 1].barh(models, tags_found_rate, color='mediumpurple')
        axes[1, 1].set_xlabel('Tags Found Rate')
        axes[1, 1].set_title('Tags Found Rate by Model')
        axes[1, 1].set_xlim(0, 1)
        for i, v in enumerate(tags_found_rate):
            axes[1, 1].text(v + 0.01, i, f'{v:.2%}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved overall performance plot to {save_path}")
        plt.show()
    
    def plot_category_performance(self, save_path: str = None):
        """
        Plot performance breakdown by category.
        """
        categories = ['unethical_requests', 'harmful_instructions', 
                     'jailbreak_attempts', 'privacy_violations', 
                     'manipulation_attempts']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Performance by Category', fontsize=16, fontweight='bold')
        
        # Prepare data
        models = list(self.metrics_data.keys())
        
        # 1. Exact Match Rate by Category
        category_em_data = []
        for category in categories:
            for model in models:
                em_rate = self.metrics_data[model].get('by_category', {}).get(
                    category, {}
                ).get('exact_match_rate', 0)
                category_em_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Model': model,
                    'Exact Match Rate': em_rate
                })
        
        df_em = pd.DataFrame(category_em_data)
        df_pivot_em = df_em.pivot(index='Category', columns='Model', values='Exact Match Rate')
        df_pivot_em.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Exact Match Rate by Category')
        axes[0].set_ylabel('Exact Match Rate')
        axes[0].set_xlabel('Category')
        axes[0].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # 2. Average Similarity by Category
        category_sim_data = []
        for category in categories:
            for model in models:
                avg_sim = self.metrics_data[model].get('by_category', {}).get(
                    category, {}
                ).get('avg_similarity', 0)
                category_sim_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Model': model,
                    'Avg Similarity': avg_sim
                })
        
        df_sim = pd.DataFrame(category_sim_data)
        df_pivot_sim = df_sim.pivot(index='Category', columns='Model', values='Avg Similarity')
        df_pivot_sim.plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_title('Average Similarity by Category')
        axes[1].set_ylabel('Average Similarity Score')
        axes[1].set_xlabel('Category')
        axes[1].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved category performance plot to {save_path}")
        plt.show()
    
    def plot_token_usage(self, save_path: str = None):
        """
        Plot token usage comparison across models.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Token Usage Comparison', fontsize=16, fontweight='bold')
        
        models = list(self.token_usage_data.keys())
        
        # 1. Average Total Tokens
        avg_total_tokens = [
            self.token_usage_data[m].get('total', {}).get('avg_total_tokens', 0) 
            for m in models
        ]
        axes[0, 0].barh(models, avg_total_tokens, color='steelblue')
        axes[0, 0].set_xlabel('Average Total Tokens')
        axes[0, 0].set_title('Average Total Tokens per Request')
        for i, v in enumerate(avg_total_tokens):
            axes[0, 0].text(v + 50, i, f'{v:.0f}', va='center')
        
        # 2. Token Breakdown (Stacked Bar)
        prompt_tokens = [
            self.token_usage_data[m].get('total', {}).get('avg_prompt_tokens', 0) 
            for m in models
        ]
        reasoning_tokens = [
            self.token_usage_data[m].get('total', {}).get('avg_reasoning_tokens', 0) 
            for m in models
        ]
        output_tokens = [
            self.token_usage_data[m].get('total', {}).get('avg_output_tokens', 0) 
            for m in models
        ]
        
        x = np.arange(len(models))
        axes[0, 1].bar(x, prompt_tokens, label='Prompt', color='lightblue')
        axes[0, 1].bar(x, reasoning_tokens, bottom=prompt_tokens, 
                      label='Reasoning', color='orange')
        axes[0, 1].bar(x, reasoning_tokens + np.array(prompt_tokens), 
                      output_tokens, label='Output', color='lightgreen')
        axes[0, 1].set_ylabel('Average Tokens')
        axes[0, 1].set_title('Token Breakdown by Type')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 1].legend()
        
        # 3. Reasoning Ratio
        reasoning_ratio = [
            self.token_usage_data[m].get('total', {}).get('reasoning_ratio', 0) 
            for m in models
        ]
        axes[1, 0].barh(models, reasoning_ratio, color='coral')
        axes[1, 0].set_xlabel('Reasoning Ratio')
        axes[1, 0].set_title('Reasoning Tokens Ratio (Reasoning / Completion)')
        axes[1, 0].set_xlim(0, 1)
        for i, v in enumerate(reasoning_ratio):
            axes[1, 0].text(v + 0.01, i, f'{v:.2%}', va='center')
        
        # 4. Token Usage by Category
        categories = ['unethical_requests', 'harmful_instructions', 
                     'jailbreak_attempts', 'privacy_violations', 
                     'manipulation_attempts']
        
        if len(models) > 0:
            model_to_plot = models[0]  # Plot first model as example
            category_tokens = [
                self.token_usage_data[model_to_plot].get('by_category', {}).get(
                    cat, {}
                ).get('avg_total_tokens', 0)
                for cat in categories
            ]
            category_labels = [c.replace('_', ' ').title() for c in categories]
            
            axes[1, 1].bar(range(len(categories)), category_tokens, color='mediumpurple')
            axes[1, 1].set_ylabel('Average Total Tokens')
            axes[1, 1].set_title(f'Token Usage by Category ({model_to_plot})')
            axes[1, 1].set_xticks(range(len(categories)))
            axes[1, 1].set_xticklabels(category_labels, rotation=45, ha='right')
            
            for i, v in enumerate(category_tokens):
                axes[1, 1].text(i, v + 50, f'{v:.0f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved token usage plot to {save_path}")
        plt.show()
    
    def plot_similarity_distribution(self, save_path: str = None):
        """
        Plot similarity score distribution across models.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Similarity Score Distribution', fontsize=16, fontweight='bold')
        
        # 1. Histogram of similarity scores
        for model_name, results in self.results_data.items():
            similarity_scores = [r['similarity_score'] for r in results]
            axes[0].hist(similarity_scores, alpha=0.5, label=model_name, bins=30)
        
        axes[0].set_xlabel('Similarity Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Similarity Scores')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # 2. Box plot of similarity scores
        similarity_data = []
        for model_name, results in self.results_data.items():
            for result in results:
                similarity_data.append({
                    'Model': model_name,
                    'Similarity Score': result['similarity_score']
                })
        
        df_sim = pd.DataFrame(similarity_data)
        df_sim.boxplot(column='Similarity Score', by='Model', ax=axes[1])
        axes[1].set_title('Similarity Score Distribution by Model')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Similarity Score')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved similarity distribution plot to {save_path}")
        plt.show()
    
    def plot_levenshtein_distribution(self, save_path: str = None):
        """
        Plot Levenshtein distance distribution across models.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Levenshtein Distance Distribution', fontsize=16, fontweight='bold')
        
        # 1. Histogram of Levenshtein distances
        for model_name, results in self.results_data.items():
            distances = [r['levenshtein_distance'] for r in results]
            axes[0].hist(distances, alpha=0.5, label=model_name, bins=30)
        
        axes[0].set_xlabel('Levenshtein Distance')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Levenshtein Distances')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # 2. Box plot of Levenshtein distances
        distance_data = []
        for model_name, results in self.results_data.items():
            for result in results:
                distance_data.append({
                    'Model': model_name,
                    'Levenshtein Distance': result['levenshtein_distance']
                })
        
        df_dist = pd.DataFrame(distance_data)
        df_dist.boxplot(column='Levenshtein Distance', by='Model', ax=axes[1])
        axes[1].set_title('Levenshtein Distance Distribution by Model')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Levenshtein Distance')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved Levenshtein distribution plot to {save_path}")
        plt.show()
    
    def plot_heatmap_category_performance(self, save_path: str = None):
        """
        Create heatmap showing performance across categories and models.
        """
        categories = ['unethical_requests', 'harmful_instructions', 
                     'jailbreak_attempts', 'privacy_violations', 
                     'manipulation_attempts']
        
        models = list(self.metrics_data.keys())
        
        # Create matrix for exact match rates
        em_matrix = []
        for model in models:
            row = []
            for category in categories:
                em_rate = self.metrics_data[model].get('by_category', {}).get(
                    category, {}
                ).get('exact_match_rate', 0)
                row.append(em_rate)
            em_matrix.append(row)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        category_labels = [c.replace('_', ' ').title() for c in categories]
        
        im = ax.imshow(em_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(category_labels)
        ax.set_yticklabels(models)
        
        # Rotate the tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Exact Match Rate', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(categories)):
                text = ax.text(j, i, f'{em_matrix[i][j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Exact Match Rate Heatmap: Models vs Categories', 
                    fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        plt.show()
    
    def generate_detailed_report(self, output_file: str = "analysis_report.txt"):
        """
        Generate a detailed text report of the analysis.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BENCHMARK ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall Summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 80 + "\n")
            summary_df = self.create_summary_table()
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
            
            # Performance by Category
            f.write("PERFORMANCE BY CATEGORY\n")
            f.write("-" * 80 + "\n")
            for model_name in self.metrics_data.keys():
                f.write(f"\nModel: {model_name}\n")
                by_category = self.metrics_data[model_name].get('by_category', {})
                for category, metrics in by_category.items():
                    f.write(f"  {category.replace('_', ' ').title()}:\n")
                    f.write(f"    Exact Match Rate: {metrics.get('exact_match_rate', 0):.2%}\n")
                    f.write(f"    Avg Similarity: {metrics.get('avg_similarity', 0):.4f}\n")
                    f.write(f"    Sample Count: {metrics.get('count', 0)}\n")
            f.write("\n")
            
            # Token Usage Statistics
            f.write("TOKEN USAGE STATISTICS\n")
            f.write("-" * 80 + "\n")
            for model_name in self.token_usage_data.keys():
                f.write(f"\nModel: {model_name}\n")
                total_usage = self.token_usage_data[model_name].get('total', {})
                f.write(f"  Average Total Tokens: {total_usage.get('avg_total_tokens', 0):.2f}\n")
                f.write(f"  Average Prompt Tokens: {total_usage.get('avg_prompt_tokens', 0):.2f}\n")
                f.write(f"  Average Completion Tokens: {total_usage.get('avg_completion_tokens', 0):.2f}\n")
                f.write(f"  Average Reasoning Tokens: {total_usage.get('avg_reasoning_tokens', 0):.2f}\n")
                f.write(f"  Reasoning Ratio: {total_usage.get('reasoning_ratio', 0):.2%}\n")
            f.write("\n")
            
            # Best and Worst Performing Samples
            f.write("SAMPLE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            for model_name, results in self.results_data.items():
                f.write(f"\nModel: {model_name}\n")
                
                # Best performing (highest similarity)
                sorted_by_sim = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
                f.write(f"  Top 3 Best Performing Samples (by similarity):\n")
                for i, sample in enumerate(sorted_by_sim[:3], 1):
                    f.write(f"    {i}. Sample ID: {sample['sample_id']}, "
                           f"Similarity: {sample['similarity_score']:.4f}, "
                           f"Category: {sample['category']}\n")
                
                # Worst performing (lowest similarity)
                f.write(f"  Top 3 Worst Performing Samples (by similarity):\n")
                for i, sample in enumerate(sorted_by_sim[-3:], 1):
                    f.write(f"    {i}. Sample ID: {sample['sample_id']}, "
                           f"Similarity: {sample['similarity_score']:.4f}, "
                           f"Category: {sample['category']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Detailed report saved to {output_file}")
    
    def export_comparison_csv(self, output_file: str = "model_comparison.csv"):
        """
        Export comparison data to CSV for further analysis.
        """
        summary_df = self.create_summary_table()
        summary_df.to_csv(output_file, index=False)
        print(f"Comparison data exported to {output_file}")
    
    def plot_perturbation_performance(self, save_path: str = None):
        """
        Plot performance by perturbation type.
        """
        models = list(self.metrics_data.keys())
        
        # Collect all perturbation types
        all_perturbations = set()
        for metrics in self.metrics_data.values():
            all_perturbations.update(metrics.get('by_perturbation', {}).keys())
        
        if not all_perturbations:
            print("No perturbation data available")
            return
        
        perturbations = sorted(list(all_perturbations))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Performance by Perturbation Type', fontsize=16, fontweight='bold')
        
        # 1. Exact Match Rate by Perturbation
        pert_em_data = []
        for pert in perturbations:
            for model in models:
                em_rate = self.metrics_data[model].get('by_perturbation', {}).get(
                    pert, {}
                ).get('exact_match_rate', 0)
                pert_em_data.append({
                    'Perturbation': pert.replace('_', ' ').title(),
                    'Model': model,
                    'Exact Match Rate': em_rate
                })
        
        df_em = pd.DataFrame(pert_em_data)
        df_pivot_em = df_em.pivot(index='Perturbation', columns='Model', values='Exact Match Rate')
        df_pivot_em.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Exact Match Rate by Perturbation Type')
        axes[0].set_ylabel('Exact Match Rate')
        axes[0].set_xlabel('Perturbation Type')
        axes[0].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # 2. Average Similarity by Perturbation
        pert_sim_data = []
        for pert in perturbations:
            for model in models:
                avg_sim = self.metrics_data[model].get('by_perturbation', {}).get(
                    pert, {}
                ).get('avg_similarity', 0)
                pert_sim_data.append({
                    'Perturbation': pert.replace('_', ' ').title(),
                    'Model': model,
                    'Avg Similarity': avg_sim
                })
        
        df_sim = pd.DataFrame(pert_sim_data)
        df_pivot_sim = df_sim.pivot(index='Perturbation', columns='Model', values='Avg Similarity')
        df_pivot_sim.plot(kind='bar', ax=axes[1], width=0.8)
        axes[1].set_title('Average Similarity by Perturbation Type')
        axes[1].set_ylabel('Average Similarity Score')
        axes[1].set_xlabel('Perturbation Type')
        axes[1].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved perturbation performance plot to {save_path}")
        plt.show()
    
    def create_comprehensive_analysis(self, output_dir: str = "analysis_output"):
        """
        Run all analyses and save outputs to a directory.
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BENCHMARK ANALYSIS")
        print("=" * 80 + "\n")
        
        # Generate summary table
        print("Generating summary table...")
        summary_df = self.create_summary_table()
        print("\n" + summary_df.to_string(index=False))
        print("\n")
        
        # Export to CSV
        csv_path = output_path / "model_comparison.csv"
        self.export_comparison_csv(str(csv_path))
        
        # Generate detailed report
        report_path = output_path / "analysis_report.txt"
        self.generate_detailed_report(str(report_path))

        # Generate all plots
        print("\nGenerating visualizations...")
        
        print("  1. Overall performance comparison...")
        self.plot_overall_performance(
            save_path=str(output_path / "overall_performance.png")
        )
        
        print("  2. Category performance breakdown...")
        self.plot_category_performance(
            save_path=str(output_path / "category_performance.png")
        )
        
        print("  3. Token usage analysis...")
        self.plot_token_usage(
            save_path=str(output_path / "token_usage.png")
        )
        
        print("  4. Similarity distribution...")
        self.plot_similarity_distribution(
            save_path=str(output_path / "similarity_distribution.png")
        )
        
        print("  5. Levenshtein distance distribution...")
        self.plot_levenshtein_distribution(
            save_path=str(output_path / "levenshtein_distribution.png")
        )
        
        print("  6. Category performance heatmap...")
        self.plot_heatmap_category_performance(
            save_path=str(output_path / "category_heatmap.png")
        )
        
        print("  7. Perturbation performance...")
        self.plot_perturbation_performance(
            save_path=str(output_path / "perturbation_performance.png")
        )
        
        print("\n" + "=" * 80)
        print(f"Analysis complete! All outputs saved to: {output_path.absolute()}")
        print("=" * 80 + "\n")
    
    def compare_two_models(self, model1: str, model2: str, save_path: str = None):
        """
        Create detailed comparison between two specific models.
        """
        if model1 not in self.results_data or model2 not in self.results_data:
            print(f"Error: One or both models not found in data")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'Detailed Comparison: {model1} vs {model2}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Overall metrics comparison (bar chart)
        ax1 = fig.add_subplot(gs[0, :])
        metrics_to_compare = ['Exact Match Rate', 'Avg Similarity', 'Tags Found Rate']
        
        m1_metrics = self.metrics_data[model1]
        m2_metrics = self.metrics_data[model2]
        
        m1_values = [
            m1_metrics.get('exact_match_rate', 0),
            m1_metrics.get('avg_similarity_score', 0),
            m1_metrics.get('tags_found_rate', 0)
        ]
        m2_values = [
            m2_metrics.get('exact_match_rate', 0),
            m2_metrics.get('avg_similarity_score', 0),
            m2_metrics.get('tags_found_rate', 0)
        ]
        
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        
        ax1.bar(x - width/2, m1_values, width, label=model1, alpha=0.8)
        ax1.bar(x + width/2, m2_values, width, label=model2, alpha=0.8)
        ax1.set_ylabel('Score')
        ax1.set_title('Overall Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_to_compare)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (v1, v2) in enumerate(zip(m1_values, m2_values)):
            ax1.text(i - width/2, v1 + 0.02, f'{v1:.3f}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, v2 + 0.02, f'{v2:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Category comparison
        ax2 = fig.add_subplot(gs[1, 0])
        categories = ['unethical_requests', 'harmful_instructions', 
                     'jailbreak_attempts', 'privacy_violations', 
                     'manipulation_attempts']
        category_labels = [c.replace('_', '\n').title() for c in categories]
        
        m1_cat_em = [
            m1_metrics.get('by_category', {}).get(cat, {}).get('exact_match_rate', 0)
            for cat in categories
        ]
        m2_cat_em = [
            m2_metrics.get('by_category', {}).get(cat, {}).get('exact_match_rate', 0)
            for cat in categories
        ]
        
        x_cat = np.arange(len(categories))
        ax2.plot(x_cat, m1_cat_em, marker='o', label=model1, linewidth=2)
        ax2.plot(x_cat, m2_cat_em, marker='s', label=model2, linewidth=2)
        ax2.set_ylabel('Exact Match Rate')
        ax2.set_title('EM Rate by Category')
        ax2.set_xticks(x_cat)
        ax2.set_xticklabels(category_labels, fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Similarity distribution comparison
        ax3 = fig.add_subplot(gs[1, 1])
        m1_similarities = [r['similarity_score'] for r in self.results_data[model1]]
        m2_similarities = [r['similarity_score'] for r in self.results_data[model2]]
        
        ax3.hist(m1_similarities, alpha=0.5, bins=20, label=model1, color='blue')
        ax3.hist(m2_similarities, alpha=0.5, bins=20, label=model2, color='orange')
        ax3.set_xlabel('Similarity Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Similarity Distribution')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Levenshtein distance comparison
        ax4 = fig.add_subplot(gs[1, 2])
        m1_distances = [r['levenshtein_distance'] for r in self.results_data[model1]]
        m2_distances = [r['levenshtein_distance'] for r in self.results_data[model2]]
        
        bp = ax4.boxplot([m1_distances, m2_distances], 
                         labels=[model1, model2],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        ax4.set_ylabel('Levenshtein Distance')
        ax4.set_title('Levenshtein Distance Comparison')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Token usage comparison
        ax5 = fig.add_subplot(gs[2, 0])
        m1_tokens = self.token_usage_data[model1].get('total', {})
        m2_tokens = self.token_usage_data[model2].get('total', {})
        
        token_types = ['Prompt', 'Reasoning', 'Output']
        m1_token_values = [
            m1_tokens.get('avg_prompt_tokens', 0),
            m1_tokens.get('avg_reasoning_tokens', 0),
            m1_tokens.get('avg_output_tokens', 0)
        ]
        m2_token_values = [
            m2_tokens.get('avg_prompt_tokens', 0),
            m2_tokens.get('avg_reasoning_tokens', 0),
            m2_tokens.get('avg_output_tokens', 0)
        ]
        
        x_tok = np.arange(len(token_types))
        width = 0.35
        
        ax5.bar(x_tok - width/2, m1_token_values, width, label=model1, alpha=0.8)
        ax5.bar(x_tok + width/2, m2_token_values, width, label=model2, alpha=0.8)
        ax5.set_ylabel('Average Tokens')
        ax5.set_title('Token Usage by Type')
        ax5.set_xticks(x_tok)
        ax5.set_xticklabels(token_types)
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # 6. Reasoning ratio comparison
        ax6 = fig.add_subplot(gs[2, 1])
        reasoning_ratios = [
            m1_tokens.get('reasoning_ratio', 0),
            m2_tokens.get('reasoning_ratio', 0)
        ]
        colors = ['steelblue', 'darkorange']
        ax6.bar([model1, model2], reasoning_ratios, color=colors, alpha=0.8)
        ax6.set_ylabel('Reasoning Ratio')
        ax6.set_title('Reasoning Token Ratio')
        ax6.set_ylim(0, 1)
        ax6.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(reasoning_ratios):
            ax6.text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 7. Win/Loss/Tie summary
        ax7 = fig.add_subplot(gs[2, 2])
        
        # Compare sample by sample
        wins = {'model1': 0, 'model2': 0, 'tie': 0}
        
        for r1 in self.results_data[model1]:
            sample_id = r1['sample_id']
            # Find corresponding sample in model2
            r2 = next((r for r in self.results_data[model2] 
                      if r['sample_id'] == sample_id), None)
            if r2:
                if r1['similarity_score'] > r2['similarity_score']:
                    wins['model1'] += 1
                elif r1['similarity_score'] < r2['similarity_score']:
                    wins['model2'] += 1
                else:
                    wins['tie'] += 1
        
        labels = [f'{model1}\nWins', 'Ties', f'{model2}\nWins']
        sizes = [wins['model1'], wins['tie'], wins['model2']]
        colors_pie = ['#66b3ff', '#99ff99', '#ff9999']
        explode = (0.1, 0, 0.1)
        
        ax7.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax7.set_title('Sample-by-Sample Win/Loss/Tie\n(by Similarity Score)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        plt.show()
    
    def get_error_analysis(self, model_name: str, top_n: int = 10):
        """
        Analyze the worst performing samples for a specific model.
        """
        if model_name not in self.results_data:
            print(f"Error: Model {model_name} not found in data")
            return None
        
        results = self.results_data[model_name]
        
        # Sort by similarity score (ascending) to get worst performers
        sorted_results = sorted(results, key=lambda x: x['similarity_score'])
        
        print(f"\n{'='*80}")
        print(f"ERROR ANALYSIS FOR: {model_name}")
        print(f"{'='*80}\n")
        
        print(f"Top {top_n} Worst Performing Samples:\n")
        
        error_data = []
        for i, sample in enumerate(sorted_results[:top_n], 1):
            print(f"{i}. Sample ID: {sample['sample_id']}")
            print(f"   Category: {sample['category']}")
            print(f"   Difficulty: {sample['difficulty']}")
            print(f"   Perturbation: {sample['perturbation_type']}")
            print(f"   Similarity Score: {sample['similarity_score']:.4f}")
            print(f"   Levenshtein Distance: {sample['levenshtein_distance']}")
            print(f"   Exact Match: {sample['exact_match']}")
            print(f"   Original: {sample['original'][:100]}...")
            print(f"   Prediction: {sample['prediction'][:100]}...")
            print()
            
            error_data.append({
                'Sample ID': sample['sample_id'],
                'Category': sample['category'],
                'Difficulty': sample['difficulty'],
                'Perturbation': sample['perturbation_type'],
                'Similarity': sample['similarity_score'],
                'Levenshtein': sample['levenshtein_distance'],
                'Exact Match': sample['exact_match']
            })
        
        return pd.DataFrame(error_data)
    
    def plot_performance_trends(self, save_path: str = None):
        """
        Plot performance trends across sample IDs (if temporal ordering exists).
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Performance Trends Across Samples', fontsize=16, fontweight='bold')
        
        for model_name, results in self.results_data.items():
            # Sort by sample_id
            sorted_results = sorted(results, key=lambda x: x['sample_id'])
            sample_ids = [r['sample_id'] for r in sorted_results]
            similarities = [r['similarity_score'] for r in sorted_results]
            levenshtein = [r['levenshtein_distance'] for r in sorted_results]
            
            axes[0].plot(sample_ids, similarities, marker='o', 
                        markersize=3, alpha=0.6, label=model_name)
            axes[1].plot(sample_ids, levenshtein, marker='o', 
                        markersize=3, alpha=0.6, label=model_name)
        
        axes[0].set_ylabel('Similarity Score')
        axes[0].set_title('Similarity Score Across Samples')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0.9, 1.0)
        
        axes[1].set_xlabel('Sample ID')
        axes[1].set_ylabel('Levenshtein Distance')
        axes[1].set_title('Levenshtein Distance Across Samples')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved performance trends plot to {save_path}")
        plt.show()


def main():
    """
    Main function to run the benchmark analysis.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                  BENCHMARK RESULTS ANALYZER                           ║
    ║                                                                       ║
    ║  Comprehensive analysis and visualization tool for benchmark results  ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Example usage - modify the base_dir path to your results directory
    base_dir = input("Enter the path to your results directory (or press Enter for current directory): ").strip()
    
    if not base_dir:
        base_dir = "."
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(base_dir)
    
    # Load all results
    analyzer.load_all_results()
    
    if not analyzer.results_data:
        print("\nNo valid results found in the specified directory.")
        print("Please ensure your directory contains subfolders with results.json, metrics.json, and token_usage.json files.")
        return
    
    # Interactive menu
    while True:
        print("\n" + "="*80)
        print("ANALYSIS OPTIONS:")
        print("="*80)
        print("1. Generate comprehensive analysis (all plots + reports)")
        print("2. Show summary table")
        print("3. Plot overall performance comparison")
        print("4. Plot category performance breakdown")
        print("5. Plot token usage analysis")
        print("6. Plot similarity distribution")
        print("7. Plot Levenshtein distance distribution")
        print("8. Plot category performance heatmap")
        print("9. Plot perturbation performance")
        print("10. Plot performance trends")
        print("11. Compare two specific models")
        print("12. Error analysis for a specific model")
        print("13. Export comparison to CSV")
        print("14. Generate detailed text report")
        print("0. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (0-14): ").strip()
        
        if choice == "0":
            print("\nExiting... Goodbye!")
            break
        
        elif choice == "1":
            output_dir = input("Enter output directory name (default: 'analysis_output'): ").strip()
            if not output_dir:
                output_dir = "analysis_output"
            analyzer.create_comprehensive_analysis(output_dir)
        
        elif choice == "2":
            print("\n" + "="*80)
            print("SUMMARY TABLE")
            print("="*80)
            summary = analyzer.create_summary_table()
            print(summary.to_string(index=False))
            print()
        
        elif choice == "3":
            save = input("Save plot? (y/n): ").strip().lower()
            save_path = "overall_performance.png" if save == 'y' else None
            analyzer.plot_overall_performance(save_path)
        
        elif choice == "4":
            save = input("Save plot? (y/n): ").strip().lower()
            save_path = "category_performance.png" if save == 'y' else None
            analyzer.plot_category_performance(save_path)
        
        elif choice == "5":
            save = input("Save plot? (y/n): ").strip().lower()
            save_path = "token_usage.png" if save == 'y' else None
            analyzer.plot_token_usage(save_path)
        
        elif choice == "6":
            save = input("Save plot? (y/n): ").strip().lower()
            save_path = "similarity_distribution.png" if save == 'y' else None
            analyzer.plot_similarity_distribution(save_path)
        
        elif choice == "7":
            save = input("Save plot? (y/n): ").strip().lower()
            save_path = "levenshtein_distribution.png" if save == 'y' else None
            analyzer.plot_levenshtein_distribution(save_path)
        
        elif choice == "8":
            save = input("Save plot? (y/n): ").strip().lower()
            save_path = "category_heatmap.png" if save == 'y' else None
            analyzer.plot_heatmap_category_performance(save_path)
        
        elif choice == "9":
            save = input("Save plot? (y/n): ").strip().lower()
            save_path = "perturbation_performance.png" if save == 'y' else None
            analyzer.plot_perturbation_performance(save_path)
        
        elif choice == "10":
            save = input("Save plot? (y/n): ").strip().lower()
            save_path = "performance_trends.png" if save == 'y' else None
            analyzer.plot_performance_trends(save_path)
        
        elif choice == "11":
            models = list(analyzer.results_data.keys())
            print("\nAvailable models:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            
            if len(models) < 2:
                print("\nNeed at least 2 models for comparison.")
                continue
            
            model1_idx = int(input(f"\nSelect first model (1-{len(models)}): ")) - 1
            model2_idx = int(input(f"Select second model (1-{len(models)}): ")) - 1
            
            if 0 <= model1_idx < len(models) and 0 <= model2_idx < len(models):
                model1 = models[model1_idx]
                model2 = models[model2_idx]
                save = input("Save plot? (y/n): ").strip().lower()
                save_path = f"comparison_{model1}_vs_{model2}.png" if save == 'y' else None
                analyzer.compare_two_models(model1, model2, save_path)
            else:
                print("Invalid model selection.")
        
        elif choice == "12":
            models = list(analyzer.results_data.keys())
            print("\nAvailable models:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            
            model_idx = int(input(f"\nSelect model (1-{len(models)}): ")) - 1
            
            if 0 <= model_idx < len(models):
                model_name = models[model_idx]
                top_n = input("Number of worst samples to show (default: 10): ").strip()
                top_n = int(top_n) if top_n else 10
                
                error_df = analyzer.get_error_analysis(model_name, top_n)
                
                if error_df is not None:
                    save = input("\nSave error analysis to CSV? (y/n): ").strip().lower()
                    if save == 'y':
                        filename = f"error_analysis_{model_name}.csv"
                        error_df.to_csv(filename, index=False)
                        print(f"Error analysis saved to {filename}")
            else:
                print("Invalid model selection.")
        
        elif choice == "13":
            filename = input("Enter CSV filename (default: 'model_comparison.csv'): ").strip()
            if not filename:
                filename = "model_comparison.csv"
            analyzer.export_comparison_csv(filename)
        
        elif choice == "14":
            filename = input("Enter report filename (default: 'analysis_report.txt'): ").strip()
            if not filename:
                filename = "analysis_report.txt"
            analyzer.generate_detailed_report(filename)
        
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


# Additional utility functions

def batch_compare_models(base_dir: str, output_dir: str = "batch_comparison"):
    """
    Batch comparison utility - compares all pairs of models.
    
    Args:
        base_dir: Directory containing result folders
        output_dir: Directory to save comparison outputs
    """
    analyzer = BenchmarkAnalyzer(base_dir)
    analyzer.load_all_results()
    
    models = list(analyzer.results_data.keys())
    
    if len(models) < 2:
        print("Need at least 2 models for comparison.")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nGenerating pairwise comparisons for {len(models)} models...")
    print(f"Total comparisons: {len(models) * (len(models) - 1) // 2}\n")
    
    comparison_count = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1 = models[i]
            model2 = models[j]
            
            comparison_count += 1
            print(f"[{comparison_count}] Comparing {model1} vs {model2}...")
            
            save_path = output_path / f"comparison_{model1}_vs_{model2}.png"
            analyzer.compare_two_models(model1, model2, str(save_path))
    
    print(f"\nAll comparisons saved to {output_path.absolute()}")


def generate_leaderboard(base_dir: str, output_file: str = "leaderboard.md"):
    """
    Generate a markdown leaderboard from benchmark results.
    
    Args:
        base_dir: Directory containing result folders
        output_file: Output markdown file
    """
    analyzer = BenchmarkAnalyzer(base_dir)
    analyzer.load_all_results()
    
    summary_df = analyzer.create_summary_table()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 🏆 Benchmark Leaderboard\n\n")
        f.write(f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## 📊 Overall Rankings\n\n")
        f.write("| Rank | Model | Exact Match Rate | Avg Similarity | Avg Levenshtein | Valid Samples |\n")
        f.write("|------|-------|------------------|----------------|-----------------|---------------|\n")
        
        for idx, row in summary_df.iterrows():
            rank = idx + 1
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
            f.write(f"| {medal} | {row['Model']} | {row['Exact Match Rate']:.2%} | "
                   f"{row['Avg Similarity']:.4f} | {row['Avg Levenshtein Dist']:.2f} | "
                   f"{int(row['Valid Samples'])} |\n")
        
        f.write("\n## 📈 Detailed Metrics\n\n")
        
        # Category breakdown
        f.write("### Performance by Category\n\n")
        categories = ['unethical_requests', 'harmful_instructions', 
                     'jailbreak_attempts', 'privacy_violations', 
                     'manipulation_attempts']
        
        for category in categories:
            f.write(f"\n#### {category.replace('_', ' ').title()}\n\n")
            f.write("| Model | Exact Match Rate | Avg Similarity | Count |\n")
            f.write("|-------|------------------|-------------------|-------|\n")
            
            cat_data = []
            for model in analyzer.metrics_data.keys():
                cat_metrics = analyzer.metrics_data[model].get('by_category', {}).get(category, {})
                cat_data.append({
                    'Model': model,
                    'EM': cat_metrics.get('exact_match_rate', 0),
                    'Sim': cat_metrics.get('avg_similarity', 0),
                    'Count': cat_metrics.get('count', 0)
                })
            
            cat_data = sorted(cat_data, key=lambda x: x['EM'], reverse=True)
            
            for data in cat_data:
                f.write(f"| {data['Model']} | {data['EM']:.2%} | "
                       f"{data['Sim']:.4f} | {data['Count']} |\n")
        
        # Token usage
        f.write("\n### Token Usage Statistics\n\n")
        f.write("| Model | Avg Total Tokens | Avg Reasoning Tokens | Reasoning Ratio |\n")
        f.write("|-------|------------------|----------------------|-----------------|\n")
        
        for model in analyzer.token_usage_data.keys():
            total_usage = analyzer.token_usage_data[model].get('total', {})
            f.write(f"| {model} | {total_usage.get('avg_total_tokens', 0):.0f} | "
                   f"{total_usage.get('avg_reasoning_tokens', 0):.0f} | "
                   f"{total_usage.get('reasoning_ratio', 0):.2%} |\n")
        
        f.write("\n---\n")
        f.write("\n*End of Leaderboard*\n")
    
    print(f"Leaderboard saved to {output_file}")


if __name__ == "__main__":
    # Run the main interactive analyzer
    main()
    
    # Uncomment below to run batch comparison or generate leaderboard
    # batch_compare_models("path/to/results", "batch_comparison_output")
    # generate_leaderboard("path/to/results", "leaderboard.md")