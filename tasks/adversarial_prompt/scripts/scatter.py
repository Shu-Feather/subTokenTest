import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


class AccuracyTokenCostVisualizer:
    """
    Visualizer for model accuracy vs token cost analysis.
    Creates scatter plots showing the trade-off between accuracy and computational cost.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the visualizer with a base directory containing result folders.
        
        Args:
            base_dir: Path to directory containing result folders
        """
        self.base_dir = Path(base_dir)
        self.results_data = {}
        self.metrics_data = {}
        self.token_usage_data = {}
        
        # Define color palette for categories
        self.category_colors = {
            'unethical_requests': '#FF6B6B',      # Red
            'harmful_instructions': '#4ECDC4',    # Teal
            'jailbreak_attempts': '#FFD93D',      # Yellow
            'privacy_violations': '#95E1D3',      # Mint
            'manipulation_attempts': '#A8E6CF',   # Light green
            'benchmark_average': '#888888'        # Gray for average
        }
        
        # Define marker styles for different model types
        self.model_markers = {
            'gpt-5': 'o',                    # Circle
            'gpt-4': 's',                    # Square
            'deepseek-reasoner': '^',        # Triangle up
            'deepseek-chat': 'D',            # Diamond
            'o4-mini': 'p',                  # Pentagon
            'claude': 'v',                   # Triangle down
            'gemini': 'h',                   # Hexagon
            'llama': '*',                    # Star
            'qwen': '<',                     # Triangle left
            'default': 'X'                   # X marker
        }
        
    def extract_model_name(self, folder_name: str) -> str:
        """
        Extract model name from folder name (e.g., 'gpt-4o_20251205_055624' -> 'gpt-4o').
        
        Args:
            folder_name: Full folder name
            
        Returns:
            Extracted model name
        """
        # Split by underscore and take the first part (before timestamp)
        parts = folder_name.split('_')
        if len(parts) > 0:
            return parts[0]
        return folder_name
    
    def get_model_type(self, model_name: str) -> str:
        """
        Determine model type from model name for marker assignment.
        
        Args:
            model_name: Model name (e.g., 'gpt-4o', 'deepseek-chat')
            
        Returns:
            Model type for marker selection
        """
        model_lower = model_name.lower()
        
        # Check for specific patterns in order of specificity
        if 'deepseek-reasoner' in model_lower or 'deepseek-r1' in model_lower:
            return 'deepseek-reasoner'
        elif 'deepseek-chat' in model_lower or 'deepseek-v3' in model_lower:
            return 'deepseek-chat'
        elif 'gpt-5' in model_lower or 'gpt5' in model_lower:
            return 'gpt-5'
        elif 'gpt-4' in model_lower or 'gpt4' in model_lower:
            return 'gpt-4'
        elif 'o4-mini' in model_lower or 'o4mini' in model_lower:
            return 'o4-mini'
        elif 'claude' in model_lower:
            return 'claude'
        elif 'gemini' in model_lower:
            return 'gemini'
        elif 'llama' in model_lower:
            return 'llama'
        elif 'qwen' in model_lower:
            return 'qwen'
        
        return 'default'
    
    def load_all_results(self):
        """
        Load all results from subdirectories in base_dir.
        """
        print("Loading benchmark results for visualization...")
        
        for folder in self.base_dir.iterdir():
            if not folder.is_dir():
                continue
            
            folder_name = folder.name
            model_name = self.extract_model_name(folder_name)
            
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
            
            # Load metrics.json
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    self.metrics_data[model_name] = json.load(f)
            
            # Load token_usage.json
            if token_usage_file.exists():
                with open(token_usage_file, 'r', encoding='utf-8') as f:
                    self.token_usage_data[model_name] = json.load(f)
        
        print(f"Loaded data for {len(self.results_data)} models")
        for model in self.results_data.keys():
            print(f"  - {model}: {len(self.results_data[model])} valid samples")
        print()
    
    def prepare_plot_data(self) -> pd.DataFrame:
        """
        Prepare data for plotting by extracting relevant metrics for each model and category.
        
        Returns:
            DataFrame with columns: model, category, exact_match_rate, avg_total_tokens, reasoning_ratio
        """
        plot_data = []
        
        categories = ['unethical_requests', 'harmful_instructions', 
                     'jailbreak_attempts', 'privacy_violations', 
                     'manipulation_attempts']
        
        for model_name in self.metrics_data.keys():
            metrics = self.metrics_data[model_name]
            token_usage = self.token_usage_data[model_name]
            
            # Extract data for each category
            for category in categories:
                category_metrics = metrics.get('by_category', {}).get(category, {})
                category_tokens = token_usage.get('by_category', {}).get(category, {})
                
                exact_match_rate = category_metrics.get('exact_match_rate', 0)
                avg_total_tokens = category_tokens.get('avg_total_tokens', 0)
                
                # Get reasoning ratio (from total, as it's typically consistent across categories)
                reasoning_ratio = token_usage.get('total', {}).get('reasoning_ratio', 0)
                
                if avg_total_tokens > 0:  # Only include if we have valid token data
                    plot_data.append({
                        'model': model_name,
                        'category': category,
                        'exact_match_rate': exact_match_rate,
                        'avg_total_tokens': avg_total_tokens,
                        'reasoning_ratio': reasoning_ratio,
                        'model_type': self.get_model_type(model_name)
                    })
            
            # Add benchmark average for each model
            overall_exact_match = metrics.get('exact_match_rate', 0)
            overall_avg_tokens = token_usage.get('total', {}).get('avg_total_tokens', 0)
            reasoning_ratio = token_usage.get('total', {}).get('reasoning_ratio', 0)
            
            if overall_avg_tokens > 0:
                plot_data.append({
                    'model': model_name,
                    'category': 'benchmark_average',
                    'exact_match_rate': overall_exact_match,
                    'avg_total_tokens': overall_avg_tokens,
                    'reasoning_ratio': reasoning_ratio,
                    'model_type': self.get_model_type(model_name)
                })
        
        df = pd.DataFrame(plot_data)
        return df
    
    def plot_accuracy_vs_token_cost(self, save_path: str = None, 
                                   figsize: Tuple[int, int] = (14, 8),
                                   show_average_line: bool = True):
        """
        Create the main scatter plot: Accuracy vs Token Cost.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            show_average_line: Whether to show the benchmark average line connecting models
        """
        df = self.prepare_plot_data()
        
        if df.empty:
            print("No data available for plotting.")
            return
        
        # Create figure with space for legends on the right
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate point sizes based on reasoning_ratio
        # Base size for reasoning_ratio = 0, then scale linearly
        base_size = 100
        max_size = 500
        size_range = max_size - base_size
        
        # Separate benchmark average data from category data
        df_categories = df[df['category'] != 'benchmark_average']
        df_average = df[df['category'] == 'benchmark_average']
        
        # Plot category data
        for category in df_categories['category'].unique():
            category_data = df_categories[df_categories['category'] == category]
            
            for model_type in category_data['model_type'].unique():
                model_data = category_data[category_data['model_type'] == model_type]
                
                # Calculate sizes: base_size + (reasoning_ratio * size_range)
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Plot with log scale on x-axis
                ax.scatter(
                    model_data['avg_total_tokens'],
                    model_data['exact_match_rate'],
                    c=self.category_colors[category],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=2
                )
        
        # Plot benchmark average points
        if not df_average.empty and show_average_line:
            for model_type in df_average['model_type'].unique():
                model_data = df_average[df_average['model_type'] == model_type]
                
                # Calculate sizes for average points
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Plot average points in gray
                ax.scatter(
                    model_data['avg_total_tokens'],
                    model_data['exact_match_rate'],
                    c=self.category_colors['benchmark_average'],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=3
                )
            
            # Sort average data by avg_total_tokens for line plot
            df_average_sorted = df_average.sort_values('avg_total_tokens')
            
            # Draw connecting line through benchmark average points
            ax.plot(
                df_average_sorted['avg_total_tokens'],
                df_average_sorted['exact_match_rate'],
                color='gray',
                linestyle='--',
                linewidth=2,
                alpha=0.4,
                zorder=1,
                label='Benchmark Average Trend'
            )
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        
        # Labels and title
        ax.set_xlabel('Average Total Tokens (log scale)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Exact Match Rate', fontsize=13, fontweight='bold')
        ax.set_title('Model Accuracy vs Token Cost Trade-off', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Set y-axis limits and formatting
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Create custom legends
        self._create_legends(ax, df)
        
        # Adjust layout to accommodate legends
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _create_legends(self, ax, df):
        """
        Create custom legends for categories, model types, and reasoning ratios.
        
        Args:
            ax: Matplotlib axes object
            df: DataFrame with plot data
        """
        legend_y_start = 1.0
        legend_spacing = 0.30
        
        # Legend 1: Categories (colors) - including benchmark average
        category_patches = []
        
        # Add regular categories
        regular_categories = ['unethical_requests', 'harmful_instructions', 
                             'jailbreak_attempts', 'privacy_violations', 
                             'manipulation_attempts']
        
        for category in regular_categories:
            if category in df['category'].values:
                color = self.category_colors[category]
                label = category.replace('_', ' ').title()
                # Shorten labels
                label = (label.replace('Requests', 'Req.')
                        .replace('Instructions', 'Instr.')
                        .replace('Attempts', 'Att.')
                        .replace('Violations', 'Viol.'))
                category_patches.append(mpatches.Patch(color=color, label=label))
        
        # Add benchmark average
        if 'benchmark_average' in df['category'].values:
            category_patches.append(
                mpatches.Patch(color=self.category_colors['benchmark_average'], 
                             label='Benchmark Avg.', alpha=0.6)
            )
        
        legend1 = ax.legend(handles=category_patches, 
                          title='Categories',
                          loc='upper left',
                          bbox_to_anchor=(1.02, legend_y_start),
                          frameon=True,
                          fontsize=9,
                          title_fontsize=10,
                          borderaxespad=0)
        ax.add_artist(legend1)
        
        # Legend 2: Model Types (markers)
        model_type_lines = []
        unique_model_types = sorted(df['model_type'].unique())
        
        for model_type in unique_model_types:
            marker = self.model_markers.get(model_type, 'X')
            # Format label
            if model_type == 'default':
                label = 'Other'
            else:
                label = model_type.upper().replace('-', ' ').title()
            
            model_type_lines.append(Line2D([0], [0], marker=marker, color='gray', 
                                          linestyle='', markersize=9, 
                                          markeredgewidth=1.3, markeredgecolor='black',
                                          label=label, alpha=0.7))
        
        legend2 = ax.legend(handles=model_type_lines,
                          title='Model Types',
                          loc='upper left',
                          bbox_to_anchor=(1.02, legend_y_start - legend_spacing),
                          frameon=True,
                          fontsize=9,
                          title_fontsize=10,
                          borderaxespad=0)
        ax.add_artist(legend2)
        
        # Legend 3: Reasoning Ratio (sizes)
        base_size = 100
        max_size = 500
        size_range = max_size - base_size
        
        # Show three size examples
        reasoning_ratios = [0, 0.5, 1.0]
        reasoning_lines = []
        
        for ratio in reasoning_ratios:
            size = base_size + (ratio * size_range)
            reasoning_lines.append(Line2D([0], [0], marker='o', color='gray',
                                        linestyle='', markersize=np.sqrt(size/10),
                                        markeredgewidth=1.3, markeredgecolor='black',
                                        label=f'{ratio:.0%}', alpha=0.7))
        
        legend3 = ax.legend(handles=reasoning_lines,
                          title='Reasoning Ratio',
                          loc='upper left',
                          bbox_to_anchor=(1.02, legend_y_start - 2 * legend_spacing),
                          frameon=True,
                          fontsize=9,
                          title_fontsize=10,
                          borderaxespad=0)
        ax.add_artist(legend3)


def main():
    """
    Main function for creating the visualization.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║         MODEL ACCURACY VS TOKEN COST VISUALIZER                          ║
    ║                                                                          ║
    ║  Visualize the trade-off between model accuracy and token consumption    ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get base directory
    base_dir = input("Enter the path to your results directory (or press Enter for current directory): ").strip()
    
    if not base_dir:
        base_dir = "."
    
    # Initialize visualizer
    visualizer = AccuracyTokenCostVisualizer(base_dir)
    
    # Load data
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("\nNo valid results found in the specified directory.")
        return
    
    # Ask for options
    show_avg = input("\nShow benchmark average line? (y/n, default: y): ").strip().lower()
    show_average_line = show_avg == '' or show_avg == 'y'
    
    # Ask for output path
    save_plot = input("Save plot? (y/n, default: y): ").strip().lower()
    
    if save_plot == '' or save_plot == 'y':
        output_file = input("Enter output filename (default: 'accuracy_vs_token_cost.png'): ").strip()
        if not output_file:
            output_file = "accuracy_vs_token_cost.png"
        save_path = output_file
    else:
        save_path = None
    
    # Generate the plot
    print("\nGenerating accuracy vs token cost visualization...")
    visualizer.plot_accuracy_vs_token_cost(save_path=save_path, 
                                          show_average_line=show_average_line)
    
    print("\nVisualization complete!")


def quick_generate(base_dir: str, 
                  output_file: str = "accuracy_vs_token_cost.png",
                  figsize: Tuple[int, int] = (14, 8),
                  show_average_line: bool = True):
    """
    Quick function to generate the plot with one command.
    
    Args:
        base_dir: Directory containing result folders
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        show_average_line: Whether to show benchmark average line
    
    Example:
        quick_generate("./results", "my_plot.png")
        quick_generate("./results", "plot.png", figsize=(18, 10), show_average_line=True)
    """
    visualizer = AccuracyTokenCostVisualizer(base_dir)
    visualizer.load_all_results()
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file, 
                                          figsize=figsize,
                                          show_average_line=show_average_line)


def customize_visualization(base_dir: str, 
                           output_file: str = "accuracy_vs_token_cost.png",
                           figsize: Tuple[int, int] = (14, 8),
                           custom_colors: Dict[str, str] = None,
                           custom_markers: Dict[str, str] = None,
                           show_average_line: bool = True):
    """
    Generate plot with custom colors and markers.
    
    Args:
        base_dir: Directory containing result folders
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        custom_colors: Custom color mapping for categories
        custom_markers: Custom marker mapping for model types
        show_average_line: Whether to show benchmark average line
    
    Example:
        custom_colors = {
            'harmful_instructions': '#FF0000',
            'jailbreak_attempts': '#00FF00',
            'benchmark_average': '#666666'
        }
        custom_markers = {
            'gpt-4': 'o',
            'deepseek-chat': '^'
        }
        customize_visualization("./results", "custom.png", 
                              custom_colors=custom_colors, 
                              custom_markers=custom_markers)
    """
    visualizer = AccuracyTokenCostVisualizer(base_dir)
    
    # Override colors if provided
    if custom_colors:
        visualizer.category_colors.update(custom_colors)
    
    # Override markers if provided
    if custom_markers:
        visualizer.model_markers.update(custom_markers)
    
    visualizer.load_all_results()
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file, 
                                          figsize=figsize,
                                          show_average_line=show_average_line)


def export_plot_data(base_dir: str, output_file: str = "plot_data.csv"):
    """
    Export the prepared plot data to CSV for external analysis.
    
    Args:
        base_dir: Directory containing result folders
        output_file: Output CSV file path
    
    Example:
        export_plot_data("./results", "my_data.csv")
    """
    visualizer = AccuracyTokenCostVisualizer(base_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    if df.empty:
        print("No data available to export.")
        return None
    
    df.to_csv(output_file, index=False)
    print(f"Plot data exported to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"Total data points: {len(df)}")
    print(f"Number of models: {df['model'].nunique()}")
    print(f"Categories: {', '.join(df['category'].unique())}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    
    # Summary for benchmark averages
    df_avg = df[df['category'] == 'benchmark_average']
    if not df_avg.empty:
        print("\n" + "-"*80)
        print("BENCHMARK AVERAGE SUMMARY")
        print("-"*80)
        avg_summary = df_avg[['model', 'exact_match_rate', 'avg_total_tokens', 'reasoning_ratio']].copy()
        avg_summary.columns = ['Model', 'Avg EM Rate', 'Avg Total Tokens', 'Reasoning Ratio']
        avg_summary = avg_summary.sort_values('Avg EM Rate', ascending=False)
        print(avg_summary.to_string(index=False))
    
    print("\n" + "-"*80)
    print("PER-CATEGORY SUMMARY")
    print("-"*80)
    df_categories = df[df['category'] != 'benchmark_average']
    summary = df_categories.groupby(['model', 'category']).agg({
        'exact_match_rate': 'mean',
        'avg_total_tokens': 'mean'
    }).round(4)
    print(summary)
    print("="*80 + "\n")
    
    return df


def compare_models_summary(base_dir: str):
    """
    Print a comparison summary of all models based on benchmark averages.
    
    Args:
        base_dir: Directory containing result folders
    
    Example:
        compare_models_summary("./results")
    """
    visualizer = AccuracyTokenCostVisualizer(base_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    # Filter for benchmark averages only
    df_avg = df[df['category'] == 'benchmark_average'].copy()
    
    if df_avg.empty:
        print("No benchmark average data available.")
        return
    
    # Sort by exact match rate
    df_avg = df_avg.sort_values('exact_match_rate', ascending=False)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON - BENCHMARK AVERAGES")
    print("="*100)
    print(f"{'Rank':<6} {'Model':<25} {'Avg EM Rate':<15} {'Avg Tokens':<15} {'Reasoning Ratio':<18} {'Model Type':<20}")
    print("-"*100)
    
    for idx, (_, row) in enumerate(df_avg.iterrows(), 1):
        print(f"{idx:<6} {row['model']:<25} {row['exact_match_rate']:<15.2%} "
              f"{row['avg_total_tokens']:<15.1f} {row['reasoning_ratio']:<18.2%} {row['model_type']:<20}")
    
    print("="*100)
    
    # Calculate efficiency metric
    df_avg['efficiency'] = df_avg['exact_match_rate'] / df_avg['avg_total_tokens'] * 1000
    df_avg_eff = df_avg.sort_values('efficiency', ascending=False)
    
    print("\n" + "="*100)
    print("EFFICIENCY RANKING (Accuracy per 1000 Tokens)")
    print("="*100)
    print(f"{'Rank':<6} {'Model':<25} {'Efficiency':<15} {'Avg EM Rate':<15} {'Avg Tokens':<15}")
    print("-"*100)
    
    for idx, (_, row) in enumerate(df_avg_eff.iterrows(), 1):
        print(f"{idx:<6} {row['model']:<25} {row['efficiency']:<15.4f} "
              f"{row['exact_match_rate']:<15.2%} {row['avg_total_tokens']:<15.1f}")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    # Run interactive main function
    main()

    # Uncomment below for quick standalone usage examples:
    
    # Example 1: Quick generation with default settings
    # quick_generate("./results", "accuracy_vs_token_cost.png")
    
    # Example 2: Generate without average line
    # quick_generate("./results", "plot_no_avg.png", show_average_line=False)
    
    # Example 3: Generate larger plot with average line
    # quick_generate("./results", "large_plot.png", figsize=(18, 10), show_average_line=True)
    
    # Example 4: Custom colors including benchmark average
    # custom_colors = {
    #     'harmful_instructions': '#E74C3C',
    #     'jailbreak_attempts': '#3498DB',
    #     'benchmark_average': '#555555'
    # }
    # customize_visualization("./results", "custom_plot.png", 
    #                        custom_colors=custom_colors,
    #                        show_average_line=True)
    
    # Example 5: Export data to CSV
    # export_plot_data("./results", "analysis_data.csv")
    
    # Example 6: Print model comparison summary
    # compare_models_summary("./results")