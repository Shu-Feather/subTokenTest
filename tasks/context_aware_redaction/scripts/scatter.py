import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
import re

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


class SensitiveDataRedactionVisualizer:
    """
    Visualizer for sensitive data redaction benchmark results.
    Creates scatter plots showing model accuracy vs token cost trade-off.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize the visualizer with a directory containing JSON result files.
        
        Args:
            results_dir: Path to directory containing result JSON files
        """
        self.results_dir = Path(results_dir)
        self.results_data = []
        
        # Define color palette for different task types/difficulties
        self.task_colors = {
            'short': '#FF6B6B',       # Red
            'medium': '#4ECDC4',      # Teal
            'long': '#FFD93D',        # Yellow
            'mixed': '#95E1D3',       # Mint
            'default': '#A8E6CF',     # Light green
            'benchmark_average': '#888888'  # Gray for average
        }
        
        # Define marker styles for different model families
        self.model_markers = {
            'gpt-4': 's',                    # Square
            'gpt-5': 'o',                    # Circle
            'deepseek-reasoner': '^',        # Triangle up
            'deepseek-chat': 'D',            # Diamond
            'deepseek-v3': 'D',              # Diamond
            'o1': 'p',                       # Pentagon
            'o4-mini': 'p',                  # Pentagon
            'claude': 'v',                   # Triangle down
            'gemini': 'h',                   # Hexagon
            'llama': '*',                    # Star
            'qwen': '<',                     # Triangle left
            'default': 'X'                   # X marker
        }
    
    def extract_model_name(self, full_name: str) -> str:
        """
        Extract simplified model name (e.g., 'deepseek-v3', 'gpt-4').
        
        Args:
            full_name: Full model name from JSON
            
        Returns:
            Simplified model name
        """
        # Common patterns to match
        patterns = [
            r'(gpt-[0-9](?:[a-z0-9-]*)?)',
            r'(o[0-9](?:-[a-z]+)?)',
            r'(deepseek-(?:v[0-9]+|chat|reasoner|r[0-9]+))',
            r'(claude-[0-9](?:[a-z0-9-]*)?)',
            r'(gemini-[a-z0-9-]+)',
            r'(llama-[0-9]+)',
            r'(qwen-[0-9]+)',
        ]
        
        full_name_lower = full_name.lower()
        
        for pattern in patterns:
            match = re.search(pattern, full_name_lower)
            if match:
                return match.group(1)
        
        # Fallback: take first two parts separated by dash or underscore
        parts = re.split(r'[-_]', full_name_lower)
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        
        return full_name_lower
    
    def get_model_type(self, model_name: str) -> str:
        """
        Determine model family from model name for marker assignment.
        
        Args:
            model_name: Simplified model name
            
        Returns:
            Model family type
        """
        model_lower = model_name.lower()
        
        # Check for specific patterns in order of specificity
        if 'deepseek-reasoner' in model_lower or 'deepseek-r' in model_lower:
            return 'deepseek-reasoner'
        elif 'deepseek-v3' in model_lower or 'deepseek-v' in model_lower:
            return 'deepseek-v3'
        elif 'deepseek-chat' in model_lower or 'deepseek-c' in model_lower:
            return 'deepseek-chat'
        elif 'gpt-5' in model_lower:
            return 'gpt-5'
        elif 'gpt-4' in model_lower:
            return 'gpt-4'
        elif 'o4-mini' in model_lower:
            return 'o4-mini'
        elif 'o1' in model_lower or 'o-1' in model_lower:
            return 'o1'
        elif 'claude' in model_lower:
            return 'claude'
        elif 'gemini' in model_lower:
            return 'gemini'
        elif 'llama' in model_lower:
            return 'llama'
        elif 'qwen' in model_lower:
            return 'qwen'
        
        return 'default'
    
    def get_task_type(self, data: Dict) -> str:
        """
        Extract task type/difficulty from result data.
        
        Args:
            data: Result JSON data
            
        Returns:
            Task type/difficulty
        """
        # Try to get from difficulty_metrics keys
        difficulty_metrics = data.get('difficulty_metrics', {})
        if difficulty_metrics:
            # Return the first key (usually the difficulty level)
            return list(difficulty_metrics.keys())[0]
        
        # Fallback to 'default'
        return 'default'
    
    def load_all_results(self):
        """
        Load all JSON result files from the results directory.
        """
        print("Loading benchmark results for visualization...")
        
        json_files = list(self.results_dir.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {self.results_dir}")
            return
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.results_data.append({
                        'filename': json_file.name,
                        'data': data
                    })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.results_data)} result files")
        print()
    
    def prepare_plot_data(self) -> pd.DataFrame:
        """
        Prepare data for plotting by extracting relevant metrics.
        
        Returns:
            DataFrame with columns: model, task_type, exact_match_rate, 
                                   avg_total_tokens, reasoning_ratio, model_type
        """
        plot_data = []
        
        # Group by model to calculate benchmark averages
        model_data_dict = {}
        
        for result in self.results_data:
            data = result['data']
            
            # Extract basic information
            full_model_name = data.get('model', 'unknown')
            model_name = self.extract_model_name(full_model_name)
            task_type = self.get_task_type(data)
            
            # Extract metrics
            exact_match_rate = data.get('overall_metrics', {}).get('exact_match_rate', 0)
            
            # Extract token usage
            token_stats = data.get('token_usage_stats', {})
            avg_total_tokens = token_stats.get('avg_tokens_per_sample', 0)
            avg_reasoning_tokens = token_stats.get('avg_reasoning_tokens_per_sample', 0)
            
            # Calculate reasoning ratio
            reasoning_ratio = (avg_reasoning_tokens / avg_total_tokens) if avg_total_tokens > 0 else 0
            
            # Get model type for marker
            model_type = self.get_model_type(model_name)
            
            # Add to plot data
            if avg_total_tokens > 0:  # Only include if we have valid token data
                data_point = {
                    'model': model_name,
                    'task_type': task_type,
                    'exact_match_rate': exact_match_rate,
                    'avg_total_tokens': avg_total_tokens,
                    'reasoning_ratio': reasoning_ratio,
                    'model_type': model_type
                }
                plot_data.append(data_point)
                
                # Store for benchmark average calculation
                if model_name not in model_data_dict:
                    model_data_dict[model_name] = []
                model_data_dict[model_name].append(data_point)
        
        # Calculate benchmark averages for each model
        for model_name, data_points in model_data_dict.items():
            if len(data_points) > 0:
                avg_exact_match = np.mean([d['exact_match_rate'] for d in data_points])
                avg_tokens = np.mean([d['avg_total_tokens'] for d in data_points])
                avg_reasoning_ratio = np.mean([d['reasoning_ratio'] for d in data_points])
                model_type = data_points[0]['model_type']
                
                plot_data.append({
                    'model': model_name,
                    'task_type': 'benchmark_average',
                    'exact_match_rate': avg_exact_match,
                    'avg_total_tokens': avg_tokens,
                    'reasoning_ratio': avg_reasoning_ratio,
                    'model_type': model_type
                })
        
        df = pd.DataFrame(plot_data)
        return df
    
    def plot_accuracy_vs_token_cost(self, 
                                   save_path: str = None,
                                   figsize: Tuple[int, int] = (16, 9),
                                   show_average_line: bool = True,
                                   add_model_labels: bool = False):
        """
        Create the main scatter plot: Accuracy vs Token Cost.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            show_average_line: Whether to show the benchmark average line
            add_model_labels: Whether to add model name labels to points
        """
        df = self.prepare_plot_data()
        
        if df.empty:
            print("No data available for plotting.")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate point sizes based on reasoning_ratio
        base_size = 100
        max_size = 600
        size_range = max_size - base_size
        
        # Separate benchmark average data from task-specific data
        df_tasks = df[df['task_type'] != 'benchmark_average']
        df_average = df[df['task_type'] == 'benchmark_average']
        
        # Plot task-specific data points
        for task_type in df_tasks['task_type'].unique():
            task_data = df_tasks[df_tasks['task_type'] == task_type]
            
            for model_type in task_data['model_type'].unique():
                model_data = task_data[task_data['model_type'] == model_type]
                
                # Calculate sizes
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Plot with log scale on x-axis
                ax.scatter(
                    model_data['avg_total_tokens'],
                    model_data['exact_match_rate'],
                    c=self.task_colors.get(task_type, self.task_colors['default']),
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=3,
                    label=f"{task_type}_{model_type}"  # For debugging
                )
        
        # Plot benchmark average points and connecting line
        if not df_average.empty and show_average_line:
            # Sort by avg_total_tokens for smooth line
            df_average_sorted = df_average.sort_values('avg_total_tokens')
            
            # Draw connecting line (semi-transparent gray)
            ax.plot(
                df_average_sorted['avg_total_tokens'],
                df_average_sorted['exact_match_rate'],
                color='gray',
                linestyle='--',
                linewidth=2.5,
                alpha=0.35,
                zorder=1,
                label='Benchmark Average Trend'
            )
            
            # Plot average points
            for model_type in df_average['model_type'].unique():
                model_data = df_average[df_average['model_type'] == model_type]
                
                # Calculate sizes
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Plot in gray
                scatter = ax.scatter(
                    model_data['avg_total_tokens'],
                    model_data['exact_match_rate'],
                    c=self.task_colors['benchmark_average'],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.5,
                    edgecolors='black',
                    linewidths=2,
                    zorder=2
                )
                
                # Add model name labels if requested
                if add_model_labels:
                    for _, row in model_data.iterrows():
                        ax.annotate(
                            row['model'],
                            (row['avg_total_tokens'], row['exact_match_rate']),
                            xytext=(8, 8),
                            textcoords='offset points',
                            fontsize=8,
                            alpha=0.7,
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', 
                                    edgecolor='gray', 
                                    alpha=0.7)
                        )
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        
        # Labels and title
        ax.set_xlabel('log(Avg Total Tokens per Sample)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Exact Match Rate', fontsize=13, fontweight='bold')
        ax.set_title('Model Accuracy vs Token Cost Trade-off\n(Sensitive Data Redaction Benchmark)', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Set y-axis limits and formatting
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # Create custom legends
        self._create_legends(ax, df)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _create_legends(self, ax, df):
        """
        Create custom legends for task types, model types, and reasoning ratios.
        
        Args:
            ax: Matplotlib axes object
            df: DataFrame with plot data
        """
        legend_y_start = 1.0
        legend_spacing = 0.28
        
        # Legend 1: Task Types (colors)
        task_patches = []
        
        # Get unique task types (excluding benchmark_average)
        unique_tasks = sorted([t for t in df['task_type'].unique() 
                              if t != 'benchmark_average'])
        
        for task_type in unique_tasks:
            color = self.task_colors.get(task_type, self.task_colors['default'])
            label = task_type.replace('_', ' ').title()
            task_patches.append(mpatches.Patch(color=color, label=label, alpha=0.7))
        
        # Add benchmark average
        if 'benchmark_average' in df['task_type'].values:
            task_patches.append(
                mpatches.Patch(
                    color=self.task_colors['benchmark_average'],
                    label='Benchmark Avg.',
                    alpha=0.5
                )
            )
        
        legend1 = ax.legend(
            handles=task_patches,
            title='Task Type',
            loc='upper left',
            bbox_to_anchor=(1.02, legend_y_start),
            frameon=True,
            fontsize=9,
            title_fontsize=11,
            borderaxespad=0,
            edgecolor='gray',
            fancybox=True,
            shadow=True
        )
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
                label = model_type.replace('-', ' ').replace('_', ' ').title()
                # Clean up common patterns
                label = (label.replace('Gpt', 'GPT')
                        .replace('Deepseek', 'DeepSeek')
                        .replace('V3', 'V3')
                        .replace('O1', 'O1')
                        .replace('O4', 'O4'))
            
            model_type_lines.append(
                Line2D([0], [0], 
                      marker=marker, 
                      color='gray',
                      linestyle='',
                      markersize=9,
                      markeredgewidth=1.5,
                      markeredgecolor='black',
                      label=label,
                      alpha=0.7)
            )
        
        legend2 = ax.legend(
            handles=model_type_lines,
            title='Model Family',
            loc='upper left',
            bbox_to_anchor=(1.02, legend_y_start - legend_spacing),
            frameon=True,
            fontsize=9,
            title_fontsize=11,
            borderaxespad=0,
            edgecolor='gray',
            fancybox=True,
            shadow=True
        )
        ax.add_artist(legend2)
        
        # Legend 3: Reasoning Ratio (sizes)
        base_size = 100
        max_size = 600
        size_range = max_size - base_size
        
        # Show representative size examples
        reasoning_ratios = [0, 0.25, 0.5, 0.75, 1.0]
        reasoning_lines = []
        
        for ratio in reasoning_ratios:
            size = base_size + (ratio * size_range)
            reasoning_lines.append(
                Line2D([0], [0],
                      marker='o',
                      color='gray',
                      linestyle='',
                      markersize=np.sqrt(size/10),
                      markeredgewidth=1.5,
                      markeredgecolor='black',
                      label=f'{ratio:.2f}',
                      alpha=0.7)
            )
        
        legend3 = ax.legend(
            handles=reasoning_lines,
            title='Reasoning Ratio',
            loc='upper left',
            bbox_to_anchor=(1.02, legend_y_start - 2 * legend_spacing),
            frameon=True,
            fontsize=9,
            title_fontsize=11,
            borderaxespad=0,
            edgecolor='gray',
            fancybox=True,
            shadow=True
        )
        ax.add_artist(legend3)
    
    def print_summary_statistics(self):
        """
        Print summary statistics of the loaded data.
        """
        df = self.prepare_plot_data()
        
        if df.empty:
            print("No data available.")
            return
        
        print("\n" + "="*100)
        print("BENCHMARK SUMMARY STATISTICS")
        print("="*100)
        
        # Overall statistics
        print(f"\nTotal data points: {len(df)}")
        print(f"Number of models: {df['model'].nunique()}")
        print(f"Task types: {', '.join(sorted(df['task_type'].unique()))}")
        print(f"Model families: {', '.join(sorted(df['model_type'].unique()))}")
        
        # Benchmark average statistics
        df_avg = df[df['task_type'] == 'benchmark_average'].copy()
        
        if not df_avg.empty:
            print("\n" + "-"*100)
            print("BENCHMARK AVERAGE - MODEL RANKING")
            print("-"*100)
            
            df_avg = df_avg.sort_values('exact_match_rate', ascending=False)
            
            print(f"\n{'Rank':<6} {'Model':<25} {'Exact Match':<15} {'Avg Tokens':<15} "
                  f"{'Reasoning Ratio':<18} {'Model Family':<20}")
            print("-"*100)
            
            for idx, (_, row) in enumerate(df_avg.iterrows(), 1):
                print(f"{idx:<6} {row['model']:<25} {row['exact_match_rate']:<15.2%} "
                      f"{row['avg_total_tokens']:<15.1f} {row['reasoning_ratio']:<18.2%} "
                      f"{row['model_type']:<20}")
            
            # Efficiency ranking
            df_avg['efficiency'] = df_avg['exact_match_rate'] / df_avg['avg_total_tokens'] * 10000
            df_avg_eff = df_avg.sort_values('efficiency', ascending=False)
            
            print("\n" + "-"*100)
            print("EFFICIENCY RANKING (Accuracy per 10K Tokens)")
            print("-"*100)
            
            print(f"\n{'Rank':<6} {'Model':<25} {'Efficiency':<15} {'Exact Match':<15} {'Avg Tokens':<15}")
            print("-"*100)
            
            for idx, (_, row) in enumerate(df_avg_eff.iterrows(), 1):
                print(f"{idx:<6} {row['model']:<25} {row['efficiency']:<15.4f} "
                      f"{row['exact_match_rate']:<15.2%} {row['avg_total_tokens']:<15.1f}")
        
        # Per-task statistics
        df_tasks = df[df['task_type'] != 'benchmark_average']
        
        if not df_tasks.empty:
            print("\n" + "-"*100)
            print("PER-TASK TYPE STATISTICS")
            print("-"*100)
            
            for task_type in sorted(df_tasks['task_type'].unique()):
                task_data = df_tasks[df_tasks['task_type'] == task_type]
                
                print(f"\n{task_type.upper()}:")
                print(f"  Number of results: {len(task_data)}")
                print(f"  Avg Exact Match: {task_data['exact_match_rate'].mean():.2%}")
                print(f"  Avg Tokens: {task_data['avg_total_tokens'].mean():.1f}")
                print(f"  Avg Reasoning Ratio: {task_data['reasoning_ratio'].mean():.2%}")
                
                # Top 3 models for this task
                top_models = task_data.nlargest(3, 'exact_match_rate')
                print(f"  Top 3 models:")
                for idx, (_, row) in enumerate(top_models.iterrows(), 1):
                    print(f"    {idx}. {row['model']}: {row['exact_match_rate']:.2%} "
                          f"({row['avg_total_tokens']:.0f} tokens)")
        
        print("\n" + "="*100 + "\n")


def export_plot_data(results_dir: str, output_file: str = "plot_data.csv"):
    """
    Export the prepared plot data to CSV for external analysis.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output CSV file path
    
    Example:
        export_plot_data("./results", "benchmark_data.csv")
    """
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    if df.empty:
        print("No data available to export.")
        return None
    
    df.to_csv(output_file, index=False)
    print(f"Plot data exported to {output_file}")
    print(f"Total rows: {len(df)}")
    
    return df


def quick_generate(results_dir: str,
                  output_file: str = "accuracy_vs_token_cost.png",
                  figsize: Tuple[int, int] = (16, 9),
                  show_average_line: bool = True,
                  add_model_labels: bool = False,
                  show_summary: bool = True):
    """
    Quick function to generate the plot with one command.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        show_average_line: Whether to show benchmark average line
        add_model_labels: Whether to add model name labels
        show_summary: Whether to print summary statistics
    
    Example:
        quick_generate("./results")
        quick_generate("./results", "my_plot.png", figsize=(18, 10))
        quick_generate("./results", add_model_labels=True)
    """
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    visualizer.load_all_results()
    
    if show_summary:
        visualizer.print_summary_statistics()
    
    visualizer.plot_accuracy_vs_token_cost(
        save_path=output_file,
        figsize=figsize,
        show_average_line=show_average_line,
        add_model_labels=add_model_labels
    )


def customize_visualization(results_dir: str,
                           output_file: str = "accuracy_vs_token_cost.png",
                           figsize: Tuple[int, int] = (16, 9),
                           custom_colors: Dict[str, str] = None,
                           custom_markers: Dict[str, str] = None,
                           show_average_line: bool = True,
                           add_model_labels: bool = False):
    """
    Generate plot with custom colors and markers.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        custom_colors: Custom color mapping for task types
        custom_markers: Custom marker mapping for model families
        show_average_line: Whether to show benchmark average line
        add_model_labels: Whether to add model name labels
    
    Example:
        custom_colors = {
            'long': '#FF0000',
            'short': '#00FF00',
            'benchmark_average': '#666666'
        }
        custom_markers = {
            'gpt-4': 'o',
            'deepseek-v3': '^'
        }
        customize_visualization(
            "./results",
            "custom_plot.png",
            custom_colors=custom_colors,
            custom_markers=custom_markers
        )
    """
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    
    # Override colors if provided
    if custom_colors:
        visualizer.task_colors.update(custom_colors)
    
    # Override markers if provided
    if custom_markers:
        visualizer.model_markers.update(custom_markers)
    
    visualizer.load_all_results()
    visualizer.plot_accuracy_vs_token_cost(
        save_path=output_file,
        figsize=figsize,
        show_average_line=show_average_line,
        add_model_labels=add_model_labels
    )


def batch_visualize(results_dir: str,
                   output_dir: str = "./plots",
                   formats: List[str] = ['png', 'pdf'],
                   show_average_line: bool = True):
    """
    Generate multiple visualizations with different formats.
    
    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save output plots
        formats: List of output formats (e.g., ['png', 'pdf', 'svg'])
        show_average_line: Whether to show benchmark average line
    
    Example:
        batch_visualize("./results", "./plots", formats=['png', 'pdf', 'svg'])
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    visualizer.load_all_results()
    
    print("Generating summary statistics...")
    visualizer.print_summary_statistics()
    
    print("\nGenerating visualizations...")
    
    for fmt in formats:
        output_file = output_path / f"accuracy_vs_token_cost.{fmt}"
        print(f"Creating {fmt.upper()} plot: {output_file}")
        
        visualizer.plot_accuracy_vs_token_cost(
            save_path=str(output_file),
            show_average_line=show_average_line
        )
    
    # Also export data to CSV
    csv_file = output_path / "plot_data.csv"
    print(f"\nExporting data to CSV: {csv_file}")
    export_plot_data(results_dir, str(csv_file))
    
    print("\nBatch visualization complete!")
    print(f"All outputs saved to: {output_dir}")


def main():
    """
    Main interactive function for creating visualizations.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║      SENSITIVE DATA REDACTION BENCHMARK VISUALIZER                       ║
    ║                                                                          ║
    ║  Visualize model accuracy vs token cost trade-off                        ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get results directory
    results_dir = input("Enter the path to your results directory (default: './results'): ").strip()
    
    if not results_dir:
        results_dir = "./results"
    
    if not Path(results_dir).exists():
        print(f"\nError: Directory '{results_dir}' does not exist.")
        return
    
    # Initialize visualizer
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    
    # Load data
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("\nNo valid result files found in the specified directory.")
        return
    
    # Show summary statistics
    show_stats = input("\nShow summary statistics? (y/n, default: y): ").strip().lower()
    if show_stats == '' or show_stats == 'y':
        visualizer.print_summary_statistics()
    
    # Ask for visualization options
    print("\n" + "-"*80)
    print("VISUALIZATION OPTIONS")
    print("-"*80)
    
    show_avg = input("Show benchmark average line? (y/n, default: y): ").strip().lower()
    show_average_line = show_avg == '' or show_avg == 'y'
    
    add_labels = input("Add model name labels to points? (y/n, default: n): ").strip().lower()
    add_model_labels = add_labels == 'y'
    
    # Figure size
    custom_size = input("Enter custom figure size as 'width,height' (default: 16,9): ").strip()
    if custom_size and ',' in custom_size:
        try:
            width, height = map(float, custom_size.split(','))
            figsize = (width, height)
        except:
            print("Invalid size format. Using default (16, 9)")
            figsize = (16, 9)
    else:
        figsize = (16, 9)
    
    # Ask for output options
    save_plot = input("\nSave plot? (y/n, default: y): ").strip().lower()
    
    if save_plot == '' or save_plot == 'y':
        output_file = input("Enter output filename (default: 'accuracy_vs_token_cost.png'): ").strip()
        if not output_file:
            output_file = "accuracy_vs_token_cost.png"
        
        # Ask for multiple formats
        multiple_formats = input("Save in multiple formats? (y/n, default: n): ").strip().lower()
        
        if multiple_formats == 'y':
            formats_input = input("Enter formats separated by comma (e.g., 'png,pdf,svg', default: 'png,pdf'): ").strip()
            if formats_input:
                formats = [f.strip() for f in formats_input.split(',')]
            else:
                formats = ['png', 'pdf']
            
            # Use batch visualize
            output_dir = input("Enter output directory (default: './plots'): ").strip()
            if not output_dir:
                output_dir = "./plots"
            
            batch_visualize(
                results_dir,
                output_dir,
                formats=formats,
                show_average_line=show_average_line
            )
            return
        
        save_path = output_file
    else:
        save_path = None
    
    # Export data to CSV
    export_csv = input("\nExport plot data to CSV? (y/n, default: n): ").strip().lower()
    if export_csv == 'y':
        csv_file = input("Enter CSV filename (default: 'plot_data.csv'): ").strip()
        if not csv_file:
            csv_file = "plot_data.csv"
        export_plot_data(results_dir, csv_file)
    
    # Generate the plot
    print("\n" + "="*80)
    print("Generating visualization...")
    print("="*80 + "\n")
    
    visualizer.plot_accuracy_vs_token_cost(
        save_path=save_path,
        figsize=figsize,
        show_average_line=show_average_line,
        add_model_labels=add_model_labels
    )
    
    print("\nVisualization complete!")


def analyze_model_performance(results_dir: str, 
                             model_name: str = None,
                             task_type: str = None):
    """
    Analyze performance for specific model or task type.
    
    Args:
        results_dir: Directory containing result JSON files
        model_name: Specific model to analyze (optional)
        task_type: Specific task type to analyze (optional)
    
    Example:
        analyze_model_performance("./results", model_name="gpt-4")
        analyze_model_performance("./results", task_type="long")
        analyze_model_performance("./results", model_name="deepseek-v3", task_type="long")
    """
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    if df.empty:
        print("No data available.")
        return
    
    # Filter by model if specified
    if model_name:
        df = df[df['model'].str.contains(model_name, case=False, na=False)]
        if df.empty:
            print(f"No data found for model: {model_name}")
            return
    
    # Filter by task type if specified
    if task_type:
        df = df[df['task_type'].str.contains(task_type, case=False, na=False)]
        if df.empty:
            print(f"No data found for task type: {task_type}")
            return
    
    print("\n" + "="*100)
    print("PERFORMANCE ANALYSIS")
    print("="*100)
    
    if model_name:
        print(f"Model: {model_name}")
    if task_type:
        print(f"Task Type: {task_type}")
    
    print(f"\nTotal results: {len(df)}")
    
    print("\n" + "-"*100)
    print(f"{'Model':<25} {'Task Type':<20} {'Exact Match':<15} {'Avg Tokens':<15} {'Reasoning Ratio':<18}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{row['model']:<25} {row['task_type']:<20} {row['exact_match_rate']:<15.2%} "
              f"{row['avg_total_tokens']:<15.1f} {row['reasoning_ratio']:<18.2%}")
    
    print("-"*100)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print(f"  Mean Exact Match Rate: {df['exact_match_rate'].mean():.2%}")
    print(f"  Std Exact Match Rate: {df['exact_match_rate'].std():.2%}")
    print(f"  Mean Avg Tokens: {df['avg_total_tokens'].mean():.1f}")
    print(f"  Std Avg Tokens: {df['avg_total_tokens'].std():.1f}")
    print(f"  Mean Reasoning Ratio: {df['reasoning_ratio'].mean():.2%}")
    print(f"  Std Reasoning Ratio: {df['reasoning_ratio'].std():.2%}")
    
    print("="*100 + "\n")


def compare_models(results_dir: str, 
                  model1: str, 
                  model2: str,
                  task_type: str = None):
    """
    Compare two models side by side.
    
    Args:
        results_dir: Directory containing result JSON files
        model1: First model name
        model2: Second model name
        task_type: Specific task type to compare (optional)
    
    Example:
        compare_models("./results", "gpt-4", "deepseek-v3")
        compare_models("./results", "gpt-4", "deepseek-v3", task_type="long")
    """
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    if df.empty:
        print("No data available.")
        return
    
    # Filter for the two models
    df1 = df[df['model'].str.contains(model1, case=False, na=False)]
    df2 = df[df['model'].str.contains(model2, case=False, na=False)]
    
    if df1.empty:
        print(f"No data found for model: {model1}")
        return
    
    if df2.empty:
        print(f"No data found for model: {model2}")
        return
    
    # Filter by task type if specified
    if task_type:
        df1 = df1[df1['task_type'].str.contains(task_type, case=False, na=False)]
        df2 = df2[df2['task_type'].str.contains(task_type, case=False, na=False)]
    
    print("\n" + "="*120)
    print(f"MODEL COMPARISON: {model1} vs {model2}")
    if task_type:
        print(f"Task Type: {task_type}")
    print("="*120)
    
    # Get benchmark averages
    avg1 = df1[df1['task_type'] == 'benchmark_average']
    avg2 = df2[df2['task_type'] == 'benchmark_average']
    
    if not avg1.empty and not avg2.empty:
        print("\nBENCHMARK AVERAGE COMPARISON:")
        print("-"*120)
        print(f"{'Metric':<30} {model1:<40} {model2:<40} {'Difference':<10}")
        print("-"*120)
        
        em1 = avg1.iloc[0]['exact_match_rate']
        em2 = avg2.iloc[0]['exact_match_rate']
        print(f"{'Exact Match Rate':<30} {em1:<40.2%} {em2:<40.2%} {em1-em2:>+9.2%}")
        
        tok1 = avg1.iloc[0]['avg_total_tokens']
        tok2 = avg2.iloc[0]['avg_total_tokens']
        print(f"{'Avg Total Tokens':<30} {tok1:<40.1f} {tok2:<40.1f} {tok1-tok2:>+9.1f}")
        
        rr1 = avg1.iloc[0]['reasoning_ratio']
        rr2 = avg2.iloc[0]['reasoning_ratio']
        print(f"{'Reasoning Ratio':<30} {rr1:<40.2%} {rr2:<40.2%} {rr1-rr2:>+9.2%}")
        
        eff1 = em1 / tok1 * 10000
        eff2 = em2 / tok2 * 10000
        print(f"{'Efficiency (EM/10K tokens)':<30} {eff1:<40.4f} {eff2:<40.4f} {eff1-eff2:>+9.4f}")
        
        print("-"*120)
    
    # Per-task comparison
    df1_tasks = df1[df1['task_type'] != 'benchmark_average']
    df2_tasks = df2[df2['task_type'] != 'benchmark_average']
    
    if not df1_tasks.empty and not df2_tasks.empty:
        print("\nPER-TASK TYPE COMPARISON:")
        print("-"*120)
        
        # Get common task types
        common_tasks = set(df1_tasks['task_type'].unique()) & set(df2_tasks['task_type'].unique())
        
        for task in sorted(common_tasks):
            task1 = df1_tasks[df1_tasks['task_type'] == task].iloc[0]
            task2 = df2_tasks[df2_tasks['task_type'] == task].iloc[0]
            
            print(f"\n{task.upper()}:")
            print(f"  {model1}: EM={task1['exact_match_rate']:.2%}, Tokens={task1['avg_total_tokens']:.1f}, RR={task1['reasoning_ratio']:.2%}")
            print(f"  {model2}: EM={task2['exact_match_rate']:.2%}, Tokens={task2['avg_total_tokens']:.1f}, RR={task2['reasoning_ratio']:.2%}")
            print(f"  Difference: EM={task1['exact_match_rate']-task2['exact_match_rate']:+.2%}, "
                  f"Tokens={task1['avg_total_tokens']-task2['avg_total_tokens']:+.1f}, "
                  f"RR={task1['reasoning_ratio']-task2['reasoning_ratio']:+.2%}")
    
    print("\n" + "="*120 + "\n")


# Example usage and utility functions
def create_presentation_plot(results_dir: str,
                            output_file: str = "presentation_plot.png",
                            dpi: int = 300):
    """
    Create a high-quality plot suitable for presentations.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename
        dpi: DPI for high-resolution output
    
    Example:
        create_presentation_plot("./results", "presentation.png", dpi=600)
    """
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    visualizer.load_all_results()
    
    # Create plot with larger figure size
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Increase font sizes for better readability
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 10
    
    df = visualizer.prepare_plot_data()
    
    if df.empty:
        print("No data available for plotting.")
        return
    
    # Plot with enhanced styling
    base_size = 150
    max_size = 800
    size_range = max_size - base_size
    
    df_tasks = df[df['task_type'] != 'benchmark_average']
    df_average = df[df['task_type'] == 'benchmark_average']
    
    # Plot task-specific data
    for task_type in df_tasks['task_type'].unique():
        task_data = df_tasks[df_tasks['task_type'] == task_type]
        
        for model_type in task_data['model_type'].unique():
            model_data = task_data[task_data['model_type'] == model_type]
            sizes = base_size + (model_data['reasoning_ratio'] * size_range)
            
            ax.scatter(
                model_data['avg_total_tokens'],
                model_data['exact_match_rate'],
                c=visualizer.task_colors.get(task_type, visualizer.task_colors['default']),
                marker=visualizer.model_markers.get(model_type, 'X'),
                s=sizes,
                alpha=0.75,
                edgecolors='black',
                linewidths=2,
                zorder=3
            )
    
    # Plot benchmark average
    if not df_average.empty:
        df_average_sorted = df_average.sort_values('avg_total_tokens')
        
        ax.plot(
            df_average_sorted['avg_total_tokens'],
            df_average_sorted['exact_match_rate'],
            color='gray',
            linestyle='--',
            linewidth=3,
            alpha=0.4,
            zorder=1
        )
        
        for model_type in df_average['model_type'].unique():
            model_data = df_average[df_average['model_type'] == model_type]
            sizes = base_size + (model_data['reasoning_ratio'] * size_range)
            
            ax.scatter(
                model_data['avg_total_tokens'],
                model_data['exact_match_rate'],
                c=visualizer.task_colors['benchmark_average'],
                marker=visualizer.model_markers.get(model_type, 'X'),
                s=sizes,
                alpha=0.6,
                edgecolors='black',
                linewidths=2.5,
                zorder=2
            )
    
    # Set log scale
    ax.set_xscale('log')
    
    # Enhanced labels and title
    ax.set_xlabel('log(Avg Total Tokens per Sample)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Exact Match Rate', fontsize=16, fontweight='bold')
    ax.set_title('Model Accuracy vs Token Cost Trade-off\nSensitive Data Redaction Benchmark',
                fontsize=18, fontweight='bold', pad=25)
    
    # Set y-axis formatting
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Enhanced grid
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)
    ax.set_axisbelow(True)
    
    # Create legends
    visualizer._create_legends(ax, df)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    
    # Save with high DPI
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"High-quality presentation plot saved to {output_file} (DPI: {dpi})")
    
    # Reset font sizes
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 9
    
    plt.show()


def generate_report(results_dir: str, 
                   output_dir: str = "./report",
                   include_plots: bool = True,
                   include_csv: bool = True):
    """
    Generate a complete analysis report with plots and data exports.
    
    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save report files
        include_plots: Whether to generate plot images
        include_csv: Whether to export data to CSV
    
    Example:
        generate_report("./results", "./report")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No data found to generate report.")
        return
    
    print("\n" + "="*100)
    print("GENERATING COMPREHENSIVE BENCHMARK REPORT")
    print("="*100 + "\n")
    
    # 1. Generate summary statistics and save to text file
    print("1. Generating summary statistics...")
    summary_file = output_path / "summary_statistics.txt"
    
    import sys
    from io import StringIO
    
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    visualizer.print_summary_statistics()
    
    summary_content = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print(f"   Summary saved to: {summary_file}")
    
    # 2. Export data to CSV
    if include_csv:
        print("2. Exporting data to CSV...")
        csv_file = output_path / "benchmark_data.csv"
        df = export_plot_data(results_dir, str(csv_file))
        print(f"   CSV saved to: {csv_file}")
    
    # 3. Generate plots
    if include_plots:
        print("3. Generating visualizations...")
        
        # Standard plot
        standard_plot = output_path / "accuracy_vs_token_cost.png"
        visualizer.plot_accuracy_vs_token_cost(
            save_path=str(standard_plot),
            show_average_line=True
        )
        print(f"   Standard plot saved to: {standard_plot}")
        
        # High-res presentation plot
        presentation_plot = output_path / "accuracy_vs_token_cost_presentation.png"
        create_presentation_plot(results_dir, str(presentation_plot), dpi=600)
        print(f"   Presentation plot saved to: {presentation_plot}")
        
        # PDF version
        pdf_plot = output_path / "accuracy_vs_token_cost.pdf"
        visualizer.plot_accuracy_vs_token_cost(
            save_path=str(pdf_plot),
            show_average_line=True
        )
        print(f"   PDF plot saved to: {pdf_plot}")
    
    # 4. Generate detailed analysis for each model
    print("4. Generating per-model analysis...")
    df = visualizer.prepare_plot_data()
    
    models = df['model'].unique()
    model_analysis_dir = output_path / "model_analysis"
    model_analysis_dir.mkdir(exist_ok=True)
    
    for model in models:
        model_file = model_analysis_dir / f"{model}_analysis.txt"
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        analyze_model_performance(results_dir, model_name=model)
        
        model_content = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(model_content)
    
    print(f"   Model analyses saved to: {model_analysis_dir}")
    
    # 5. Create README
    print("5. Creating README...")
    readme_file = output_path / "README.md"
    
    readme_content = f"""# Sensitive Data Redaction Benchmark Report

## Overview

This report contains comprehensive analysis of model performance on the Sensitive Data Redaction Benchmark.

**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Results directory:** {results_dir}  
**Number of models analyzed:** {len(models)}  
**Total data points:** {len(df)}

## Files in This Report

### Summary Statistics
- `summary_statistics.txt` - Overall benchmark statistics and rankings

### Data Exports
- `benchmark_data.csv` - Complete dataset in CSV format for further analysis

### Visualizations
- `accuracy_vs_token_cost.png` - Standard visualization (300 DPI)
- `accuracy_vs_token_cost_presentation.png` - High-resolution version (600 DPI)
- `accuracy_vs_token_cost.pdf` - Vector format for publications

### Model-Specific Analysis
- `model_analysis/` - Detailed performance analysis for each model

## Key Findings

### Top 3 Models by Accuracy
"""
    
    # Add top models
    df_avg = df[df['task_type'] == 'benchmark_average'].sort_values('exact_match_rate', ascending=False)
    
    for idx, (_, row) in enumerate(df_avg.head(3).iterrows(), 1):
        readme_content += f"{idx}. **{row['model']}**: {row['exact_match_rate']:.2%} exact match rate\n"
    
    readme_content += "\n### Top 3 Models by Efficiency (Accuracy per 10K Tokens)\n"
    
    df_avg['efficiency'] = df_avg['exact_match_rate'] / df_avg['avg_total_tokens'] * 10000
    df_avg_eff = df_avg.sort_values('efficiency', ascending=False)
    
    for idx, (_, row) in enumerate(df_avg_eff.head(3).iterrows(), 1):
        readme_content += f"{idx}. **{row['model']}**: {row['efficiency']:.4f} efficiency score\n"
    
    readme_content += f"""
### Model Families Analyzed
{', '.join(sorted(df['model_type'].unique()))}

### Task Types Covered
{', '.join(sorted([t for t in df['task_type'].unique() if t != 'benchmark_average']))}

## How to Use This Report

1. **Quick Overview**: Start with `summary_statistics.txt` for high-level insights
2. **Visual Analysis**: Check the plots in the root directory
3. **Detailed Comparison**: Use `benchmark_data.csv` for custom analysis
4. **Model-Specific**: Review individual model reports in `model_analysis/`

## Metrics Explanation

- **Exact Match Rate**: Percentage of samples where model output exactly matches ground truth
- **Avg Total Tokens**: Average number of tokens used per sample (input + output)
- **Reasoning Ratio**: Proportion of reasoning tokens to total output tokens
- **Efficiency**: Exact match rate per 10,000 tokens (higher is better)

## Contact

For questions about this benchmark or to report issues, please contact the benchmark maintainers.
"""
    
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"   README saved to: {readme_file}")
    
    print("\n" + "="*100)
    print("REPORT GENERATION COMPLETE")
    print("="*100)
    print(f"\nAll files saved to: {output_dir}")
    print("\nReport contents:")
    print(f"  - Summary statistics: summary_statistics.txt")
    if include_csv:
        print(f"  - Data export: benchmark_data.csv")
    if include_plots:
        print(f"  - Visualizations: accuracy_vs_token_cost.* (png, pdf)")
        print(f"  - Presentation plot: accuracy_vs_token_cost_presentation.png")
    print(f"  - Model analyses: model_analysis/*.txt")
    print(f"  - Documentation: README.md")
    print("\n" + "="*100 + "\n")


# CLI-style helper functions
def list_available_models(results_dir: str):
    """
    List all models found in the results directory.
    
    Args:
        results_dir: Directory containing result JSON files
    
    Example:
        list_available_models("./results")
    """
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    if df.empty:
        print("No data available.")
        return
    
    models = sorted(df['model'].unique())
    
    print("\n" + "="*80)
    print("AVAILABLE MODELS")
    print("="*80)
    print(f"\nTotal: {len(models)} models\n")
    
    for idx, model in enumerate(models, 1):
        model_data = df[df['model'] == model]
        num_results = len(model_data)
        avg_em = model_data['exact_match_rate'].mean()
        
        print(f"{idx:2d}. {model:<30} ({num_results} results, avg EM: {avg_em:.2%})")
    
    print("\n" + "="*80 + "\n")


def list_available_tasks(results_dir: str):
    """
    List all task types found in the results directory.
    
    Args:
        results_dir: Directory containing result JSON files
    
    Example:
        list_available_tasks("./results")
    """
    visualizer = SensitiveDataRedactionVisualizer(results_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    if df.empty:
        print("No data available.")
        return
    
    tasks = sorted([t for t in df['task_type'].unique() if t != 'benchmark_average'])
    
    print("\n" + "="*80)
    print("AVAILABLE TASK TYPES")
    print("="*80)
    print(f"\nTotal: {len(tasks)} task types\n")
    
    for idx, task in enumerate(tasks, 1):
        task_data = df[df['task_type'] == task]
        num_results = len(task_data)
        avg_em = task_data['exact_match_rate'].mean()
        
        print(f"{idx}. {task:<30} ({num_results} results, avg EM: {avg_em:.2%})")
    
    print("\n" + "="*80 + "\n")


# Main entry point
if __name__ == "__main__":
    import sys
    
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick" and len(sys.argv) >= 3:
            # Quick generation: python script/scatter.py quick ./results
            results_dir = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else "accuracy_vs_token_cost.png"
            quick_generate(results_dir, output_file)
        
        elif command == "report" and len(sys.argv) >= 3:
            # Generate report: python script/scatter.py report ./results ./output
            results_dir = sys.argv[2]
            output_dir = sys.argv[3] if len(sys.argv) > 3 else "./report"
            generate_report(results_dir, output_dir)
        
        elif command == "list-models" and len(sys.argv) >= 3:
            # List models: python script/scatter.py list-models ./results
            results_dir = sys.argv[2]
            list_available_models(results_dir)
        
        elif command == "list-tasks" and len(sys.argv) >= 3:
            # List tasks: python script/scatter.py list-tasks ./results
            results_dir = sys.argv[2]
            list_available_tasks(results_dir)
        
        elif command == "compare" and len(sys.argv) >= 5:
            # Compare models: python script/scatter.py compare ./results model1 model2
            results_dir = sys.argv[2]
            model1 = sys.argv[3]
            model2 = sys.argv[4]
            task_type = sys.argv[5] if len(sys.argv) > 5 else None
            compare_models(results_dir, model1, model2, task_type)
        
        elif command == "analyze" and len(sys.argv) >= 3:
            # Analyze model: python script/scatter.py analyze ./results model_name
            results_dir = sys.argv[2]
            model_name = sys.argv[3] if len(sys.argv) > 3 else None
            task_type = sys.argv[4] if len(sys.argv) > 4 else None
            analyze_model_performance(results_dir, model_name, task_type)
        
        elif command == "export" and len(sys.argv) >= 3:
            # Export data: python script/scatter.py export ./results output.csv
            results_dir = sys.argv[2]
            output_file = sys.argv[3] if len(sys.argv) > 3 else "plot_data.csv"
            export_plot_data(results_dir, output_file)
        
        elif command == "help" or command == "-h" or command == "--help":
            print("""
Sensitive Data Redaction Benchmark Visualizer

Usage:
  python script/scatter.py <command> [arguments]

Commands:
  quick <results_dir> [output_file]
      Generate a quick visualization
      Example: python script/scatter.py quick ./results my_plot.png

  report <results_dir> [output_dir]
      Generate a comprehensive report with plots and analysis
      Example: python script/scatter.py report ./results ./my_report

  list-models <results_dir>
      List all available models in the results
      Example: python script/scatter.py list-models ./results

  list-tasks <results_dir>
      List all available task types
      Example: python script/scatter.py list-tasks ./results

  compare <results_dir> <model1> <model2> [task_type]
      Compare two models side by side
      Example: python script/scatter.py compare ./results gpt-4 deepseek-v3
      Example: python script/scatter.py compare ./results gpt-4 deepseek-v3 long

  analyze <results_dir> [model_name] [task_type]
      Analyze performance for specific model or task
      Example: python script/scatter.py analyze ./results gpt-4
      Example: python script/scatter.py analyze ./results gpt-4 long

  export <results_dir> [output_file]
      Export plot data to CSV
      Example: python script/scatter.py export ./results data.csv

  help, -h, --help
      Show this help message

Interactive Mode:
  python script/scatter.py
      Run without arguments for interactive mode with prompts

Advanced Usage (in Python):
  from script_name import quick_generate, customize_visualization, batch_visualize
  
  # Quick generation
  quick_generate("./results", "plot.png")
  
  # Custom colors and markers
  customize_visualization(
      "./results",
      custom_colors={'long': '#FF0000'},
      custom_markers={'gpt-4': 'o'}
  )
  
  # Batch generation
  batch_visualize("./results", "./plots", formats=['png', 'pdf', 'svg'])
  
  # Generate complete report
  generate_report("./results", "./report")

For more information, visit: https://github.com/your-repo/benchmark
            """)
        
        else:
            print(f"Unknown command: {command}")
            print("Use 'python script/scatter.py help' for usage information")
    
    else:
        # No arguments - run interactive mode
        main()