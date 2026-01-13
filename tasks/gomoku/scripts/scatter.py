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
import re

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


class GomokuBenchmarkVisualizer:
    """
    Visualizer for Gomoku benchmark accuracy vs token cost analysis.
    Creates scatter plots showing the trade-off between task accuracy and computational cost.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the visualizer with a base directory containing result JSON files.
        
        Args:
            base_dir: Path to directory containing JSON result files
        """
        self.base_dir = Path(base_dir)
        self.results_data = []
        
        # Define color palette for task types
        self.task_colors = {
            'linear': '#FF6B6B',           # Red
            'diagonal': '#4ECDC4',         # Teal
            'benchmark_average': '#888888'  # Gray for average
        }
        
        # Define marker styles for different model types
        self.model_markers = {
            'gpt-5': 'o',                    # Circle
            'gpt-4': 's',                    # Square
            'o1': '^',                       # Triangle up
            'o4': 'v',                       # Triangle down
            'deepseek-reasoner': 'D',        # Diamond
            'deepseek-chat': 'p',            # Pentagon
            'deepseek-v3': 'p',              # Pentagon
            'claude': 'h',                   # Hexagon
            'gemini': '*',                   # Star
            'llama': '<',                    # Triangle left
            'qwen': '>',                     # Triangle right
            'default': 'X'                   # X marker
        }
        
    def extract_task_type(self, filename: str) -> str:
        """
        Extract task type (linear/diagonal) from filename.
        
        Args:
            filename: JSON filename
            
        Returns:
            Task type ('linear', 'diagonal', or 'unknown')
        """
        filename_lower = filename.lower()
        if 'linear' in filename_lower:
            return 'linear'
        elif 'diagonal' in filename_lower:
            return 'diagonal'
        return 'unknown'
    
    def extract_model_name(self, full_model_name: str) -> str:
        """
        Extract simplified model name (e.g., 'gpt-4o-mini' -> 'gpt-4o').
        
        Args:
            full_model_name: Full model name from JSON
            
        Returns:
            Simplified model name
        """
        # Common patterns to extract model names
        patterns = [
            r'(o1-[\w.-]+)',
            r'(o4-[\w.-]+)',
            r'(gpt-[\w]+)',
            r'(claude-[\w.-]+)',
            r'(deepseek-reasoner)',
            r'(deepseek-r1)',
            r'(deepseek-v3)',
            r'(deepseek-chat)',
            r'(deepseek-[\w.-]+)',
            r'(gemini-[\w.-]+)',
            r'(llama-[\w.-]+)',
            r'(qwen-[\w.-]+)',
            r'(mistral-[\w.-]+)',
        ]
        
        model_lower = full_model_name.lower()
        for pattern in patterns:
            match = re.search(pattern, model_lower)
            if match:
                return match.group(1)
        
        # If no pattern matches, return first two parts separated by dash
        parts = full_model_name.split('-')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return full_model_name
    
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
        if 'o1-' in model_lower or model_lower.startswith('o1'):
            return 'o1'
        elif 'o4-' in model_lower or model_lower.startswith('o4'):
            return 'o4'
        elif 'deepseek-reasoner' in model_lower or 'deepseek-r1' in model_lower:
            return 'deepseek-reasoner'
        elif 'deepseek-v3' in model_lower:
            return 'deepseek-v3'
        elif 'deepseek-chat' in model_lower:
            return 'deepseek-chat'
        elif 'gpt-5' in model_lower or 'gpt5' in model_lower:
            return 'gpt-5'
        elif 'gpt-4' in model_lower or 'gpt4' in model_lower:
            return 'gpt-4'
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
        Load all JSON results from the base directory.
        """
        print("Loading Gomoku benchmark results for visualization...")
        
        json_files = list(self.base_dir.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {self.base_dir}")
            return
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    task_type = self.extract_task_type(json_file.name)
                    model_name = self.extract_model_name(data.get('model_name', ''))
                    
                    result = {
                        'model_name': model_name,
                        'full_model_name': data.get('model_name', ''),
                        'task_type': task_type,
                        'accuracy': data.get('results', {}).get('accuracy', 0),
                        'avg_total_tokens': data.get('token_usage', {}).get('avg_tokens_per_case', 0),
                        'avg_reasoning_tokens': data.get('token_usage', {}).get('avg_reasoning_per_case', 0),
                        'avg_output_tokens': data.get('token_usage', {}).get('avg_output_per_case', 0),
                        'filename': json_file.name,
                        'model_type': ''  # Will be set later
                    }
                    
                    # Calculate reasoning ratio
                    if result['avg_total_tokens'] > 0:
                        result['reasoning_ratio'] = result['avg_reasoning_tokens'] / result['avg_total_tokens']
                    else:
                        result['reasoning_ratio'] = 0
                    
                    result['model_type'] = self.get_model_type(model_name)
                    
                    self.results_data.append(result)
                    
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
        
        print(f"Loaded {len(self.results_data)} benchmark results")
        
        # Group by model and task type
        model_task_summary = defaultdict(list)
        for result in self.results_data:
            model_task_summary[result['model_name']].append(result['task_type'])
        
        for model, tasks in sorted(model_task_summary.items()):
            print(f"  - {model}: {len(tasks)} tasks ({', '.join(sorted(set(tasks)))})")
        print()
    
    def prepare_plot_data(self) -> pd.DataFrame:
        """
        Prepare data for plotting by extracting relevant metrics.
        
        Returns:
            DataFrame with columns: model_name, task_type, accuracy, avg_total_tokens, reasoning_ratio, model_type
        """
        if not self.results_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results_data)
        
        # Calculate benchmark averages for each model
        benchmark_averages = []
        
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            
            avg_result = {
                'model_name': model_name,
                'task_type': 'benchmark_average',
                'accuracy': model_data['accuracy'].mean(),
                'avg_total_tokens': model_data['avg_total_tokens'].mean(),
                'avg_reasoning_tokens': model_data['avg_reasoning_tokens'].mean(),
                'avg_output_tokens': model_data['avg_output_tokens'].mean(),
                'reasoning_ratio': model_data['reasoning_ratio'].mean(),
                'model_type': model_data['model_type'].iloc[0],
                'filename': 'benchmark_average'
            }
            
            benchmark_averages.append(avg_result)
        
        # Combine original data with benchmark averages
        df_with_averages = pd.concat([df, pd.DataFrame(benchmark_averages)], ignore_index=True)
        
        return df_with_averages
    
    def plot_accuracy_vs_token_cost(self, save_path: str = None, 
                                   figsize: Tuple[int, int] = (14, 8),
                                   show_average_line: bool = True):
        """
        Create the main scatter plot: Task Accuracy vs Token Cost.
        
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
        
        # Separate benchmark average data from task data
        df_tasks = df[df['task_type'] != 'benchmark_average']
        df_average = df[df['task_type'] == 'benchmark_average']
        
        # Plot task data (linear and diagonal)
        for task_type in df_tasks['task_type'].unique():
            task_data = df_tasks[df_tasks['task_type'] == task_type]
            
            for model_type in task_data['model_type'].unique():
                model_data = task_data[task_data['model_type'] == model_type]
                
                # Calculate sizes: base_size + (reasoning_ratio * size_range)
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Plot with log scale on x-axis
                ax.scatter(
                    model_data['avg_total_tokens'],
                    model_data['accuracy'],
                    c=self.task_colors[task_type],
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
                    model_data['accuracy'],
                    c=self.task_colors['benchmark_average'],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=3
                )
                
                # Add model name annotations
                for _, row in model_data.iterrows():
                    ax.annotate(
                        row['model_name'],
                        (row['avg_total_tokens'], row['accuracy']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7
                    )
            
            # Sort average data by avg_total_tokens for line plot
            df_average_sorted = df_average.sort_values('avg_total_tokens')
            
            # Draw connecting line through benchmark average points
            ax.plot(
                df_average_sorted['avg_total_tokens'],
                df_average_sorted['accuracy'],
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
        ax.set_xlabel('log(Average Tokens per Case)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Task Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Gomoku Benchmark: Model Task Accuracy vs Token Cost', 
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
        Create custom legends for task types, model types, and reasoning ratios.
        
        Args:
            ax: Matplotlib axes object
            df: DataFrame with plot data
        """
        legend_y_start = 1.0
        legend_spacing = 0.28
        
        # Legend 1: Task Types (colors)
        task_patches = []
        
        # Add task types
        for task_type in ['linear', 'diagonal', 'benchmark_average']:
            if task_type in df['task_type'].values:
                color = self.task_colors[task_type]
                label = task_type.replace('_', ' ').title()
                alpha = 0.6 if task_type == 'benchmark_average' else 1.0
                task_patches.append(
                    mpatches.Patch(color=color, label=label, alpha=alpha)
                )
        
        legend1 = ax.legend(handles=task_patches, 
                          title='Task Type',
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
                # Simplify some labels
                label = (label.replace('Deepseek V3', 'DeepSeek-V3')
                        .replace('Deepseek Chat', 'DeepSeek-Chat')
                        .replace('Deepseek Reasoner', 'DeepSeek-R1'))
            
            model_type_lines.append(
                Line2D([0], [0], marker=marker, color='gray', 
                      linestyle='', markersize=9, 
                      markeredgewidth=1.3, markeredgecolor='black',
                      label=label, alpha=0.7)
            )
        
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
        
        # Show size examples
        reasoning_ratios = [0, 0.25, 0.5, 0.75, 1.0]
        reasoning_lines = []
        
        for ratio in reasoning_ratios:
            size = base_size + (ratio * size_range)
            reasoning_lines.append(
                Line2D([0], [0], marker='o', color='gray',
                      linestyle='', markersize=np.sqrt(size/10),
                      markeredgewidth=1.3, markeredgecolor='black',
                      label=f'{ratio:.2f}', alpha=0.7)
            )
        
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
    ║         GOMOKU BENCHMARK: ACCURACY VS TOKEN COST VISUALIZER              ║
    ║                                                                          ║
    ║  Visualize the trade-off between task accuracy and token consumption     ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get base directory
    base_dir = input("Enter the path to your results directory (or press Enter for current directory): ").strip()
    
    if not base_dir:
        base_dir = "."
    
    # Initialize visualizer
    visualizer = GomokuBenchmarkVisualizer(base_dir)
    
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
        output_file = input("Enter output filename (default: 'gomoku_accuracy_vs_token_cost.png'): ").strip()
        if not output_file:
            output_file = "gomoku_accuracy_vs_token_cost.png"
        save_path = output_file
    else:
        save_path = None
    
    # Generate the plot
    print("\nGenerating accuracy vs token cost visualization...")
    visualizer.plot_accuracy_vs_token_cost(save_path=save_path, 
                                          show_average_line=show_average_line)
    
    print("\nVisualization complete!")


def quick_generate(base_dir: str, 
                  output_file: str = "gomoku_accuracy_vs_token_cost.png",
                  figsize: Tuple[int, int] = (14, 8),
                  show_average_line: bool = True):
    """
    Quick function to generate the plot with one command.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        show_average_line: Whether to show benchmark average line
    
    Example:
        quick_generate("./benchmark_results", "my_plot.png")
        quick_generate("./results", "plot.png", figsize=(18, 10), show_average_line=True)
    """
    visualizer = GomokuBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file, 
                                          figsize=figsize,
                                          show_average_line=show_average_line)


def customize_visualization(base_dir: str, 
                           output_file: str = "gomoku_accuracy_vs_token_cost.png",
                           figsize: Tuple[int, int] = (14, 8),
                           custom_colors: Dict[str, str] = None,
                           custom_markers: Dict[str, str] = None,
                           show_average_line: bool = True):
    """
    Generate plot with custom colors and markers.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        custom_colors: Custom color mapping for task types
        custom_markers: Custom marker mapping for model types
        show_average_line: Whether to show benchmark average line
    
    Example:
        custom_colors = {
            'linear': '#FF0000',
            'diagonal': '#00FF00',
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
    visualizer = GomokuBenchmarkVisualizer(base_dir)
    
    # Override colors if provided
    if custom_colors:
        visualizer.task_colors.update(custom_colors)
    
    # Override markers if provided
    if custom_markers:
        visualizer.model_markers.update(custom_markers)
    
    visualizer.load_all_results()
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file, 
                                          figsize=figsize,
                                          show_average_line=show_average_line)


def export_plot_data(base_dir: str, output_file: str = "gomoku_plot_data.csv"):
    """
    Export the prepared plot data to CSV for external analysis.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output CSV file path
    
    Example:
        export_plot_data("./results", "my_data.csv")
    """
    visualizer = GomokuBenchmarkVisualizer(base_dir)
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
    print(f"Number of models: {df['model_name'].nunique()}")
    print(f"Task types: {', '.join(df['task_type'].unique())}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    
    # Summary for benchmark averages
    df_avg = df[df['task_type'] == 'benchmark_average']
    if not df_avg.empty:
        print("\n" + "-"*80)
        print("BENCHMARK AVERAGE SUMMARY")
        print("-"*80)
        avg_summary = df_avg[['model_name', 'accuracy', 'avg_total_tokens', 'reasoning_ratio']].copy()
        avg_summary.columns = ['Model', 'Avg Accuracy', 'Avg Total Tokens', 'Reasoning Ratio']
        avg_summary = avg_summary.sort_values('Avg Accuracy', ascending=False)
        print(avg_summary.to_string(index=False))
    
    print("\n" + "-"*80)
    print("PER-TASK SUMMARY")
    print("-"*80)
    df_tasks = df[df['task_type'] != 'benchmark_average']
    if not df_tasks.empty:
        summary = df_tasks.groupby(['model_name', 'task_type']).agg({
            'accuracy': 'mean',
            'avg_total_tokens': 'mean',
            'reasoning_ratio': 'mean'
        }).round(4)
        print(summary)
    print("="*80 + "\n")
    
    return df


def compare_models_summary(base_dir: str):
    """
    Print a comparison summary of all models based on benchmark averages.
    
    Args:
        base_dir: Directory containing result JSON files
    
    Example:
        compare_models_summary("./results")
    """
    visualizer = GomokuBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    # Filter for benchmark averages only
    df_avg = df[df['task_type'] == 'benchmark_average'].copy()
    
    if df_avg.empty:
        print("No benchmark average data available.")
        return
    
    # Sort by accuracy
    df_avg = df_avg.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*110)
    print("MODEL COMPARISON - BENCHMARK AVERAGES")
    print("="*110)
    print(f"{'Rank':<6} {'Model':<25} {'Avg Accuracy':<15} {'Avg Tokens':<15} {'Reasoning Ratio':<18} {'Model Type':<20}")
    print("-"*110)
    
    for idx, (_, row) in enumerate(df_avg.iterrows(), 1):
        print(f"{idx:<6} {row['model_name']:<25} {row['accuracy']:<15.2%} "
              f"{row['avg_total_tokens']:<15.1f} {row['reasoning_ratio']:<18.2%} {row['model_type']:<20}")
    
    print("="*110)
    
    # Calculate efficiency metric (accuracy per 1000 tokens)
    df_avg['efficiency'] = df_avg['accuracy'] / df_avg['avg_total_tokens'] * 1000
    df_avg_eff = df_avg.sort_values('efficiency', ascending=False)
    
    print("\n" + "="*110)
    print("EFFICIENCY RANKING (Accuracy per 1000 Tokens)")
    print("="*110)
    print(f"{'Rank':<6} {'Model':<25} {'Efficiency':<15} {'Avg Accuracy':<15} {'Avg Tokens':<15}")
    print("-"*110)
    
    for idx, (_, row) in enumerate(df_avg_eff.iterrows(), 1):
        print(f"{idx:<6} {row['model_name']:<25} {row['efficiency']:<15.4f} "
              f"{row['accuracy']:<15.2%} {row['avg_total_tokens']:<15.1f}")
    
    print("="*110 + "\n")
    
    # Task-specific performance
    df_tasks = df[df['task_type'] != 'benchmark_average']
    if not df_tasks.empty:
        print("\n" + "="*110)
        print("TASK-SPECIFIC PERFORMANCE")
        print("="*110)
        
        for task in ['linear', 'diagonal']:
            task_data = df_tasks[df_tasks['task_type'] == task].sort_values('accuracy', ascending=False)
            if not task_data.empty:
                print(f"\n{task.upper()} Task:")
                print(f"{'Rank':<6} {'Model':<25} {'Accuracy':<15} {'Avg Tokens':<15}")
                print("-"*70)
                for idx, (_, row) in enumerate(task_data.iterrows(), 1):
                    print(f"{idx:<6} {row['model_name']:<25} {row['accuracy']:<15.2%} "
                          f"{row['avg_total_tokens']:<15.1f}")
        
        print("="*110 + "\n")


def analyze_reasoning_impact(base_dir: str):
    """
    Analyze the impact of reasoning tokens on accuracy.
    
    Args:
        base_dir: Directory containing result JSON files
    
    Example:
        analyze_reasoning_impact("./results")
    """
    visualizer = GomokuBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    # Filter for benchmark averages
    df_avg = df[df['task_type'] == 'benchmark_average'].copy()
    
    if df_avg.empty:
        print("No benchmark average data available.")
        return
    
    print("\n" + "="*100)
    print("REASONING TOKEN IMPACT ANALYSIS")
    print("="*100)
    
    # Categorize models by reasoning usage
    df_avg['reasoning_category'] = df_avg['reasoning_ratio'].apply(
        lambda x: 'No Reasoning' if x == 0 else ('Low (<50%)' if x < 0.5 else 'High (≥50%)')
    )
    
    print("\nAverage Performance by Reasoning Category:")
    print("-"*100)
    
    category_stats = df_avg.groupby('reasoning_category').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'avg_total_tokens': ['mean', 'std'],
        'reasoning_ratio': 'mean',
        'model_name': 'count'
    }).round(4)
    
    print(category_stats)
    print()
    
    # Detailed breakdown
    print("\nDetailed Breakdown:")
    print("-"*100)
    print(f"{'Model':<25} {'Accuracy':<15} {'Reasoning Ratio':<18} {'Avg Tokens':<15} {'Category':<20}")
    print("-"*100)
    
    df_avg_sorted = df_avg.sort_values('reasoning_ratio', ascending=False)
    for _, row in df_avg_sorted.iterrows():
        print(f"{row['model_name']:<25} {row['accuracy']:<15.2%} "
              f"{row['reasoning_ratio']:<18.2%} {row['avg_total_tokens']:<15.1f} "
              f"{row['reasoning_category']:<20}")
    
    print("="*100 + "\n")
    
    # Correlation analysis
    if len(df_avg) > 2:
        correlation = df_avg['reasoning_ratio'].corr(df_avg['accuracy'])
        print(f"Correlation between reasoning ratio and accuracy: {correlation:.4f}")
        
        if abs(correlation) < 0.3:
            interpretation = "weak"
        elif abs(correlation) < 0.7:
            interpretation = "moderate"
        else:
            interpretation = "strong"
        
        direction = "positive" if correlation > 0 else "negative"
        print(f"Interpretation: {interpretation} {direction} correlation\n")


def generate_task_comparison(base_dir: str, save_path: str = "task_comparison.png"):
    """
    Generate a comparison plot for different task types.
    
    Args:
        base_dir: Directory containing result JSON files
        save_path: Path to save the comparison plot
    
    Example:
        generate_task_comparison("./results", "comparison.png")
    """
    visualizer = GomokuBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    df_tasks = df[df['task_type'] != 'benchmark_average']
    
    if df_tasks.empty:
        print("No task data available for comparison.")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    task_types = ['linear', 'diagonal']
    
    for idx, task_type in enumerate(task_types):
        ax = axes[idx]
        task_data = df_tasks[df_tasks['task_type'] == task_type]
        
        if task_data.empty:
            continue
        
        # Sort by accuracy
        task_data_sorted = task_data.sort_values('accuracy', ascending=True)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(task_data_sorted))
        
        bars = ax.barh(y_pos, task_data_sorted['accuracy'], 
                      color=visualizer.task_colors[task_type], alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(task_data_sorted['model_name'])
        ax.set_xlabel('Accuracy', fontweight='bold')
        ax.set_title(f'{task_type.title()} Task Performance', fontweight='bold', fontsize=13)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, task_data_sorted['accuracy'])):
            ax.text(acc + 0.02, i, f'{acc:.1%}', 
                   va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Task comparison plot saved to {save_path}")
    plt.show()


def generate_comprehensive_report(base_dir: str, output_dir: str = "./report"):
    """
    Generate a comprehensive report including plots, tables, and analysis.
    
    Args:
        base_dir: Directory containing result JSON files
        output_dir: Directory to save report files
    
    Example:
        generate_comprehensive_report("./results", "./analysis_report")
    """
    import os
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              GENERATING COMPREHENSIVE BENCHMARK REPORT                   ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    visualizer = GomokuBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No data available for report generation.")
        return
    
    # 1. Generate main accuracy vs token cost plot
    print("\n[1/5] Generating accuracy vs token cost plot...")
    main_plot_path = output_path / "accuracy_vs_token_cost.png"
    visualizer.plot_accuracy_vs_token_cost(save_path=str(main_plot_path))
    
    # 2. Export data to CSV
    print("\n[2/5] Exporting data to CSV...")
    csv_path = output_path / "benchmark_data.csv"
    df = export_plot_data(base_dir, str(csv_path))
    
    # 3. Generate task comparison
    print("\n[3/5] Generating task comparison plot...")
    comparison_path = output_path / "task_comparison.png"
    generate_task_comparison(base_dir, str(comparison_path))
    
    # 4. Save text summaries
    print("\n[4/5] Generating text summaries...")
    
    import sys
    from io import StringIO
    
    # Capture model comparison summary
    old_stdout = sys.stdout
    sys.stdout = summary_output = StringIO()
    compare_models_summary(base_dir)
    model_summary = summary_output.getvalue()
    sys.stdout = old_stdout
    
    # Capture reasoning impact analysis
    sys.stdout = reasoning_output = StringIO()
    analyze_reasoning_impact(base_dir)
    reasoning_analysis = reasoning_output.getvalue()
    sys.stdout = old_stdout
    
    # Write to file
    summary_path = output_path / "benchmark_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*110 + "\n")
        f.write("GOMOKU BENCHMARK COMPREHENSIVE REPORT\n")
        f.write("="*110 + "\n\n")
        f.write(model_summary)
        f.write("\n\n")
        f.write(reasoning_analysis)
    
    print(f"Text summary saved to {summary_path}")
    
    # 5. Generate README
    print("\n[5/5] Generating report README...")
    readme_path = output_path / "README.md"
    
    # Extract key statistics
    df_avg = df[df['task_type'] == 'benchmark_average']
    
    # Prepare statistics strings
    if not df_avg.empty:
        best_accuracy_idx = df_avg['accuracy'].idxmax()
        best_accuracy = df_avg.loc[best_accuracy_idx]
        best_acc_model = best_accuracy['model_name']
        best_acc_value = f"{best_accuracy['accuracy']:.2%}"
        best_acc_tokens = f"{best_accuracy['avg_total_tokens']:.1f}"
        
        df_avg['efficiency'] = df_avg['accuracy'] / df_avg['avg_total_tokens']
        best_eff_idx = df_avg['efficiency'].idxmax()
        best_efficiency = df_avg.loc[best_eff_idx]
        best_eff_model = best_efficiency['model_name']
        best_eff_acc = f"{best_efficiency['accuracy']:.2%}"
        best_eff_tokens = f"{best_efficiency['avg_total_tokens']:.1f}"
        best_eff_value = f"{best_efficiency['efficiency'] * 1000:.4f}"
    else:
        best_acc_model = best_acc_value = best_acc_tokens = 'N/A'
        best_eff_model = best_eff_acc = best_eff_tokens = best_eff_value = 'N/A'
    
    total_models = df['model_name'].nunique()
    total_runs = len(df[df['task_type'] != 'benchmark_average'])
    task_types = ', '.join(sorted(df[df['task_type'] != 'benchmark_average']['task_type'].unique()))
    
    readme_content = f"""# Gomoku Benchmark Report

## Overview

This report contains a comprehensive analysis of model performance on the Gomoku benchmark tasks.

## Files in This Report

- `accuracy_vs_token_cost.png`: Main visualization showing accuracy vs token cost trade-off
- `task_comparison.png`: Side-by-side comparison of model performance on different tasks
- `benchmark_data.csv`: Complete dataset in CSV format for further analysis
- `benchmark_summary.txt`: Detailed text summary with rankings and statistics

## Key Findings

### Top Performers

**Best Accuracy:**
- Model: {best_acc_model}
- Accuracy: {best_acc_value}
- Avg Tokens: {best_acc_tokens}

**Best Efficiency (Accuracy per 1000 tokens):**
- Model: {best_eff_model}
- Accuracy: {best_eff_acc}
- Avg Tokens: {best_eff_tokens}
- Efficiency: {best_eff_value}

### Model Statistics

- Total models evaluated: {total_models}
- Total benchmark runs: {total_runs}
- Task types: {task_types}

## Visualization Guide

### Main Plot (accuracy_vs_token_cost.png)

- **X-axis (log scale)**: Average tokens per case
- **Y-axis**: Task accuracy (0-100%)
- **Point colors**: Different task types (linear, diagonal, benchmark average)
- **Point shapes**: Different model families (GPT, DeepSeek, Claude, etc.)
- **Point sizes**: Reasoning token ratio (larger = more reasoning tokens)
- **Gray dashed line**: Connects benchmark average points across models

### Task Comparison (task_comparison.png)

Shows model rankings for each individual task type (linear and diagonal).

## Data Dictionary (benchmark_data.csv)

- `model_name`: Simplified model identifier
- `full_model_name`: Complete model name from benchmark
- `task_type`: Type of task (linear, diagonal, or benchmark_average)
- `accuracy`: Task accuracy (0-1)
- `avg_total_tokens`: Average total tokens per case
- `avg_reasoning_tokens`: Average reasoning tokens per case
- `avg_output_tokens`: Average output tokens per case
- `reasoning_ratio`: Ratio of reasoning tokens to total tokens
- `model_type`: Model family/type
- `filename`: Source JSON filename

## Notes

- Benchmark averages are calculated as the mean across all task types for each model
- Efficiency metric is defined as: (accuracy / avg_total_tokens) × 1000
- Models with reasoning_ratio = 0 do not use explicit reasoning tokens
- Log scale is used for token axis to better visualize the wide range of token usage

---

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README saved to {readme_path}")
    
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll files saved to: {output_path.absolute()}")
    print("\nGenerated files:")
    print(f"  1. {main_plot_path.name}")
    print(f"  2. {comparison_path.name}")
    print(f"  3. {csv_path.name}")
    print(f"  4. {summary_path.name}")
    print(f"  5. {readme_path.name}")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run interactive main function
    main()
    
    # Uncomment below for quick testing:
    # quick_generate("./benchmark_results")
    
    # Uncomment below to generate comprehensive report:
    # generate_comprehensive_report("./benchmark_results", "./analysis_report")