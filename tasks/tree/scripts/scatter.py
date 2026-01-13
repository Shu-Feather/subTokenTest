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


class TreeBenchmarkVisualizer:
    """
    Visualizer for tree benchmark accuracy vs token cost analysis.
    Creates scatter plots showing the trade-off between accuracy and computational cost.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the visualizer with a base directory containing result JSON files.
        
        Args:
            base_dir: Path to directory containing result JSON files
        """
        self.base_dir = Path(base_dir)
        self.results_data = []
        
        # Define color palette for task types
        self.task_colors = {
            'task1': '#3498db',      # Blue - Tree Knowledge
            'task2': '#e74c3c',      # Red - Tree Path
            'benchmark_average': '#888888'  # Gray for average
        }
        
        # Task type labels
        self.task_labels = {
            'task1': 'Tree Knowledge',
            'task2': 'Tree Path',
            'benchmark_average': 'Benchmark Average'
        }
        
        # Define marker styles for different model types
        self.model_markers = {
            'gpt-5': 'o',                    # Circle
            'gpt-4': 's',                    # Square
            'deepseek-v3': '^',              # Triangle up
            'deepseek-r1': 'D',              # Diamond
            'o4': 'p',                       # Pentagon
            'o3': 'h',                       # Hexagon
            'o1': 'H',                       # Rotated hexagon
            'claude': 'v',                   # Triangle down
            'gemini': '*',                   # Star
            'llama': '<',                    # Triangle left
            'qwen': '>',                     # Triangle right
            'default': 'X'                   # X marker
        }
        
    def parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Parse filename to extract model name and task type.
        Expected format: results_{model}_{task}_loaded_{timestamp}.json
        
        Args:
            filename: JSON filename
            
        Returns:
            Tuple of (model_name, task_type)
        """
        basename = os.path.basename(filename)
        pattern = r'results_(.+?)_(task\d+)_loaded_\d+_\d+\.json'
        match = re.match(pattern, basename)
        
        if match:
            model_name = match.group(1)
            task_type = match.group(2)
            return model_name, task_type
        return None, None
    
    def get_model_type(self, model_name: str) -> str:
        """
        Determine model type from model name for marker assignment.
        Extract pattern like 'deepseek-v3', 'gpt-4', etc.
        
        Args:
            model_name: Full model name
            
        Returns:
            Model type for marker selection
        """
        model_lower = model_name.lower()
        
        # Check for specific patterns in order of specificity
        if 'deepseek-v3' in model_lower or 'deepseek_v3' in model_lower:
            return 'deepseek-v3'
        elif 'deepseek-r1' in model_lower or 'deepseek_r1' in model_lower or 'deepseek-reasoner' in model_lower:
            return 'deepseek-r1'
        elif 'gpt-5' in model_lower or 'gpt5' in model_lower:
            return 'gpt-5'
        elif 'gpt-4' in model_lower or 'gpt4' in model_lower:
            return 'gpt-4'
        elif 'o4-mini' in model_lower or 'o4mini' in model_lower:
            return 'o4'
        elif 'o3-mini' in model_lower or 'o3mini' in model_lower:
            return 'o3'
        elif 'o1-mini' in model_lower or 'o1mini' in model_lower or 'o1-preview' in model_lower:
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
    
    def calculate_reasoning_ratio(self, token_usage_summary: Dict) -> float:
        """
        Calculate reasoning token ratio.
        reasoning_ratio = total_reasoning_tokens / total_tokens
        
        Args:
            token_usage_summary: Token usage summary dictionary
            
        Returns:
            Reasoning ratio (0.0 to 1.0)
        """
        total_tokens = token_usage_summary.get('total_tokens', 0)
        reasoning_tokens = token_usage_summary.get('total_reasoning_tokens', 0)
        
        if total_tokens == 0:
            return 0.0
        
        return reasoning_tokens / total_tokens
    
    def load_all_results(self):
        """
        Load all result JSON files from the base directory.
        """
        print("Loading tree benchmark results for visualization...")
        
        json_files = list(self.base_dir.glob('results_*.json'))
        
        if not json_files:
            print(f"No result files found in {self.base_dir}")
            return
        
        for filepath in json_files:
            model_name, task_type = self.parse_filename(str(filepath))
            
            if model_name is None or task_type is None:
                print(f"Warning: Could not parse filename {filepath.name}, skipping...")
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Determine accuracy based on task type
                if task_type == 'task1':
                    accuracy = data.get('task1_accuracy', 0.0)
                elif task_type == 'task2':
                    accuracy = data.get('task2_accuracy', 0.0)
                else:
                    accuracy = 0.0
                
                # Get token usage
                token_summary = data.get('token_usage_summary', {})
                avg_tokens = token_summary.get('average_tokens_per_sample', 0)
                reasoning_ratio = self.calculate_reasoning_ratio(token_summary)
                
                # Extract model type
                model_type = self.get_model_type(model_name)
                
                self.results_data.append({
                    'model_name': model_name,
                    'model_type': model_type,
                    'task_type': task_type,
                    'accuracy': accuracy,
                    'avg_tokens': avg_tokens,
                    'reasoning_ratio': reasoning_ratio,
                    'filepath': str(filepath)
                })
                
            except Exception as e:
                print(f"Error loading {filepath.name}: {e}")
                continue
        
        print(f"Loaded {len(self.results_data)} result files")
        
        # Print summary by model and task
        models = sorted(list(set(r['model_name'] for r in self.results_data)))
        print(f"Models found: {len(models)}")
        for model in models:
            model_results = [r for r in self.results_data if r['model_name'] == model]
            tasks = [r['task_type'] for r in model_results]
            print(f"  - {model}: {', '.join(sorted(tasks))}")
        print()
    
    def prepare_plot_data(self) -> pd.DataFrame:
        """
        Prepare data for plotting by extracting relevant metrics for each model and task.
        Also calculate benchmark averages.
        
        Returns:
            DataFrame with columns: model_name, model_type, task_type, accuracy, avg_tokens, reasoning_ratio
        """
        if not self.results_data:
            return pd.DataFrame()
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results_data)
        
        # Calculate benchmark averages for each model
        benchmark_averages = []
        
        for model_name in df['model_name'].unique():
            model_data = df[df['model_name'] == model_name]
            
            avg_accuracy = model_data['accuracy'].mean()
            avg_tokens_overall = model_data['avg_tokens'].mean()
            
            # Get model type and reasoning ratio from first entry
            model_type = model_data.iloc[0]['model_type']
            reasoning_ratio = model_data.iloc[0]['reasoning_ratio']
            
            benchmark_averages.append({
                'model_name': model_name,
                'model_type': model_type,
                'task_type': 'benchmark_average',
                'accuracy': avg_accuracy,
                'avg_tokens': avg_tokens_overall,
                'reasoning_ratio': reasoning_ratio
            })
        
        # Combine original data with benchmark averages
        df_avg = pd.DataFrame(benchmark_averages)
        df_combined = pd.concat([df, df_avg], ignore_index=True)
        
        return df_combined
    
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
        
        # Plot task data
        for task_type in df_tasks['task_type'].unique():
            task_data = df_tasks[df_tasks['task_type'] == task_type]
            
            for model_type in task_data['model_type'].unique():
                model_data = task_data[task_data['model_type'] == model_type]
                
                # Calculate sizes: base_size + (reasoning_ratio * size_range)
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Plot with log scale on x-axis
                ax.scatter(
                    model_data['avg_tokens'],
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
                    model_data['avg_tokens'],
                    model_data['accuracy'],
                    c=self.task_colors['benchmark_average'],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=3
                )
            
            # Sort average data by avg_tokens for line plot
            df_average_sorted = df_average.sort_values('avg_tokens')
            
            # Draw connecting line through benchmark average points
            ax.plot(
                df_average_sorted['avg_tokens'],
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
        ax.set_xlabel('Average Total Tokens (log scale)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Task Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Tree Benchmark: Model Accuracy vs Token Cost Trade-off', 
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
        for task_type in ['task1', 'task2']:
            if task_type in df['task_type'].values:
                color = self.task_colors[task_type]
                label = self.task_labels[task_type]
                task_patches.append(mpatches.Patch(color=color, label=label))
        
        # Add benchmark average
        if 'benchmark_average' in df['task_type'].values:
            task_patches.append(
                mpatches.Patch(color=self.task_colors['benchmark_average'], 
                             label=self.task_labels['benchmark_average'], 
                             alpha=0.6)
            )
        
        legend1 = ax.legend(handles=task_patches, 
                          title='Task Types',
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
                # Capitalize and format nicely
                label = model_type.replace('-', ' ').title()
                # Special formatting for specific models
                if 'Gpt' in label:
                    label = label.replace('Gpt', 'GPT')
                elif 'Deepseek' in label:
                    label = label.replace('Deepseek', 'DeepSeek')
            
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
        
        # Show example sizes for reasoning ratio
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
    
    def print_summary_statistics(self):
        """
        Print detailed summary statistics of loaded results.
        """
        if not self.results_data:
            print("No data loaded.")
            return
        
        df = pd.DataFrame(self.results_data)
        
        print("\n" + "="*100)
        print("TREE BENCHMARK SUMMARY STATISTICS")
        print("="*100)
        
        print(f"\nTotal result files loaded: {len(self.results_data)}")
        print(f"Number of unique models: {df['model_name'].nunique()}")
        print(f"Task types tested: {', '.join(sorted(df['task_type'].unique()))}")
        
        # Detailed results by model and task
        print("\n" + "-"*100)
        print(f"{'Model':<30} {'Task':<15} {'Accuracy':<12} {'Avg Tokens':<15} {'Reasoning Ratio'}")
        print("-"*100)
        
        for _, row in df.sort_values(['model_name', 'task_type']).iterrows():
            print(f"{row['model_name']:<30} {row['task_type']:<15} "
                  f"{row['accuracy']:<12.4f} {row['avg_tokens']:<15.2f} "
                  f"{row['reasoning_ratio']:.4f}")
        
        print("="*100 + "\n")


def main():
    """
    Main function for creating the visualization.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║         TREE BENCHMARK: ACCURACY VS TOKEN COST VISUALIZER                ║
    ║                                                                          ║
    ║  Visualize the trade-off between model accuracy and token consumption    ║
    ║  for tree structure understanding tasks                                  ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get base directory
    base_dir = input("Enter the path to your results directory (or press Enter for current directory): ").strip()
    
    if not base_dir:
        base_dir = "."
    
    # Initialize visualizer
    visualizer = TreeBenchmarkVisualizer(base_dir)
    
    # Load data
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("\nNo valid results found in the specified directory.")
        print("Expected filename format: results_{model}_{task}_loaded_{timestamp}.json")
        return
    
    # Print summary statistics
    visualizer.print_summary_statistics()
    
    # Ask for options
    show_avg = input("\nShow benchmark average line? (y/n, default: y): ").strip().lower()
    show_average_line = show_avg == '' or show_avg == 'y'
    
    # Ask for output path
    save_plot = input("Save plot? (y/n, default: y): ").strip().lower()
    
    if save_plot == '' or save_plot == 'y':
        output_file = input("Enter output filename (default: 'tree_accuracy_vs_token_cost.png'): ").strip()
        if not output_file:
            output_file = "tree_accuracy_vs_token_cost.png"
        save_path = output_file
    else:
        save_path = None
    
    # Generate the plot
    print("\nGenerating accuracy vs token cost visualization...")
    visualizer.plot_accuracy_vs_token_cost(save_path=save_path, 
                                          show_average_line=show_average_line)
    
    print("\nVisualization complete!")


def quick_generate(base_dir: str, 
                  output_file: str = "tree_accuracy_vs_token_cost.png",
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
        quick_generate("./results", "my_plot.png")
        quick_generate("./results", "plot.png", figsize=(18, 10), show_average_line=True)
    """
    visualizer = TreeBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No results found!")
        return
    
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file, 
                                          figsize=figsize,
                                          show_average_line=show_average_line)


def customize_visualization(base_dir: str, 
                           output_file: str = "tree_accuracy_vs_token_cost.png",
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
            'task1': '#FF0000',
            'task2': '#00FF00',
            'benchmark_average': '#666666'
        }
        custom_markers = {
            'gpt-4': 'o',
            'deepseek-v3': '^'
        }
        customize_visualization("./results", "custom.png", 
                              custom_colors=custom_colors, 
                              custom_markers=custom_markers)
    """
    visualizer = TreeBenchmarkVisualizer(base_dir)
    
    # Override colors if provided
    if custom_colors:
        visualizer.task_colors.update(custom_colors)
    
    # Override markers if provided
    if custom_markers:
        visualizer.model_markers.update(custom_markers)
    
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No results found!")
        return
    
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file, 
                                          figsize=figsize,
                                          show_average_line=show_average_line)


def export_plot_data(base_dir: str, output_file: str = "tree_plot_data.csv"):
    """
    Export the prepared plot data to CSV for external analysis.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output CSV file path
    
    Example:
        export_plot_data("./results", "my_data.csv")
    """
    visualizer = TreeBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No results found!")
        return None
    
    df = visualizer.prepare_plot_data()
    
    if df.empty:
        print("No data available to export.")
        return None
    
    df.to_csv(output_file, index=False)
    print(f"Plot data exported to {output_file}")
    
    # Print summary
    print("\n" + "="*100)
    print("DATA SUMMARY")
    print("="*100)
    print(f"Total data points: {len(df)}")
    print(f"Number of models: {df['model_name'].nunique()}")
    print(f"Task types: {', '.join(df['task_type'].unique())}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    
    # Summary for benchmark averages
    df_avg = df[df['task_type'] == 'benchmark_average'].copy()
    if not df_avg.empty:
        print("\n" + "-"*100)
        print("BENCHMARK AVERAGE SUMMARY")
        print("-"*100)
        df_avg = df_avg.sort_values('accuracy', ascending=False)
        print(f"{'Model':<30} {'Avg Accuracy':<15} {'Avg Tokens':<15} {'Reasoning Ratio':<18} {'Model Type'}")
        print("-"*100)
        for _, row in df_avg.iterrows():
            print(f"{row['model_name']:<30} {row['accuracy']:<15.4f} "
                  f"{row['avg_tokens']:<15.2f} {row['reasoning_ratio']:<18.4f} {row['model_type']}")
    
    # Per-task summary
    print("\n" + "-"*100)
    print("PER-TASK SUMMARY")
    print("-"*100)
    df_tasks = df[df['task_type'] != 'benchmark_average']
    
    for task in sorted(df_tasks['task_type'].unique()):
        print(f"\n{task.upper()} ({visualizer.task_labels.get(task, task)}):")
        task_data = df_tasks[df_tasks['task_type'] == task].sort_values('accuracy', ascending=False)
        print(f"{'Model':<30} {'Accuracy':<15} {'Avg Tokens':<15} {'Reasoning Ratio'}")
        print("-"*100)
        for _, row in task_data.iterrows():
            print(f"{row['model_name']:<30} {row['accuracy']:<15.4f} "
                  f"{row['avg_tokens']:<15.2f} {row['reasoning_ratio']:<18.4f}")
    
    print("\n" + "="*100 + "\n")
    
    return df


def compare_models_summary(base_dir: str):
    """
    Print a detailed comparison summary of all models.
    
    Args:
        base_dir: Directory containing result JSON files
    
    Example:
        compare_models_summary("./results")
    """
    visualizer = TreeBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No results found!")
        return
    
    df = visualizer.prepare_plot_data()
    
    # Filter for benchmark averages only
    df_avg = df[df['task_type'] == 'benchmark_average'].copy()
    
    if df_avg.empty:
        print("No benchmark average data available.")
        return
    
    # Sort by accuracy
    df_avg = df_avg.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON - BENCHMARK AVERAGES")
    print("="*100)
    print(f"{'Rank':<6} {'Model':<30} {'Avg Accuracy':<15} {'Avg Tokens':<15} {'Reasoning Ratio':<18} {'Model Type'}")
    print("-"*100)
    
    for idx, (_, row) in enumerate(df_avg.iterrows(), 1):
        print(f"{idx:<6} {row['model_name']:<30} {row['accuracy']:<15.2%} "
              f"{row['avg_tokens']:<15.1f} {row['reasoning_ratio']:<18.2%} {row['model_type']:<20}")
    
    print("="*100)
    
    # Calculate efficiency metric (accuracy per 1000 tokens)
    df_avg['efficiency'] = df_avg['accuracy'] / df_avg['avg_tokens'] * 1000
    df_avg_eff = df_avg.sort_values('efficiency', ascending=False)
    
    print("\n" + "="*100)
    print("EFFICIENCY RANKING (Accuracy per 1000 Tokens)")
    print("="*100)
    print(f"{'Rank':<6} {'Model':<30} {'Efficiency':<15} {'Avg Accuracy':<15} {'Avg Tokens'}")
    print("-"*100)
    
    for idx, (_, row) in enumerate(df_avg_eff.iterrows(), 1):
        print(f"{idx:<6} {row['model_name']:<30} {row['efficiency']:<15.4f} "
              f"{row['accuracy']:<15.2%} {row['avg_tokens']:<15.1f}")
    
    print("="*100)
    
    # Task-specific comparisons
    df_tasks = df[df['task_type'] != 'benchmark_average']
    
    for task in sorted(df_tasks['task_type'].unique()):
        print("\n" + "="*100)
        print(f"TASK-SPECIFIC RANKING: {visualizer.task_labels.get(task, task).upper()}")
        print("="*100)
        
        task_data = df_tasks[df_tasks['task_type'] == task].sort_values('accuracy', ascending=False)
        
        print(f"{'Rank':<6} {'Model':<30} {'Accuracy':<15} {'Avg Tokens':<15} {'Reasoning Ratio'}")
        print("-"*100)
        
        for idx, (_, row) in enumerate(task_data.iterrows(), 1):
            print(f"{idx:<6} {row['model_name']:<30} {row['accuracy']:<15.2%} "
                  f"{row['avg_tokens']:<15.1f} {row['reasoning_ratio']:<18.2%}")
        
        print("="*100)
    
    print("\n")


def analyze_task_difficulty(base_dir: str):
    """
    Analyze and compare task difficulty across models.
    
    Args:
        base_dir: Directory containing result JSON files
    
    Example:
        analyze_task_difficulty("./results")
    """
    visualizer = TreeBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No results found!")
        return
    
    df = pd.DataFrame(visualizer.results_data)
    df_tasks = df[df['task_type'].isin(['task1', 'task2'])]
    
    if df_tasks.empty:
        print("No task data available.")
        return
    
    print("\n" + "="*100)
    print("TASK DIFFICULTY ANALYSIS")
    print("="*100)
    
    # Overall statistics by task
    print("\nOVERALL TASK STATISTICS:")
    print("-"*100)
    print(f"{'Task Type':<20} {'Task Name':<25} {'Avg Accuracy':<15} {'Avg Tokens':<15} {'Std Accuracy':<15}")
    print("-"*100)
    
    for task in sorted(df_tasks['task_type'].unique()):
        task_data = df_tasks[df_tasks['task_type'] == task]
        task_name = visualizer.task_labels.get(task, task)
        avg_acc = task_data['accuracy'].mean()
        std_acc = task_data['accuracy'].std()
        avg_tokens = task_data['avg_tokens'].mean()
        
        print(f"{task:<20} {task_name:<25} {avg_acc:<15.4f} {avg_tokens:<15.2f} {std_acc:<15.4f}")
    
    print("="*100)
    
    # Per-model task comparison
    print("\nPER-MODEL TASK COMPARISON:")
    print("-"*100)
    print(f"{'Model':<30} {'Task1 Acc':<15} {'Task2 Acc':<15} {'Difference':<15} {'Better At'}")
    print("-"*100)
    
    for model in sorted(df_tasks['model_name'].unique()):
        model_data = df_tasks[df_tasks['model_name'] == model]
        
        task1_data = model_data[model_data['task_type'] == 'task1']
        task2_data = model_data[model_data['task_type'] == 'task2']
        
        task1_acc = task1_data['accuracy'].values[0] if len(task1_data) > 0 else None
        task2_acc = task2_data['accuracy'].values[0] if len(task2_data) > 0 else None
        
        if task1_acc is not None and task2_acc is not None:
            diff = task1_acc - task2_acc
            better_at = "Tree Knowledge" if diff > 0 else "Tree Path" if diff < 0 else "Equal"
            
            print(f"{model:<30} {task1_acc:<15.4f} {task2_acc:<15.4f} {diff:<15.4f} {better_at}")
    
    print("="*100 + "\n")


def generate_comparison_table(base_dir: str, output_file: str = "model_comparison.csv"):
    """
    Generate a comprehensive comparison table and export to CSV.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output CSV filename
    
    Example:
        generate_comparison_table("./results", "comparison.csv")
    """
    visualizer = TreeBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No results found!")
        return None
    
    df = pd.DataFrame(visualizer.results_data)
    
    # Pivot data to create comparison table
    comparison_data = []
    
    for model in sorted(df['model_name'].unique()):
        model_data = df[df['model_name'] == model]
        
        row = {'model_name': model}
        row['model_type'] = model_data.iloc[0]['model_type']
        
        # Add task-specific metrics
        for task in ['task1', 'task2']:
            task_data = model_data[model_data['task_type'] == task]
            if len(task_data) > 0:
                row[f'{task}_accuracy'] = task_data.iloc[0]['accuracy']
                row[f'{task}_avg_tokens'] = task_data.iloc[0]['avg_tokens']
            else:
                row[f'{task}_accuracy'] = None
                row[f'{task}_avg_tokens'] = None
        
        # Calculate averages
        accuracies = [row.get('task1_accuracy'), row.get('task2_accuracy')]
        accuracies = [a for a in accuracies if a is not None]
        row['avg_accuracy'] = np.mean(accuracies) if accuracies else None
        
        tokens = [row.get('task1_avg_tokens'), row.get('task2_avg_tokens')]
        tokens = [t for t in tokens if t is not None]
        row['avg_tokens'] = np.mean(tokens) if tokens else None
        
        row['reasoning_ratio'] = model_data.iloc[0]['reasoning_ratio']
        
        # Calculate efficiency
        if row['avg_accuracy'] is not None and row['avg_tokens'] is not None and row['avg_tokens'] > 0:
            row['efficiency'] = row['avg_accuracy'] / row['avg_tokens'] * 1000
        else:
            row['efficiency'] = None
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by average accuracy
    comparison_df = comparison_df.sort_values('avg_accuracy', ascending=False, na_position='last')
    
    # Export to CSV
    comparison_df.to_csv(output_file, index=False)
    print(f"Comparison table exported to {output_file}")
    
    # Print formatted table
    print("\n" + "="*140)
    print("COMPREHENSIVE MODEL COMPARISON TABLE")
    print("="*140)
    print(f"{'Model':<30} {'Type':<15} {'Task1 Acc':<12} {'Task2 Acc':<12} {'Avg Acc':<12} "
          f"{'Avg Tokens':<12} {'Reasoning':<12} {'Efficiency':<12}")
    print("-"*140)
    
    for _, row in comparison_df.iterrows():
        task1_acc = f"{row['task1_accuracy']:.4f}" if pd.notna(row['task1_accuracy']) else "N/A"
        task2_acc = f"{row['task2_accuracy']:.4f}" if pd.notna(row['task2_accuracy']) else "N/A"
        avg_acc = f"{row['avg_accuracy']:.4f}" if pd.notna(row['avg_accuracy']) else "N/A"
        avg_tokens = f"{row['avg_tokens']:.1f}" if pd.notna(row['avg_tokens']) else "N/A"
        reasoning = f"{row['reasoning_ratio']:.4f}" if pd.notna(row['reasoning_ratio']) else "N/A"
        efficiency = f"{row['efficiency']:.4f}" if pd.notna(row['efficiency']) else "N/A"
        
        print(f"{row['model_name']:<30} {row['model_type']:<15} {task1_acc:<12} {task2_acc:<12} "
              f"{avg_acc:<12} {avg_tokens:<12} {reasoning:<12} {efficiency:<12}")
    
    print("="*140 + "\n")
    
    return comparison_df


def create_multi_panel_plot(base_dir: str, 
                           output_file: str = "tree_benchmark_multi_panel.png",
                           figsize: Tuple[int, int] = (18, 12)):
    """
    Create a multi-panel visualization with different views of the data.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
    
    Example:
        create_multi_panel_plot("./results", "multi_panel.png")
    """
    visualizer = TreeBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No results found!")
        return
    
    df = visualizer.prepare_plot_data()
    
    if df.empty:
        print("No data available for plotting.")
        return
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Panel 1: Main scatter plot (accuracy vs tokens)
    df_tasks = df[df['task_type'] != 'benchmark_average']
    df_avg = df[df['task_type'] == 'benchmark_average']
    
    base_size = 100
    max_size = 500
    size_range = max_size - base_size
    
    for task_type in df_tasks['task_type'].unique():
        task_data = df_tasks[df_tasks['task_type'] == task_type]
        sizes = base_size + (task_data['reasoning_ratio'] * size_range)
        
        ax1.scatter(task_data['avg_tokens'], task_data['accuracy'],
                   c=visualizer.task_colors[task_type], s=sizes, alpha=0.7,
                   label=visualizer.task_labels[task_type], edgecolors='black', linewidths=1)
    
    if not df_avg.empty:
        sizes = base_size + (df_avg['reasoning_ratio'] * size_range)
        ax1.scatter(df_avg['avg_tokens'], df_avg['accuracy'],
                   c=visualizer.task_colors['benchmark_average'], s=sizes, alpha=0.6,
                   label='Benchmark Avg', edgecolors='black', linewidths=1)
        
        df_avg_sorted = df_avg.sort_values('avg_tokens')
        ax1.plot(df_avg_sorted['avg_tokens'], df_avg_sorted['accuracy'],
                color='gray', linestyle='--', linewidth=2, alpha=0.4)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Average Tokens (log scale)', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy vs Token Cost', fontweight='bold', fontsize=12)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Panel 2: Accuracy comparison by task
    models = sorted(df_tasks['model_name'].unique())
    task1_accs = []
    task2_accs = []
    
    for model in models:
        model_data = df_tasks[df_tasks['model_name'] == model]
        task1 = model_data[model_data['task_type'] == 'task1']
        task2 = model_data[model_data['task_type'] == 'task2']
        
        task1_accs.append(task1['accuracy'].values[0] if len(task1) > 0 else 0)
        task2_accs.append(task2['accuracy'].values[0] if len(task2) > 0 else 0)
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    ax2.bar(x_pos - width/2, task1_accs, width, label='Tree Knowledge',
           color=visualizer.task_colors['task1'], alpha=0.7, edgecolor='black')
    ax2.bar(x_pos + width/2, task2_accs, width, label='Tree Path',
           color=visualizer.task_colors['task2'], alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Task Accuracy Comparison', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.05)
    
    # Panel 3: Token usage comparison
    avg_tokens_list = []
    for model in models:
        model_data = df_tasks[df_tasks['model_name'] == model]
        avg_tokens_list.append(model_data['avg_tokens'].mean())
    
    colors_3 = [visualizer.task_colors['benchmark_average']] * len(models)
    ax3.barh(models, avg_tokens_list, color=colors_3, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Average Tokens', fontweight='bold')
    ax3.set_ylabel('Models', fontweight='bold')
    ax3.set_title('Token Usage Comparison', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Panel 4: Reasoning ratio vs accuracy
    if not df_avg.empty:
        for model_type in df_avg['model_type'].unique():
            type_data = df_avg[df_avg['model_type'] == model_type]
            marker = visualizer.model_markers.get(model_type, 'X')
            
            ax4.scatter(type_data['reasoning_ratio'], type_data['accuracy'],
                       marker=marker, s=200, alpha=0.7, edgecolors='black',
                       linewidths=1.5, label=model_type, color='steelblue')
        
        ax4.set_xlabel('Reasoning Ratio', fontweight='bold')
        ax4.set_ylabel('Average Accuracy', fontweight='bold')
        ax4.set_title('Reasoning Ratio vs Accuracy', fontweight='bold', fontsize=12)
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-0.05, 1.05)
        ax4.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Multi-panel plot saved to {output_file}")
    
    plt.show()


def batch_analysis(base_dir: str, output_dir: str = "./analysis_output"):
    """
    Perform comprehensive batch analysis and generate all outputs.
    
    Args:
        base_dir: Directory containing result JSON files
        output_dir: Directory to save all analysis outputs
    
    Example:
        batch_analysis("./results", "./analysis_output")
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    BATCH ANALYSIS STARTED                                ║
    ║                                                                          ║
    ║  Generating comprehensive analysis outputs...                            ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize visualizer
    visualizer = TreeBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No results found!")
        return
    
    print("✓ Data loaded successfully\n")
    
    # 1. Generate main scatter plot
    print("1. Generating main accuracy vs token cost plot...")
    main_plot_path = output_path / "accuracy_vs_token_cost.png"
    visualizer.plot_accuracy_vs_token_cost(
        save_path=str(main_plot_path),
        show_average_line=True
    )
    print(f"   ✓ Saved to {main_plot_path}\n")
    
    # 2. Generate multi-panel plot
    print("2. Generating multi-panel visualization...")
    multi_panel_path = output_path / "multi_panel_analysis.png"
    create_multi_panel_plot(base_dir, str(multi_panel_path))
    print(f"   ✓ Saved to {multi_panel_path}\n")
    
    # 3. Export plot data
    print("3. Exporting plot data to CSV...")
    plot_data_path = output_path / "plot_data.csv"
    export_plot_data(base_dir, str(plot_data_path))
    print(f"   ✓ Saved to {plot_data_path}\n")
    
    # 4. Generate comparison table
    print("4. Generating model comparison table...")
    comparison_path = output_path / "model_comparison.csv"
    generate_comparison_table(base_dir, str(comparison_path))
    print(f"   ✓ Saved to {comparison_path}\n")
    
    # 5. Save text summaries to file
    print("5. Generating text summaries...")
    summary_path = output_path / "analysis_summary.txt"
    
    import sys
    from io import StringIO
    
    # Redirect stdout to capture print outputs
    old_stdout = sys.stdout
    sys.stdout = summary_buffer = StringIO()
    
    print("="*100)
    print("TREE BENCHMARK COMPREHENSIVE ANALYSIS REPORT")
    print("="*100)
    print(f"\nGenerated at: {pd.Timestamp.now()}")
    print(f"Results directory: {base_dir}")
    print("\n")
    
    # Print summary statistics
    visualizer.print_summary_statistics()
    
    # Print model comparison
    compare_models_summary(base_dir)
    
    # Print task difficulty analysis
    analyze_task_difficulty(base_dir)
    
    # Restore stdout
    sys.stdout = old_stdout
    
    # Save to file
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_buffer.getvalue())
    
    print(f"   ✓ Saved to {summary_path}\n")
    
    print("="*80)
    print("BATCH ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_path}")
    print("\nGenerated files:")
    print(f"  1. {main_plot_path.name} - Main scatter plot")
    print(f"  2. {multi_panel_path.name} - Multi-panel visualization")
    print(f"  3. {plot_data_path.name} - Plot data (CSV)")
    print(f"  4. {comparison_path.name} - Model comparison table (CSV)")
    print(f"  5. {summary_path.name} - Text summary report")
    print()


def interactive_explorer(base_dir: str):
    """
    Interactive command-line explorer for benchmark results.
    
    Args:
        base_dir: Directory containing result JSON files
    
    Example:
        interactive_explorer("./results")
    """
    visualizer = TreeBenchmarkVisualizer(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No results found!")
        return
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              TREE BENCHMARK INTERACTIVE EXPLORER                         ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\n" + "="*80)
        print("AVAILABLE COMMANDS:")
        print("="*80)
        print("  1. Show summary statistics")
        print("  2. Compare models")
        print("  3. Analyze task difficulty")
        print("  4. Generate main plot")
        print("  5. Generate multi-panel plot")
        print("  6. Export data to CSV")
        print("  7. Generate comparison table")
        print("  8. Batch analysis (generate all)")
        print("  9. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            visualizer.print_summary_statistics()
        
        elif choice == '2':
            compare_models_summary(base_dir)
        
        elif choice == '3':
            analyze_task_difficulty(base_dir)
        
        elif choice == '4':
            output_file = input("Enter output filename (default: 'plot.png'): ").strip()
            if not output_file:
                output_file = "plot.png"
            show_avg = input("Show benchmark average line? (y/n, default: y): ").strip().lower()
            show_average_line = show_avg == '' or show_avg == 'y'
            visualizer.plot_accuracy_vs_token_cost(
                save_path=output_file,
                show_average_line=show_average_line
            )
        
        elif choice == '5':
            output_file = input("Enter output filename (default: 'multi_panel.png'): ").strip()
            if not output_file:
                output_file = "multi_panel.png"
            create_multi_panel_plot(base_dir, output_file)
        
        elif choice == '6':
            output_file = input("Enter output filename (default: 'plot_data.csv'): ").strip()
            if not output_file:
                output_file = "plot_data.csv"
            export_plot_data(base_dir, output_file)
        
        elif choice == '7':
            output_file = input("Enter output filename (default: 'comparison.csv'): ").strip()
            if not output_file:
                output_file = "comparison.csv"
            generate_comparison_table(base_dir, output_file)
        
        elif choice == '8':
            output_dir = input("Enter output directory (default: './analysis_output'): ").strip()
            if not output_dir:
                output_dir = "./analysis_output"
            batch_analysis(base_dir, output_dir)
        
        elif choice == '9':
            print("\nExiting interactive explorer. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number between 1 and 9.")


# Convenience functions for common use cases
def quick_plot(results_dir: str = "./results", 
               output_file: str = "tree_benchmark.png"):
    """
    Quickest way to generate a plot - one line of code.
    
    Args:
        results_dir: Directory with result JSON files
        output_file: Output image filename
    
    Example:
        quick_plot()
        quick_plot("./my_results", "my_plot.png")
    """
    quick_generate(results_dir, output_file)


def full_analysis(results_dir: str = "./results",
                 output_dir: str = "./analysis"):
    """
    Generate complete analysis with all outputs.
    
    Args:
        results_dir: Directory with result JSON files
        output_dir: Directory to save outputs
    
    Example:
        full_analysis()
        full_analysis("./my_results", "./my_analysis")
    """
    batch_analysis(results_dir, output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tree Benchmark Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python script.py
  
  # Quick plot generation
  python script.py --quick-plot --input ./results --output plot.png
  
  # Full batch analysis
  python script.py --batch --input ./results --output-dir ./analysis
  
  # Interactive explorer
  python script.py --explorer --input ./results
  
  # Export data only
  python script.py --export-data --input ./results --output data.csv
  
  # Comparison summary
  python script.py --compare --input ./results
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default='./results',
                       help='Input directory containing result JSON files (default: ./results)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path')
    parser.add_argument('--output-dir', '-d', type=str,
                       help='Output directory for batch analysis')
    parser.add_argument('--quick-plot', action='store_true',
                       help='Generate quick plot and exit')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch analysis')
    parser.add_argument('--explorer', action='store_true',
                       help='Start interactive explorer')
    parser.add_argument('--export-data', action='store_true',
                       help='Export plot data to CSV')
    parser.add_argument('--compare', action='store_true',
                       help='Print model comparison summary')
    parser.add_argument('--task-analysis', action='store_true',
                       help='Analyze task difficulty')
    parser.add_argument('--no-average-line', action='store_true',
                       help='Do not show benchmark average line in plots')
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.quick_plot:
        output_file = args.output if args.output else "tree_benchmark.png"
        quick_generate(args.input, output_file, show_average_line=not args.no_average_line)
    
    elif args.batch:
        output_dir = args.output_dir if args.output_dir else "./analysis_output"
        batch_analysis(args.input, output_dir)
    
    elif args.explorer:
        interactive_explorer(args.input)
    
    elif args.export_data:
        output_file = args.output if args.output else "plot_data.csv"
        export_plot_data(args.input, output_file)
    
    elif args.compare:
        compare_models_summary(args.input)
    
    elif args.task_analysis:
        analyze_task_difficulty(args.input)
    
    else:
        # Default: run interactive main
        main()