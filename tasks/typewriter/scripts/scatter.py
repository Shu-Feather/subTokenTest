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
import warnings
import re

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


class BenchmarkVisualizationTool:
    """
    Visualizer for benchmark results showing task accuracy vs token cost.
    Creates scatter plots with color-coded tasks, size-coded reasoning ratios, 
    and shape-coded model types.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the visualizer with a base directory containing JSON result files.
        
        Args:
            base_dir: Path to directory containing JSON result files
        """
        self.base_dir = Path(base_dir)
        self.results_data = []
        
        # Define color palette for tasks
        self.task_colors = {
            'Progressive Typing': '#2E86AB',      # Blue
            'Backspace Operation': '#A23B72',     # Purple
            'Benchmark Average': '#888888'        # Gray
        }
        
        # Define marker styles for different model types
        self.model_markers = {
            'gpt-4': 's',                    # Square
            'gpt-5': 'o',                    # Circle
            'o4-mini': 'p',                  # Pentagon
            'o1-mini': 'h',                  # Hexagon
            'o1-pro': 'H',                   # Hexagon2
            'deepseek-reasoner': '^',        # Triangle up
            'deepseek-v3': 'D',              # Diamond
            'deepseek-r1': '*',              # Star
            'claude': 'v',                   # Triangle down
            'gemini': '<',                   # Triangle left
            'qwen': '>',                     # Triangle right
            'default': 'X'                   # X marker
        }
        
    def extract_model_name(self, full_name: str) -> str:
        """
        Extract simplified model name (e.g., 'gpt-4o-mini' from full name).
        Keeps pattern like xxx-xxx or xxx-xxx-xxx.
        
        Args:
            full_name: Full model name from JSON
            
        Returns:
            Simplified model name
        """
        name = full_name.lower()
        
        # Try to match pattern like xxx-xxx or xxx-xxx-xxx
        match = re.search(r'([a-z0-9]+-[a-z0-9]+(?:-[a-z0-9]+)?)', name)
        if match:
            return match.group(1)
        
        # Fallback: remove timestamp and other suffixes
        # Split by underscore and take the first part
        parts = name.split('_')
        if len(parts) > 0:
            return parts[0]
        
        return name
    
    def get_model_type(self, model_name: str) -> str:
        """
        Determine model type from model name for marker assignment.
        
        Args:
            model_name: Simplified model name (e.g., 'gpt-4o', 'deepseek-v3')
            
        Returns:
            Model type for marker selection
        """
        model_lower = model_name.lower()
        
        # Check for specific patterns in order of specificity
        if 'deepseek-r1' in model_lower or 'deepseek-reasoner' in model_lower:
            return 'deepseek-r1'
        elif 'deepseek-v3' in model_lower or 'deepseek-chat' in model_lower:
            return 'deepseek-v3'
        elif 'gpt-5' in model_lower or 'gpt5' in model_lower:
            return 'gpt-5'
        elif 'gpt-4' in model_lower or 'gpt4' in model_lower:
            return 'gpt-4'
        elif 'o4-mini' in model_lower:
            return 'o4-mini'
        elif 'o1-pro' in model_lower:
            return 'o1-pro'
        elif 'o1-mini' in model_lower:
            return 'o1-mini'
        elif 'claude' in model_lower:
            return 'claude'
        elif 'gemini' in model_lower:
            return 'gemini'
        elif 'qwen' in model_lower:
            return 'qwen'
        
        return 'default'
    
    def load_all_results(self):
        """
        Load all JSON result files from the base directory.
        """
        print("Loading benchmark results for visualization...")
        
        json_files = list(self.base_dir.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {self.base_dir}")
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
                print(f"Error loading {json_file.name}: {e}")
        
        print(f"Loaded {len(self.results_data)} result files")
        for result in self.results_data:
            model_name = result['data']['model_info']['name']
            print(f"  - {model_name}")
        print()
    
    def prepare_plot_data(self) -> pd.DataFrame:
        """
        Prepare data for plotting by extracting relevant metrics.
        
        Returns:
            DataFrame with columns: model, task, accuracy, avg_total_tokens, 
                                   reasoning_ratio, model_type
        """
        plot_data = []
        
        for result in self.results_data:
            data = result['data']
            model_name = self.extract_model_name(data['model_info']['name'])
            model_type = self.get_model_type(model_name)
            
            # Extract task metrics
            metrics = data.get('metrics', {})
            usage = data.get('total_usage', {})
            
            # Task 1: Progressive Typing
            task1_metrics = metrics.get('task1_metrics', {})
            task1_accuracy = task1_metrics.get('accuracy', 0)
            
            # Task 2: Backspace Operation
            task2_metrics = metrics.get('task2_metrics', {})
            task2_accuracy = task2_metrics.get('accuracy', 0)
            
            # Calculate average total tokens
            avg_total_tokens = usage.get('avg_tokens_per_sample', 0)
            
            # Calculate reasoning ratio
            avg_reasoning = usage.get('avg_output_per_sample', 0)
            reasoning_tokens = usage.get('reasoning_tokens', 0)
            total_samples = usage.get('total_samples', 1)
            
            # Try to get reasoning ratio more accurately
            if 'avg_output_per_sample' in usage and avg_total_tokens > 0:
                # If we have reasoning tokens separately
                if reasoning_tokens > 0:
                    avg_reasoning_per_sample = reasoning_tokens / total_samples
                    reasoning_ratio = avg_reasoning_per_sample / avg_total_tokens
                else:
                    reasoning_ratio = 0
            else:
                reasoning_ratio = 0
            
            # Ensure reasoning_ratio is between 0 and 1
            reasoning_ratio = max(0, min(1, reasoning_ratio))
            
            if avg_total_tokens > 0:
                # Add Task 1 data
                plot_data.append({
                    'model': model_name,
                    'task': 'Progressive Typing',
                    'accuracy': task1_accuracy,
                    'avg_total_tokens': avg_total_tokens,
                    'reasoning_ratio': reasoning_ratio,
                    'model_type': model_type
                })
                
                # Add Task 2 data
                plot_data.append({
                    'model': model_name,
                    'task': 'Backspace Operation',
                    'accuracy': task2_accuracy,
                    'avg_total_tokens': avg_total_tokens,
                    'reasoning_ratio': reasoning_ratio,
                    'model_type': model_type
                })
                
                # Add Benchmark Average
                avg_accuracy = (task1_accuracy + task2_accuracy) / 2
                plot_data.append({
                    'model': model_name,
                    'task': 'Benchmark Average',
                    'accuracy': avg_accuracy,
                    'avg_total_tokens': avg_total_tokens,
                    'reasoning_ratio': reasoning_ratio,
                    'model_type': model_type
                })
        
        df = pd.DataFrame(plot_data)
        return df
    
    def plot_accuracy_vs_token_cost(self, save_path: str = None,
                                   figsize: Tuple[int, int] = (16, 10),
                                   show_average_line: bool = True):
        """
        Create the main scatter plot: Task Accuracy vs Token Cost.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            show_average_line: Whether to show the benchmark average line
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
        max_size = 600
        size_range = max_size - base_size
        
        # Separate benchmark average data from task data
        df_tasks = df[df['task'] != 'Benchmark Average']
        df_average = df[df['task'] == 'Benchmark Average']
        
        # Plot task data
        for task in ['Progressive Typing', 'Backspace Operation']:
            task_data = df_tasks[df_tasks['task'] == task]
            
            for model_type in task_data['model_type'].unique():
                model_data = task_data[task_data['model_type'] == model_type]
                
                # Calculate sizes: base_size + (reasoning_ratio * size_range)
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Plot with log scale on x-axis
                ax.scatter(
                    model_data['avg_total_tokens'],
                    model_data['accuracy'],
                    c=self.task_colors[task],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.7,
                    edgecolors='white',
                    linewidths=1.5,
                    zorder=2
                )
        
        # Plot benchmark average points and line
        if not df_average.empty:
            for model_type in df_average['model_type'].unique():
                model_data = df_average[df_average['model_type'] == model_type]
                
                # Calculate sizes for average points
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Plot average points in gray
                ax.scatter(
                    model_data['avg_total_tokens'],
                    model_data['accuracy'],
                    c=self.task_colors['Benchmark Average'],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.5,
                    edgecolors='white',
                    linewidths=1.5,
                    zorder=3
                )
            
            if show_average_line:
                # Sort average data by avg_total_tokens for line plot
                df_average_sorted = df_average.sort_values('avg_total_tokens')
                
                # Draw connecting line through benchmark average points
                ax.plot(
                    df_average_sorted['avg_total_tokens'],
                    df_average_sorted['accuracy'],
                    color='gray',
                    linestyle='--',
                    linewidth=2.5,
                    alpha=0.3,
                    zorder=1
                )
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        
        # Labels and title
        ax.set_xlabel('Log₁₀(Average Total Tokens per Sample)', 
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Task Accuracy', fontsize=13, fontweight='bold')
        ax.set_title('Model Task Accuracy vs Token Cost Trade-off\n' + 
                    '(Point size represents Reasoning Ratio)',
                    fontsize=15, fontweight='bold', pad=20)
        
        # Set y-axis limits and formatting
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Create custom legends
        self._create_legends(ax, df)
        
        # Adjust layout to accommodate legends
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _create_legends(self, ax, df):
        """
        Create custom legends for tasks, model types, and reasoning ratios.
        
        Args:
            ax: Matplotlib axes object
            df: DataFrame with plot data
        """
        legend_y_start = 1.0
        legend_spacing = 0.28
        
        # Legend 1: Task Types (colors)
        task_patches = []
        for task in ['Progressive Typing', 'Backspace Operation', 'Benchmark Average']:
            if task in df['task'].values:
                color = self.task_colors[task]
                alpha = 0.5 if task == 'Benchmark Average' else 0.7
                task_patches.append(
                    mpatches.Patch(color=color, label=task, alpha=alpha)
                )
        
        legend1 = ax.legend(handles=task_patches,
                          title='Task Type',
                          loc='upper left',
                          bbox_to_anchor=(1.02, legend_y_start),
                          frameon=True,
                          fontsize=10,
                          title_fontsize=11,
                          borderaxespad=0,
                          edgecolor='black',
                          fancybox=False)
        ax.add_artist(legend1)
        
        # Legend 2: Model Types (markers)
        model_type_lines = []
        unique_model_types = sorted(df['model_type'].unique())
        
        for model_type in unique_model_types:
            marker = self.model_markers.get(model_type, 'X')
            
            # Format label
            if model_type == 'default':
                label = 'Other Models'
            else:
                # Capitalize and format nicely
                label = model_type.replace('-', ' ').title()
                label = label.replace('Gpt','GPT').replace('V3', 'V3').replace('R1', 'R1')
            
            model_type_lines.append(
                Line2D([0], [0], marker=marker, color='gray',
                      linestyle='', markersize=10,
                      markeredgewidth=1.5, markeredgecolor='black',
                      label=label, alpha=0.7)
            )
        
        legend2 = ax.legend(handles=model_type_lines,
                          title='Model Type',
                          loc='upper left',
                          bbox_to_anchor=(1.02, legend_y_start - legend_spacing),
                          frameon=True,
                          fontsize=9,
                          title_fontsize=11,
                          borderaxespad=0,
                          edgecolor='black',
                          fancybox=False,
                          ncol=1)
        ax.add_artist(legend2)
        
        # Legend 3: Reasoning Ratio (sizes)
        base_size = 100
        max_size = 600
        size_range = max_size - base_size
        
        # Show size examples for reasoning ratios
        reasoning_ratios = [0, 0.25, 0.5, 0.75, 1.0]
        reasoning_lines = []
        
        for ratio in reasoning_ratios:
            size = base_size + (ratio * size_range)
            reasoning_lines.append(
                Line2D([0], [0], marker='o', color='gray',
                      linestyle='', markersize=np.sqrt(size/10),
                      markeredgewidth=1.5, markeredgecolor='black',
                      label=f'{ratio:.2f}', alpha=0.7)
            )
        
        legend3 = ax.legend(handles=reasoning_lines,
                          title='Reasoning Ratio',
                          loc='upper left',
                          bbox_to_anchor=(1.02, legend_y_start - 2 * legend_spacing),
                          frameon=True,
                          fontsize=9,
                          title_fontsize=11,
                          borderaxespad=0,
                          edgecolor='black',
                          fancybox=False)
        ax.add_artist(legend3)


def main():
    """
    Main function for creating the visualization.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║            BENCHMARK ACCURACY VS TOKEN COST VISUALIZER                   ║
    ║                                                                          ║
    ║  Visualize task accuracy vs token consumption with reasoning ratio       ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get base directory
    base_dir = input("Enter the path to your results directory (or press Enter for current directory): ").strip()
    
    if not base_dir:
        base_dir = "."
    
    # Initialize visualizer
    visualizer = BenchmarkVisualizationTool(base_dir)
    
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
        output_file = input("Enter output filename (default: 'task_accuracy_vs_token_cost.png'): ").strip()
        if not output_file:
            output_file = "task_accuracy_vs_token_cost.png"
        save_path = output_file
    else:
        save_path = None
    
    # Generate the plot
    print("\nGenerating task accuracy vs token cost visualization...")
    visualizer.plot_accuracy_vs_token_cost(save_path=save_path,
                                          show_average_line=show_average_line)
    
    print("\nVisualization complete!")


def quick_generate(base_dir: str,
                  output_file: str = "task_accuracy_vs_token_cost.png",
                  figsize: Tuple[int, int] = (16, 10),
                  show_average_line: bool = True):
    """
    Quick function to generate the plot with one command.
    
    Args:
        base_dir: Directory containing JSON result files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        show_average_line: Whether to show benchmark average line
    
    Example:
        quick_generate("./benchmark_results", "my_plot.png")
        quick_generate("./results", "plot.png", figsize=(18, 12))
    """
    visualizer = BenchmarkVisualizationTool(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No data found!")
        return
    
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file,
                                          figsize=figsize,
                                          show_average_line=show_average_line)


def export_plot_data(base_dir: str, output_file: str = "benchmark_plot_data.csv"):
    """
    Export the prepared plot data to CSV for external analysis.
    
    Args:
        base_dir: Directory containing JSON result files
        output_file: Output CSV file path
    
    Example:
        export_plot_data("./benchmark_results", "data.csv")
    """
    visualizer = BenchmarkVisualizationTool(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No data found!")
        return None
    
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
    print(f"Tasks: {', '.join(df['task'].unique())}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    
    # Summary for benchmark averages
    df_avg = df[df['task'] == 'Benchmark Average']
    if not df_avg.empty:
        print("\n" + "-"*80)
        print("BENCHMARK AVERAGE SUMMARY")
        print("-"*80)
        avg_summary = df_avg[['model', 'accuracy', 'avg_total_tokens', 'reasoning_ratio']].copy()
        avg_summary.columns = ['Model', 'Avg Accuracy', 'Avg Total Tokens', 'Reasoning Ratio']
        avg_summary = avg_summary.sort_values('Avg Accuracy', ascending=False)
        print(avg_summary.to_string(index=False))
    
    # Per-task summary
    print("\n" + "-"*80)
    print("PER-TASK SUMMARY")
    print("-"*80)
    df_tasks = df[df['task'] != 'Benchmark Average']
    
    for task in ['Progressive Typing', 'Backspace Operation']:
        task_data = df_tasks[df_tasks['task'] == task]
        if not task_data.empty:
            print(f"\n{task}:")
            task_summary = task_data[['model', 'accuracy', 'avg_total_tokens']].copy()
            task_summary.columns = ['Model', 'Accuracy', 'Avg Tokens']
            task_summary = task_summary.sort_values('Accuracy', ascending=False)
            print(task_summary.to_string(index=False))
    
    print("\n" + "="*80 + "\n")
    
    return df


def compare_models_summary(base_dir: str):
    """
    Print a comparison summary of all models based on benchmark averages.
    
    Args:
        base_dir: Directory containing JSON result files
    
    Example:
        compare_models_summary("./benchmark_results")
    """
    visualizer = BenchmarkVisualizationTool(base_dir)
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No data found!")
        return
    
    df = visualizer.prepare_plot_data()
    
    # Filter for benchmark averages only
    df_avg = df[df['task'] == 'Benchmark Average'].copy()
    
    if df_avg.empty:
        print("No benchmark average data available.")
        return
    
    # Sort by accuracy
    df_avg = df_avg.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON - BENCHMARK AVERAGES (SORTED BY ACCURACY)")
    print("="*100)
    print(f"{'Rank':<6} {'Model':<30} {'Avg Accuracy':<15} {'Avg Tokens':<15} "
          f"{'Reasoning Ratio':<18} {'Model Type':<20}")
    print("-"*100)
    
    for idx, (_, row) in enumerate(df_avg.iterrows(), 1):
        print(f"{idx:<6} {row['model']:<30} {row['accuracy']:<15.2%} "
              f"{row['avg_total_tokens']:<15.1f} {row['reasoning_ratio']:<18.2%} "
              f"{row['model_type']:<20}")
    
    print("="*100)
    
    # Calculate efficiency metric (accuracy per 1000 tokens)
    df_avg['efficiency'] = df_avg['accuracy'] / df_avg['avg_total_tokens'] * 1000
    df_avg_eff = df_avg.sort_values('efficiency', ascending=False)
    
    print("\n" + "="*100)
    print("EFFICIENCY RANKING (Accuracy per 1000 Tokens)")
    print("="*100)
    print(f"{'Rank':<6} {'Model':<30} {'Efficiency':<15} {'Avg Accuracy':<15} {'Avg Tokens':<15}")
    print("-"*100)
    
    for idx, (_, row) in enumerate(df_avg_eff.iterrows(), 1):
        print(f"{idx:<6} {row['model']:<30} {row['efficiency']:<15.4f} "
              f"{row['accuracy']:<15.2%} {row['avg_total_tokens']:<15.1f}")
    
    print("="*100)
    
    # Task-specific rankings
    print("\n" + "="*100)
    print("TASK-SPECIFIC RANKINGS")
    print("="*100)
    
    df_tasks = df[df['task'] != 'Benchmark Average']
    
    for task in ['Progressive Typing', 'Backspace Operation']:
        task_data = df_tasks[df_tasks['task'] == task].copy()
        if not task_data.empty:
            task_data = task_data.sort_values('accuracy', ascending=False)
            
            print(f"\n{task}:")
            print("-"*100)
            print(f"{'Rank':<6} {'Model':<30} {'Accuracy':<15} {'Avg Tokens':<15}")
            print("-"*100)
            
            for idx, (_, row) in enumerate(task_data.iterrows(), 1):
                print(f"{idx:<6} {row['model']:<30} {row['accuracy']:<15.2%} "
                      f"{row['avg_total_tokens']:<15.1f}")
    
    print("\n" + "="*100 + "\n")


def customize_visualization(base_dir: str,
                           output_file: str = "task_accuracy_vs_token_cost.png",
                           figsize: Tuple[int, int] = (16, 10),
                           custom_colors: Dict[str, str] = None,
                           custom_markers: Dict[str, str] = None,
                           show_average_line: bool = True):
    """
    Generate plot with custom colors and markers.
    
    Args:
        base_dir: Directory containing JSON result files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        custom_colors: Custom color mapping for tasks
        custom_markers: Custom marker mapping for model types
        show_average_line: Whether to show benchmark average line
    
    Example:
        custom_colors = {
            'Progressive Typing': '#FF0000',
            'Backspace Operation': '#00FF00',
            'Benchmark Average': '#666666'
        }
        custom_markers = {
            'gpt-4': 'o',
            'deepseek-v3': '^'
        }
        customize_visualization("./results", "custom.png",
                              custom_colors=custom_colors,
                              custom_markers=custom_markers)
    """
    visualizer = BenchmarkVisualizationTool(base_dir)
    
    # Override colors if provided
    if custom_colors:
        visualizer.task_colors.update(custom_colors)
    
    # Override markers if provided
    if custom_markers:
        visualizer.model_markers.update(custom_markers)
    
    visualizer.load_all_results()
    
    if not visualizer.results_data:
        print("No data found!")
        return
    
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file,
                                          figsize=figsize,
                                          show_average_line=show_average_line)


if __name__ == "__main__":
    # Run interactive main function
    main()