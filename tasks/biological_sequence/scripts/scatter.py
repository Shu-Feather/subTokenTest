import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100


class BenchmarkVisualizer:
    """
    Visualizer for benchmark accuracy vs token cost analysis.
    Creates scatter plots showing the trade-off between accuracy and computational cost.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the visualizer with a base directory containing result folders.
        
        Args:
            base_dir: Path to directory containing result folders
        """
        self.base_dir = Path(base_dir)
        self.results = []
        self.overall_results = []  # Store overall benchmark averages
        
        # Define color palette for task types
        self.task_colors = {
            'dna_complement': '#1f77b4',           # Blue
            'rna_complement': '#ff7f0e',           # Orange
            'protein_three_to_one': '#2ca02c',     # Green
            'protein_one_to_three': '#d62728',     # Red
            'benchmark_average': '#808080'         # Gray for overall average
        }
        
        # Define marker styles for different model types
        self.model_markers = {
            'gpt-5': 'o',           # Circle
            'gpt-4': 's',           # Square
            'deepseek-r1': '^',     # Triangle up
            'deepseek-v3': 'D',     # Diamond
            'o4-mini': 'p',         # Pentagon
            'o1': 'v',              # Triangle down
            'claude': 'h',          # Hexagon
            'gemini': '*',          # Star
            'llama': '<',           # Triangle left
            'qwen': '>',            # Triangle right
            'default': 'X'          # X marker
        }
    
    def extract_model_name(self, folder_name: str) -> str:
        """
        Extract simplified model name from folder name.
        Examples: 'deepseek-r1_20251120_124745' -> 'deepseek-r1'
                  'gpt-4-turbo_20251120_124745' -> 'gpt-4-turbo'
        
        Args:
            folder_name: Full folder name
            
        Returns:
            Extracted model name
        """
        # Remove timestamp pattern (_YYYYMMDD_HHMMSS)
        model_name = re.sub(r'_\d{8}_\d{6}$', '', folder_name)
        return model_name
    
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
        if 'deepseek-r1' in model_lower or 'deepseek-reasoner' in model_lower:
            return 'deepseek-r1'
        elif 'deepseek-v3' in model_lower or 'deepseek-chat' in model_lower or 'deepseek' in model_lower:
            return 'deepseek-v3'
        elif 'gpt-5' in model_lower or 'gpt5' in model_lower:
            return 'gpt-5'
        elif 'gpt-4' in model_lower or 'gpt4' in model_lower:
            return 'gpt-4'
        elif 'o4-mini' in model_lower or 'o4mini' in model_lower:
            return 'o4-mini'
        elif 'o1-mini' in model_lower or 'o1mini' in model_lower or 'o1-preview' in model_lower or model_lower == 'o1':
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
    
    def load_benchmark_results(self):
        """
        Load all benchmark results from subdirectories.
        """
        print("Loading benchmark results for visualization...")
        
        # Find all subdirectories containing metrics.json
        for metrics_file in glob.glob(os.path.join(self.base_dir, '*', 'metrics.json')):
            folder_path = os.path.dirname(metrics_file)
            folder_name = os.path.basename(folder_path)
            model_name = self.extract_model_name(folder_name)
            
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Extract overall average data
                overall_metrics = metrics.get('overall', {})
                overall_usage = metrics.get('usage_statistics', {}).get('overall', {})
                
                if overall_metrics and overall_usage:
                    overall_result = {
                        'model_name': model_name,
                        'model_type': self.get_model_type(model_name),
                        'task_type': 'benchmark_average',
                        'accuracy': overall_metrics.get('accuracy', 0),
                        'avg_total_tokens': overall_usage.get('avg_total_tokens', 0),
                        'reasoning_ratio': overall_usage.get('reasoning_ratio', 0),
                        'folder_name': folder_name
                    }
                    self.overall_results.append(overall_result)
                
                # Extract data for each task type
                for task_type, task_data in metrics['by_task'].items():
                    # Get usage statistics for this task
                    usage_stats = metrics['usage_statistics']['per_task'][task_type]
                    
                    result = {
                        'model_name': model_name,
                        'model_type': self.get_model_type(model_name),
                        'task_type': task_type,
                        'accuracy': task_data['accuracy'],
                        'avg_total_tokens': usage_stats['avg_total_tokens'],
                        'reasoning_ratio': usage_stats['reasoning_ratio'],
                        'folder_name': folder_name
                    }
                    self.results.append(result)
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")
        
        print(f"Loaded {len(self.results)} task-specific data points from {len(set([r['folder_name'] for r in self.results]))} result folders")
        print(f"Loaded {len(self.overall_results)} benchmark average data points")
        
        # Print per-model summary
        models = {}
        for r in self.results:
            if r['model_name'] not in models:
                models[r['model_name']] = 0
            models[r['model_name']] += 1
        
        for model, count in sorted(models.items()):
            print(f"  - {model}: {count} task evaluations")
        print()
    
    def plot_accuracy_vs_token_cost(self, save_path: str = None, 
                                   figsize: tuple = (14, 8),
                                   num_ticks: int = 15,
                                   show_benchmark_average: bool = True):
        """
        Create the main scatter plot: Accuracy vs Token Cost.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            num_ticks: Number of major ticks on x-axis (default: 15)
            show_benchmark_average: Whether to show benchmark average points and line (default: True)
        """
        if not self.results:
            print("No data available for plotting.")
            return
        
        # Import ticker for custom tick formatting
        from matplotlib.ticker import LogLocator, FuncFormatter
        
        # Create figure with space for legends on the right
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate point sizes based on reasoning_ratio
        # Base size for reasoning_ratio = 0, then scale linearly
        base_size = 100
        max_size = 500
        size_range = max_size - base_size
        
        # Create scatter plot for each task and model type
        for task_type in self.task_colors.keys():
            if task_type == 'benchmark_average':
                continue  # Handle benchmark average separately
            
            task_results = [r for r in self.results if r['task_type'] == task_type]
            
            for model_type in set([r['model_type'] for r in task_results]):
                model_data = [r for r in task_results if r['model_type'] == model_type]
                
                if not model_data:
                    continue
                
                # Extract data
                x_data = [r['avg_total_tokens'] for r in model_data]
                y_data = [r['accuracy'] for r in model_data]
                reasoning_ratios = [r['reasoning_ratio'] for r in model_data]
                
                # Calculate sizes: base_size + (reasoning_ratio * size_range)
                sizes = [base_size + (ratio * size_range) for ratio in reasoning_ratios]
                
                # Plot with log scale on x-axis
                ax.scatter(
                    x_data,
                    y_data,
                    c=self.task_colors[task_type],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=3
                )
        
        # Plot benchmark average points and connecting line
        if show_benchmark_average and self.overall_results:
            # Sort by model name for consistent line drawing
            sorted_overall = sorted(self.overall_results, 
                                   key=lambda x: x['avg_total_tokens'])
            
            # Extract data for line plot
            line_x = [r['avg_total_tokens'] for r in sorted_overall]
            line_y = [r['accuracy'] for r in sorted_overall]
            
            # Draw connecting line (semi-transparent gray)
            ax.plot(line_x, line_y, 
                   color='#808080', 
                   alpha=0.3, 
                   linewidth=2.5, 
                   linestyle='-',
                   zorder=1,
                   label='Benchmark Average Trend')
            
            # Plot benchmark average points
            for result in self.overall_results:
                model_type = result['model_type']
                x = result['avg_total_tokens']
                y = result['accuracy']
                reasoning_ratio = result['reasoning_ratio']
                
                # Calculate size
                size = base_size + (reasoning_ratio * size_range)
                
                # Plot point
                ax.scatter(
                    x, y,
                    c=self.task_colors['benchmark_average'],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=size,
                    alpha=0.5,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=2
                )
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        
        # Customize x-axis ticks
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=num_ticks))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
        
        # Format tick labels for readability
        def format_func(value, tick_number):
            if value >= 10000:
                return f'{value/1000:.0f}k'
            elif value >= 1000:
                return f'{value/1000:.1f}k'
            else:
                return f'{int(value)}'
        
        ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Labels and title
        ax.set_xlabel('Average Total Tokens (log scale)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (Exact Match Rate)', fontsize=13, fontweight='bold')
        
        title = 'Model Accuracy vs Token Cost Trade-off'
        if show_benchmark_average:
            title += '\n(Gray points & line = Benchmark Average across all tasks)'
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        # Set y-axis limits and formatting
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Grid - separate for major and minor
        ax.grid(True, which='major', alpha=0.3, linestyle='--', linewidth=1.0, zorder=0)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5, zorder=0)
        
        # Create custom legends
        self._create_legends(ax, show_benchmark_average)
        
        # Adjust layout to accommodate legends
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _create_legends(self, ax, show_benchmark_average=True):
        """
        Create custom legends for task types, model types, and reasoning ratios.
        
        Args:
            ax: Matplotlib axes object
            show_benchmark_average: Whether benchmark average is shown
        """
        legend_y_start = 1.0
        legend_spacing = 0.30
        
        # Legend 1: Task Types (colors)
        task_patches = []
        for task_type, color in self.task_colors.items():
            if task_type == 'benchmark_average':
                if show_benchmark_average and self.overall_results:
                    label = 'Benchmark Average'
                    task_patches.append(Patch(color=color, label=label, alpha=0.5))
            elif any(r['task_type'] == task_type for r in self.results):
                # Format label
                label = task_type.replace('_', ' ').title()
                label = label.replace('Dna', 'DNA').replace('Rna', 'RNA')
                task_patches.append(Patch(color=color, label=label))
        
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
        unique_model_types = sorted(set([r['model_type'] for r in self.results] + 
                                        [r['model_type'] for r in self.overall_results]))
        
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
    
    def print_summary_statistics(self):
        """Print summary statistics of the results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        print(f"\nTotal data points: {len(self.results)}")
        print(f"Unique models: {len(set([r['model_name'] for r in self.results]))}")
        print(f"Models: {', '.join(sorted(set([r['model_name'] for r in self.results])))}")
        
        # Print benchmark average statistics
        if self.overall_results:
            print("\n" + "-"*80)
            print("Benchmark Average Statistics (across all tasks):")
            print("-"*80)
            
            for result in sorted(self.overall_results, key=lambda x: x['model_name']):
                print(f"\n{result['model_name']}:")
                print(f"  Overall Accuracy: {result['accuracy']:.3f}")
                print(f"  Avg Total Tokens: {result['avg_total_tokens']:.1f}")
                print(f"  Reasoning Ratio: {result['reasoning_ratio']:.3f}")
        
        print("\n" + "-"*80)
        print("Per-task statistics:")
        print("-"*80)
        
        for task in ['dna_complement', 'rna_complement', 'protein_three_to_one', 'protein_one_to_three']:
            task_results = [r for r in self.results if r['task_type'] == task]
            if task_results:
                accuracies = [r['accuracy'] for r in task_results]
                tokens = [r['avg_total_tokens'] for r in task_results]
                reasoning_ratios = [r['reasoning_ratio'] for r in task_results]
                
                print(f"\n{task.replace('_', ' ').title()}:")
                print(f"  Number of models: {len(task_results)}")
                print(f"  Accuracy: mean={np.mean(accuracies):.3f}, "
                      f"min={np.min(accuracies):.3f}, max={np.max(accuracies):.3f}")
                print(f"  Avg tokens: mean={np.mean(tokens):.1f}, "
                      f"min={np.min(tokens):.1f}, max={np.max(tokens):.1f}")
                print(f"  Reasoning ratio: mean={np.mean(reasoning_ratios):.3f}, "
                      f"min={np.min(reasoning_ratios):.3f}, max={np.max(reasoning_ratios):.3f}")
        
        print("\n" + "="*80 + "\n")


def main():
    """
    Main function for creating the visualization.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║         BENCHMARK ACCURACY VS TOKEN COST VISUALIZER                      ║
    ║                                                                          ║
    ║  Visualize the trade-off between model accuracy and token consumption    ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get base directory
    base_dir = input("Enter the path to your results directory (or press Enter for current directory): ").strip()
    
    if not base_dir:
        base_dir = "."
    
    if not os.path.exists(base_dir):
        print(f"\nError: Directory '{base_dir}' does not exist!")
        return
    
    # Initialize visualizer
    visualizer = BenchmarkVisualizer(base_dir)
    
    # Load data
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("\nNo valid results found in the specified directory.")
        print("Expected structure: base_dir/model_name_timestamp/metrics.json")
        return
    
    # Print summary statistics
    visualizer.print_summary_statistics()
    
    # Ask whether to show benchmark average
    show_avg = input("\nShow benchmark average points and trend line? (y/n, default: y): ").strip().lower()
    show_benchmark_average = (show_avg == '' or show_avg == 'y')
    
    # Ask for output path
    save_plot = input("\nSave plot? (y/n, default: y): ").strip().lower()
    
    if save_plot == '' or save_plot == 'y':
        output_file = input("Enter output filename (default: 'accuracy_vs_token_cost.png'): ").strip()
        if not output_file:
            output_file = os.path.join(base_dir, 'accuracy_vs_token_cost.png')
        save_path = output_file
    else:
        save_path = None
    
    # Generate the plot
    print("\nGenerating accuracy vs token cost visualization...")
    visualizer.plot_accuracy_vs_token_cost(save_path=save_path, 
                                          show_benchmark_average=show_benchmark_average)
    
    print("\nVisualization complete!")


def quick_generate(base_dir: str, output_file: str = "accuracy_vs_token_cost.png",
                  figsize: tuple = (14, 8), num_ticks: int = 15,
                  show_benchmark_average: bool = True):
    """
    Quick function to generate the plot with one command.
    
    Args:
        base_dir: Directory containing result folders
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        num_ticks: Number of major ticks on x-axis (default: 15)
        show_benchmark_average: Whether to show benchmark average (default: True)
    
    Example:
        >>> quick_generate("./results", "my_plot.png")
        >>> quick_generate("./results", "large_plot.png", figsize=(18, 10))
        >>> quick_generate("./results", "plot.png", show_benchmark_average=False)
    """
    visualizer = BenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    visualizer.print_summary_statistics()
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file, 
                                          figsize=figsize, 
                                          num_ticks=num_ticks,
                                          show_benchmark_average=show_benchmark_average)


def customize_visualization(base_dir: str, 
                           output_file: str = "accuracy_vs_token_cost.png",
                           figsize: tuple = (14, 8),
                           custom_colors: dict = None,
                           custom_markers: dict = None,
                           show_benchmark_average: bool = True):
    """
    Generate plot with custom colors and markers.
    
    Args:
        base_dir: Directory containing result folders
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        custom_colors: Custom color mapping for task types
        custom_markers: Custom marker mapping for model types
        show_benchmark_average: Whether to show benchmark average (default: True)
    
    Example:
        >>> custom_colors = {
        ...     'dna_complement': '#FF0000',
        ...     'rna_complement': '#00FF00',
        ...     'benchmark_average': '#888888'
        ... }
        >>> custom_markers = {
        ...     'gpt-4': 'o',
        ...     'deepseek-r1': '^'
        ... }
        >>> customize_visualization("./results", "custom.png", 
        ...                        custom_colors=custom_colors, 
        ...                        custom_markers=custom_markers)
    """
    visualizer = BenchmarkVisualizer(base_dir)
    
    # Override colors if provided
    if custom_colors:
        visualizer.task_colors.update(custom_colors)
    
    # Override markers if provided
    if custom_markers:
        visualizer.model_markers.update(custom_markers)
    
    visualizer.load_benchmark_results()
    visualizer.print_summary_statistics()
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file, 
                                          figsize=figsize,
                                          show_benchmark_average=show_benchmark_average)


def export_plot_data(base_dir: str, output_file: str = "plot_data.csv"):
    """
    Export the prepared plot data to CSV for external analysis.
    
    Args:
        base_dir: Directory containing result folders
        output_file: Output CSV file path
    
    Example:
        >>> df = export_plot_data("./results", "my_data.csv")
        >>> print(df.head())
    
    Returns:
        DataFrame with plot data
    """
    import pandas as pd
    
    visualizer = BenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available to export.")
        return None
    
    # Combine task-specific and overall results
    all_data = visualizer.results + visualizer.overall_results
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Plot data exported to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"Total data points: {len(df)}")
    print(f"  - Task-specific: {len(visualizer.results)}")
    print(f"  - Benchmark average: {len(visualizer.overall_results)}")
    print(f"Number of models: {df['model_name'].nunique()}")
    print(f"Task types: {', '.join(df['task_type'].unique())}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    
    print("\nPer-model summary:")
    print("-"*80)
    summary = df.groupby(['model_name', 'task_type']).agg({
        'accuracy': 'mean',
        'avg_total_tokens': 'mean',
        'reasoning_ratio': 'mean'
    }).round(4)
    print(summary)
    print("="*80 + "\n")
    
    return df


def compare_models(base_dir: str, model_names: list = None, 
                  output_file: str = "model_comparison.png",
                  show_benchmark_average: bool = True):
    """
    Create a comparison plot for specific models.
    
    Args:
        base_dir: Directory containing result folders
        model_names: List of model names to compare (None = all models)
        output_file: Output filename for the plot
        show_benchmark_average: Whether to show benchmark average (default: True)
    
    Example:
        >>> compare_models("./results", 
        ...               model_names=['gpt-4o', 'deepseek-r1', 'claude-3'],
        ...               output_file="top_models.png")
    """
    visualizer = BenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    # Filter results if specific models requested
    if model_names:
        visualizer.results = [r for r in visualizer.results 
                             if r['model_name'] in model_names]
        visualizer.overall_results = [r for r in visualizer.overall_results
                                     if r['model_name'] in model_names]
        print(f"\nFiltered to {len(visualizer.results)} task-specific data points and "
              f"{len(visualizer.overall_results)} benchmark average points from models: {', '.join(model_names)}")
    
    if not visualizer.results:
        print("No data available for the specified models.")
        return
    
    visualizer.print_summary_statistics()
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file,
                                          show_benchmark_average=show_benchmark_average)


def analyze_task_performance(base_dir: str, task_type: str, 
                            output_file: str = None):
    """
    Analyze performance for a specific task type.
    
    Args:
        base_dir: Directory containing result folders
        task_type: Task type to analyze ('dna_complement', 'rna_complement', etc.)
        output_file: Output filename for the plot
    
    Example:
        >>> analyze_task_performance("./results", "dna_complement", 
        ...                         "dna_analysis.png")
    """
    visualizer = BenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    # Filter for specific task
    visualizer.results = [r for r in visualizer.results 
                         if r['task_type'] == task_type]
    
    if not visualizer.results:
        print(f"No data available for task: {task_type}")
        return
    
    print(f"\nAnalyzing task: {task_type}")
    print(f"Found {len(visualizer.results)} model evaluations")
    
    # Create output filename if not provided
    if output_file is None:
        output_file = f"{task_type}_analysis.png"
    
    # Print task-specific statistics
    accuracies = [r['accuracy'] for r in visualizer.results]
    tokens = [r['avg_total_tokens'] for r in visualizer.results]
    reasoning_ratios = [r['reasoning_ratio'] for r in visualizer.results]
    
    print("\n" + "="*80)
    print(f"TASK ANALYSIS: {task_type.upper().replace('_', ' ')}")
    print("="*80)
    print(f"\nAccuracy Statistics:")
    print(f"  Mean: {np.mean(accuracies):.3f}")
    print(f"  Std:  {np.std(accuracies):.3f}")
    print(f"  Min:  {np.min(accuracies):.3f}")
    print(f"  Max:  {np.max(accuracies):.3f}")
    
    print(f"\nToken Usage Statistics:")
    print(f"  Mean: {np.mean(tokens):.1f}")
    print(f"  Std:  {np.std(tokens):.1f}")
    print(f"  Min:  {np.min(tokens):.1f}")
    print(f"  Max:  {np.max(tokens):.1f}")
    
    print(f"\nReasoning Ratio Statistics:")
    print(f"  Mean: {np.mean(reasoning_ratios):.3f}")
    print(f"  Std:  {np.std(reasoning_ratios):.3f}")
    print(f"  Min:  {np.min(reasoning_ratios):.3f}")
    print(f"  Max:  {np.max(reasoning_ratios):.3f}")
    
    print("\n" + "="*80 + "\n")
    
    # Generate plot (without benchmark average for single task)
    visualizer.plot_accuracy_vs_token_cost(save_path=output_file, 
                                          show_benchmark_average=False)


def generate_benchmark_average_only(base_dir: str, 
                                   output_file: str = "benchmark_average.png",
                                   figsize: tuple = (14, 8)):
    """
    Generate a plot showing only the benchmark average points and trend line.
    
    Args:
        base_dir: Directory containing result folders
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
    
    Example:
        >>> generate_benchmark_average_only("./results", "avg_only.png")
    """
    from matplotlib.ticker import LogLocator, FuncFormatter
    
    visualizer = BenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.overall_results:
        print("No benchmark average data available.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate point sizes
    base_size = 100
    max_size = 500
    size_range = max_size - base_size
    
    # Sort by token usage for line plot
    sorted_overall = sorted(visualizer.overall_results, 
                           key=lambda x: x['avg_total_tokens'])
    
    # Extract data for line plot
    line_x = [r['avg_total_tokens'] for r in sorted_overall]
    line_y = [r['accuracy'] for r in sorted_overall]
    
    # Draw connecting line
    ax.plot(line_x, line_y, 
           color='#808080', 
           alpha=0.4, 
           linewidth=3, 
           linestyle='-',
           zorder=1,
           label='Benchmark Average Trend')
    
    # Plot benchmark average points
    for result in visualizer.overall_results:
        model_type = result['model_type']
        x = result['avg_total_tokens']
        y = result['accuracy']
        reasoning_ratio = result['reasoning_ratio']
        
        # Calculate size
        size = base_size + (reasoning_ratio * size_range)
        
        # Plot point
        ax.scatter(
            x, y,
            c='#808080',
            marker=visualizer.model_markers.get(model_type, 'X'),
            s=size,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.5,
            zorder=2
        )
        
        # Add model name label
        ax.annotate(result['model_name'], 
                   xy=(x, y), 
                   xytext=(5, 5),
                   textcoords='offset points',
                   fontsize=8,
                   alpha=0.7)
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    # Customize x-axis ticks
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    
    # Format tick labels
    def format_func(value, tick_number):
        if value >= 10000:
            return f'{value/1000:.0f}k'
        elif value >= 1000:
            return f'{value/1000:.1f}k'
        else:
            return f'{int(value)}'
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Labels and title
    ax.set_xlabel('Average Total Tokens (log scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Overall Accuracy (Exact Match Rate)', fontsize=13, fontweight='bold')
    ax.set_title('Benchmark Average: Model Accuracy vs Token Cost\n(Averaged across all tasks)', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Set y-axis limits and formatting
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Grid
    ax.grid(True, which='major', alpha=0.3, linestyle='--', linewidth=1.0, zorder=0)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5, zorder=0)
    
    # Create legend for model types
    unique_model_types = sorted(set([r['model_type'] for r in visualizer.overall_results]))
    model_type_lines = []
    
    for model_type in unique_model_types:
        marker = visualizer.model_markers.get(model_type, 'X')
        if model_type == 'default':
            label = 'Other'
        else:
            label = model_type.upper().replace('-', ' ').title()
        
        model_type_lines.append(Line2D([0], [0], marker=marker, color='gray', 
                                      linestyle='', markersize=9, 
                                      markeredgewidth=1.3, markeredgecolor='black',
                                      label=label, alpha=0.7))
    
    ax.legend(handles=model_type_lines,
             title='Model Types',
             loc='best',
             frameon=True,
             fontsize=9,
             title_fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Benchmark average plot saved to {output_file}")
    
    plt.show()


def create_combined_report(base_dir: str, output_dir: str = "./report"):
    """
    Generate a comprehensive report with multiple visualizations.
    
    Args:
        base_dir: Directory containing result folders
        output_dir: Directory to save report files
    
    Example:
        >>> create_combined_report("./results", "./my_report")
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE BENCHMARK REPORT")
    print("="*80 + "\n")
    
    visualizer = BenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available for report generation.")
        return
    
    # 1. Full visualization with benchmark average
    print("1. Generating full visualization with benchmark average...")
    full_plot_path = os.path.join(output_dir, "full_analysis.png")
    visualizer.plot_accuracy_vs_token_cost(save_path=full_plot_path, 
                                          show_benchmark_average=True,
                                          figsize=(16, 10))
    
    # 2. Benchmark average only
    print("\n2. Generating benchmark average plot...")
    avg_plot_path = os.path.join(output_dir, "benchmark_average.png")
    generate_benchmark_average_only(base_dir, avg_plot_path, figsize=(14, 8))
    
    # 3. Per-task analyses
    print("\n3. Generating per-task analyses...")
    tasks = ['dna_complement', 'rna_complement', 'protein_three_to_one', 'protein_one_to_three']
    for task in tasks:
        task_plot_path = os.path.join(output_dir, f"{task}_analysis.png")
        print(f"   - Analyzing {task}...")
        analyze_task_performance(base_dir, task, task_plot_path)
    
    # 4. Export data to CSV
    print("\n4. Exporting data to CSV...")
    csv_path = os.path.join(output_dir, "benchmark_data.csv")
    export_plot_data(base_dir, csv_path)
    
    # 5. Generate summary statistics text file
    print("\n5. Generating summary statistics...")
    import sys
    from io import StringIO
    
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = summary_output = StringIO()
    
    visualizer.print_summary_statistics()
    
    sys.stdout = old_stdout
    summary_text = summary_output.getvalue()
    
    summary_path = os.path.join(output_dir, "summary_statistics.txt")
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"Summary statistics saved to {summary_path}")
    
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)
    print(f"\nReport files saved to: {output_dir}")
    print("Generated files:")
    print(f"  - full_analysis.png")
    print(f"  - benchmark_average.png")
    for task in tasks:
        print(f"  - {task}_analysis.png")
    print(f"  - benchmark_data.csv")
    print(f"  - summary_statistics.txt")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Run interactive main function
    main()