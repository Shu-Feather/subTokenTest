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


class MorseCaesarBenchmarkVisualizer:
    """
    Visualizer for Morse/Caesar cipher benchmark: Success Rate vs Token Cost analysis.
    Creates scatter plots showing the trade-off between success rate and computational cost.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the visualizer with a base directory containing JSON result files.
        
        Args:
            base_dir: Path to directory containing result JSON files
        """
        self.base_dir = Path(base_dir)
        self.results = []
        self.model_averages = []  # Store average metrics for each model
        
        # Define color palette for task types
        self.task_colors = {
            'morse_encode': '#1f77b4',      # Blue
            'morse_decode': '#ff7f0e',      # Orange
            'caesar_encode': '#2ca02c',     # Green
            'caesar_decode': '#d62728',     # Red
            'benchmark_average': '#808080'  # Gray for averages
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
    
    def extract_model_name(self, filename: str) -> str:
        """
        Extract simplified model name from filename.
        Examples: 'deepseek-r1_experiment_1209.json' -> 'deepseek-r1'
                  'gpt-4o_experiment_1209.json' -> 'gpt-4o'
        
        Args:
            filename: Full filename
            
        Returns:
            Extracted model name
        """
        basename = os.path.basename(filename)
        # Remove '_experiment_*' and '.json'
        model_name = re.sub(r'_experiment_.*\.json$', '', basename)
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
    
    def calculate_metrics_per_task(self, data: list) -> dict:
        """
        Calculate metrics for each task type separately from JSON data.
        
        Args:
            data: List of evaluation results
            
        Returns:
            Dictionary with task-specific metrics
        """
        from collections import defaultdict
        
        # Organize data by task type
        task_data = defaultdict(lambda: {
            'correct': 0, 
            'total': 0, 
            'total_tokens': [], 
            'reasoning_tokens': []
        })
        
        for item in data:
            task_type = item['task_type']
            is_correct = item['is_correct']
            
            task_data[task_type]['total'] += 1
            if is_correct:
                task_data[task_type]['correct'] += 1
            
            # Collect token usage for this specific task
            if 'token_usage' in item:
                total_tokens = item['token_usage'].get('total_tokens', 0)
                reasoning_tokens = item['token_usage'].get('reasoning_tokens', 0)
                
                task_data[task_type]['total_tokens'].append(total_tokens)
                task_data[task_type]['reasoning_tokens'].append(reasoning_tokens)
        
        # Calculate metrics for each task
        results = {}
        for task_type, metrics in task_data.items():
            success_rate = (
                metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0
            )
            
            avg_total_tokens = (
                np.mean(metrics['total_tokens']) if metrics['total_tokens'] else 0
            )
            avg_reasoning_tokens = (
                np.mean(metrics['reasoning_tokens']) if metrics['reasoning_tokens'] else 0
            )
            
            reasoning_ratio = (
                avg_reasoning_tokens / avg_total_tokens 
                if avg_total_tokens > 0 else 0
            )
            
            results[task_type] = {
                'success_rate': success_rate,
                'avg_total_tokens': avg_total_tokens,
                'reasoning_ratio': reasoning_ratio
            }
        
        return results
    
    def load_benchmark_results(self):
        """
        Load all benchmark results from JSON files in the directory.
        """
        print("Loading benchmark results for visualization...")
        
        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(self.base_dir, '*.json'))
        
        if not json_files:
            print(f"No JSON files found in {self.base_dir}")
            return
        
        for json_file in json_files:
            model_name = self.extract_model_name(json_file)
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Calculate metrics per task
                task_metrics = self.calculate_metrics_per_task(data)
                
                # Store individual task results
                for task_type, metrics in task_metrics.items():
                    result = {
                        'model_name': model_name,
                        'model_type': self.get_model_type(model_name),
                        'task_type': task_type,
                        'success_rate': metrics['success_rate'],
                        'avg_total_tokens': metrics['avg_total_tokens'],
                        'reasoning_ratio': metrics['reasoning_ratio'],
                        'filename': os.path.basename(json_file)
                    }
                    self.results.append(result)
                
                # Calculate and store model average across all tasks
                avg_success_rate = np.mean([m['success_rate'] for m in task_metrics.values()])
                avg_total_tokens = np.mean([m['avg_total_tokens'] for m in task_metrics.values()])
                avg_reasoning_ratio = np.mean([m['reasoning_ratio'] for m in task_metrics.values()])
                
                model_avg = {
                    'model_name': model_name,
                    'model_type': self.get_model_type(model_name),
                    'task_type': 'benchmark_average',
                    'success_rate': avg_success_rate,
                    'avg_total_tokens': avg_total_tokens,
                    'reasoning_ratio': avg_reasoning_ratio,
                    'filename': os.path.basename(json_file)
                }
                self.model_averages.append(model_avg)
                    
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Loaded {len(self.results)} data points from {len(json_files)} result files")
        print(f"Calculated {len(self.model_averages)} model averages")
        
        # Print per-model summary
        models = {}
        for r in self.results:
            if r['model_name'] not in models:
                models[r['model_name']] = 0
            models[r['model_name']] += 1
        
        for model, count in sorted(models.items()):
            print(f"  - {model}: {count} task evaluations")
        print()
    
    def plot_success_rate_vs_token_cost(self, save_path: str = None, 
                                        figsize: tuple = (14, 8)):
        """
        Create the main scatter plot: Success Rate vs Token Cost.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
        """
        if not self.results:
            print("No data available for plotting.")
            return
        
        # Create figure with space for legends on the right
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate point sizes based on reasoning_ratio
        # Base size for reasoning_ratio = 0, then scale linearly
        base_size = 100
        max_size = 500
        size_range = max_size - base_size
        
        # Plot individual task results
        for task_type in ['morse_encode', 'morse_decode', 'caesar_encode', 'caesar_decode']:
            task_results = [r for r in self.results if r['task_type'] == task_type]
            
            for model_type in set([r['model_type'] for r in task_results]):
                model_data = [r for r in task_results if r['model_type'] == model_type]
                
                if not model_data:
                    continue
                
                # Extract data
                x_data = [r['avg_total_tokens'] for r in model_data]
                y_data = [r['success_rate'] for r in model_data]
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
        
        # Plot benchmark averages (gray points)
        # Sort by avg_total_tokens for line connection
        sorted_averages = sorted(self.model_averages, key=lambda x: x['avg_total_tokens'])
        
        for avg_data in sorted_averages:
            model_type = avg_data['model_type']
            size = base_size + (avg_data['reasoning_ratio'] * size_range)
            
            ax.scatter(
                avg_data['avg_total_tokens'],
                avg_data['success_rate'],
                c=self.task_colors['benchmark_average'],
                marker=self.model_markers.get(model_type, 'X'),
                s=size,
                alpha=0.5,
                edgecolors='black',
                linewidths=1.5,
                zorder=2
            )
        
        # Connect benchmark average points with gray line
        avg_x = [r['avg_total_tokens'] for r in sorted_averages]
        avg_y = [r['success_rate'] for r in sorted_averages]
        ax.plot(avg_x, avg_y, 
               color='gray', 
               linestyle='--', 
               linewidth=2, 
               alpha=0.3,
               zorder=1,
               label='Benchmark Average Trend')
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        
        # Manually set specific tick locations
        # Adjust these based on your data range
        tick_locations = [1000, 2000, 5000, 10000, 20000]
        ax.set_xticks(tick_locations)
        
        # Format the labels
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Add minor ticks
        from matplotlib.ticker import LogLocator
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.grid(True, which='minor', alpha=0.2, linestyle=':')
        ax.grid(True, which='major', alpha=0.3, linestyle='--')
        
        # Labels and title
        ax.set_xlabel('Average Total Tokens (log scale)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Success Rate (Exact Match Rate)', fontsize=13, fontweight='bold')
        ax.set_title('Model Success Rate vs Token Cost Trade-off', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Set y-axis limits and formatting
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Create custom legends
        self._create_legends(ax)
        
        # Adjust layout to accommodate legends
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _create_legends(self, ax):
        """
        Create custom legends for task types, model types, and reasoning ratios.
        
        Args:
            ax: Matplotlib axes object
        """
        legend_y_start = 1.0
        legend_spacing = 0.30
        
        # Legend 1: Task Types (colors) - including benchmark average
        task_patches = []
        for task_type, color in self.task_colors.items():
            if task_type == 'benchmark_average':
                # Check if we have model averages
                if self.model_averages:
                    task_patches.append(Patch(color=color, label='Benchmark Average', alpha=0.5))
            else:
                # Check if we have data for this task
                if any(r['task_type'] == task_type for r in self.results):
                    # Format label
                    label = task_type.replace('_', ' ').title()
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
        unique_model_types = sorted(set([r['model_type'] for r in self.results]))
        
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
        
        print("\n" + "-"*80)
        print("Per-task statistics:")
        print("-"*80)
        
        for task in ['morse_encode', 'morse_decode', 'caesar_encode', 'caesar_decode']:
            task_results = [r for r in self.results if r['task_type'] == task]
            if task_results:
                success_rates = [r['success_rate'] for r in task_results]
                tokens = [r['avg_total_tokens'] for r in task_results]
                reasoning_ratios = [r['reasoning_ratio'] for r in task_results]
                
                print(f"\n{task.replace('_', ' ').title()}:")
                print(f"  Number of models: {len(task_results)}")
                print(f"  Success rate: mean={np.mean(success_rates):.3f}, "
                      f"min={np.min(success_rates):.3f}, max={np.max(success_rates):.3f}")
                print(f"  Avg tokens: mean={np.mean(tokens):.1f}, "
                      f"min={np.min(tokens):.1f}, max={np.max(tokens):.1f}")
                print(f"  Reasoning ratio: mean={np.mean(reasoning_ratios):.3f}, "
                      f"min={np.min(reasoning_ratios):.3f}, max={np.max(reasoning_ratios):.3f}")
        
        # Benchmark average statistics
        print("\n" + "-"*80)
        print("Benchmark Average (across all tasks):")
        print("-"*80)
        
        if self.model_averages:
            avg_success_rates = [r['success_rate'] for r in self.model_averages]
            avg_tokens = [r['avg_total_tokens'] for r in self.model_averages]
            avg_reasoning_ratios = [r['reasoning_ratio'] for r in self.model_averages]
            
            print(f"\n  Number of models: {len(self.model_averages)}")
            print(f"  Success rate: mean={np.mean(avg_success_rates):.3f}, "
                  f"min={np.min(avg_success_rates):.3f}, max={np.max(avg_success_rates):.3f}")
            print(f"  Avg tokens: mean={np.mean(avg_tokens):.1f}, "
                  f"min={np.min(avg_tokens):.1f}, max={np.max(avg_tokens):.1f}")
            print(f"  Reasoning ratio: mean={np.mean(avg_reasoning_ratios):.3f}, "
                  f"min={np.min(avg_reasoning_ratios):.3f}, max={np.max(avg_reasoning_ratios):.3f}")
        
        # Per-model statistics
        print("\n" + "-"*80)
        print("Per-model statistics:")
        print("-"*80)
        
        for model_name in sorted(set([r['model_name'] for r in self.results])):
            model_results = [r for r in self.results if r['model_name'] == model_name]
            model_avg = [r for r in self.model_averages if r['model_name'] == model_name]
            
            if model_results:
                success_rates = [r['success_rate'] for r in model_results]
                tokens = [r['avg_total_tokens'] for r in model_results]
                reasoning_ratios = [r['reasoning_ratio'] for r in model_results]
                
                print(f"\n{model_name}:")
                print(f"  Tasks evaluated: {len(model_results)}")
                
                # Show per-task breakdown
                for task in ['morse_encode', 'morse_decode', 'caesar_encode', 'caesar_decode']:
                    task_data = [r for r in model_results if r['task_type'] == task]
                    if task_data:
                        td = task_data[0]
                        print(f"    {task}: SR={td['success_rate']:.3f}, "
                              f"Tokens={td['avg_total_tokens']:.1f}, "
                              f"RR={td['reasoning_ratio']:.3f}")
                
                # Show average
                if model_avg:
                    ma = model_avg[0]
                    print(f"  Benchmark Average:")
                    print(f"    Success Rate: {ma['success_rate']:.3f}")
                    print(f"    Avg Tokens: {ma['avg_total_tokens']:.1f}")
                    print(f"    Reasoning Ratio: {ma['reasoning_ratio']:.3f}")
        
        print("\n" + "="*80 + "\n")


def main():
    """
    Main function for creating the visualization.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║         MORSE/CAESAR BENCHMARK SUCCESS RATE VS TOKEN COST VISUALIZER     ║
    ║                                                                          ║
    ║  Visualize the trade-off between model success rate and token usage      ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get base directory
    base_dir = input("Enter the path to your output directory (or press Enter for './output'): ").strip()
    
    if not base_dir:
        base_dir = "./output"
    
    if not os.path.exists(base_dir):
        print(f"\nError: Directory '{base_dir}' does not exist!")
        return
    
    # Initialize visualizer
    visualizer = MorseCaesarBenchmarkVisualizer(base_dir)
    
    # Load data
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("\nNo valid results found in the specified directory.")
        print("Expected structure: *.json files in the output directory")
        return
    
    # Print summary statistics
    visualizer.print_summary_statistics()
    
    # Ask for output path
    save_plot = input("\nSave plot? (y/n, default: y): ").strip().lower()
    
    if save_plot == '' or save_plot == 'y':
        output_file = input("Enter output filename (default: 'success_rate_vs_token_cost.png'): ").strip()
        if not output_file:
            output_file = os.path.join(base_dir, 'success_rate_vs_token_cost.png')
        save_path = output_file
    else:
        save_path = None
    
    # Generate the plot
    print("\nGenerating success rate vs token cost visualization...")
    visualizer.plot_success_rate_vs_token_cost(save_path=save_path)
    
    print("\nVisualization complete!")


def quick_generate(base_dir: str = "./output", 
                  output_file: str = "success_rate_vs_token_cost.png",
                  figsize: tuple = (14, 8)):
    """
    Quick function to generate the plot with one command.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
    
    Example:
        >>> quick_generate("./output", "my_plot.png")
        >>> quick_generate("./output", "large_plot.png", figsize=(18, 10))
    """
    visualizer = MorseCaesarBenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    visualizer.print_summary_statistics()
    visualizer.plot_success_rate_vs_token_cost(save_path=output_file, figsize=figsize)


def customize_visualization(base_dir: str = "./output", 
                           output_file: str = "success_rate_vs_token_cost.png",
                           figsize: tuple = (14, 8),
                           custom_colors: dict = None,
                           custom_markers: dict = None):
    """
    Generate plot with custom colors and markers.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        custom_colors: Custom color mapping for task types
        custom_markers: Custom marker mapping for model types
    
    Example:
        >>> custom_colors = {
        ...     'morse_encode': '#FF0000',
        ...     'morse_decode': '#00FF00'
        ... }
        >>> custom_markers = {
        ...     'gpt-4': 'o',
        ...     'deepseek-r1': '^'
        ... }
        >>> customize_visualization("./output", "custom.png", 
        ...                        custom_colors=custom_colors, 
        ...                        custom_markers=custom_markers)
    """
    visualizer = MorseCaesarBenchmarkVisualizer(base_dir)
    
    # Override colors if provided
    if custom_colors:
        visualizer.task_colors.update(custom_colors)
    
    # Override markers if provided
    if custom_markers:
        visualizer.model_markers.update(custom_markers)
    
    visualizer.load_benchmark_results()
    visualizer.print_summary_statistics()
    visualizer.plot_success_rate_vs_token_cost(save_path=output_file, figsize=figsize)


def export_plot_data(base_dir: str = "./output", output_file: str = "plot_data.csv"):
    """
    Export the prepared plot data to CSV for external analysis.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output CSV file path
    
    Example:
        >>> df = export_plot_data("./output", "my_data.csv")
        >>> print(df.head())
    
    Returns:
        DataFrame with plot data
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required for data export. Install with: pip install pandas")
        return None
    
    visualizer = MorseCaesarBenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available to export.")
        return None
    
    # Combine regular results and model averages
    all_data = self.results + visualizer.model_averages
    
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
    print(f"Number of models: {df['model_name'].nunique()}")
    print(f"Task types: {', '.join(df['task_type'].unique())}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    print("\nPer-model summary:")
    print("-"*80)
    summary = df.groupby('model_name').agg({
        'success_rate': 'mean',
        'avg_total_tokens': 'mean',
        'reasoning_ratio': 'mean'
    }).round(4)
    print(summary)
    print("="*80 + "\n")
    
    return df


def compare_models(base_dir: str = "./output", 
                  model_names: list = None, 
                  output_file: str = "model_comparison.png"):
    """
    Create a comparison plot for specific models.
    
    Args:
        base_dir: Directory containing result JSON files
        model_names: List of model names to compare (None = all models)
        output_file: Output filename for the plot
    
    Example:
    >>> compare_models("./output", 
        ...               model_names=['gpt-4o', 'deepseek-r1', 'claude-3'],
        ...               output_file="top_models.png")
    """
    visualizer = MorseCaesarBenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    # Filter results if specific models requested
    if model_names:
        visualizer.results = [r for r in visualizer.results 
                             if r['model_name'] in model_names]
        visualizer.model_averages = [r for r in visualizer.model_averages 
                                     if r['model_name'] in model_names]
        print(f"\nFiltered to {len(visualizer.results)} data points from models: {', '.join(model_names)}")
    
    if not visualizer.results:
        print("No data available for the specified models.")
        return
    
    visualizer.print_summary_statistics()
    visualizer.plot_success_rate_vs_token_cost(save_path=output_file)


def analyze_task_performance(base_dir: str = "./output", 
                            task_type: str = 'morse_encode', 
                            output_file: str = None):
    """
    Analyze performance for a specific task type.
    
    Args:
        base_dir: Directory containing result JSON files
        task_type: Task type to analyze ('morse_encode', 'morse_decode', etc.)
        output_file: Output filename for the plot
    
    Example:
        >>> analyze_task_performance("./output", "morse_encode", 
        ...                         "morse_encode_analysis.png")
    """
    visualizer = MorseCaesarBenchmarkVisualizer(base_dir)
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
    success_rates = [r['success_rate'] for r in visualizer.results]
    tokens = [r['avg_total_tokens'] for r in visualizer.results]
    reasoning_ratios = [r['reasoning_ratio'] for r in visualizer.results]
    
    print("\n" + "="*80)
    print(f"TASK ANALYSIS: {task_type.upper().replace('_', ' ')}")
    print("="*80)
    print(f"\nSuccess Rate Statistics:")
    print(f"  Mean: {np.mean(success_rates):.3f}")
    print(f"  Std:  {np.std(success_rates):.3f}")
    print(f"  Min:  {np.min(success_rates):.3f}")
    print(f"  Max:  {np.max(success_rates):.3f}")
    
    print(f"\nToken Usage Statistics (task-specific):")
    print(f"  Mean: {np.mean(tokens):.1f}")
    print(f"  Std:  {np.std(tokens):.1f}")
    print(f"  Min:  {np.min(tokens):.1f}")
    print(f"  Max:  {np.max(tokens):.1f}")
    
    print(f"\nReasoning Ratio Statistics:")
    print(f"  Mean: {np.mean(reasoning_ratios):.3f}")
    print(f"  Std:  {np.std(reasoning_ratios):.3f}")
    print(f"  Min:  {np.min(reasoning_ratios):.3f}")
    print(f"  Max:  {np.max(reasoning_ratios):.3f}")
    
    # Per-model performance for this task
    print(f"\nPer-model performance on {task_type}:")
    print("-"*80)
    for model_name in sorted(set([r['model_name'] for r in visualizer.results])):
        model_result = [r for r in visualizer.results if r['model_name'] == model_name][0]
        print(f"  {model_name}:")
        print(f"    Success Rate: {model_result['success_rate']:.3f}")
        print(f"    Avg Tokens: {model_result['avg_total_tokens']:.1f}")
        print(f"    Reasoning Ratio: {model_result['reasoning_ratio']:.3f}")
    
    print("\n" + "="*80 + "\n")
    
    # Don't include model averages for single task analysis
    visualizer.model_averages = []
    
    # Generate plot
    visualizer.plot_success_rate_vs_token_cost(save_path=output_file)


def batch_analyze_all_tasks(base_dir: str = "./output", 
                           output_dir: str = None):
    """
    Generate individual analysis plots for all task types.
    
    Args:
        base_dir: Directory containing result JSON files
        output_dir: Directory to save output plots (defaults to base_dir)
    
    Example:
        >>> batch_analyze_all_tasks("./output", "./analysis_plots")
    """
    if output_dir is None:
        output_dir = base_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    task_types = ['morse_encode', 'morse_decode', 'caesar_encode', 'caesar_decode']
    
    print("\n" + "="*80)
    print("BATCH ANALYSIS: ALL TASK TYPES")
    print("="*80 + "\n")
    
    for task_type in task_types:
        output_file = os.path.join(output_dir, f"{task_type}_analysis.png")
        print(f"\nAnalyzing {task_type}...")
        analyze_task_performance(base_dir, task_type, output_file)
    
    print("\n" + "="*80)
    print("BATCH ANALYSIS COMPLETE")
    print("="*80)
    print(f"All plots saved to: {output_dir}\n")


def create_comparison_table(base_dir: str = "./output", 
                           output_file: str = "model_comparison_table.txt"):
    """
    Create a formatted comparison table of all models.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output text file for the table
    
    Example:
        >>> create_comparison_table("./output", "comparison.txt")
    """
    visualizer = MorseCaesarBenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available.")
        return
    
    # Organize data by model and task
    from collections import defaultdict
    model_task_data = defaultdict(dict)
    
    for result in visualizer.results:
        model = result['model_name']
        task = result['task_type']
        model_task_data[model][task] = {
            'success_rate': result['success_rate'],
            'tokens': result['avg_total_tokens'],
            'reasoning_ratio': result['reasoning_ratio']
        }
    
    # Add model averages
    for avg_result in visualizer.model_averages:
        model = avg_result['model_name']
        model_task_data[model]['benchmark_average'] = {
            'success_rate': avg_result['success_rate'],
            'tokens': avg_result['avg_total_tokens'],
            'reasoning_ratio': avg_result['reasoning_ratio']
        }
    
    # Create table
    task_types = ['morse_encode', 'morse_decode', 'caesar_encode', 'caesar_decode', 'benchmark_average']
    
    output_lines = []
    output_lines.append("="*120)
    output_lines.append("MODEL COMPARISON TABLE")
    output_lines.append("="*120)
    output_lines.append("")
    
    # Header
    header = f"{'Model':<20} | {'Task':<18} | {'Success Rate':<12} | {'Avg Tokens':<12} | {'Reasoning Ratio':<15}"
    output_lines.append(header)
    output_lines.append("-"*120)
    
    # Data rows
    for model in sorted(model_task_data.keys()):
        for i, task in enumerate(task_types):
            if task in model_task_data[model]:
                data = model_task_data[model][task]
                model_col = model if i == 0 else ""
                
                # Format task name
                task_display = task.replace('_', ' ').title() if task != 'benchmark_average' else 'BENCHMARK AVG'
                
                row = (f"{model_col:<20} | {task_display:<18} | "
                      f"{data['success_rate']:>11.1%} | "
                      f"{data['tokens']:>11.1f} | "
                      f"{data['reasoning_ratio']:>14.3f}")
                output_lines.append(row)
        output_lines.append("-"*120)
    
    output_lines.append("="*120)
    
    # Write to file
    table_text = "\n".join(output_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(table_text)
    
    # Also print to console
    print(table_text)
    print(f"\nTable saved to: {output_file}")


def find_pareto_frontier(base_dir: str = "./output",
                        task_type: str = None,
                        use_average: bool = False,
                        output_file: str = "pareto_frontier.png"):
    """
    Identify and visualize the Pareto frontier (models with best success rate for given token cost).
    
    Args:
        base_dir: Directory containing result JSON files
        task_type: Specific task to analyze (None = all tasks)
        use_average: If True, use benchmark averages; if False, use individual task results
        output_file: Output filename for the plot
    
    Example:
        >>> find_pareto_frontier("./output", "morse_encode", output_file="pareto_morse.png")
        >>> find_pareto_frontier("./output", use_average=True, output_file="pareto_avg.png")
    """
    visualizer = MorseCaesarBenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available.")
        return
    
    # Select data source
    if use_average:
        results = visualizer.model_averages
        title_suffix = " - Benchmark Average"
    elif task_type:
        results = [r for r in visualizer.results if r['task_type'] == task_type]
        title_suffix = f" - {task_type.replace('_', ' ').title()}"
    else:
        results = visualizer.results
        title_suffix = " - All Tasks"
    
    if not results:
        print(f"No data available for the specified configuration.")
        return
    
    # Extract coordinates
    tokens = np.array([r['avg_total_tokens'] for r in results])
    success_rates = np.array([r['success_rate'] for r in results])
    model_names = [r['model_name'] for r in results]
    
    # Find Pareto frontier
    # A point is on the Pareto frontier if no other point has both lower tokens AND higher success rate
    pareto_indices = []
    
    for i in range(len(results)):
        is_pareto = True
        for j in range(len(results)):
            if i != j:
                # Check if j dominates i (lower tokens AND higher/equal success rate)
                if tokens[j] <= tokens[i] and success_rates[j] > success_rates[i]:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    
    # Sort Pareto points by tokens for plotting
    pareto_indices = sorted(pareto_indices, key=lambda i: tokens[i])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot all points
    for i, result in enumerate(results):
        color = visualizer.task_colors.get(result['task_type'], 'gray')
        marker = visualizer.model_markers.get(result['model_type'], 'o')
        
        ax.scatter(tokens[i], success_rates[i], 
                  c=color, marker=marker, s=200,
                  alpha=0.5, edgecolors='black', linewidths=1.5,
                  zorder=2)
    
    # Highlight Pareto frontier points
    pareto_tokens = [tokens[i] for i in pareto_indices]
    pareto_success = [success_rates[i] for i in pareto_indices]
    pareto_names = [model_names[i] for i in pareto_indices]
    
    ax.scatter(pareto_tokens, pareto_success,
              s=400, facecolors='none', edgecolors='red',
              linewidths=3, zorder=3, label='Pareto Frontier')
    
    # Connect Pareto points
    ax.plot(pareto_tokens, pareto_success,
           'r--', linewidth=2, alpha=0.5, zorder=1)
    
    # Annotate Pareto points
    for i, (t, s, name) in enumerate(zip(pareto_tokens, pareto_success, pareto_names)):
        ax.annotate(name, (t, s), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Labels and title
    ax.set_xlabel('Average Total Tokens (log scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=13, fontweight='bold')
    ax.set_title(f'Pareto Frontier Analysis{title_suffix}', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Set y-axis formatting
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Pareto frontier plot saved to {output_file}")
    
    plt.show()
    
    # Print Pareto frontier models
    print("\n" + "="*80)
    print("PARETO FRONTIER MODELS")
    print("="*80)
    print(f"Configuration: {title_suffix}")
    print(f"Number of Pareto optimal models: {len(pareto_indices)}")
    print("\nModels on the Pareto frontier (sorted by token usage):")
    print("-"*80)
    
    for i, idx in enumerate(pareto_indices):
        result = results[idx]
        print(f"\n{i+1}. {result['model_name']}")
        print(f"   Success Rate: {result['success_rate']:.1%}")
        print(f"   Avg Tokens: {result['avg_total_tokens']:.1f}")
        print(f"   Reasoning Ratio: {result['reasoning_ratio']:.3f}")
        print(f"   Task: {result['task_type']}")
    
    print("\n" + "="*80 + "\n")


def create_detailed_report(base_dir: str = "./output",
                          output_file: str = "detailed_report.txt"):
    """
    Generate a comprehensive text report with all statistics.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output text file for the report
    
    Example:
        >>> create_detailed_report("./output", "full_report.txt")
    """
    visualizer = MorseCaesarBenchmarkVisualizer(base_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available.")
        return
    
    output_lines = []
    
    # Header
    output_lines.append("="*100)
    output_lines.append("MORSE/CAESAR CIPHER BENCHMARK - DETAILED REPORT")
    output_lines.append("="*100)
    output_lines.append("")
    
    # Overview
    output_lines.append("OVERVIEW")
    output_lines.append("-"*100)
    output_lines.append(f"Total evaluations: {len(visualizer.results)}")
    output_lines.append(f"Number of models: {len(set([r['model_name'] for r in visualizer.results]))}")
    output_lines.append(f"Task types: morse_encode, morse_decode, caesar_encode, caesar_decode")
    output_lines.append("")
    
    # Model list
    output_lines.append("MODELS EVALUATED")
    output_lines.append("-"*100)
    for model in sorted(set([r['model_name'] for r in visualizer.results])):
        output_lines.append(f"  • {model}")
    output_lines.append("")
    
    # Per-model detailed statistics
    output_lines.append("="*100)
    output_lines.append("DETAILED MODEL STATISTICS")
    output_lines.append("="*100)
    output_lines.append("")
    
    for model_name in sorted(set([r['model_name'] for r in visualizer.results])):
        output_lines.append(f"\n{'='*100}")
        output_lines.append(f"MODEL: {model_name}")
        output_lines.append('='*100)
        
        model_results = [r for r in visualizer.results if r['model_name'] == model_name]
        model_avg = [r for r in visualizer.model_averages if r['model_name'] == model_name]
        
        # Task-by-task breakdown
        for task in ['morse_encode', 'morse_decode', 'caesar_encode', 'caesar_decode']:
            task_data = [r for r in model_results if r['task_type'] == task]
            if task_data:
                td = task_data[0]
                output_lines.append(f"\n{task.replace('_', ' ').title()}:")
                output_lines.append(f"  Success Rate:     {td['success_rate']:>6.1%}")
                output_lines.append(f"  Avg Total Tokens: {td['avg_total_tokens']:>8.1f}")
                output_lines.append(f"  Reasoning Ratio:  {td['reasoning_ratio']:>6.3f}")
        
        # Benchmark average
        if model_avg:
            ma = model_avg[0]
            output_lines.append(f"\nBenchmark Average (across all tasks):")
            output_lines.append(f"  Success Rate:     {ma['success_rate']:>6.1%}")
            output_lines.append(f"  Avg Total Tokens: {ma['avg_total_tokens']:>8.1f}")
            output_lines.append(f"  Reasoning Ratio:  {ma['reasoning_ratio']:>6.3f}")
        
        output_lines.append("")
    
    # Task-level statistics
    output_lines.append("\n" + "="*100)
    output_lines.append("TASK-LEVEL STATISTICS")
    output_lines.append("="*100)
    output_lines.append("")
    
    for task in ['morse_encode', 'morse_decode', 'caesar_encode', 'caesar_decode']:
        task_results = [r for r in visualizer.results if r['task_type'] == task]
        if task_results:
            success_rates = [r['success_rate'] for r in task_results]
            tokens = [r['avg_total_tokens'] for r in task_results]
            reasoning_ratios = [r['reasoning_ratio'] for r in task_results]
            
            output_lines.append(f"\n{task.replace('_', ' ').title()}:")
            output_lines.append(f"  Models evaluated: {len(task_results)}")
            output_lines.append(f"  Success Rate:")
            output_lines.append(f"    Mean: {np.mean(success_rates):.3f}")
            output_lines.append(f"    Std:  {np.std(success_rates):.3f}")
            output_lines.append(f"    Min:  {np.min(success_rates):.3f}")
            output_lines.append(f"    Max:  {np.max(success_rates):.3f}")
            output_lines.append(f"  Avg Tokens (task-specific):")
            output_lines.append(f"    Mean: {np.mean(tokens):.1f}")
            output_lines.append(f"    Std:  {np.std(tokens):.1f}")
            output_lines.append(f"    Min:  {np.min(tokens):.1f}")
            output_lines.append(f"    Max:  {np.max(tokens):.1f}")
            output_lines.append(f"  Reasoning Ratio:")
            output_lines.append(f"    Mean: {np.mean(reasoning_ratios):.3f}")
            output_lines.append(f"    Std:  {np.std(reasoning_ratios):.3f}")
            output_lines.append(f"    Min:  {np.min(reasoning_ratios):.3f}")
            output_lines.append(f"    Max:  {np.max(reasoning_ratios):.3f}")
    
    # Rankings
    output_lines.append("\n" + "="*100)
    output_lines.append("RANKINGS")
    output_lines.append("="*100)
    
    # Rank by average success rate
    output_lines.append("\nBy Average Success Rate (Descending):")
    output_lines.append("-"*100)
    sorted_by_sr = sorted(visualizer.model_averages, 
                         key=lambda x: x['success_rate'], 
                         reverse=True)
    for i, model in enumerate(sorted_by_sr, 1):
        output_lines.append(f"  {i:2d}. {model['model_name']:<25} SR: {model['success_rate']:.1%}  "
                          f"Tokens: {model['avg_total_tokens']:>8.1f}  "
                          f"RR: {model['reasoning_ratio']:.3f}")
    
    # Rank by token efficiency (lowest tokens)
    output_lines.append("\nBy Token Efficiency (Ascending):")
    output_lines.append("-"*100)
    sorted_by_tokens = sorted(visualizer.model_averages, 
                             key=lambda x: x['avg_total_tokens'])
    for i, model in enumerate(sorted_by_tokens, 1):
        output_lines.append(f"  {i:2d}. {model['model_name']:<25} Tokens: {model['avg_total_tokens']:>8.1f}  "
                          f"SR: {model['success_rate']:.1%}  "
                          f"RR: {model['reasoning_ratio']:.3f}")
    
    # Rank by reasoning ratio
    output_lines.append("\nBy Reasoning Ratio (Descending):")
    output_lines.append("-"*100)
    sorted_by_rr = sorted(visualizer.model_averages, 
                         key=lambda x: x['reasoning_ratio'], 
                         reverse=True)
    for i, model in enumerate(sorted_by_rr, 1):
        output_lines.append(f"  {i:2d}. {model['model_name']:<25} RR: {model['reasoning_ratio']:.3f}  "
                          f"SR: {model['success_rate']:.1%}  "
                          f"Tokens: {model['avg_total_tokens']:>8.1f}")
    
    output_lines.append("\n" + "="*100)
    
    # Write to file
    report_text = "\n".join(output_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Also print to console
    print(report_text)
    print(f"\nDetailed report saved to: {output_file}")


if __name__ == "__main__":
    # Run interactive main function
    main()

    # Uncomment below for quick testing:
    # quick_generate("./output")
    
    # Example: Generate comparison table
    # create_comparison_table("./output")
    
    # Example: Generate detailed report
    # create_detailed_report("./output")
    
    # Example: Find Pareto frontier using benchmark averages
    # find_pareto_frontier("./output", use_average=True, output_file="pareto_avg.png")
    
    # Example: Find Pareto frontier for a specific task
    # find_pareto_frontier("./output", task_type="morse_encode", output_file="pareto_morse.png")
    
    # Example: Batch analyze all tasks
    # batch_analyze_all_tasks("./output")