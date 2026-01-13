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


class ReplacementF1TokenCostVisualizer:
    """
    Visualizer for model replacement F1 score vs token cost analysis.
    Creates scatter plots showing the trade-off between F1 performance and computational cost.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the visualizer with a base directory containing result JSON files.
        
        Args:
            base_dir: Path to directory containing result JSON files
        """
        self.base_dir = Path(base_dir)
        self.results_data = []
        
        # Define color palette for different num_differences (task types)
        self.task_colors = {
            3: '#FF6B6B',      # Red
            5: '#4ECDC4',      # Teal
            7: '#FFD93D',      # Yellow
            9: '#95E1D3',      # Mint
            11: '#A8E6CF',     # Light green
            13: '#FF8C94',     # Pink
            15: '#6C5CE7',     # Purple
            'benchmark_average': '#888888'  # Gray for average
        }
        
        # Define marker styles for different model types
        self.model_markers = {
            'gpt-5': 'o',                    # Circle
            'gpt-4': 's',                    # Square
            'deepseek-reasoner': '^',        # Triangle up
            'deepseek-r1': '^',              # Triangle up
            'deepseek-chat': 'D',            # Diamond
            'deepseek-v3': 'D',              # Diamond
            'o4-mini': 'p',                  # Pentagon
            'claude': 'v',                   # Triangle down
            'gemini': 'h',                   # Hexagon
            'llama': '*',                    # Star
            'qwen': '<',                     # Triangle left
            'default': 'X'                   # X marker
        }
        
    def extract_info_from_filename(self, filename: str) -> Tuple[str, int]:
        """
        Extract model name and num_differences from filename.
        Expected format: modelname_results_difficult_X_difference_MMDD.json
        
        Args:
            filename: JSON filename
            
        Returns:
            Tuple of (model_name, num_differences)
        """
        pattern = r'(.+?)_results_difficult_(\d+)_difference'
        match = re.search(pattern, filename)
        if match:
            model_name = match.group(1)
            num_diff = int(match.group(2))
            return model_name, num_diff
        return None, None
    
    def simplify_model_name(self, model_name: str) -> str:
        """
        Simplify model name to xxx-xxx format.
        
        Args:
            model_name: Full model name
            
        Returns:
            Simplified model name
        """
        # Split by underscore or dash and take first two parts
        parts = re.split(r'[-_]', model_name)
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
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
        if 'deepseek-reasoner' in model_lower or 'deepseek-r1' in model_lower:
            return 'deepseek-r1'
        elif 'deepseek-chat' in model_lower or 'deepseek-v3' in model_lower:
            return 'deepseek-v3'
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
    
    def calculate_replacement_f1(self, total_predictions: int, 
                                 total_ground_truth: int, 
                                 total_correct_replacements: int) -> float:
        """
        Calculate F1 score for replacements.
        
        Args:
            total_predictions: Total number of predictions
            total_ground_truth: Total number of ground truth items
            total_correct_replacements: Total number of correct replacements
            
        Returns:
            F1 score
        """
        if total_predictions == 0:
            precision = 0
        else:
            precision = total_correct_replacements / total_predictions
        
        if total_ground_truth == 0:
            recall = 0
        else:
            recall = total_correct_replacements / total_ground_truth
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def load_all_results(self):
        """
        Load all results from JSON files in base_dir.
        """
        print("Loading benchmark results for visualization...")
        
        json_files = list(self.base_dir.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {self.base_dir}")
            return
        
        for json_file in json_files:
            filename = json_file.name
            model_name, num_diff = self.extract_info_from_filename(filename)
            
            if model_name is None:
                print(f"Warning: Could not parse filename {filename}, skipping...")
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            # Calculate replacement F1
            replacement_f1 = self.calculate_replacement_f1(
                result['total_predictions'],
                result['total_ground_truth'],
                result['total_correct_replacements']
            )
            
            # Extract token usage
            if 'token_usage' in result and 'average_per_sample' in result['token_usage']:
                avg_tokens = result['token_usage']['average_per_sample']
                total_tokens = avg_tokens.get('total_tokens', 0)
                reasoning_tokens = avg_tokens.get('reasoning_tokens', 0)
                
                if total_tokens > 0:
                    reasoning_ratio = reasoning_tokens / total_tokens
                else:
                    reasoning_ratio = 0
            else:
                total_tokens = 0
                reasoning_ratio = 0
            
            simplified_name = self.simplify_model_name(model_name)
            
            self.results_data.append({
                'model_name': model_name,
                'simplified_model_name': simplified_name,
                'num_differences': num_diff,
                'replacement_f1': replacement_f1,
                'avg_total_tokens': total_tokens,
                'reasoning_ratio': reasoning_ratio,
                'total_predictions': result['total_predictions'],
                'total_ground_truth': result['total_ground_truth'],
                'total_correct_replacements': result['total_correct_replacements'],
                'model_type': self.get_model_type(simplified_name)
            })
        
        print(f"Loaded {len(self.results_data)} result files")
        
        # Group by model
        models = defaultdict(int)
        for item in self.results_data:
            models[item['simplified_model_name']] += 1
        
        for model, count in sorted(models.items()):
            print(f"  - {model}: {count} tasks")
        print()
    
    def prepare_plot_data(self) -> pd.DataFrame:
        """
        Prepare data for plotting by extracting relevant metrics.
        
        Returns:
            DataFrame with plot data
        """
        if not self.results_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results_data)
        
        # Calculate benchmark averages for each model
        avg_data = []
        for model_name in df['simplified_model_name'].unique():
            model_data = df[df['simplified_model_name'] == model_name]
            
            avg_f1 = model_data['replacement_f1'].mean()
            avg_tokens = model_data['avg_total_tokens'].mean()
            avg_reasoning_ratio = model_data['reasoning_ratio'].mean()
            model_type = model_data['model_type'].iloc[0]
            
            avg_data.append({
                'simplified_model_name': model_name,
                'num_differences': 'benchmark_average',
                'replacement_f1': avg_f1,
                'avg_total_tokens': avg_tokens,
                'reasoning_ratio': avg_reasoning_ratio,
                'model_type': model_type,
                'is_average': True
            })
        
        # Add is_average flag to original data
        df['is_average'] = False
        
        # Combine original data with averages
        avg_df = pd.DataFrame(avg_data)
        combined_df = pd.concat([df, avg_df], ignore_index=True)
        
        return combined_df
    
    def plot_f1_vs_token_cost(self, save_path: str = None, 
                             figsize: Tuple[int, int] = (14, 8),
                             show_average_line: bool = True):
        """
        Create the main scatter plot: Replacement F1 vs Token Cost.
        
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
        base_size = 100
        max_size = 500
        size_range = max_size - base_size
        
        # Separate benchmark average data from task data
        df_tasks = df[~df['is_average']]
        df_average = df[df['is_average']]
        
        # Plot task data
        for num_diff in df_tasks['num_differences'].unique():
            task_data = df_tasks[df_tasks['num_differences'] == num_diff]
            
            for model_type in task_data['model_type'].unique():
                model_data = task_data[task_data['model_type'] == model_type]
                
                # Calculate sizes: base_size + (reasoning_ratio * size_range)
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Get color for this task type
                color = self.task_colors.get(num_diff, '#CCCCCC')
                marker = self.model_markers.get(model_type, 'X')
                
                # Plot with log scale on x-axis
                ax.scatter(
                    model_data['avg_total_tokens'],
                    model_data['replacement_f1'],
                    c=color,
                    marker=marker,
                    s=sizes,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=2
                )
        
        # Plot benchmark average points and line
        if not df_average.empty and show_average_line:
            for model_type in df_average['model_type'].unique():
                model_data = df_average[df_average['model_type'] == model_type]
                
                # Calculate sizes for average points
                sizes = base_size + (model_data['reasoning_ratio'] * size_range)
                
                # Plot average points in gray
                ax.scatter(
                    model_data['avg_total_tokens'],
                    model_data['replacement_f1'],
                    c=self.task_colors['benchmark_average'],
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
                df_average_sorted['replacement_f1'],
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
        ax.set_ylabel('Replacement F1 Score', fontsize=13, fontweight='bold')
        ax.set_title('Model Replacement F1 vs Token Cost Trade-off', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Set y-axis limits
        ax.set_ylim(-0.05, 1.05)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
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
        Create custom legends for task types, model types, and reasoning ratios.
        
        Args:
            ax: Matplotlib axes object
            df: DataFrame with plot data
        """
        legend_y_start = 1.0
        legend_spacing = 0.28
        
        # Legend 1: Task Types (colors) - num_differences
        task_patches = []
        
        # Get unique task types (excluding benchmark average)
        df_tasks = df[~df['is_average']]
        unique_tasks = sorted(df_tasks['num_differences'].unique())
        
        for task in unique_tasks:
            color = self.task_colors.get(task, '#CCCCCC')
            label = f'{task}-difference'
            task_patches.append(mpatches.Patch(color=color, label=label))
        
        # Add benchmark average
        if df['is_average'].any():
            task_patches.append(
                mpatches.Patch(color=self.task_colors['benchmark_average'], 
                             label='Benchmark Avg.', alpha=0.6)
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
                label = model_type.replace('-', ' ').title()
            
            model_type_lines.append(Line2D([0], [0], marker=marker, color='gray', 
                                          linestyle='', markersize=9, 
                                          markeredgewidth=1.3, markeredgecolor='black',
                                          label=label, alpha=0.7))
        
        legend2 = ax.legend(handles=model_type_lines,
                          title='Model Type',
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
    ║         REPLACEMENT F1 VS TOKEN COST VISUALIZER                          ║
    ║                                                                          ║
    ║  Visualize the trade-off between replacement F1 and token consumption    ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get base directory
    base_dir = input("Enter the path to your results directory (or press Enter for current directory): ").strip()
    
    if not base_dir:
        base_dir = "."
    
    # Initialize visualizer
    visualizer = ReplacementF1TokenCostVisualizer(base_dir)
    
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
        output_file = input("Enter output filename (default: 'replacement_f1_vs_token_cost.png'): ").strip()
        if not output_file:
            output_file = "replacement_f1_vs_token_cost.png"
        save_path = output_file
    else:
        save_path = None
    
    # Generate the plot
    print("\nGenerating replacement F1 vs token cost visualization...")
    visualizer.plot_f1_vs_token_cost(save_path=save_path, 
                                    show_average_line=show_average_line)
    
    print("\nVisualization complete!")


def quick_generate(base_dir: str, 
                  output_file: str = "replacement_f1_vs_token_cost.png",
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
    visualizer = ReplacementF1TokenCostVisualizer(base_dir)
    visualizer.load_all_results()
    visualizer.plot_f1_vs_token_cost(save_path=output_file, 
                                    figsize=figsize,
                                    show_average_line=show_average_line)


def customize_visualization(base_dir: str, 
                           output_file: str = "replacement_f1_vs_token_cost.png",
                           figsize: Tuple[int, int] = (14, 8),
                           custom_colors: Dict[int, str] = None,
                           custom_markers: Dict[str, str] = None,
                           show_average_line: bool = True):
    """
    Generate plot with custom colors and markers.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        custom_colors: Custom color mapping for task types (num_differences)
        custom_markers: Custom marker mapping for model types
        show_average_line: Whether to show benchmark average line
    
    Example:
        custom_colors = {
            5: '#FF0000',
            7: '#00FF00',
            9: '#0000FF'
        }
        custom_markers = {
            'gpt-4': 'o',
            'deepseek-v3': '^'
        }
        customize_visualization("./results", "custom.png", 
                              custom_colors=custom_colors, 
                              custom_markers=custom_markers)
    """
    visualizer = ReplacementF1TokenCostVisualizer(base_dir)
    
    # Override colors if provided
    if custom_colors:
        visualizer.task_colors.update(custom_colors)
    
    # Override markers if provided
    if custom_markers:
        visualizer.model_markers.update(custom_markers)
    
    visualizer.load_all_results()
    visualizer.plot_f1_vs_token_cost(save_path=output_file, 
                                    figsize=figsize,
                                    show_average_line=show_average_line)


def export_plot_data(base_dir: str, output_file: str = "plot_data.csv"):
    """
    Export the prepared plot data to CSV for external analysis.
    
    Args:
        base_dir: Directory containing result JSON files
        output_file: Output CSV file path
    
    Example:
        export_plot_data("./results", "my_data.csv")
    """
    visualizer = ReplacementF1TokenCostVisualizer(base_dir)
    visualizer.load_all_results()
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
    print(f"Number of models: {df['simplified_model_name'].nunique()}")
    
    # Task types
    df_tasks = df[~df['is_average']]
    if not df_tasks.empty:
        task_types = sorted(df_tasks['num_differences'].unique())
        print(f"Task types (num_differences): {', '.join(map(str, task_types))}")
    
    model_types = sorted(df['model_type'].unique())
    print(f"Model types: {', '.join(model_types)}")
    
    # Summary for benchmark averages
    df_avg = df[df['is_average']]
    if not df_avg.empty:
        print("\n" + "-"*100)
        print("BENCHMARK AVERAGE SUMMARY")
        print("-"*100)
        avg_summary = df_avg[['simplified_model_name', 'replacement_f1', 
                              'avg_total_tokens', 'reasoning_ratio']].copy()
        avg_summary.columns = ['Model', 'Avg F1', 'Avg Total Tokens', 'Reasoning Ratio']
        avg_summary = avg_summary.sort_values('Avg F1', ascending=False)
        print(avg_summary.to_string(index=False))
    
    print("\n" + "-"*100)
    print("PER-TASK SUMMARY")
    print("-"*100)
    
    if not df_tasks.empty:
        summary = df_tasks.groupby(['simplified_model_name', 'num_differences']).agg({
            'replacement_f1': 'mean',
            'avg_total_tokens': 'mean',
            'reasoning_ratio': 'mean'
        }).round(4)
        print(summary)
    
    print("="*100 + "\n")
    
    return df


def compare_models_summary(base_dir: str):
    """
    Print a comparison summary of all models based on benchmark averages.
    
    Args:
        base_dir: Directory containing result JSON files
    
    Example:
        compare_models_summary("./results")
    """
    visualizer = ReplacementF1TokenCostVisualizer(base_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    # Filter for benchmark averages only
    df_avg = df[df['is_average']].copy()
    
    if df_avg.empty:
        print("No benchmark average data available.")
        return
    
    # Sort by replacement F1
    df_avg = df_avg.sort_values('replacement_f1', ascending=False)
    
    print("\n" + "="*110)
    print("MODEL COMPARISON - BENCHMARK AVERAGES")
    print("="*110)
    print(f"{'Rank':<6} {'Model':<25} {'Avg F1':<12} {'Avg Tokens':<15} "
          f"{'Reasoning Ratio':<18} {'Model Type':<20}")
    print("-"*110)
    
    for idx, (_, row) in enumerate(df_avg.iterrows(), 1):
        print(f"{idx:<6} {row['simplified_model_name']:<25} {row['replacement_f1']:<12.4f} "
              f"{row['avg_total_tokens']:<15.1f} {row['reasoning_ratio']:<18.2%} "
              f"{row['model_type']:<20}")
    
    print("="*110)
    
    # Calculate efficiency metric (F1 per 1000 tokens)
    df_avg['efficiency'] = df_avg['replacement_f1'] / df_avg['avg_total_tokens'] * 1000
    df_avg_eff = df_avg.sort_values('efficiency', ascending=False)
    
    print("\n" + "="*110)
    print("EFFICIENCY RANKING (F1 Score per 1000 Tokens)")
    print("="*110)
    print(f"{'Rank':<6} {'Model':<25} {'Efficiency':<15} {'Avg F1':<12} {'Avg Tokens':<15}")
    print("-"*110)
    
    for idx, (_, row) in enumerate(df_avg_eff.iterrows(), 1):
        print(f"{idx:<6} {row['simplified_model_name']:<25} {row['efficiency']:<15.6f} "
              f"{row['replacement_f1']:<12.4f} {row['avg_total_tokens']:<15.1f}")
    
    print("="*110 + "\n")


def analyze_task_difficulty(base_dir: str):
    """
    Analyze how models perform across different task difficulties (num_differences).
    
    Args:
        base_dir: Directory containing result JSON files
    
    Example:
        analyze_task_difficulty("./results")
    """
    visualizer = ReplacementF1TokenCostVisualizer(base_dir)
    visualizer.load_all_results()
    df = visualizer.prepare_plot_data()
    
    # Filter out benchmark averages
    df_tasks = df[~df['is_average']].copy()
    
    if df_tasks.empty:
        print("No task data available.")
        return
    
    print("\n" + "="*100)
    print("TASK DIFFICULTY ANALYSIS")
    print("="*100)
    
    # Group by num_differences
    for num_diff in sorted(df_tasks['num_differences'].unique()):
        task_data = df_tasks[df_tasks['num_differences'] == num_diff]
        
        print(f"\n{num_diff}-DIFFERENCE TASK:")
        print("-"*100)
        print(f"{'Model':<25} {'F1 Score':<12} {'Avg Tokens':<15} {'Reasoning Ratio':<18}")
        print("-"*100)
        
        task_sorted = task_data.sort_values('replacement_f1', ascending=False)
        for _, row in task_sorted.iterrows():
            print(f"{row['simplified_model_name']:<25} {row['replacement_f1']:<12.4f} "
                  f"{row['avg_total_tokens']:<15.1f} {row['reasoning_ratio']:<18.2%}")
        
        # Statistics for this task
        print("-"*100)
        print(f"Average F1: {task_data['replacement_f1'].mean():.4f}")
        print(f"F1 Std Dev: {task_data['replacement_f1'].std():.4f}")
        print(f"Average Tokens: {task_data['avg_total_tokens'].mean():.1f}")
        print(f"Token Std Dev: {task_data['avg_total_tokens'].std():.1f}")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    # Run interactive main function
    main()