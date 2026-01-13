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


class TableBenchmarkVisualizer:
    """
    Visualizer for table generation benchmark: alignment rate vs token cost analysis.
    Creates scatter plots showing the trade-off between alignment quality and computational cost.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize the visualizer with a results directory.
        
        Args:
            results_dir: Path to directory containing result JSON files
        """
        self.results_dir = Path(results_dir)
        self.results = []
        self.model_averages = []  # Store model-level averages
        
        # Define color palette for table formats
        self.format_colors = {
            'latex': '#e74c3c',      # Red
            'markdown': '#3498db',   # Blue
            'text': '#2ecc71',       # Green
            'html': '#f39c12',       # Orange
            'csv': '#9b59b6',        # Purple
            'benchmark_average': '#808080'  # Gray for averages
        }
        
        # Define marker styles for different model types
        self.model_markers = {
            'gpt-5': 'o',           # Circle
            'gpt-4': 's',           # Square
            'deepseek-r1': '^',     # Triangle up
            'deepseek-reasoner': '^', # Triangle up
            'deepseek-v3': 'D',     # Diamond
            'deepseek-chat': 'D',   # Diamond
            'o4-mini': 'p',         # Pentagon
            'o1': 'v',              # Triangle down
            'o1-mini': 'v',         # Triangle down
            'claude': 'h',          # Hexagon
            'gemini': '*',          # Star
            'llama': '<',           # Triangle left
            'qwen': '>',            # Triangle right
            'default': 'X'          # X marker
        }
    
    def extract_model_name(self, filename: str) -> str:
        """
        Extract simplified model name from filename.
        Expected format: {task_type}_{model_name}_results_{date}.json
        Examples: 'hard_deepseek-r1_results_1209.json' -> 'deepseek-r1'
                  'easy_gpt-4o_results_1210.json' -> 'gpt-4o'
        
        Args:
            filename: Result filename
            
        Returns:
            Extracted model name
        """
        basename = os.path.basename(filename)
        # Match pattern: {prefix}_{model_name}_results_{date}.json
        match = re.search(r'^[^_]+_([a-zA-Z0-9\-]+)_results_\d+\.json$', basename)
        if match:
            return match.group(1)
        
        # Fallback: remove .json and try to extract middle part
        name = basename.replace('.json', '')
        parts = name.split('_')
        if len(parts) >= 2:
            # Return second part (should be model name)
            return parts[1]
        
        return name
    
    def extract_difficulty_level(self, filename: str) -> str:
        """
        Extract difficulty level from filename.
        Expected format: {difficulty}_{model_name}_results_{date}.json
        
        Args:
            filename: Result filename
            
        Returns:
            Difficulty level (e.g., 'easy', 'hard')
        """
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) > 0:
            difficulty = parts[0].lower()
            # Common difficulty levels
            if difficulty in ['easy', 'medium', 'hard', 'simple', 'complex']:
                return difficulty
        return 'unknown'
    
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
        elif 'o1-mini' in model_lower or 'o1mini' in model_lower:
            return 'o1'
        elif 'o1-preview' in model_lower or model_lower == 'o1':
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
        Load all benchmark results from JSON files.
        Extracts data for each table format (latex, markdown, text) separately.
        Also calculates model-level averages across all formats.
        """
        print("Loading table generation benchmark results...")
        
        # Find all result JSON files matching pattern
        json_pattern = os.path.join(self.results_dir, '*_results_*.json')
        json_files = glob.glob(json_pattern)
        
        if not json_files:
            print(f"No result files found in {self.results_dir}")
            print(f"Looking for pattern: *_results_*.json")
            return
        
        print(f"Found {len(json_files)} result files")
        
        # Dictionary to accumulate model-level statistics
        model_stats = {}
        
        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract model name and difficulty
                model_name = self.extract_model_name(filepath)
                difficulty = self.extract_difficulty_level(filepath)
                model_type = self.get_model_type(model_name)
                
                # Check if format_statistics exists
                format_stats = data.get('format_statistics', {})
                format_token_stats = data.get('format_token_statistics', {})
                
                if format_stats and format_token_stats:
                    # Initialize model stats if not exists
                    if model_name not in model_stats:
                        model_stats[model_name] = {
                            'model_type': model_type,
                            'alignment_rates': [],
                            'total_tokens': [],
                            'reasoning_ratios': [],
                            'content_scores': [],
                            'total_cases': 0,
                            'valid_cases': 0
                        }
                    
                    # Extract data for each format
                    for table_format in format_stats.keys():
                        fmt_stats = format_stats[table_format]
                        fmt_tokens = format_token_stats.get(table_format, {})
                        
                        alignment_rate = fmt_stats.get('alignment_rate', 0)
                        avg_total_tokens = fmt_tokens.get('avg_tokens_per_case', 0)
                        avg_reasoning_tokens = fmt_tokens.get('avg_reasoning_tokens_per_case', 0)
                        
                        # Calculate reasoning ratio
                        if avg_total_tokens > 0:
                            reasoning_ratio = avg_reasoning_tokens / avg_total_tokens
                        else:
                            reasoning_ratio = 0
                        
                        result = {
                            'filepath': filepath,
                            'filename': os.path.basename(filepath),
                            'model_name': model_name,
                            'model_type': model_type,
                            'difficulty': difficulty,
                            'table_format': table_format,
                            'alignment_rate': alignment_rate,
                            'avg_total_tokens': avg_total_tokens,
                            'reasoning_ratio': reasoning_ratio,
                            'avg_content_score': fmt_stats.get('avg_content_score', 0),
                            'avg_alignment_score': fmt_stats.get('avg_alignment_score', 0),
                            'total_cases': fmt_stats.get('total_cases', 0),
                            'valid_cases': fmt_stats.get('valid_cases', 0)
                        }
                        
                        self.results.append(result)
                        
                        # Accumulate for model average
                        model_stats[model_name]['alignment_rates'].append(alignment_rate)
                        model_stats[model_name]['total_tokens'].append(avg_total_tokens)
                        model_stats[model_name]['reasoning_ratios'].append(reasoning_ratio)
                        model_stats[model_name]['content_scores'].append(fmt_stats.get('avg_content_score', 0))
                        model_stats[model_name]['total_cases'] += fmt_stats.get('total_cases', 0)
                        model_stats[model_name]['valid_cases'] += fmt_stats.get('valid_cases', 0)
                else:
                    # No format-specific statistics, use overall statistics
                    overall_stats = data.get('overall_statistics', {})
                    token_stats = data.get('token_statistics', {})
                    
                    alignment_rate = overall_stats.get('alignment_rate', 0)
                    avg_total_tokens = token_stats.get('avg_tokens_per_case', 0)
                    avg_reasoning_tokens = token_stats.get('avg_reasoning_tokens_per_case', 0)
                    
                    # Calculate reasoning ratio
                    if avg_total_tokens > 0:
                        reasoning_ratio = avg_reasoning_tokens / avg_total_tokens
                    else:
                        reasoning_ratio = 0
                    
                    result = {
                        'filepath': filepath,
                        'filename': os.path.basename(filepath),
                        'model_name': model_name,
                        'model_type': model_type,
                        'difficulty': difficulty,
                        'table_format': 'mixed',  # Unknown format
                        'alignment_rate': alignment_rate,
                        'avg_total_tokens': avg_total_tokens,
                        'reasoning_ratio': reasoning_ratio,
                        'avg_content_score': overall_stats.get('avg_content_score', 0),
                        'avg_alignment_score': overall_stats.get('avg_alignment_score', 0),
                        'total_cases': overall_stats.get('total_cases', 0),
                        'valid_cases': overall_stats.get('valid_cases', 0)
                    }
                    
                    self.results.append(result)
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Calculate model-level averages
        for model_name, stats in model_stats.items():
            if stats['alignment_rates']:
                avg_result = {
                    'model_name': model_name,
                    'model_type': stats['model_type'],
                    'table_format': 'benchmark_average',
                    'alignment_rate': np.mean(stats['alignment_rates']),
                    'avg_total_tokens': np.mean(stats['total_tokens']),
                    'reasoning_ratio': np.mean(stats['reasoning_ratios']),
                    'avg_content_score': np.mean(stats['content_scores']),
                    'total_cases': stats['total_cases'],
                    'valid_cases': stats['valid_cases']
                }
                self.model_averages.append(avg_result)
        
        print(f"Successfully loaded {len(self.results)} format-specific data points")
        print(f"Calculated {len(self.model_averages)} model-level averages")
        
        # Print per-model and per-format summary
        if self.results:
            models = {}
            formats = {}
            for r in self.results:
                if r['model_name'] not in models:
                    models[r['model_name']] = 0
                models[r['model_name']] += 1
                
                if r['table_format'] not in formats:
                    formats[r['table_format']] = 0
                formats[r['table_format']] += 1
            
            print("\nPer-model summary:")
            for model, count in sorted(models.items()):
                print(f"  - {model}: {count} evaluation(s)")
            
            print("\nPer-format summary:")
            for fmt, count in sorted(formats.items()):
                print(f"  - {fmt}: {count} evaluation(s)")
        print()
    
    def plot_alignment_vs_token_cost(self, save_path: str = None, 
                                    figsize: tuple = (14, 8),
                                    show_averages: bool = True):
        """
        Create the main scatter plot: Alignment Rate vs Token Cost.
        Color by table format, shape by model type, size by reasoning ratio.
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            show_averages: Whether to show model-level averages
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
        
        # Create scatter plot for each table format and model type
        for table_format in self.format_colors.keys():
            if table_format == 'benchmark_average':
                continue  # Skip average, will plot separately
                
            format_results = [r for r in self.results if r['table_format'] == table_format]
            
            if not format_results:
                continue
            
            for model_type in set([r['model_type'] for r in format_results]):
                model_data = [r for r in format_results if r['model_type'] == model_type]
                
                if not model_data:
                    continue
                
                # Extract data
                x_data = [r['avg_total_tokens'] for r in model_data]
                y_data = [r['alignment_rate'] for r in model_data]
                reasoning_ratios = [r['reasoning_ratio'] for r in model_data]
                
                # Calculate sizes: base_size + (reasoning_ratio * size_range)
                sizes = [base_size + (ratio * size_range) for ratio in reasoning_ratios]
                
                # Plot with log scale on x-axis
                ax.scatter(
                    x_data,
                    y_data,
                    c=self.format_colors[table_format],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes,
                    alpha=0.7,
                    edgecolors='black',
                    linewidths=1.5,
                    zorder=2
                )
        
        # Plot model-level averages if requested
        if show_averages and self.model_averages:
            # Sort averages by model name for consistent line plotting
            sorted_averages = sorted(self.model_averages, 
                                    key=lambda x: x['avg_total_tokens'])
            
            # Group by model type for plotting
            model_type_groups = {}
            for avg in sorted_averages:
                model_type = avg['model_type']
                if model_type not in model_type_groups:
                    model_type_groups[model_type] = []
                model_type_groups[model_type].append(avg)
            
            # Plot each model type separately
            for model_type, avgs in model_type_groups.items():
                x_avg = [r['avg_total_tokens'] for r in avgs]
                y_avg = [r['alignment_rate'] for r in avgs]
                reasoning_ratios_avg = [r['reasoning_ratio'] for r in avgs]
                
                # Calculate sizes for average points
                sizes_avg = [base_size + (ratio * size_range) for ratio in reasoning_ratios_avg]
                
                # Plot average points
                ax.scatter(
                    x_avg,
                    y_avg,
                    c=self.format_colors['benchmark_average'],
                    marker=self.model_markers.get(model_type, 'X'),
                    s=sizes_avg,
                    alpha=0.5,
                    edgecolors='gray',
                    linewidths=2,
                    zorder=3,
                    label='_nolegend_'  # Don't add to legend (will add custom entry)
                )
            
            # Draw connecting line for all averages
            x_all_avg = [r['avg_total_tokens'] for r in sorted_averages]
            y_all_avg = [r['alignment_rate'] for r in sorted_averages]
            
            ax.plot(
                x_all_avg,
                y_all_avg,
                color='gray',
                linestyle='--',
                linewidth=2,
                alpha=0.4,
                zorder=1,
                label='_nolegend_'
            )
            
            # Add model name annotations for average points
            for avg in sorted_averages:
                ax.annotate(
                    avg['model_name'],
                    xy=(avg['avg_total_tokens'], avg['alignment_rate']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7,
                    color='gray'
                )
        
        # Set log scale for x-axis
        ax.set_xscale('log')
        
        # Manually set specific tick locations based on data range
        all_tokens = [r['avg_total_tokens'] for r in self.results if r['avg_total_tokens'] > 0]
        if show_averages and self.model_averages:
            all_tokens.extend([r['avg_total_tokens'] for r in self.model_averages])
        
        if all_tokens:
            min_tokens = min(all_tokens)
            max_tokens = max(all_tokens)
            
            # Create reasonable tick locations
            tick_locations = []
            for power in range(int(np.log10(min_tokens)), int(np.log10(max_tokens)) + 2):
                for mult in [1, 2, 5]:
                    val = mult * (10 ** power)
                    if min_tokens / 2 <= val <= max_tokens * 2:
                        tick_locations.append(val)
            
            ax.set_xticks(tick_locations)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Add minor ticks
        from matplotlib.ticker import LogLocator
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.grid(True, which='minor', alpha=0.2, linestyle=':')
        ax.grid(True, which='major', alpha=0.3, linestyle='--')
        
        # Labels and title
        ax.set_xlabel('Average Total Tokens (log scale)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Alignment Rate', fontsize=13, fontweight='bold')
        title = 'Table Generation: Alignment Rate vs Token Cost by Format'
        if show_averages:
            title += '\n(Gray points show benchmark averages across all formats)'
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        # Set y-axis limits and formatting
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Create custom legends
        self._create_legends(ax, show_averages)
        
        # Adjust layout to accommodate legends
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _create_legends(self, ax, show_averages=True):
        """
        Create custom legends for table formats, model types, and reasoning ratios.
        
        Args:
            ax: Matplotlib axes object
            show_averages: Whether averages are shown in the plot
        """
        legend_y_start = 1.0
        legend_spacing = 0.30
        
        # Legend 1: Table Formats (colors)
        format_patches = []
        for table_format, color in self.format_colors.items():
            if table_format == 'benchmark_average':
                continue  # Will add separately if needed
            if any(r['table_format'] == table_format for r in self.results):
                # Format label
                label = table_format.upper() if table_format in ['latex', 'html', 'csv'] else table_format.capitalize()
                format_patches.append(Patch(color=color, label=label))
        
        # Add benchmark average to format legend if shown
        if show_averages and self.model_averages:
            format_patches.append(Patch(color=self.format_colors['benchmark_average'], 
                                       label='Benchmark Average', 
                                       alpha=0.5))
        
        if format_patches:
            legend1 = ax.legend(handles=format_patches, 
                              title='Table Format',
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
                # Special formatting
                label = label.replace('Gpt', 'GPT').replace('O1', 'O1').replace('O4', 'O4')
            
            model_type_lines.append(Line2D([0], [0], marker=marker, color='gray', 
                                          linestyle='', markersize=9, 
                                          markeredgewidth=1.3, markeredgecolor='black',
                                          label=label, alpha=0.7))
        
        if model_type_lines:
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
        print("SUMMARY STATISTICS - TABLE GENERATION BENCHMARK")
        print("="*80)
        
        print(f"\nTotal data points: {len(self.results)}")
        print(f"Unique models: {len(set([r['model_name'] for r in self.results]))}")
        print(f"Models: {', '.join(sorted(set([r['model_name'] for r in self.results])))}")
        print(f"Table formats: {', '.join(sorted(set([r['table_format'] for r in self.results])))}")
        
        # Print model averages
        if self.model_averages:
            print("\n" + "-"*80)
            print("Model-level averages (across all formats):")
            print("-"*80)
            print(f"{'Model':<25} {'Avg Alignment':<18} {'Avg Tokens':<18} {'Avg Reasoning Ratio':<20}")
            print("-"*80)
            
            sorted_avgs = sorted(self.model_averages, key=lambda x: x['alignment_rate'], reverse=True)
            for avg in sorted_avgs:
                print(f"{avg['model_name']:<25} "
                      f"{avg['alignment_rate']:<18.3f} "
                      f"{avg['avg_total_tokens']:<18.1f} "
                      f"{avg['reasoning_ratio']:<20.3f}")
        
        print("\n" + "-"*80)
        print("Per-format statistics:")
        print("-"*80)
        
        for table_format in sorted(set([r['table_format'] for r in self.results])):
            format_results = [r for r in self.results if r['table_format'] == table_format]
            if format_results:
                alignment_rates = [r['alignment_rate'] for r in format_results]
                tokens = [r['avg_total_tokens'] for r in format_results]
                reasoning_ratios = [r['reasoning_ratio'] for r in format_results]
                content_scores = [r['avg_content_score'] for r in format_results]
                
                print(f"\n{table_format.upper() if table_format in ['latex', 'html', 'csv'] else table_format.capitalize()}:")
                print(f"  Number of evaluations: {len(format_results)}")
                print(f"  Alignment Rate: mean={np.mean(alignment_rates):.3f}, "
                      f"min={np.min(alignment_rates):.3f}, max={np.max(alignment_rates):.3f}")
                print(f"  Content Score: mean={np.mean(content_scores):.3f}, "
                      f"min={np.min(content_scores):.3f}, max={np.max(content_scores):.3f}")
                print(f"  Avg tokens: mean={np.mean(tokens):.1f}, "
                      f"min={np.min(tokens):.1f}, max={np.max(tokens):.1f}")
                print(f"  Reasoning ratio: mean={np.mean(reasoning_ratios):.3f}, "
                      f"min={np.min(reasoning_ratios):.3f}, max={np.max(reasoning_ratios):.3f}")
        
        print("\n" + "-"*80)
        print("Per-model statistics:")
        print("-"*80)
        
        for model in sorted(set([r['model_name'] for r in self.results])):
            model_results = [r for r in self.results if r['model_name'] == model]
            if model_results:
                alignment_rates = [r['alignment_rate'] for r in model_results]
                tokens = [r['avg_total_tokens'] for r in model_results]
                reasoning_ratios = [r['reasoning_ratio'] for r in model_results]
                formats = sorted(set([r['table_format'] for r in model_results]))
                
                print(f"\n{model}:")
                print(f"  Formats evaluated: {len(model_results)} ({', '.join(formats)})")
                print(f"  Avg Alignment Rate: {np.mean(alignment_rates):.3f}")
                print(f"  Avg Tokens: {np.mean(tokens):.1f}")
                print(f"  Avg Reasoning Ratio: {np.mean(reasoning_ratios):.3f}")
        
        print("\n" + "="*80 + "\n")


def main():
    """
    Main function for creating the visualization.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║         TABLE GENERATION BENCHMARK VISUALIZER                            ║
    ║                                                                          ║
    ║  Visualize alignment rate vs token cost by table format                  ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get results directory
    results_dir = input("Enter the path to your results directory (or press Enter for './output'): ").strip()
    
    if not results_dir:
        results_dir = "./output"
    
    if not os.path.exists(results_dir):
        print(f"\nError: Directory '{results_dir}' does not exist!")
        return
    
    # Initialize visualizer
    visualizer = TableBenchmarkVisualizer(results_dir)
    
    # Load data
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("\nNo valid results found in the specified directory.")
        print("Expected file format: {difficulty}_{model_name}_results_{date}.json")
        print("Results should contain 'format_statistics' with data for latex/markdown/text formats")
        return
    
    # Print summary statistics
    visualizer.print_summary_statistics()
    # Ask about showing averages
    show_avg = input("\nShow benchmark averages? (y/n, default: y): ").strip().lower()
    show_averages = show_avg == '' or show_avg == 'y'
    
    # Ask for output path
    save_plot = input("\nSave plot? (y/n, default: y): ").strip().lower()
    
    if save_plot == '' or save_plot == 'y':
        output_file = input("Enter output filename (default: 'alignment_token_cost.png'): ").strip()
        if not output_file:
            output_file = os.path.join(results_dir, 'alignment_token_cost.png')
        save_path = output_file
    else:
        save_path = None
    
    # Generate the plot
    print("\nGenerating alignment rate vs token cost visualization...")
    visualizer.plot_alignment_vs_token_cost(save_path=save_path, show_averages=show_averages)
    
    print("\nVisualization complete!")


def quick_generate(results_dir: str = "./output", 
                  output_file: str = "alignment_token_cost.png",
                  figsize: tuple = (14, 8),
                  show_averages: bool = True):
    """
    Quick function to generate the plot with one command.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        show_averages: Whether to show benchmark averages
    
    Example:
        >>> quick_generate("./output", "my_plot.png")
        >>> quick_generate("./output", "large_plot.png", figsize=(18, 10))
        >>> quick_generate("./output", "no_avg.png", show_averages=False)
    """
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    visualizer.print_summary_statistics()
    visualizer.plot_alignment_vs_token_cost(save_path=output_file, 
                                           figsize=figsize,
                                           show_averages=show_averages)


def customize_visualization(results_dir: str = "./output",
                           output_file: str = "alignment_token_cost.png",
                           figsize: tuple = (14, 8),
                           custom_colors: dict = None,
                           custom_markers: dict = None,
                           show_averages: bool = True):
    """
    Generate plot with custom colors and markers.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        figsize: Figure size (width, height)
        custom_colors: Custom color mapping for table formats
        custom_markers: Custom marker mapping for model types
        show_averages: Whether to show benchmark averages
    
    Example:
        >>> custom_colors = {
        ...     'latex': '#FF0000',
        ...     'markdown': '#00FF00'
        ... }
        >>> custom_markers = {
        ...     'gpt-4': 'o',
        ...     'deepseek-r1': '^'
        ... }
        >>> customize_visualization("./output", "custom.png", 
        ...                        custom_colors=custom_colors, 
        ...                        custom_markers=custom_markers)
    """
    visualizer = TableBenchmarkVisualizer(results_dir)
    
    # Override colors if provided
    if custom_colors:
        visualizer.format_colors.update(custom_colors)
    
    # Override markers if provided
    if custom_markers:
        visualizer.model_markers.update(custom_markers)
    
    visualizer.load_benchmark_results()
    visualizer.print_summary_statistics()
    visualizer.plot_alignment_vs_token_cost(save_path=output_file, 
                                           figsize=figsize,
                                           show_averages=show_averages)


def export_plot_data(results_dir: str = "./output", 
                    output_file: str = "plot_data.csv",
                    include_averages: bool = True):
    """
    Export the prepared plot data to CSV for external analysis.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output CSV file path
        include_averages: Whether to include benchmark averages in export
    
    Example:
        >>> df = export_plot_data("./output", "my_data.csv")
        >>> print(df.head())
    
    Returns:
        DataFrame with plot data
    """
    import pandas as pd
    
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available to export.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(visualizer.results)
    
    # Add averages if requested
    if include_averages and visualizer.model_averages:
        df_avg = pd.DataFrame(visualizer.model_averages)
        df = pd.concat([df, df_avg], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Plot data exported to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("DATA SUMMARY")
    print("="*80)
    print(f"Total data points: {len(df)}")
    print(f"Number of models: {df['model_name'].nunique()}")
    print(f"Table formats: {', '.join(df['table_format'].unique())}")
    print(f"Model types: {', '.join(df['model_type'].unique())}")
    print("\nPer-model summary:")
    print("-"*80)
    summary = df.groupby('model_name').agg({
        'alignment_rate': 'mean',
        'avg_total_tokens': 'mean',
        'reasoning_ratio': 'mean',
        'avg_content_score': 'mean'
    }).round(4)
    print(summary)
    print("="*80 + "\n")
    
    return df


def compare_models(results_dir: str = "./output", 
                  model_names: list = None, 
                  output_file: str = "model_comparison.png",
                  show_averages: bool = True):
    """
    Create a comparison plot for specific models.
    
    Args:
        results_dir: Directory containing result JSON files
        model_names: List of model names to compare (None = all models)
        output_file: Output filename for the plot
        show_averages: Whether to show benchmark averages
    
    Example:
        >>> compare_models("./output", 
        ...               model_names=['gpt-4o', 'deepseek-r1', 'claude-3'],
        ...               output_file="top_models.png")
    """
    visualizer = TableBenchmarkVisualizer(results_dir)
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
    visualizer.plot_alignment_vs_token_cost(save_path=output_file, show_averages=show_averages)


def analyze_format_performance(results_dir: str = "./output", 
                              table_format: str = None,
                              output_file: str = None,
                              show_averages: bool = False):
    """
    Analyze performance for a specific table format.
    
    Args:
        results_dir: Directory containing result JSON files
        table_format: Table format to analyze ('latex', 'markdown', 'text', 'html')
        output_file: Output filename for the plot
        show_averages: Whether to show benchmark averages (usually False for single format)
    
    Example:
        >>> analyze_format_performance("./output", "latex", "latex_analysis.png")
    """
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    # Filter for specific format if provided
    if table_format:
        visualizer.results = [r for r in visualizer.results 
                             if r['table_format'] == table_format]
        
        if not visualizer.results:
            print(f"No data available for format: {table_format}")
            return
        
        print(f"\nAnalyzing table format: {table_format}")
        print(f"Found {len(visualizer.results)} model evaluations")
        
        # Create output filename if not provided
        if output_file is None:
            output_file = f"{table_format}_analysis.png"
        
        # Print format-specific statistics
        alignment_rates = [r['alignment_rate'] for r in visualizer.results]
        tokens = [r['avg_total_tokens'] for r in visualizer.results]
        reasoning_ratios = [r['reasoning_ratio'] for r in visualizer.results]
        content_scores = [r['avg_content_score'] for r in visualizer.results]
        
        print("\n" + "="*80)
        print(f"FORMAT ANALYSIS: {table_format.upper()}")
        print("="*80)
        print(f"\nAlignment Rate Statistics:")
        print(f"  Mean: {np.mean(alignment_rates):.3f}")
        print(f"  Std:  {np.std(alignment_rates):.3f}")
        print(f"  Min:  {np.min(alignment_rates):.3f}")
        print(f"  Max:  {np.max(alignment_rates):.3f}")
        
        print(f"\nContent Score Statistics:")
        print(f"  Mean: {np.mean(content_scores):.3f}")
        print(f"  Std:  {np.std(content_scores):.3f}")
        print(f"  Min:  {np.min(content_scores):.3f}")
        print(f"  Max:  {np.max(content_scores):.3f}")
        
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
    else:
        # No filter, use default filename
        if output_file is None:
            output_file = "all_formats_analysis.png"
    
    # Generate plot
    visualizer.plot_alignment_vs_token_cost(save_path=output_file, show_averages=show_averages)


def compare_format_difficulties(results_dir: str = "./output",
                                output_file: str = "format_comparison.png",
                                show_averages: bool = True):
    """
    Create a comparison showing how different table formats affect performance.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        show_averages: Whether to show benchmark averages
    
    Example:
        >>> compare_format_difficulties("./output", "format_comparison.png")
    """
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available for comparison.")
        return
    
    # Group by table format
    table_formats = sorted(set([r['table_format'] for r in visualizer.results]))
    
    print("\n" + "="*80)
    print("TABLE FORMAT COMPARISON")
    print("="*80)
    
    for fmt in table_formats:
        format_results = [r for r in visualizer.results if r['table_format'] == fmt]
        if format_results:
            alignment_rates = [r['alignment_rate'] for r in format_results]
            tokens = [r['avg_total_tokens'] for r in format_results]
            
            print(f"\n{fmt.upper() if fmt in ['latex', 'html', 'csv'] else fmt.capitalize()}:")
            print(f"  Models tested: {len(format_results)}")
            print(f"  Avg alignment rate: {np.mean(alignment_rates):.3f}")
            print(f"  Avg token usage: {np.mean(tokens):.1f}")
            print(f"  Best alignment: {np.max(alignment_rates):.3f}")
            print(f"  Worst alignment: {np.min(alignment_rates):.3f}")
    
    print("\n" + "="*80 + "\n")
    
    # Generate plot
    visualizer.print_summary_statistics()
    visualizer.plot_alignment_vs_token_cost(save_path=output_file, show_averages=show_averages)


def generate_efficiency_report(results_dir: str = "./output",
                               output_file: str = "efficiency_report.txt"):
    """
    Generate a detailed efficiency report comparing models and formats.
    Efficiency = Alignment Rate / log10(Avg Tokens)
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output text file path
    
    Example:
        >>> generate_efficiency_report("./output", "efficiency.txt")
    """
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available for efficiency report.")
        return
    
    # Calculate efficiency scores for format-specific results
    efficiency_data = []
    for r in visualizer.results:
        if r['avg_total_tokens'] > 0:
            efficiency = r['alignment_rate'] / np.log10(r['avg_total_tokens'])
        else:
            efficiency = 0
        
        efficiency_data.append({
            'model_name': r['model_name'],
            'table_format': r['table_format'],
            'difficulty': r.get('difficulty', 'unknown'),
            'alignment_rate': r['alignment_rate'],
            'avg_total_tokens': r['avg_total_tokens'],
            'reasoning_ratio': r['reasoning_ratio'],
            'efficiency_score': efficiency
        })
    
    # Calculate efficiency for model averages
    avg_efficiency_data = []
    for avg in visualizer.model_averages:
        if avg['avg_total_tokens'] > 0:
            efficiency = avg['alignment_rate'] / np.log10(avg['avg_total_tokens'])
        else:
            efficiency = 0
        
        avg_efficiency_data.append({
            'model_name': avg['model_name'],
            'alignment_rate': avg['alignment_rate'],
            'avg_total_tokens': avg['avg_total_tokens'],
            'reasoning_ratio': avg['reasoning_ratio'],
            'efficiency_score': efficiency
        })
    
    # Sort by efficiency score
    efficiency_data.sort(key=lambda x: x['efficiency_score'], reverse=True)
    avg_efficiency_data.sort(key=lambda x: x['efficiency_score'], reverse=True)
    
    # Generate report
    report_lines = []
    report_lines.append("="*100)
    report_lines.append("MODEL EFFICIENCY REPORT BY TABLE FORMAT")
    report_lines.append("Efficiency Score = Alignment Rate / log10(Avg Tokens)")
    report_lines.append("="*100)
    report_lines.append("")
    
    # Header for format-specific results
    report_lines.append("FORMAT-SPECIFIC EFFICIENCY RANKINGS:")
    report_lines.append("-"*100)
    report_lines.append(f"{'Rank':<6} {'Model':<20} {'Format':<12} {'Difficulty':<12} "
                       f"{'Alignment':<12} {'Tokens':<12} {'Reasoning':<12} {'Efficiency':<12}")
    report_lines.append("-"*100)
    
    # Data rows
    for i, data in enumerate(efficiency_data, 1):
        report_lines.append(
            f"{i:<6} "
            f"{data['model_name']:<20} "
            f"{data['table_format']:<12} "
            f"{data['difficulty']:<12} "
            f"{data['alignment_rate']:<12.3f} "
            f"{data['avg_total_tokens']:<12.1f} "
            f"{data['reasoning_ratio']:<12.3f} "
            f"{data['efficiency_score']:<12.4f}"
        )
    
    report_lines.append("="*100)
    report_lines.append("")
    
    # Benchmark average efficiency rankings
    if avg_efficiency_data:
        report_lines.append("BENCHMARK AVERAGE EFFICIENCY RANKINGS (across all formats):")
        report_lines.append("-"*100)
        report_lines.append(f"{'Rank':<6} {'Model':<30} {'Avg Alignment':<18} "
                           f"{'Avg Tokens':<18} {'Avg Reasoning':<18} {'Efficiency':<12}")
        report_lines.append("-"*100)
        
        for i, data in enumerate(avg_efficiency_data, 1):
            report_lines.append(
                f"{i:<6} "
                f"{data['model_name']:<30} "
                f"{data['alignment_rate']:<18.3f} "
                f"{data['avg_total_tokens']:<18.1f} "
                f"{data['reasoning_ratio']:<18.3f} "
                f"{data['efficiency_score']:<12.4f}"
            )
        
        report_lines.append("="*100)
        report_lines.append("")
    
    # Add summary by model
    report_lines.append("AVERAGE EFFICIENCY BY MODEL (across all formats):")
    report_lines.append("-"*100)
    
    model_efficiencies = {}
    for data in efficiency_data:
        model = data['model_name']
        if model not in model_efficiencies:
            model_efficiencies[model] = []
        model_efficiencies[model].append(data['efficiency_score'])
    
    model_avg = [(model, np.mean(scores)) for model, scores in model_efficiencies.items()]
    model_avg.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, avg_eff) in enumerate(model_avg, 1):
        report_lines.append(f"{i}. {model:<30} {avg_eff:.4f}")
    
    report_lines.append("="*100)
    report_lines.append("")
    
    # Add summary by format
    report_lines.append("AVERAGE EFFICIENCY BY TABLE FORMAT:")
    report_lines.append("-"*100)
    
    format_efficiencies = {}
    for data in efficiency_data:
        fmt = data['table_format']
        if fmt not in format_efficiencies:
            format_efficiencies[fmt] = []
        format_efficiencies[fmt].append(data['efficiency_score'])
    
    format_avg = [(fmt, np.mean(scores)) for fmt, scores in format_efficiencies.items()]
    format_avg.sort(key=lambda x: x[1], reverse=True)
    
    for i, (fmt, avg_eff) in enumerate(format_avg, 1):
        fmt_label = fmt.upper() if fmt in ['latex', 'html', 'csv'] else fmt.capitalize()
        report_lines.append(f"{i}. {fmt_label:<30} {avg_eff:.4f}")
    
    report_lines.append("="*100)
    
    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nEfficiency report saved to {output_file}")
    
    return efficiency_data


def create_pareto_frontier(results_dir: str = "./output",
                          output_file: str = "pareto_frontier.png",
                          show_averages: bool = True):
    """
    Create a plot highlighting the Pareto frontier (models that are not dominated
    by any other model in terms of both alignment rate and token efficiency).
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename for the plot
        show_averages: Whether to show and calculate Pareto for averages
    
    Example:
        >>> create_pareto_frontier("./output", "pareto.png")
    """
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available for Pareto frontier analysis.")
        return
    
    # Calculate Pareto frontier for each table format
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for table_format in sorted(set([r['table_format'] for r in visualizer.results])):
        format_results = [r for r in visualizer.results if r['table_format'] == table_format]
        
        # Sort by tokens (ascending) for Pareto calculation
        format_results.sort(key=lambda x: x['avg_total_tokens'])
        
        # Find Pareto frontier points
        pareto_points = []
        max_alignment_so_far = -1
        
        for r in format_results:
            if r['alignment_rate'] > max_alignment_so_far:
                pareto_points.append(r)
                max_alignment_so_far = r['alignment_rate']
        
        # Plot all points
        x_all = [r['avg_total_tokens'] for r in format_results]
        y_all = [r['alignment_rate'] for r in format_results]
        ax.scatter(x_all, y_all, 
                  c=visualizer.format_colors.get(table_format, '#95a5a6'),
                  alpha=0.3,
                  s=100,
                  edgecolors='none',
                  zorder=1)
        
        # Highlight Pareto frontier
        x_pareto = [r['avg_total_tokens'] for r in pareto_points]
        y_pareto = [r['alignment_rate'] for r in pareto_points]
        
        fmt_label = table_format.upper() if table_format in ['latex', 'html', 'csv'] else table_format.capitalize()
        
        ax.scatter(x_pareto, y_pareto,
                  c=visualizer.format_colors.get(table_format, '#95a5a6'),
                  alpha=0.9,
                  s=200,
                  edgecolors='black',
                  linewidths=2,
                  label=f'{fmt_label} (Pareto)',
                  marker='*',
                  zorder=3)
        
        # Draw Pareto line
        if len(pareto_points) > 1:
            ax.plot(x_pareto, y_pareto,
                   c=visualizer.format_colors.get(table_format, '#95a5a6'),
                   linestyle='--',
                   linewidth=2,
                   alpha=0.6,
                   zorder=2)
        
        # Annotate Pareto points
        for r in pareto_points:
            ax.annotate(r['model_name'],
                       xy=(r['avg_total_tokens'], r['alignment_rate']),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8,
                       alpha=0.7)
    
    # Add benchmark averages Pareto frontier if requested
    if show_averages and visualizer.model_averages:
        avg_sorted = sorted(visualizer.model_averages, key=lambda x: x['avg_total_tokens'])
        
        # Find Pareto frontier for averages
        pareto_avg = []
        max_alignment_so_far = -1
        
        for r in avg_sorted:
            if r['alignment_rate'] > max_alignment_so_far:
                pareto_avg.append(r)
                max_alignment_so_far = r['alignment_rate']
        
        # Plot all average points
        x_all_avg = [r['avg_total_tokens'] for r in visualizer.model_averages]
        y_all_avg = [r['alignment_rate'] for r in visualizer.model_averages]
        ax.scatter(x_all_avg, y_all_avg,
                  c=visualizer.format_colors['benchmark_average'],
                  alpha=0.3,
                  s=100,
                  edgecolors='none',
                  zorder=1)
        
        # Highlight Pareto frontier for averages
        x_pareto_avg = [r['avg_total_tokens'] for r in pareto_avg]
        y_pareto_avg = [r['alignment_rate'] for r in pareto_avg]
        
        ax.scatter(x_pareto_avg, y_pareto_avg,
                  c=visualizer.format_colors['benchmark_average'],
                  alpha=0.7,
                  s=250,
                  edgecolors='darkgray',
                  linewidths=2.5,
                  label='Benchmark Avg (Pareto)',
                  marker='*',
                  zorder=4)
        
        # Draw Pareto line for averages
        if len(pareto_avg) > 1:
            ax.plot(x_pareto_avg, y_pareto_avg,
                   c=visualizer.format_colors['benchmark_average'],
                   linestyle='-',
                   linewidth=2.5,
                   alpha=0.5,
                   zorder=2)
        
        # Annotate average Pareto points
        for r in pareto_avg:
            ax.annotate(r['model_name'],
                       xy=(r['avg_total_tokens'], r['alignment_rate']),
                       xytext=(5, -10),
                       textcoords='offset points',
                       fontsize=9,
                       fontweight='bold',
                       alpha=0.8,
                       color='darkgray')
    
    ax.set_xscale('log')
    ax.set_xlabel('Average Total Tokens (log scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Alignment Rate', fontsize=13, fontweight='bold')
    ax.set_title('Pareto Frontier: Optimal Alignment-Token Trade-offs by Format', 
                fontsize=15, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Pareto frontier plot saved to {output_file}")
    plt.show()


def create_format_comparison_heatmap(results_dir: str = "./output",
                                     output_file: str = "format_heatmap.png"):
    """
    Create a heatmap showing alignment rates across models and formats.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename for the plot
    
    Example:
        >>> create_format_comparison_heatmap("./output", "heatmap.png")
    """
    import pandas as pd
    
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available for heatmap.")
        return
    
    # Create pivot table
    df = pd.DataFrame(visualizer.results)
    pivot = df.pivot_table(values='alignment_rate', 
                           index='model_name', 
                           columns='table_format',
                           aggfunc='mean')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([col.upper() if col in ['latex', 'html', 'csv'] else col.capitalize() 
                        for col in pivot.columns])
    ax.set_yticklabels(pivot.index)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Alignment Rate', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title('Alignment Rate Heatmap: Models vs Table Formats', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Table Format', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")
    plt.show()


def create_token_comparison_heatmap(results_dir: str = "./output",
                                    output_file: str = "token_heatmap.png"):
    """
    Create a heatmap showing average token usage across models and formats.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename for the plot
    
    Example:
        >>> create_token_comparison_heatmap("./output", "token_heatmap.png")
    """
    import pandas as pd
    
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available for heatmap.")
        return
    
    # Create pivot table
    df = pd.DataFrame(visualizer.results)
    pivot = df.pivot_table(values='avg_total_tokens', 
                           index='model_name', 
                           columns='table_format',
                           aggfunc='mean')
    
    # Create heatmap with log scale coloring
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use log scale for better visualization
    log_values = np.log10(pivot.values + 1)  # +1 to avoid log(0)
    im = ax.imshow(log_values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([col.upper() if col in ['latex', 'html', 'csv'] else col.capitalize() 
                        for col in pivot.columns])
    ax.set_yticklabels(pivot.index)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Avg Tokens (log scale)', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations with actual values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{int(value)}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Average Token Usage: Models vs Table Formats', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Table Format', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Token usage heatmap saved to {output_file}")
    plt.show()


def generate_comprehensive_report(results_dir: str = "./output",
                                  output_dir: str = None,
                                  show_averages: bool = True):
    """
    Generate a comprehensive analysis with multiple plots and reports.
    
    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save all outputs (default: same as results_dir)
        show_averages: Whether to include benchmark averages in plots
    
    Example:
        >>> generate_comprehensive_report("./output", "./analysis")
    """
    if output_dir is None:
        output_dir = results_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("="*80 + "\n")
    
    # 1. Main scatter plot
    print("1. Creating main alignment vs token cost plot...")
    quick_generate(results_dir, 
                  os.path.join(output_dir, "alignment_vs_token_cost.png"),
                  show_averages=show_averages)
    
    # 2. Format comparison
    print("\n2. Creating format comparison analysis...")
    compare_format_difficulties(results_dir,
                               os.path.join(output_dir, "format_comparison.png"),
                               show_averages=show_averages)
    
    # 3. Pareto frontier
    print("\n3. Creating Pareto frontier plot...")
    create_pareto_frontier(results_dir,
                          os.path.join(output_dir, "pareto_frontier.png"),
                          show_averages=show_averages)
    
    # 4. Alignment heatmap
    print("\n4. Creating alignment rate heatmap...")
    create_format_comparison_heatmap(results_dir,
                                    os.path.join(output_dir, "alignment_heatmap.png"))
    
    # 5. Token usage heatmap
    print("\n5. Creating token usage heatmap...")
    create_token_comparison_heatmap(results_dir,
                                   os.path.join(output_dir, "token_heatmap.png"))
    
    # 6. Efficiency report
    print("\n6. Generating efficiency report...")
    generate_efficiency_report(results_dir,
                              os.path.join(output_dir, "efficiency_report.txt"))
    
    # 7. Export data
    print("\n7. Exporting plot data to CSV...")
    export_plot_data(results_dir,
                    os.path.join(output_dir, "plot_data.csv"),
                    include_averages=True)
    
    # 8. Individual format analyses
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    formats = set([r['table_format'] for r in visualizer.results])
    for fmt in formats:
        if fmt != 'mixed':
            print(f"\n8. Creating analysis for {fmt} format...")
            analyze_format_performance(results_dir, fmt,
                                     os.path.join(output_dir, f"{fmt}_analysis.png"),
                                     show_averages=False)
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE! All outputs saved to: {output_dir}")
    print("="*80 + "\n")
    
    print("Generated files:")
    print("  - alignment_vs_token_cost.png (main scatter plot)")
    print("  - format_comparison.png (format comparison)")
    print("  - pareto_frontier.png (Pareto optimal models)")
    print("  - alignment_heatmap.png (alignment rate heatmap)")
    print("  - token_heatmap.png (token usage heatmap)")
    print("  - efficiency_report.txt (detailed efficiency analysis)")
    print("  - plot_data.csv (raw data export)")
    for fmt in formats:
        if fmt != 'mixed':
            print(f"  - {fmt}_analysis.png (format-specific analysis)")
    print()


def analyze_reasoning_impact(results_dir: str = "./output",
                             output_file: str = "reasoning_impact.png"):
    """
    Analyze the impact of reasoning ratio on alignment performance.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output filename for the plot
    
    Example:
        >>> analyze_reasoning_impact("./output", "reasoning_analysis.png")
    """
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available for reasoning impact analysis.")
        return
    
    # Create scatter plot: reasoning ratio vs alignment rate
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Reasoning Ratio vs Alignment Rate
    for table_format in visualizer.format_colors.keys():
        if table_format == 'benchmark_average':
            continue
            
        format_results = [r for r in visualizer.results if r['table_format'] == table_format]
        
        if not format_results:
            continue
        
        x = [r['reasoning_ratio'] for r in format_results]
        y = [r['alignment_rate'] for r in format_results]
        
        fmt_label = table_format.upper() if table_format in ['latex', 'html', 'csv'] else table_format.capitalize()
        
        ax1.scatter(x, y,
                   c=visualizer.format_colors[table_format],
                   s=150,
                   alpha=0.7,
                   edgecolors='black',
                   linewidths=1.5,
                   label=fmt_label)
    
    # Add benchmark averages
    if visualizer.model_averages:
        x_avg = [r['reasoning_ratio'] for r in visualizer.model_averages]
        y_avg = [r['alignment_rate'] for r in visualizer.model_averages]
        
        ax1.scatter(x_avg, y_avg,
                   c=visualizer.format_colors['benchmark_average'],
                   s=200,
                   alpha=0.5,
                   edgecolors='gray',
                   linewidths=2,
                   label='Benchmark Avg',
                   marker='D')
    
    ax1.set_xlabel('Reasoning Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Alignment Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Reasoning Ratio vs Alignment Rate', fontsize=13, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(title='Table Format', fontsize=9)
    
    # Plot 2: Reasoning Ratio vs Token Usage
    for table_format in visualizer.format_colors.keys():
        if table_format == 'benchmark_average':
            continue
            
        format_results = [r for r in visualizer.results if r['table_format'] == table_format]
        
        if not format_results:
            continue
        
        x = [r['reasoning_ratio'] for r in format_results]
        y = [r['avg_total_tokens'] for r in format_results]
        
        fmt_label = table_format.upper() if table_format in ['latex', 'html', 'csv'] else table_format.capitalize()
        
        ax2.scatter(x, y,
                   c=visualizer.format_colors[table_format],
                   s=150,
                   alpha=0.7,
                   edgecolors='black',
                   linewidths=1.5,
                   label=fmt_label)
    
    # Add benchmark averages
    if visualizer.model_averages:
        x_avg = [r['reasoning_ratio'] for r in visualizer.model_averages]
        y_avg = [r['avg_total_tokens'] for r in visualizer.model_averages]
        
        ax2.scatter(x_avg, y_avg,
                   c=visualizer.format_colors['benchmark_average'],
                   s=200,
                   alpha=0.5,
                   edgecolors='gray',
                   linewidths=2,
                   label='Benchmark Avg',
                   marker='D')
    
    ax2.set_xlabel('Reasoning Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Total Tokens', fontsize=12, fontweight='bold')
    ax2.set_title('Reasoning Ratio vs Token Usage', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(title='Table Format', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Reasoning impact analysis saved to {output_file}")
    plt.show()


def create_model_comparison_table(results_dir: str = "./output",
                                  output_file: str = "model_comparison_table.txt"):
    """
    Create a formatted text table comparing all models across formats.
    
    Args:
        results_dir: Directory containing result JSON files
        output_file: Output text file path
    
    Example:
        >>> create_model_comparison_table("./output", "comparison.txt")
    """
    import pandas as pd
    
    visualizer = TableBenchmarkVisualizer(results_dir)
    visualizer.load_benchmark_results()
    
    if not visualizer.results:
        print("No data available for comparison table.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(visualizer.results)
    
    # Create pivot tables for different metrics
    alignment_pivot = df.pivot_table(values='alignment_rate', 
                                    index='model_name', 
                                    columns='table_format',
                                    aggfunc='mean')
    
    token_pivot = df.pivot_table(values='avg_total_tokens', 
                                index='model_name', 
                                columns='table_format',
                                aggfunc='mean')
    
    reasoning_pivot = df.pivot_table(values='reasoning_ratio', 
                                    index='model_name', 
                                    columns='table_format',
                                    aggfunc='mean')
    
    # Generate table
    lines = []
    lines.append("="*120)
    lines.append("MODEL COMPARISON TABLE")
    lines.append("="*120)
    lines.append("")
    
    # Alignment Rate Table
    lines.append("ALIGNMENT RATE BY FORMAT:")
    lines.append("-"*120)
    lines.append(alignment_pivot.to_string())
    lines.append("")
    
    # Token Usage Table
    lines.append("AVERAGE TOKEN USAGE BY FORMAT:")
    lines.append("-"*120)
    lines.append(token_pivot.to_string())
    lines.append("")
    
    # Reasoning Ratio Table
    lines.append("REASONING RATIO BY FORMAT:")
    lines.append("-"*120)
    lines.append(reasoning_pivot.to_string())
    lines.append("")
    
    # Benchmark Averages
    if visualizer.model_averages:
        lines.append("BENCHMARK AVERAGES (across all formats):")
        lines.append("-"*120)
        avg_df = pd.DataFrame(visualizer.model_averages)
        avg_df = avg_df[['model_name', 'alignment_rate', 'avg_total_tokens', 'reasoning_ratio']]
        avg_df.columns = ['Model', 'Avg Alignment', 'Avg Tokens', 'Avg Reasoning Ratio']
        avg_df = avg_df.sort_values('Avg Alignment', ascending=False)
        lines.append(avg_df.to_string(index=False))
        lines.append("")
    
    lines.append("="*120)
    
    # Write to file
    table_text = "\n".join(lines)
    print(table_text)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(table_text)
    
    print(f"\nComparison table saved to {output_file}")


if __name__ == "__main__":
    # Run interactive main function
    main()

    # Alternative: Quick generation examples (commented out)
    # quick_generate("./output", "my_plot.png", show_averages=True)
    # compare_models("./output", model_names=['deepseek-r1', 'gpt-4o'])
    # analyze_format_performance("./output", "latex")
    # generate_efficiency_report("./output")
    # create_pareto_frontier("./output", show_averages=True)
    # generate_comprehensive_report("./output", "./analysis", show_averages=True)
    # analyze_reasoning_impact("./output")
    # create_model_comparison_table("./output")