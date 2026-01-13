import json
import os
import glob
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

class BenchmarkAnalyzer:
    """Analyzer for LLM Gomoku benchmark results"""
    
    def __init__(self, results_dir: str = ".", output_dir: str = None):
        """
        Initialize the analyzer
        
        Args:
            results_dir: Directory containing the JSON result files
            output_dir: Directory to save output files (defaults to results_dir)
        """
        self.results_dir = results_dir
        self.output_dir = output_dir if output_dir is not None else results_dir
        self.results = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_results(self, pattern: str = "*_*x*_*_*.json") -> int:
        """
        Load all JSON result files matching the pattern
        
        Args:
            pattern: Glob pattern to match result files
            
        Returns:
            Number of files loaded
        """
        json_files = glob.glob(os.path.join(self.results_dir, pattern))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['filename'] = os.path.basename(file_path)
                    self.results.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(self.results)} result files from {self.results_dir}")
        return len(self.results)
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame for easy analysis
        
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for result in self.results:
            row = {
                'model_name': result['model_name'],
                'model_type': result['model_type'],
                'board_size': result['board_size'],
                'num_cases': result['num_cases'],
                'successful_cases': result['successful_cases'],
                'execution_time': result['execution_time'],
                'timestamp': result['timestamp'],
                'accuracy': result['results']['accuracy'],
                'white_wins_f1': result['results']['white_wins_f1'],
                'black_wins_f1': result['results']['black_wins_f1'],
                'no_winner_f1': result['results']['no_winner_f1'],
                'error_cases': result['results']['error_cases'],
                'parse_failures': result['results']['parse_failures'],
                'filename': result['filename']
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def plot_model_comparison(self, metric: str = 'accuracy', 
                             figsize: tuple = (12, 6),
                             save_path: str = None):
        """
        Plot comparison of models by specified metric
        
        Args:
            metric: Metric to compare (accuracy, white_wins_f1, etc.)
            figsize: Figure size
            save_path: Path to save the figure (absolute path)
        """
        df = self.get_summary_dataframe()
        
        plt.figure(figsize=figsize)
        
        # Group by model and calculate mean
        model_stats = df.groupby('model_name')[metric].agg(['mean', 'std', 'count'])
        model_stats = model_stats.sort_values('mean', ascending=False)
        
        # Create bar plot
        ax = model_stats['mean'].plot(kind='bar', yerr=model_stats['std'], 
                                       capsize=5, color='skyblue', edgecolor='black')
        
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Model Name', fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Add value labels on bars
        for i, v in enumerate(model_stats['mean']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        plt.show()
    
    def plot_confusion_matrix(self, model_name: str = None, 
                             index: int = 0,
                             figsize: tuple = (10, 8),
                             save_path: str = None):
        """
        Plot confusion matrix for a specific result
        
        Args:
            model_name: Name of the model to plot
            index: Index of the result if multiple exist
            figsize: Figure size
            save_path: Path to save the figure (absolute path)
        """
        if model_name:
            filtered = [r for r in self.results if r['model_name'] == model_name]
            if not filtered:
                print(f"No results found for model: {model_name}")
                return
            result = filtered[index]
        else:
            result = self.results[index]
        
        # Extract confusion matrix
        cm_dict = result['results']['confusion_matrix']
        
        # Define all possible categories
        categories = ['WHITE_WINS', 'BLACK_WINS', 'NO_WINNER', 'ERROR', 'PARSE_FAIL']
        
        # Create matrix
        cm = np.zeros((3, 5))
        true_labels = ['WHITE_WINS', 'BLACK_WINS', 'NO_WINNER']
        
        for i, true_label in enumerate(true_labels):
            for j, pred_label in enumerate(categories):
                cm[i, j] = cm_dict.get(true_label, {}).get(pred_label, 0)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                    xticklabels=categories, yticklabels=true_labels,
                    cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {result["model_name"]} '
                  f'({result["board_size"]}x{result["board_size"]})',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        plt.show()
    
    def plot_f1_scores(self, figsize: tuple = (12, 6), save_path: str = None):
        """
        Plot F1 scores for all classes across models
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure (absolute path)
        """
        df = self.get_summary_dataframe()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group by model and calculate mean for F1 scores only
        f1_columns = ['white_wins_f1', 'black_wins_f1', 'no_winner_f1']
        models = df.groupby('model_name')[f1_columns].mean()
        
        x = np.arange(len(models.index))
        width = 0.25
        
        bars1 = ax.bar(x - width, models['white_wins_f1'], width, 
                       label='White Wins', color='lightcoral')
        bars2 = ax.bar(x, models['black_wins_f1'], width, 
                       label='Black Wins', color='lightblue')
        bars3 = ax.bar(x + width, models['no_winner_f1'], width, 
                       label='No Winner', color='lightgreen')
        
        ax.set_xlabel('Model Name', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_title('F1 Scores by Class and Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        plt.show()
    
    def plot_execution_time(self, figsize: tuple = (12, 6), save_path: str = None):
        """
        Plot execution time comparison
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure (absolute path)
        """
        df = self.get_summary_dataframe()
        
        plt.figure(figsize=figsize)
        
        # Calculate average time per case
        df['time_per_case'] = df['execution_time'] / df['num_cases']
        
        model_time = df.groupby('model_name')['time_per_case'].agg(['mean', 'std'])
        model_time = model_time.sort_values('mean')
        
        ax = model_time['mean'].plot(kind='barh', xerr=model_time['std'], 
                                      capsize=5, color='coral', edgecolor='black')
        
        plt.title('Average Execution Time per Case', fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Model Name', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        plt.show()
    
    def plot_error_analysis(self, figsize: tuple = (12, 6), save_path: str = None):
        """
        Plot error cases and parse failures
        
        Args:
            figsize: Figure size
            save_path: Path to save the figure (absolute path)
        """
        df = self.get_summary_dataframe()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Error cases
        model_errors = df.groupby('model_name')['error_cases'].mean().sort_values()
        model_errors.plot(kind='barh', ax=ax1, color='salmon', edgecolor='black')
        ax1.set_title('Average Error Cases', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Number of Errors', fontsize=10)
        ax1.grid(axis='x', alpha=0.3)
        
        # Parse failures
        model_parse = df.groupby('model_name')['parse_failures'].mean().sort_values()
        model_parse.plot(kind='barh', ax=ax2, color='khaki', edgecolor='black')
        ax2.set_title('Average Parse Failures', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Parse Failures', fontsize=10)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        plt.show()
    
    def generate_report(self, output_file: str = 'benchmark_report.txt'):
        """
        Generate a text report summarizing all results
        
        Args:
            output_file: Filename to save the report (saved in output_dir)
        """
        if not os.path.isabs(output_file):
            output_file = os.path.join(self.output_dir, output_file)
            
        df = self.get_summary_dataframe()
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM GOMOKU BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {self.results_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Total Results Analyzed: {len(self.results)}\n")
            f.write(f"Unique Models: {df['model_name'].nunique()}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            # Overall stats by model
            numeric_cols = {
                'accuracy': ['mean', 'std'],
                'white_wins_f1': ['mean', 'std'],
                'black_wins_f1': ['mean', 'std'],
                'no_winner_f1': ['mean', 'std'],
                'error_cases': 'mean',
                'execution_time': 'mean'
            }
            model_stats = df.groupby('model_name').agg(numeric_cols).round(4)
            
            f.write(model_stats.to_string())
            f.write("\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("TOP PERFORMING MODELS\n")
            f.write("-" * 80 + "\n\n")
            
            top_accuracy = df.nlargest(5, 'accuracy')[['model_name', 'accuracy', 
                                                         'board_size', 'num_cases']]
            f.write("By Accuracy:\n")
            f.write(top_accuracy.to_string(index=False))
            f.write("\n\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"Report saved to {output_file}")
    
    def plot_all(self, plots_subdir: str = 'plots'):
        """
        Generate all plots and save to directory
        
        Args:
            plots_subdir: Subdirectory name within output_dir to save plots
        """
        plots_dir = os.path.join(self.output_dir, plots_subdir)
        os.makedirs(plots_dir, exist_ok=True)
        
        print(f"Generating all plots to {plots_dir}...")
        
        self.plot_model_comparison('accuracy', 
                                   save_path=os.path.join(plots_dir, 'accuracy_comparison.png'))
        self.plot_f1_scores(save_path=os.path.join(plots_dir, 'f1_scores.png'))
        self.plot_execution_time(save_path=os.path.join(plots_dir, 'execution_time.png'))
        self.plot_error_analysis(save_path=os.path.join(plots_dir, 'error_analysis.png'))
        
        # Generate confusion matrix for each unique model
        for model in set(r['model_name'] for r in self.results):
            safe_name = model.replace('/', '_').replace('\\', '_')
            self.plot_confusion_matrix(model_name=model,
                                      save_path=os.path.join(plots_dir, 
                                                            f'confusion_matrix_{safe_name}.png'))
        
        print(f"All plots saved to {plots_dir}/")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize LLM Gomoku benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results in current directory
  python benchmark_analyzer.py
  
  # Analyze results from specific directory
  python benchmark_analyzer.py --results_dir ./benchmark_results
  
  # Specify different output directory
  python benchmark_analyzer.py --results_dir ./results --output_dir ./analysis
  
  # Generate only specific plots
  python benchmark_analyzer.py --plot accuracy f1
  
  # Skip report generation
  python benchmark_analyzer.py --no_report
        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='.',
        help='Directory containing JSON result files (default: current directory)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save output files (default: same as results_dir)'
    )
    
    parser.add_argument(
        '--plots_subdir',
        type=str,
        default='plots',
        help='Subdirectory name for saving plots (default: plots)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_*x*_*_*.json',
        help='Glob pattern to match result files (default: *_*x*_*_*.json)'
    )
    
    parser.add_argument(
        '--plot',
        nargs='+',
        choices=['all', 'accuracy', 'f1', 'time', 'error', 'confusion'],
        default=['all'],
        help='Specify which plots to generate (default: all)'
    )
    
    parser.add_argument(
        '--no_report',
        action='store_true',
        help='Skip generating text report'
    )
    
    parser.add_argument(
        '--report_name',
        type=str,
        default='benchmark_report.txt',
        help='Name of the report file (default: benchmark_report.txt)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Show plots interactively (in addition to saving)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the analyzer"""
    args = parse_args()
    
    print("=" * 80)
    print("LLM GOMOKU BENCHMARK ANALYZER")
    print("=" * 80)
    print(f"\nResults Directory: {args.results_dir}")
    print(f"Output Directory: {args.output_dir or args.results_dir}")
    print(f"File Pattern: {args.pattern}\n")
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    # Load results
    num_loaded = analyzer.load_results(pattern=args.pattern)
    
    if num_loaded == 0:
        print("No result files found. Exiting.")
        return
    
    # Generate summary DataFrame
    df = analyzer.get_summary_dataframe()
    print("\nSummary Statistics:")
    print("-" * 80)
    print(df.describe())
    print("\n")
    
    # Determine which plots to generate
    plot_types = args.plot
    if 'all' in plot_types:
        plot_types = ['accuracy', 'f1', 'time', 'error', 'confusion']
    
    plots_dir = os.path.join(analyzer.output_dir, args.plots_subdir)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plots
    print(f"\nGenerating plots to {plots_dir}...")
    print("-" * 80)
    
    if 'accuracy' in plot_types:
        print("Generating accuracy comparison plot...")
        analyzer.plot_model_comparison(
            'accuracy',
            save_path=os.path.join(plots_dir, 'accuracy_comparison.png')
        )
        if not args.interactive:
            plt.close()
    
    if 'f1' in plot_types:
        print("Generating F1 scores plot...")
        analyzer.plot_f1_scores(
            save_path=os.path.join(plots_dir, 'f1_scores.png')
        )
        if not args.interactive:
            plt.close()
    
    if 'time' in plot_types:
        print("Generating execution time plot...")
        analyzer.plot_execution_time(
            save_path=os.path.join(plots_dir, 'execution_time.png')
        )
        if not args.interactive:
            plt.close()
    
    if 'error' in plot_types:
        print("Generating error analysis plot...")
        analyzer.plot_error_analysis(
            save_path=os.path.join(plots_dir, 'error_analysis.png')
        )
        if not args.interactive:
            plt.close()
    
    if 'confusion' in plot_types:
        print("Generating confusion matrices...")
        for model in set(r['model_name'] for r in analyzer.results):
            safe_name = model.replace('/', '_').replace('\\', '_')
            print(f"  - {model}")
            analyzer.plot_confusion_matrix(
                model_name=model,
                save_path=os.path.join(plots_dir, f'confusion_matrix_{safe_name}.png')
            )
            if not args.interactive:
                plt.close()
    
    print(f"\nAll plots saved to {plots_dir}/")
    
    # Generate report
    if not args.no_report:
        print("\nGenerating text report...")
        print("-" * 80)
        analyzer.generate_report(output_file=args.report_name)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


# Example usage as library
if __name__ == "__main__":
    # If run with command line arguments, use CLI mode
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        # Example: Use as library
        print("Running in example mode (use command line arguments for full functionality)\n")
        
        # Initialize analyzer with custom directories
        analyzer = BenchmarkAnalyzer(
            results_dir=".",
            output_dir="./analysis_output"
        )
        
        # Load all result files
        num_files = analyzer.load_results()
        
        if num_files > 0:
            # Generate summary DataFrame
            df = analyzer.get_summary_dataframe()
            print("\nSummary DataFrame:")
            print(df.head())
            print("\n")
            
            # Generate individual plots
            print("Generating plots...")
            analyzer.plot_model_comparison('accuracy')
            plt.close()  # Close to prevent display in non-interactive mode
            
            analyzer.plot_f1_scores()
            plt.close()
            
            analyzer.plot_execution_time()
            plt.close()
            
            analyzer.plot_error_analysis()
            plt.close()
            
            # Generate all confusion matrices
            if analyzer.results:
                analyzer.plot_confusion_matrix()
                plt.close()
            
            # Generate comprehensive analysis
            analyzer.plot_all(plots_subdir='plots')
            
            # Generate text report
            analyzer.generate_report('benchmark_report.txt')
            
            print("\nAnalysis complete! Check the 'analysis_output' directory for results.")
        else:
            print("No result files found. Please check the results_dir path.")
            print("\nUsage examples:")
            print("  python benchmark_analyzer.py --results_dir ./benchmark_results")
            print("  python benchmark_analyzer.py --results_dir ./results --output_dir ./analysis")
            print("  python benchmark_analyzer.py --plot accuracy f1 --no_report")