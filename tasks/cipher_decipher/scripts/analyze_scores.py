"""
Advanced analysis and visualization for cipher benchmark results.
Provides statistical analysis, error patterns, and performance metrics.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import logging
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

from src.evaluation.evaluator import CipherEvaluator, OutputProcessor, SimilarityCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkAnalyzer:
    """Analyzer for cipher benchmark results with advanced metrics."""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.model_name = self.log_file.stem
        self.raw_data = self._load_data()
        self.data = [item for item in self.raw_data if item.get('model_response', '').strip()]
        
        self.similarity_calc = SimilarityCalculator()
        self.output_processor = OutputProcessor()
        
        if len(self.data) < len(self.raw_data):
            logger.warning(f"Filtered {len(self.raw_data) - len(self.data)} invalid responses")
        
        logger.info(f"Loaded {len(self.data)} valid interactions from {log_file}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSON log file."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {self.log_file}: {e}")
            return []
    
    def _normalize_text(self, text: str, task_type: str) -> str:
        """Normalize text based on task type."""
        if 'decode' in task_type.lower():
            return self.output_processor.normalize_text(text, ignore_case=True, 
                                                        ignore_punctuation=True).upper()
        elif 'morse_encode' in task_type.lower():
            return self.output_processor.normalize_morse_code(text)
        return text.strip()
    
    def calculate_basic_stats(self) -> Dict[str, Any]:
        """Calculate basic performance statistics."""
        stats = {
            'total_tasks': len(self.data),
            'correct_tasks': sum(1 for d in self.data if d['is_correct']),
            'per_task': defaultdict(lambda: {'total': 0, 'correct': 0}),
            'per_difficulty': defaultdict(lambda: {'total': 0, 'correct': 0}),
        }
        
        for item in self.data:
            task_type = item['task_type']
            difficulty = item.get('difficulty', 'unknown')
            
            stats['per_task'][task_type]['total'] += 1
            stats['per_difficulty'][difficulty]['total'] += 1
            
            if item['is_correct']:
                stats['per_task'][task_type]['correct'] += 1
                stats['per_difficulty'][difficulty]['correct'] += 1
        
        stats['overall_accuracy'] = (stats['correct_tasks'] / stats['total_tasks'] * 100) \
                                    if stats['total_tasks'] > 0 else 0
        
        # Calculate accuracies
        for category in ['per_task', 'per_difficulty']:
            for key, counts in stats[category].items():
                counts['accuracy'] = (counts['correct'] / counts['total'] * 100) \
                                    if counts['total'] > 0 else 0
        
        return stats
    
    def calculate_similarity_stats(self) -> Dict[str, Any]:
        """Calculate similarity score statistics."""
        similarities = {'all': [], 'by_task': defaultdict(list)}
        
        for item in self.data:
            task_type = item['task_type']
            expected = self._normalize_text(item['golden_answer'], task_type)
            actual = self._normalize_text(item['extracted_answer'], task_type)
            
            sim = self.similarity_calc.similarity_score(expected, actual)
            similarities['all'].append(sim)
            similarities['by_task'][task_type].append(sim)
        
        def calc_stats(values):
            if not values:
                return {}
            arr = np.array(values)
            return {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'count': len(values)
            }
        
        return {
            'overall': calc_stats(similarities['all']),
            'by_task': {task: calc_stats(scores) for task, scores in similarities['by_task'].items()}
        }
    
    def analyze_length_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between input length and performance."""
        lengths, accuracies, similarities = [], [], []
        
        for item in self.data:
            task_type = item['task_type']
            expected = self._normalize_text(item['golden_answer'], task_type)
            actual = self._normalize_text(item['extracted_answer'], task_type)
            
            lengths.append(len(expected))
            accuracies.append(1 if item['is_correct'] else 0)
            similarities.append(self.similarity_calc.similarity_score(expected, actual))
        
        if len(lengths) > 1:
            acc_corr, acc_p = stats.spearmanr(lengths, accuracies)
            sim_corr, sim_p = stats.spearmanr(lengths, similarities)
        else:
            acc_corr = acc_p = sim_corr = sim_p = 0
        
        return {
            'accuracy_vs_length': {
                'coefficient': float(acc_corr),
                'p_value': float(acc_p),
                'significant': bool(acc_p < 0.05)
            },
            'similarity_vs_length': {
                'coefficient': float(sim_corr),
                'p_value': float(sim_p),
                'significant': bool(sim_p < 0.05)
            },
            'raw_data': {'lengths': lengths, 'accuracies': accuracies, 'similarities': similarities}
        }
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        errors = [item for item in self.data if not item['is_correct']]
        
        error_data = {
            'total_errors': len(errors),
            'length_diffs': [],
            'by_task': defaultdict(int),
            'by_difficulty': defaultdict(int),
            'samples': []
        }
        
        for error in errors[:10]:  # Sample top 10
            task_type = error['task_type']
            expected = self._normalize_text(error['golden_answer'], task_type)
            actual = self._normalize_text(error['extracted_answer'], task_type)
            
            error_data['length_diffs'].append(abs(len(expected) - len(actual)))
            error_data['by_task'][task_type] += 1
            error_data['by_difficulty'][error.get('difficulty', 'unknown')] += 1
            
            error_data['samples'].append({
                'task': task_type,
                'difficulty': error.get('difficulty', 'unknown'),
                'expected': expected[:50],
                'actual': actual[:50],
                'similarity': self.similarity_calc.similarity_score(expected, actual)
            })
        
        return error_data


class Visualizer:
    """Create visualizations for benchmark results."""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_overview(self, stats: Dict, similarity: Dict, model_name: str):
        """Create overview dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name} - Performance Overview', fontsize=16, fontweight='bold')
        
        # Overall accuracy
        ax = axes[0, 0]
        categories = ['Correct', 'Incorrect']
        values = [stats['correct_tasks'], stats['total_tasks'] - stats['correct_tasks']]
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Overall: {stats['overall_accuracy']:.1f}%", fontweight='bold')
        
        # Task-wise accuracy
        ax = axes[0, 1]
        tasks = list(stats['per_task'].keys())
        accs = [stats['per_task'][t]['accuracy'] for t in tasks]
        bars = ax.barh(tasks, accs, color='skyblue', edgecolor='black')
        for bar, acc in zip(bars, accs):
            ax.text(acc + 1, bar.get_y() + bar.get_height()/2, f'{acc:.1f}%', 
                   va='center', fontweight='bold')
        ax.set_xlabel('Accuracy (%)')
        ax.set_title('Accuracy by Task', fontweight='bold')
        ax.set_xlim(0, 105)
        
        # Difficulty comparison
        ax = axes[1, 0]
        diffs = ['easy', 'medium', 'hard']
        accs = [stats['per_difficulty'].get(d, {}).get('accuracy', 0) for d in diffs]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.bar(diffs, accs, color=colors, alpha=0.7, edgecolor='black')
        for i, acc in enumerate(accs):
            ax.text(i, acc + 2, f'{acc:.1f}%', ha='center', fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Difficulty', fontweight='bold')
        ax.set_ylim(0, 105)
        
        # Similarity distribution
        ax = axes[1, 1]
        if similarity['overall']:
            data = [similarity['by_task'][t] for t in tasks if t in similarity['by_task']]
            means = [d['mean'] for d in data if d]
            ax.bar(range(len(means)), means, color='coral', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(tasks)))
            ax.set_xticklabels([t.split('_')[0] for t in tasks], rotation=45)
            ax.set_ylabel('Mean Similarity')
            ax.set_title('Similarity by Task', fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.axhline(y=similarity['overall']['mean'], color='red', linestyle='--', 
                      label=f"Overall: {similarity['overall']['mean']:.3f}")
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved overview plot")
    
    def plot_correlation(self, correlation: Dict, model_name: str):
        """Plot length correlation analysis."""
        if not correlation.get('raw_data'):
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{model_name} - Input Length Analysis', fontsize=14, fontweight='bold')
        
        data = correlation['raw_data']
        lengths = np.array(data['lengths'])
        
        # Accuracy vs Length
        ax = axes[0]
        accuracies = np.array(data['accuracies'])
        ax.scatter(lengths, accuracies + np.random.normal(0, 0.02, len(accuracies)), 
                  alpha=0.5, edgecolors='black', linewidth=0.5)
        
        z = np.polyfit(lengths, accuracies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(lengths.min(), lengths.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)
        
        corr_info = correlation['accuracy_vs_length']
        text = f"ρ = {corr_info['coefficient']:.3f}\np = {corr_info['p_value']:.4f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Input Length')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Length')
        ax.set_ylim(-0.1, 1.1)
        
        # Similarity vs Length
        ax = axes[1]
        similarities = np.array(data['similarities'])
        scatter = ax.scatter(lengths, similarities, c=accuracies, cmap='RdYlGn', 
                           alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Correct/Incorrect')
        
        z = np.polyfit(lengths, similarities, 1)
        p = np.poly1d(z)
        ax.plot(x_line, p(x_line), "b--", linewidth=2, alpha=0.8)
        
        corr_info = correlation['similarity_vs_length']
        text = f"ρ = {corr_info['coefficient']:.3f}\np = {corr_info['p_value']:.4f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel('Input Length')
        ax.set_ylabel('Similarity Score')
        ax.set_title('Similarity vs Length')
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{model_name}_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation plot")
    
    def plot_multi_model_comparison(self, models: List[str], all_stats: List[Dict]):
        """Create comparison plot for multiple models."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Multi-Model Comparison', fontsize=16, fontweight='bold')
        
        # Overall accuracy
        ax = axes[0]
        accs = [s['overall_accuracy'] for s in all_stats]
        bars = ax.bar(models, accs, alpha=0.7, edgecolor='black')
        for bar, acc in zip(bars, accs):
            color = '#2ecc71' if acc >= 70 else '#f39c12' if acc >= 40 else '#e74c3c'
            bar.set_color(color)
            ax.text(bar.get_x() + bar.get_width()/2, acc + 2, f'{acc:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Overall Accuracy', fontweight='bold')
        ax.set_ylim(0, 105)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # Task comparison
        ax = axes[1]
        all_tasks = set()
        for s in all_stats:
            all_tasks.update(s['per_task'].keys())
        tasks = sorted(all_tasks)
        
        x = np.arange(len(tasks))
        width = 0.8 / len(models)
        
        for i, (model, stats) in enumerate(zip(models, all_stats)):
            accs = [stats['per_task'].get(t, {}).get('accuracy', 0) for t in tasks]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, accs, width, label=model, alpha=0.7)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Task-wise Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', '\n') for t in tasks], fontsize=9)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # Difficulty comparison
        ax = axes[2]
        difficulties = ['easy', 'medium', 'hard']
        x = np.arange(len(difficulties))
        
        for i, (model, stats) in enumerate(zip(models, all_stats)):
            accs = [stats['per_difficulty'].get(d, {}).get('accuracy', 0) for d in difficulties]
            offset = (i - len(models)/2 + 0.5) * width
            ax.bar(x + offset, accs, width, label=model, alpha=0.7)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Difficulty Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([d.title() for d in difficulties])
        ax.legend(fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'multi_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved multi-model comparison plot")


class ReportGenerator:
    """Generate text and JSON reports."""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, model_name: str, stats: Dict, similarity: Dict, 
                       correlation: Dict, errors: Dict):
        """Generate comprehensive text report."""
        report_path = self.output_dir / f'{model_name}_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"BENCHMARK ANALYSIS REPORT - {model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall performance
            f.write("1. OVERALL PERFORMANCE\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Tasks:      {stats['total_tasks']}\n")
            f.write(f"Correct:          {stats['correct_tasks']}\n")
            f.write(f"Overall Accuracy: {stats['overall_accuracy']:.2f}%\n\n")
            
            # Task breakdown
            f.write("Performance by Task:\n")
            for task, data in sorted(stats['per_task'].items()):
                f.write(f"  {task:25} {data['correct']:3}/{data['total']:3} "
                       f"({data['accuracy']:6.2f}%)\n")
            f.write("\n")
            
            # Difficulty breakdown
            f.write("Performance by Difficulty:\n")
            for diff in ['easy', 'medium', 'hard']:
                if diff in stats['per_difficulty']:
                    data = stats['per_difficulty'][diff]
                    f.write(f"  {diff.title():25} {data['correct']:3}/{data['total']:3} "
                           f"({data['accuracy']:6.2f}%)\n")
            f.write("\n\n")
            
            # Similarity analysis
            f.write("2. SIMILARITY ANALYSIS\n")
            f.write("-"*80 + "\n")
            if similarity['overall']:
                s = similarity['overall']
                f.write(f"Mean:    {s['mean']:.4f}\n")
                f.write(f"Median:  {s['median']:.4f}\n")
                f.write(f"Std Dev: {s['std']:.4f}\n")
                f.write(f"Range:   {s['min']:.4f} - {s['max']:.4f}\n\n")
            
            f.write("By Task:\n")
            for task, s in sorted(similarity['by_task'].items()):
                if s:
                    f.write(f"  {task:25} μ={s['mean']:.4f} σ={s['std']:.4f}\n")
            f.write("\n\n")
            
            # Correlation analysis
            f.write("3. LENGTH CORRELATION\n")
            f.write("-"*80 + "\n")
            
            acc_corr = correlation['accuracy_vs_length']
            f.write(f"Accuracy vs Length:\n")
            f.write(f"  Spearman ρ: {acc_corr['coefficient']:.4f}\n")
            f.write(f"  p-value:    {acc_corr['p_value']:.4f}\n")
            f.write(f"  Significant: {'Yes' if acc_corr['significant'] else 'No'}\n\n")
            
            sim_corr = correlation['similarity_vs_length']
            f.write(f"Similarity vs Length:\n")
            f.write(f"  Spearman ρ: {sim_corr['coefficient']:.4f}\n")
            f.write(f"  p-value:    {sim_corr['p_value']:.4f}\n")
            f.write(f"  Significant: {'Yes' if sim_corr['significant'] else 'No'}\n\n\n")
            
            # Error analysis
            f.write("4. ERROR ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Errors: {errors['total_errors']}\n\n")
            
            if errors['length_diffs']:
                f.write(f"Mean Length Difference: {np.mean(errors['length_diffs']):.2f} chars\n\n")
            
            f.write("Errors by Task:\n")
            for task, count in sorted(errors['by_task'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {task:25} {count:3}\n")
            f.write("\n")
            
            f.write("Sample Errors:\n")
            for i, sample in enumerate(errors['samples'][:5], 1):
                f.write(f"\n  #{i} ({sample['task']} - {sample['difficulty']}):\n")
                f.write(f"    Expected: {sample['expected']}...\n")
                f.write(f"    Actual:   {sample['actual']}...\n")
                f.write(f"    Similarity: {sample['similarity']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Saved report to {report_path}")
        return report_path
    
    def generate_json(self, model_name: str, all_data: Dict):
        """Generate JSON summary."""
        json_path = self.output_dir / f'{model_name}_summary.json'
        
        def convert(obj):
            """Convert non-serializable types."""
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, defaultdict):
                return dict(obj)
            elif isinstance(obj, dict):
                return {convert(k): convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            return obj
        
        try:
            data = convert(all_data)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON to {json_path}")
            return json_path
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")
            return None
    
    def generate_comparison_report(self, models: List[str], all_stats: List[Dict]):
        """Generate multi-model comparison report."""
        report_path = self.output_dir / 'comparison_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Models: {', '.join(models)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Rankings
            f.write("OVERALL RANKINGS\n")
            f.write("-"*80 + "\n")
            
            ranking = sorted(zip(models, all_stats), 
                           key=lambda x: x[1]['overall_accuracy'], reverse=True)
            
            for rank, (model, stats) in enumerate(ranking, 1):
                f.write(f"{rank}. {model:30} {stats['overall_accuracy']:6.2f}%\n")
            f.write("\n")
            
            # Detailed comparison
            f.write("DETAILED COMPARISON\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"{'Metric':<25}")
            for model in models:
                f.write(f"{model[:12]:>13}")
            f.write("\n" + "-"*80 + "\n")
            
            f.write(f"{'Overall Accuracy (%)':<25}")
            for stats in all_stats:
                f.write(f"{stats['overall_accuracy']:13.2f}")
            f.write("\n")
            
            # Task comparison
            all_tasks = set()
            for stats in all_stats:
                all_tasks.update(stats['per_task'].keys())
            
            f.write("\nTask-wise Accuracy (%):\n")
            for task in sorted(all_tasks):
                f.write(f"  {task:<23}")
                for stats in all_stats:
                    acc = stats['per_task'].get(task, {}).get('accuracy', 0)
                    f.write(f"{acc:13.2f}")
                f.write("\n")
            
            # Difficulty comparison
            f.write("\nDifficulty-wise Accuracy (%):\n")
            for diff in ['easy', 'medium', 'hard']:
                f.write(f"  {diff.title():<23}")
                for stats in all_stats:
                    acc = stats['per_difficulty'].get(diff, {}).get('accuracy', 0)
                    f.write(f"{acc:13.2f}")
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Saved comparison report to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze cipher benchmark results")
    parser.add_argument('log_files', nargs='+', help='Path(s) to response log JSON file(s)')
    parser.add_argument('--output', '-o', default='analysis_results', help='Output directory')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare multiple models')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--no-reports', action='store_true', help='Skip generating reports')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing {len(args.log_files)} file(s)")
    logger.info(f"Output directory: {output_dir}")
    
    visualizer = Visualizer(output_dir)
    reporter = ReportGenerator(output_dir)
    
    analyzers = []
    all_stats = []
    
    # Analyze each file
    for log_file in args.log_files:
        logger.info(f"\nProcessing: {log_file}")
        
        try:
            analyzer = BenchmarkAnalyzer(log_file)
            analyzers.append(analyzer)
            
            # Calculate metrics
            stats = analyzer.calculate_basic_stats()
            similarity = analyzer.calculate_similarity_stats()
            correlation = analyzer.analyze_length_correlation()
            errors = analyzer.analyze_errors()
            
            all_stats.append(stats)
            
            # Console output
            print(f"\n{'='*60}")
            print(f"Summary: {analyzer.model_name}")
            print(f"{'='*60}")
            print(f"Total:    {stats['total_tasks']}")
            print(f"Correct:  {stats['correct_tasks']}")
            print(f"Accuracy: {stats['overall_accuracy']:.2f}%")
            if similarity['overall']:
                print(f"Similarity: {similarity['overall']['mean']:.4f}")
            
            # Generate outputs
            if not args.no_plots:
                visualizer.plot_overview(stats, similarity, analyzer.model_name)
                visualizer.plot_correlation(correlation, analyzer.model_name)
            
            if not args.no_reports:
                reporter.generate_report(analyzer.model_name, stats, similarity, 
                                       correlation, errors)
                reporter.generate_json(analyzer.model_name, {
                    'stats': stats,
                    'similarity': similarity,
                    'correlation': correlation,
                    'errors': errors
                })
            
            logger.info(f"✓ Completed: {analyzer.model_name}")
            
        except Exception as e:
            logger.error(f"Error processing {log_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Multi-model comparison
    if args.compare and len(analyzers) > 1:
        logger.info("\n" + "="*60)
        logger.info("Generating multi-model comparison")
        logger.info("="*60)
        
        try:
            model_names = [a.model_name for a in analyzers]
            
            if not args.no_plots:
                visualizer.plot_multi_model_comparison(model_names, all_stats)
            
            if not args.no_reports:
                reporter.generate_comparison_report(model_names, all_stats)
            
            # Print comparison
            print(f"\n{'='*60}")
            print("MODEL COMPARISON")
            print(f"{'='*60}")
            
            ranking = sorted(zip(model_names, all_stats), 
                           key=lambda x: x[1]['overall_accuracy'], reverse=True)
            
            for rank, (model, stats) in enumerate(ranking, 1):
                print(f"{rank}. {model:30} {stats['overall_accuracy']:6.2f}%")
            
            logger.info("✓ Comparison completed")
            
        except Exception as e:
            logger.error(f"Error generating comparison: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    
    # List generated files
    png_files = list(output_dir.glob('*.png'))
    txt_files = list(output_dir.glob('*.txt'))
    json_files = list(output_dir.glob('*.json'))
    
    if png_files:
        print(f"\nVisualizations ({len(png_files)}):")
        for f in sorted(png_files):
            print(f"  - {f.name}")
    
    if txt_files:
        print(f"\nReports ({len(txt_files)}):")
        for f in sorted(txt_files):
            print(f"  - {f.name}")
    
    if json_files:
        print(f"\nData Files ({len(json_files)}):")
        for f in sorted(json_files):
            print(f"  - {f.name}")
    
    # Key insights
    if analyzers:
        print(f"\n{'='*60}")
        print("KEY INSIGHTS")
        print(f"{'='*60}")
        
        if len(analyzers) == 1:
            analyzer = analyzers[0]
            stats = all_stats[0]
            
            # Best/worst tasks
            task_accs = {t: d['accuracy'] for t, d in stats['per_task'].items()}
            if task_accs:
                best = max(task_accs.items(), key=lambda x: x[1])
                worst = min(task_accs.items(), key=lambda x: x[1])
                
                print(f"\nModel: {analyzer.model_name}")
                print(f"  Best Task:  {best[0]} ({best[1]:.1f}%)")
                print(f"  Worst Task: {worst[0]} ({worst[1]:.1f}%)")
                
                # Difficulty trend
                diff_accs = {d: stats['per_difficulty'][d]['accuracy'] 
                           for d in ['easy', 'medium', 'hard'] 
                           if d in stats['per_difficulty']}
                
                if len(diff_accs) == 3:
                    if diff_accs['easy'] > diff_accs['medium'] > diff_accs['hard']:
                        print(f"  Trend: Performance decreases with difficulty ✓")
                    else:
                        print(f"  Trend: Inconsistent difficulty response")
                
                # Length correlation warning
                correlation = analyzer.analyze_length_correlation()
                acc_corr = correlation['accuracy_vs_length']
                
                if acc_corr['significant']:
                    direction = "negatively" if acc_corr['coefficient'] < 0 else "positively"
                    print(f"\n  ⚠️  Performance is {direction} correlated with input length")
                    print(f"     (ρ={acc_corr['coefficient']:.3f}, p={acc_corr['p_value']:.4f})")
        
        elif len(analyzers) > 1:
            # Multi-model insights
            ranking = sorted(zip([a.model_name for a in analyzers], all_stats),
                           key=lambda x: x[1]['overall_accuracy'], reverse=True)
            
            print(f"\nTop Model: {ranking[0][0]} ({ranking[0][1]['overall_accuracy']:.2f}%)")
            
            if len(ranking) >= 2:
                gap = ranking[0][1]['overall_accuracy'] - ranking[-1][1]['overall_accuracy']
                print(f"Performance Gap: {gap:.2f}%")
                
                if gap > 20:
                    print("  ⚠️  Large performance variance detected")
            
            # Task-specific champions
            all_tasks = set()
            for stats in all_stats:
                all_tasks.update(stats['per_task'].keys())
            
            print("\nTask Champions:")
            for task in sorted(all_tasks):
                task_ranking = sorted(
                    zip([a.model_name for a in analyzers], all_stats),
                    key=lambda x: x[1]['per_task'].get(task, {}).get('accuracy', 0),
                    reverse=True
                )
                champion = task_ranking[0][0]
                acc = task_ranking[0][1]['per_task'].get(task, {}).get('accuracy', 0)
                print(f"  {task:20} {champion} ({acc:.1f}%)")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()