"""
Metrics calculation and reporting for cipher benchmark.
"""

import json
import csv
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

from .evaluator import EvaluationResult, TaskType

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various performance metrics for the benchmark."""
    
    @staticmethod
    def calculate_accuracy(evaluations: List[EvaluationResult]) -> float:
        """
        Calculate overall accuracy.
        
        Args:
            evaluations (List[EvaluationResult]): Evaluation results
            
        Returns:
            float: Accuracy percentage (0-100)
        """
        if not evaluations:
            return 0.0
        
        correct = sum(1 for eval_result in evaluations if eval_result.is_correct)
        total = len(evaluations)
        return (correct / total) * 100.0
    
    @staticmethod
    def calculate_per_task_accuracy(evaluations: List[EvaluationResult]) -> Dict[str, float]:
        """
        Calculate accuracy per task type.
        
        Args:
            evaluations (List[EvaluationResult]): Evaluation results
            
        Returns:
            Dict[str, float]: Accuracy per task type
        """
        task_accuracies = {}
        
        for task_type in TaskType:
            task_evaluations = [e for e in evaluations if e.task_type == task_type]
            if task_evaluations:
                task_accuracies[task_type.value] = MetricsCalculator.calculate_accuracy(task_evaluations)
        
        return task_accuracies
    
    @staticmethod
    def calculate_per_difficulty_accuracy(evaluations: List[EvaluationResult]) -> Dict[str, float]:
        """
        Calculate accuracy per difficulty level.
        
        Args:
            evaluations (List[EvaluationResult]): Evaluation results
            
        Returns:
            Dict[str, float]: Accuracy per difficulty level
        """
        difficulty_accuracies = {}
        
        for difficulty in ['easy', 'medium', 'hard']:
            difficulty_evaluations = [e for e in evaluations if e.difficulty == difficulty]
            if difficulty_evaluations:
                difficulty_accuracies[difficulty] = MetricsCalculator.calculate_accuracy(difficulty_evaluations)
        
        return difficulty_accuracies
    
    @staticmethod
    def calculate_similarity_metrics(evaluations: List[EvaluationResult]) -> Dict[str, float]:
        """
        Calculate similarity-based metrics.
        
        Args:
            evaluations (List[EvaluationResult]): Evaluation results
            
        Returns:
            Dict[str, float]: Similarity metrics
        """
        if not evaluations:
            return {'avg_similarity': 0.0, 'min_similarity': 0.0, 'max_similarity': 0.0}
        
        similarities = [eval_result.similarity_score for eval_result in evaluations]
        
        return {
            'avg_similarity': sum(similarities) / len(similarities),
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'median_similarity': sorted(similarities)[len(similarities) // 2]
        }
    
    @staticmethod
    def calculate_error_distribution(evaluations: List[EvaluationResult]) -> Dict[str, int]:
        """
        Calculate distribution of error types.
        
        Args:
            evaluations (List[EvaluationResult]): Evaluation results
            
        Returns:
            Dict[str, int]: Error type distribution
        """
        error_counts = {}
        
        for eval_result in evaluations:
            if not eval_result.is_correct:
                error_type = eval_result.error_type or 'incorrect_output'
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return error_counts
    
    @staticmethod
    def calculate_comprehensive_metrics(evaluations: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for the benchmark results.
        
        Args:
            evaluations (List[EvaluationResult]): Evaluation results
            
        Returns:
            Dict[str, Any]: Comprehensive metrics
        """
        if not evaluations:
            return {'error': 'No evaluations provided'}
        
        return {
            'overall_accuracy': MetricsCalculator.calculate_accuracy(evaluations),
            'per_task_accuracy': MetricsCalculator.calculate_per_task_accuracy(evaluations),
            'per_difficulty_accuracy': MetricsCalculator.calculate_per_difficulty_accuracy(evaluations),
            'similarity_metrics': MetricsCalculator.calculate_similarity_metrics(evaluations),
            'error_distribution': MetricsCalculator.calculate_error_distribution(evaluations),
            'total_samples': len(evaluations),
            'correct_samples': sum(1 for e in evaluations if e.is_correct),
            'task_distribution': {
                task_type.value: len([e for e in evaluations if e.task_type == task_type])
                for task_type in TaskType
            },
            'difficulty_distribution': {
                difficulty: len([e for e in evaluations if e.difficulty == difficulty])
                for difficulty in ['easy', 'medium', 'hard']
            }
        }


class ResultsExporter:
    """Export benchmark results in various formats."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize results exporter.
        
        Args:
            output_dir (str): Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _create_base_name(model_name: str, experiment_id: Optional[str] = None) -> str:
        """
        Create unified base name for all output files.
        
        Args:
            model_name (str): Name of the model
            experiment_id (str): Optional experiment identifier
            
        Returns:
            str: Base name in format "modelname_experiment_MMDD"
        """
        # Clean model name (replace / and \ with _)
        clean_model_name = model_name.replace('/', '_').replace('\\', '_')
        
        # Use provided experiment_id or default to current date in MMDD format
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%m%d")
        
        return f"{clean_model_name}_experiment_{experiment_id}"
    
    def export_to_json(self, evaluations: List[EvaluationResult], 
                      metrics: Dict[str, Any], filename: str) -> str:
        """
        Export results to JSON format.
        
        Args:
            evaluations (List[EvaluationResult]): Evaluation results
            metrics (Dict[str, Any]): Calculated metrics
            filename (str): Output filename
            
        Returns:
            str: Path to exported file
        """
        filepath = self.output_dir / filename
        
        # Convert evaluation results to dictionaries
        results_data = []
        for eval_result in evaluations:
            result_dict = asdict(eval_result)
            # Convert enum to string
            result_dict['task_type'] = eval_result.task_type.value
            results_data.append(result_dict)
        
        export_data = {
            'metadata': {
                'total_evaluations': len(evaluations),
                'export_timestamp': pd.Timestamp.now().isoformat()
            },
            'metrics': metrics,
            'detailed_results': results_data
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results exported to JSON: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise
    
    def export_to_csv(self, evaluations: List[EvaluationResult], 
                     filename: str) -> str:
        """
        Export results to CSV format.
        
        Args:
            evaluations (List[EvaluationResult]): Evaluation results
            filename (str): Output filename
            
        Returns:
            str: Path to exported file
        """
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = [
                    'task_type', 'input_text', 'expected_output', 'model_output',
                    'is_correct', 'similarity_score', 'error_type', 'difficulty', 'additional_info'
                ]
                writer.writerow(header)
                
                # Write data
                for eval_result in evaluations:
                    row = [
                        eval_result.task_type.value,
                        eval_result.input_text,
                        eval_result.expected_output,
                        eval_result.model_output,
                        eval_result.is_correct,
                        eval_result.similarity_score,
                        eval_result.error_type or '',
                        eval_result.difficulty or '',
                        json.dumps(eval_result.additional_info) if eval_result.additional_info else ''
                    ]
                    writer.writerow(row)
            
            logger.info(f"Results exported to CSV: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def export_summary_report(self, metrics: Dict[str, Any], 
                            model_name: str, filename: str) -> str:
        """
        Export a human-readable summary report.
        
        Args:
            metrics (Dict[str, Any]): Calculated metrics
            model_name (str): Name of the evaluated model
            filename (str): Output filename
            
        Returns:
            str: Path to exported file
        """
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Cipher & Decipher Benchmark Results\n")
                f.write(f"{'=' * 40}\n\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Timestamp: {pd.Timestamp.now().isoformat()}\n\n")
                
                # Overall metrics
                f.write("Overall Performance:\n")
                f.write(f"  Total Samples: {metrics.get('total_samples', 0)}\n")
                f.write(f"  Correct Samples: {metrics.get('correct_samples', 0)}\n")
                f.write(f"  Overall Accuracy: {metrics.get('overall_accuracy', 0):.2f}%\n\n")
                
                # Per-task accuracy
                f.write("Per-Task Accuracy:\n")
                per_task = metrics.get('per_task_accuracy', {})
                for task, accuracy in per_task.items():
                    f.write(f"  {task.replace('_', ' ').title()}: {accuracy:.2f}%\n")
                f.write("\n")
                
                # Per-difficulty accuracy
                f.write("Per-Difficulty Accuracy:\n")
                per_difficulty = metrics.get('per_difficulty_accuracy', {})
                for difficulty, accuracy in per_difficulty.items():
                    f.write(f"  {difficulty.title()}: {accuracy:.2f}%\n")
                f.write("\n")
                
                # Similarity metrics
                f.write("Similarity Metrics:\n")
                similarity = metrics.get('similarity_metrics', {})
                for metric, value in similarity.items():
                    f.write(f"  {metric.replace('_', ' ').title()}: {value:.3f}\n")
                f.write("\n")
                
                # Task distribution
                f.write("Task Distribution:\n")
                task_dist = metrics.get('task_distribution', {})
                for task, count in task_dist.items():
                    f.write(f"  {task.replace('_', ' ').title()}: {count} samples\n")
                f.write("\n")
                
                # Difficulty distribution
                f.write("Difficulty Distribution:\n")
                difficulty_dist = metrics.get('difficulty_distribution', {})
                for difficulty, count in difficulty_dist.items():
                    f.write(f"  {difficulty.title()}: {count} samples\n")
                f.write("\n")
                
                # Error distribution
                f.write("Error Distribution:\n")
                error_dist = metrics.get('error_distribution', {})
                if error_dist:
                    for error_type, count in error_dist.items():
                        f.write(f"  {error_type.replace('_', ' ').title()}: {count} errors\n")
                else:
                    f.write("  No errors recorded\n")
            
            logger.info(f"Summary report exported: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting summary report: {e}")
            raise
    
    def export_all_formats(self, evaluations: List[EvaluationResult], 
                          model_name: str, experiment_id: Optional[str] = None) -> Dict[str, str]:
        """
        Export results in all supported formats with unified naming.
        
        Args:
            evaluations (List[EvaluationResult]): Evaluation results
            model_name (str): Name of the evaluated model
            experiment_id (str): Optional experiment identifier (e.g., "1203")
                               If None, defaults to current date in MMDD format
            
        Returns:
            Dict[str, str]: Paths to exported files
        
        Example:
            For model_name="deepseek-v3" and experiment_id="1203", generates:
            - deepseek-v3_experiment_1203_results.json
            - deepseek-v3_experiment_1203_results.csv
            - deepseek-v3_experiment_1203_summary.txt
        """
        # Create unified base name
        base_name = self._create_base_name(model_name, experiment_id)
        
        # Generate filenames with unified naming
        filenames = {
            'json': f"{base_name}_results.json",
            'csv': f"{base_name}_results.csv",
            'summary': f"{base_name}_summary.txt"
        }
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_comprehensive_metrics(evaluations)
        
        exported_files = {}
        
        try:
            exported_files['json'] = self.export_to_json(
                evaluations, metrics, filenames['json']
            )
            exported_files['csv'] = self.export_to_csv(
                evaluations, filenames['csv']
            )
            exported_files['summary'] = self.export_summary_report(
                metrics, model_name, filenames['summary']
            )
            
            logger.info(f"All formats exported for model: {model_name} (base name: {base_name})")
            logger.info(f"  JSON: {filenames['json']}")
            logger.info(f"  CSV: {filenames['csv']}")
            logger.info(f"  Summary: {filenames['summary']}")
            
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting all formats: {e}")
            raise