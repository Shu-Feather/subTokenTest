from typing import List, Dict, Any, Tuple
import numpy as np

class BenchmarkMetrics:
    """Comprehensive metrics for typewriter benchmark evaluation"""
    
    @staticmethod
    def calculate_task_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for a single task"""
        if not results:
            return {}
        
        # Extract scores
        scores = [result.get('score', 0.0) for result in results]
        exact_matches = [result.get('exact_match', False) for result in results]
        
        metrics = {
            'accuracy': np.mean(exact_matches),
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'total_samples': len(results)
        }
        
        return metrics
    
    @staticmethod
    def calculate_overall_metrics(task1_results: List[Dict[str, Any]], 
                                task2_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall benchmark metrics"""
        task1_metrics = BenchmarkMetrics.calculate_task_metrics(task1_results)
        task2_metrics = BenchmarkMetrics.calculate_task_metrics(task2_results)
        
        # Calculate weighted overall score
        if task1_metrics and task2_metrics:
            overall_accuracy = (task1_metrics['accuracy'] + task2_metrics['accuracy']) / 2
            overall_score = (task1_metrics['average_score'] + task2_metrics['average_score']) / 2
        else:
            overall_accuracy = 0.0
            overall_score = 0.0
        
        return {
            'task1_metrics': task1_metrics,
            'task2_metrics': task2_metrics,
            'overall_accuracy': overall_accuracy,
            'overall_score': overall_score
        }
