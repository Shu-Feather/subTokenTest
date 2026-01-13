from abc import ABC, abstractmethod
from typing import Dict, List, Any


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.results = []
    
    @abstractmethod
    def evaluate_single(
        self, 
        predicted: str, 
        ground_truth: str,
        task_data: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a single prediction.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            task_data: Additional task information
            
        Returns:
            True if correct, False otherwise
        """
        pass
    
    def add_result(
        self,
        task_id: int,
        task_data: Dict[str, Any],
        predicted: str,
        ground_truth: str,
        is_correct: bool
    ):
        """
        Add evaluation result.
        
        Args:
            task_id: Task ID
            task_data: Task data
            predicted: Predicted answer
            ground_truth: Ground truth answer
            is_correct: Whether prediction is correct
        """
        self.results.append({
            'task_id': task_id,
            'env_type': task_data.get('env_type'),
            'task_type': task_data.get('task_type'),
            'predicted': predicted,
            'ground_truth': ground_truth,
            'is_correct': is_correct
        })
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute overall metrics.
        
        Returns:
            Dictionary of metrics
        """
        if not self.results:
            return {}
        
        # Overall accuracy
        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])
        overall_acc = correct / total if total > 0 else 0
        
        # Accuracy by environment type
        env_types = set(r['env_type'] for r in self.results)
        env_acc = {}
        for env in env_types:
            env_results = [r for r in self.results if r['env_type'] == env]
            env_correct = sum(1 for r in env_results if r['is_correct'])
            env_acc[env] = env_correct / len(env_results) if env_results else 0
        
        # Accuracy by task type
        task_types = set(r['task_type'] for r in self.results)
        task_acc = {}
        for task_type in task_types:
            task_results = [r for r in self.results if r['task_type'] == task_type]
            task_correct = sum(1 for r in task_results if r['is_correct'])
            task_acc[f'type_{task_type}'] = task_correct / len(task_results) if task_results else 0
        
        # Accuracy by environment and task type
        env_task_acc = {}
        for env in env_types:
            for task_type in task_types:
                key = f'{env}_type_{task_type}'
                env_task_results = [
                    r for r in self.results 
                    if r['env_type'] == env and r['task_type'] == task_type
                ]
                if env_task_results:
                    env_task_correct = sum(1 for r in env_task_results if r['is_correct'])
                    env_task_acc[key] = env_task_correct / len(env_task_results)
        
        return {
            'overall_accuracy': overall_acc,
            'total_tasks': total,
            'correct_tasks': correct,
            'accuracy_by_env': env_acc,
            'accuracy_by_task_type': task_acc,
            'accuracy_by_env_and_task': env_task_acc
        }