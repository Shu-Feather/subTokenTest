import json
from typing import Dict, Any
from .base_evaluator import BaseEvaluator
from utils.parser import normalize_answer, parse_coordinate, parse_json_answer


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluator using exact match criterion."""
    
    def __init__(self):
        """Initialize exact match evaluator."""
        super().__init__()
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison."""
        return normalize_answer(text)
    
    def _compare_coordinates(self, pred: str, gt: str) -> bool:
        """Compare coordinate answers."""
        pred_coord = parse_coordinate(pred)
        gt_coord = parse_coordinate(gt)
        
        if pred_coord and gt_coord:
            return pred_coord == gt_coord
        
        # Fallback to string comparison
        return self._normalize_for_comparison(pred) == self._normalize_for_comparison(gt)
    
    def _compare_json(self, pred: str, gt: str) -> bool:
        """Compare JSON answers."""
        pred_json = parse_json_answer(pred)
        gt_json = parse_json_answer(gt)
        
        if pred_json and gt_json:
            # Normalize both JSONs
            pred_str = json.dumps(pred_json, sort_keys=True)
            gt_str = json.dumps(gt_json, sort_keys=True)
            return self._normalize_for_comparison(pred_str) == self._normalize_for_comparison(gt_str)
        
        # Fallback to string comparison
        return self._normalize_for_comparison(pred) == self._normalize_for_comparison(gt)
    
    def evaluate_single(
        self, 
        predicted: str, 
        ground_truth: str,
        task_data: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a single prediction using exact match.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            task_data: Additional task information
            
        Returns:
            True if correct, False otherwise
        """
        if predicted is None or ground_truth is None:
            return False
        
        task_type = task_data.get('task_type')
        
        # Task type 2 and 4: coordinate comparison
        if task_type in [2, 4]:
            return self._compare_coordinates(predicted, ground_truth)
        
        # Task type 3: JSON comparison (surrounding elements)
        elif task_type == 3:
            return self._compare_json(predicted, ground_truth)
        
        # Other task types: simple string comparison
        else:
            return self._normalize_for_comparison(predicted) == self._normalize_for_comparison(ground_truth)