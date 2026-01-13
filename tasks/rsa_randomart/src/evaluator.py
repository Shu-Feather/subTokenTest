"""
Evaluator for RSA-Difference Benchmark
Path: src/evaluator.py
"""

from typing import Dict, List, Tuple, Set
import re


class Evaluator:
    """Evaluator for RSA-Difference Benchmark"""
    
    def __init__(
        self,
        coordinate_weight: float = 0.5,
        replacement_weight: float = 0.5
    ):
        """
        Initialize evaluator
        
        Args:
            coordinate_weight: Weight for coordinate accuracy
            replacement_weight: Weight for replacement accuracy
        """
        self.coordinate_weight = coordinate_weight
        self.replacement_weight = replacement_weight
    
    def parse_prediction(self, response: str) -> List[Dict]:
        """
        Parse model prediction from response
        
        Args:
            response: Model response containing answer tags
            
        Returns:
            List of predicted differences
        """
        # Extract content between <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        
        if not answer_match:
            return []
        
        answer_text = answer_match.group(1)
        
        # Parse differences
        # Expected format: (x, y): A -> B or (x, y)：A → B
        predictions = []
        
        # Match patterns like (7, 3): → o or (7, 3)：  → o
        # Use a more precise pattern that captures characters or empty space before arrow
        # Split by lines first to avoid cross-line matching
        lines = answer_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Pattern to match (x, y): original -> modified
            # We need to carefully capture the characters before and after arrow
            match = re.match(r'\((\d+),\s*(\d+)\)\s*[:：]\s*(.*?)\s*(?:->|→|−>|-->)\s*(.*?)$', line)
            
            if match:
                x, y, original, modified = match.groups()
                
                # Clean up the captured strings
                original = original.strip()
                modified = modified.strip()
                
                # Handle empty strings as spaces
                # If completely empty, treat as space
                if not original:
                    original = ' '
                if not modified:
                    modified = ' '
                
                predictions.append({
                    'x': int(x),
                    'y': int(y),
                    'original': original,
                    'modified': modified
                })
        
        return predictions
    
    def evaluate_sample(
        self,
        prediction: List[Dict],
        ground_truth: List[Dict]
    ) -> Dict:
        """
        Evaluate a single sample
        
        Args:
            prediction: List of predicted differences
            ground_truth: List of ground truth differences
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert to sets for comparison
        pred_coords = {(d['x'], d['y']) for d in prediction}
        gt_coords = {(d['x'], d['y']) for d in ground_truth}
        
        # Coordinate accuracy
        correct_coords = pred_coords & gt_coords
        coord_precision = len(correct_coords) / len(pred_coords) if pred_coords else 0.0
        coord_recall = len(correct_coords) / len(gt_coords) if gt_coords else 0.0
        coord_f1 = (
            2 * coord_precision * coord_recall / (coord_precision + coord_recall)
            if (coord_precision + coord_recall) > 0 else 0.0
        )
        
        # Replacement accuracy (only for coordinates that match)
        replacement_correct = 0
        
        # Create lookup dictionaries
        pred_dict = {(d['x'], d['y']): d for d in prediction}
        gt_dict = {(d['x'], d['y']): d for d in ground_truth}
        
        for coord in correct_coords:
            pred_diff = pred_dict[coord]
            gt_diff = gt_dict[coord]
            
            # Check if both original and modified elements match
            if (pred_diff['original'] == gt_diff['original'] and
                pred_diff['modified'] == gt_diff['modified']):
                replacement_correct += 1
        
        replacement_accuracy = (
            replacement_correct / len(correct_coords)
            if correct_coords else 0.0
        )
        
        # Overall score
        overall_score = (
            self.coordinate_weight * coord_f1 +
            self.replacement_weight * replacement_accuracy
        )
        
        return {
            'coordinate_precision': coord_precision,
            'coordinate_recall': coord_recall,
            'coordinate_f1': coord_f1,
            'replacement_accuracy': replacement_accuracy,
            'overall_score': overall_score,
            'num_predictions': len(prediction),
            'num_ground_truth': len(ground_truth),
            'num_correct_coords': len(correct_coords),
            'num_correct_replacements': replacement_correct
        }
    
    def evaluate_batch(
        self,
        predictions: List[List[Dict]],
        ground_truths: List[List[Dict]]
    ) -> Dict:
        """
        Evaluate a batch of samples
        
        Args:
            predictions: List of predicted differences for each sample
            ground_truths: List of ground truth differences for each sample
            
        Returns:
            Dictionary containing aggregated metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Number of predictions and ground truths must match")
        
        results = []
        for pred, gt in zip(predictions, ground_truths):
            result = self.evaluate_sample(pred, gt)
            results.append(result)
        
        # Aggregate results
        num_samples = len(results)
        
        aggregated = {
            'num_samples': num_samples,
            'avg_coordinate_precision': sum(r['coordinate_precision'] for r in results) / num_samples,
            'avg_coordinate_recall': sum(r['coordinate_recall'] for r in results) / num_samples,
            'avg_coordinate_f1': sum(r['coordinate_f1'] for r in results) / num_samples,
            'avg_replacement_accuracy': sum(r['replacement_accuracy'] for r in results) / num_samples,
            'avg_overall_score': sum(r['overall_score'] for r in results) / num_samples,
            'total_predictions': sum(r['num_predictions'] for r in results),
            'total_ground_truth': sum(r['num_ground_truth'] for r in results),
            'total_correct_coords': sum(r['num_correct_coords'] for r in results),
            'total_correct_replacements': sum(r['num_correct_replacements'] for r in results),
            'individual_results': results
        }
        
        return aggregated