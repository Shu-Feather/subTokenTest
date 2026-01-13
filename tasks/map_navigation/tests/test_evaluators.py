import unittest
import json
from evaluators.exact_match import ExactMatchEvaluator


class TestExactMatchEvaluator(unittest.TestCase):
    """Test cases for ExactMatchEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ExactMatchEvaluator()
    
    def test_simple_string_comparison(self):
        """Test simple string comparison."""
        task_data = {'task_type': 1}
        
        # Exact match
        self.assertTrue(self.evaluator.evaluate_single('X', 'X', task_data))
        
        # Case insensitive
        self.assertTrue(self.evaluator.evaluate_single('x', 'X', task_data))
        
        # With whitespace
        self.assertTrue(self.evaluator.evaluate_single(' X ', 'X', task_data))
        
        # Different answers
        self.assertFalse(self.evaluator.evaluate_single('X', 'O', task_data))
    
    def test_coordinate_comparison(self):
        """Test coordinate comparison for task type 2."""
        task_data = {'task_type': 2}
        
        # Exact match
        self.assertTrue(self.evaluator.evaluate_single('(3, 4)', '(3, 4)', task_data))
        
        # With different spacing
        self.assertTrue(self.evaluator.evaluate_single('(3,4)', '(3, 4)', task_data))
        
        # Different coordinates
        self.assertFalse(self.evaluator.evaluate_single('(3, 4)', '(4, 3)', task_data))
    
    def test_relative_position_comparison(self):
        """Test relative position comparison for task type 4."""
        task_data = {'task_type': 4}
        
        # Exact match
        self.assertTrue(self.evaluator.evaluate_single('(2, -1)', '(2, -1)', task_data))
        
        # Negative numbers
        self.assertTrue(self.evaluator.evaluate_single('(-1, -2)', '(-1, -2)', task_data))
        
        # Different positions
        self.assertFalse(self.evaluator.evaluate_single('(1, 2)', '(2, 1)', task_data))
    
    def test_json_comparison(self):
        """Test JSON comparison for task type 3."""
        task_data = {'task_type': 3}
        
        # Create JSON strings
        json1 = json.dumps({'up': 'X', 'down': '_', 'left': '#', 'right': 'O'}, sort_keys=True)
        json2 = json.dumps({'down': '_', 'up': 'X', 'right': 'O', 'left': '#'}, sort_keys=True)
        
        # Should match (same content, different order)
        self.assertTrue(self.evaluator.evaluate_single(json1, json2, task_data))
        
        # Different content
        json3 = json.dumps({'up': 'O', 'down': '_', 'left': '#', 'right': 'X'}, sort_keys=True)
        self.assertFalse(self.evaluator.evaluate_single(json1, json3, task_data))
    
    def test_none_handling(self):
        """Test handling of None values."""
        task_data = {'task_type': 1}
        
        self.assertFalse(self.evaluator.evaluate_single(None, 'X', task_data))
        self.assertFalse(self.evaluator.evaluate_single('X', None, task_data))
        self.assertFalse(self.evaluator.evaluate_single(None, None, task_data))
    
    def test_add_result(self):
        """Test adding results."""
        task_data = {'env_type': 'sokoban', 'task_type': 1}
        
        self.evaluator.add_result(0, task_data, 'X', 'X', True)
        self.evaluator.add_result(1, task_data, 'O', 'X', False)
        
        self.assertEqual(len(self.evaluator.results), 2)
        self.assertTrue(self.evaluator.results[0]['is_correct'])
        self.assertFalse(self.evaluator.results[1]['is_correct'])
    
    def test_compute_metrics(self):
        """Test metrics computation."""
        # Add some results
        self.evaluator.add_result(0, {'env_type': 'sokoban', 'task_type': 1}, 'X', 'X', True)
        self.evaluator.add_result(1, {'env_type': 'sokoban', 'task_type': 1}, 'O', 'X', False)
        self.evaluator.add_result(2, {'env_type': 'sokoban', 'task_type': 2}, '(1,2)', '(1, 2)', True)
        self.evaluator.add_result(3, {'env_type': 'frozenlake', 'task_type': 1}, 'G', 'G', True)
        
        metrics = self.evaluator.compute_metrics()
        
        # Check overall accuracy: 3/4 = 0.75
        self.assertEqual(metrics['overall_accuracy'], 0.75)
        self.assertEqual(metrics['total_tasks'], 4)
        self.assertEqual(metrics['correct_tasks'], 3)
        
        # Check accuracy by environment
        self.assertEqual(metrics['accuracy_by_env']['sokoban'], 2/3)
        self.assertEqual(metrics['accuracy_by_env']['frozenlake'], 1.0)
        
        # Check accuracy by task type
        self.assertEqual(metrics['accuracy_by_task_type']['type_1'], 2/3)
        self.assertEqual(metrics['accuracy_by_task_type']['type_2'], 1.0)
    
    def test_empty_results(self):
        """Test metrics with no results."""
        metrics = self.evaluator.compute_metrics()
        self.assertEqual(metrics, {})


if __name__ == '__main__':
    unittest.main()