import unittest
import os
import json
import tempfile
from generators.sokoban_generator import SokobanGenerator
from generators.frozenlake_generator import FrozenLakeGenerator
from evaluators.exact_match import ExactMatchEvaluator
from utils.parser import parse_answer


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_sokoban_full_pipeline(self):
        """Test complete Sokoban pipeline from generation to evaluation."""
        # Generate data
        generator = SokobanGenerator(size=8)
        tasks_per_type = {1: 2, 2: 1, 3: 1, 4: 1}
        dataset = generator.generate_dataset(2, tasks_per_type)
        
        # Save to file
        output_path = os.path.join(self.temp_dir, 'sokoban_test.json')
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {'env_type': 'sokoban'},
                'data': dataset
            }, f)
        
        # Load and verify
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(len(loaded['data']), 10)  # 2 maps * 5 tasks
        
        # Simulate evaluation with perfect answers
        evaluator = ExactMatchEvaluator()
        for idx, task in enumerate(loaded['data']):
            # Use ground truth as prediction (perfect score)
            predicted = task['answer']
            ground_truth = task['answer']
            is_correct = evaluator.evaluate_single(predicted, ground_truth, task)
            evaluator.add_result(idx, task, predicted, ground_truth, is_correct)
        
        # Check metrics
        metrics = evaluator.compute_metrics()
        self.assertEqual(metrics['overall_accuracy'], 1.0)
    
    def test_frozenlake_full_pipeline(self):
        """Test complete FrozenLake pipeline."""
        # Generate data
        generator = FrozenLakeGenerator(size=8, num_holes=5)
        tasks_per_type = {1: 2, 2: 1, 3: 1, 4: 1, 5: 1}
        dataset = generator.generate_dataset(2, tasks_per_type)
        
        # Verify dataset structure
        self.assertEqual(len(dataset), 12)  # 2 maps * 6 tasks
        
        # Evaluate with some correct and some incorrect
        evaluator = ExactMatchEvaluator()
        for idx, task in enumerate(dataset):
            # Alternate between correct and incorrect
            if idx % 2 == 0:
                predicted = task['answer']
            else:
                predicted = "WRONG"
            
            ground_truth = task['answer']
            is_correct = evaluator.evaluate_single(predicted, ground_truth, task)
            evaluator.add_result(idx, task, predicted, ground_truth, is_correct)
        
        # Check metrics
        metrics = evaluator.compute_metrics()
        self.assertEqual(metrics['overall_accuracy'], 0.5)
    
    def test_answer_parsing_integration(self):
        """Test answer parsing in realistic scenarios."""
        test_cases = [
            # Task type 1: Element identification
            {
                'response': "Looking at coordinates (2, 3), I see <answer>X</answer>",
                'expected': 'X',
                'task_type': 1
            },
            # Task type 2: Coordinates
            {
                'response': "The player P is located at <answer>(4, 5)</answer>",
                'expected': '(4, 5)',
                'task_type': 2
            },
            # Task type 3: Surrounding elements
            {
                'response': """The surrounding elements are:
                <answer>{"up": "X", "down": "_", "left": "#", "right": "O", 
                         "up-left": "#", "up-right": "_", "down-left": "_", "down-right": "#"}</answer>""",
                'expected': '{"down": "_", "down-left": "_", "down-right": "#", "left": "#", "right": "O", "up": "X", "up-left": "#", "up-right": "_"}',
                'task_type': 3
            },
            # Task type 4: Relative position
            {
                'response': "From X to O, the relative position is <answer>(2, -1)</answer>",
                'expected': '(2, -1)',
                'task_type': 4
            },
            # Task type 5: Count
            {
                'response': "I count <answer>5</answer> holes in the map.",
                'expected': '5',
                'task_type': 5
            }
        ]
        
        evaluator = ExactMatchEvaluator()
        
        for test_case in test_cases:
            parsed = parse_answer(test_case['response'])
            self.assertIsNotNone(parsed, f"Failed to parse: {test_case['response'][:50]}")
            
            task_data = {'task_type': test_case['task_type']}
            is_correct = evaluator.evaluate_single(
                parsed, 
                test_case['expected'], 
                task_data
            )
            self.assertTrue(is_correct, 
                          f"Evaluation failed for task type {test_case['task_type']}")
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        evaluator = ExactMatchEvaluator()
        
        # None prediction
        result = evaluator.evaluate_single(None, "X", {'task_type': 1})
        self.assertFalse(result)
        
        # Empty string
        result = evaluator.evaluate_single("", "X", {'task_type': 1})
        self.assertFalse(result)
        
        # Invalid coordinate format
        result = evaluator.evaluate_single("invalid", "(3, 4)", {'task_type': 2})
        self.assertFalse(result)
        
        # Invalid JSON
        result = evaluator.evaluate_single("{invalid}", '{"up": "X"}', {'task_type': 3})
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()