"""
Test script for evaluator
Path: tests/test_evaluator.py
"""

import sys
sys.path.append('..')

from src.evaluator import Evaluator


def test_parse_prediction():
    """Test prediction parsing"""
    print("Testing prediction parsing...")
    
    evaluator = Evaluator()
    
    # Test response with answer tags
    response = """
    I found the following differences:
    
    <answer>
    (7, 3):   -> o
    (8, 3): o ->  
    (3, 5): o -> .
    (13, 6):   -> o
    (14, 6): o ->  
    </answer>
    """
    
    predictions = evaluator.parse_prediction(response)
    
    print(f"Parsed {len(predictions)} predictions:")
    for pred in predictions:
        print(f"  ({pred['x']}, {pred['y']}): '{pred['original']}' -> '{pred['modified']}'")
    
    assert len(predictions) == 5, f"Expected 5 predictions, got {len(predictions)}"
    print("✓ Parsing test passed!")


def test_evaluation():
    """Test evaluation metrics"""
    print("\nTesting evaluation...")
    
    evaluator = Evaluator()
    
    # Ground truth
    ground_truth = [
        {'x': 7, 'y': 3, 'original': ' ', 'modified': 'o'},
        {'x': 8, 'y': 3, 'original': 'o', 'modified': ' '},
        {'x': 3, 'y': 5, 'original': 'o', 'modified': '.'},
    ]
    
    # Perfect prediction
    prediction = [
        {'x': 7, 'y': 3, 'original': ' ', 'modified': 'o'},
        {'x': 8, 'y': 3, 'original': 'o', 'modified': ' '},
        {'x': 3, 'y': 5, 'original': 'o', 'modified': '.'},
    ]
    
    result = evaluator.evaluate_sample(prediction, ground_truth)
    
    print(f"Perfect prediction results:")
    print(f"  Coordinate F1: {result['coordinate_f1']:.4f}")
    print(f"  Replacement Accuracy: {result['replacement_accuracy']:.4f}")
    print(f"  Overall Score: {result['overall_score']:.4f}")
    
    assert result['overall_score'] == 1.0, "Perfect prediction should score 1.0"
    
    # Partial prediction
    prediction_partial = [
        {'x': 7, 'y': 3, 'original': ' ', 'modified': 'o'},
        {'x': 8, 'y': 3, 'original': 'o', 'modified': 'x'},  # Wrong replacement
        {'x': 99, 'y': 99, 'original': 'a', 'modified': 'b'},  # Wrong coordinate
    ]
    
    result_partial = evaluator.evaluate_sample(prediction_partial, ground_truth)
    
    print(f"\nPartial prediction results:")
    print(f"  Coordinate F1: {result_partial['coordinate_f1']:.4f}")
    print(f"  Replacement Accuracy: {result_partial['replacement_accuracy']:.4f}")
    print(f"  Overall Score: {result_partial['overall_score']:.4f}")
    
    assert result_partial['overall_score'] < 1.0, "Partial prediction should score < 1.0"
    print("✓ Evaluation test passed!")


if __name__ == '__main__':
    test_parse_prediction()
    test_evaluation()
    print("\n✓ All tests passed!")