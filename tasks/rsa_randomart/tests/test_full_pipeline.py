"""
Full pipeline integration test
Path: tests/test_full_pipeline.py
"""

import sys
import os
import tempfile
import shutil
sys.path.append('..')

from src.data_generator import RSAPatternGenerator
from src.evaluator import Evaluator
from src.utils import create_prompt, load_config


def test_full_pipeline():
    """Test the complete pipeline"""
    print("Testing full pipeline...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Generate data
        print("\n1. Testing data generation...")
        generator = RSAPatternGenerator(width=19, height=9)
        samples = generator.generate_batch(num_samples=5, num_differences=3)
        
        assert len(samples) == 5, "Should generate 5 samples"
        
        data_path = os.path.join(temp_dir, 'test_data.json')
        generator.save_samples(samples, data_path)
        assert os.path.exists(data_path), "Data file should exist"
        
        # 2. Load data
        print("2. Testing data loading...")
        loaded_samples = generator.load_samples(data_path)
        assert len(loaded_samples) == 5, "Should load 5 samples"
        
        # 3. Create prompts
        print("3. Testing prompt creation...")
        for sample in samples:
            prompt = create_prompt(sample['pattern1'], sample['pattern2'])
            assert '<answer>' in prompt, "Prompt should contain answer tags"
            assert 'Pattern 1:' in prompt, "Prompt should contain Pattern 1"
            assert 'Pattern 2:' in prompt, "Prompt should contain Pattern 2"
        
        # 4. Test evaluation with mock predictions
        print("4. Testing evaluation...")
        evaluator = Evaluator()
        
        # Create mock perfect predictions
        predictions = [sample['ground_truth'] for sample in samples]
        ground_truths = [sample['ground_truth'] for sample in samples]
        
        results = evaluator.evaluate_batch(predictions, ground_truths)
        
        assert results['avg_overall_score'] == 1.0, "Perfect predictions should score 1.0"
        assert results['avg_coordinate_f1'] == 1.0, "Perfect F1 should be 1.0"
        assert results['avg_replacement_accuracy'] == 1.0, "Perfect accuracy should be 1.0"
        
        # 5. Test with imperfect predictions
        print("5. Testing imperfect predictions...")
        imperfect_predictions = []
        for gt in ground_truths:
            # Take only first half of ground truth
            imperfect_pred = gt[:len(gt)//2] if gt else []
            imperfect_predictions.append(imperfect_pred)
        
        imperfect_results = evaluator.evaluate_batch(imperfect_predictions, ground_truths)
        
        assert imperfect_results['avg_overall_score'] < 1.0, "Imperfect predictions should score < 1.0"
        assert imperfect_results['avg_coordinate_recall'] < 1.0, "Recall should be < 1.0"
        
        # 6. Test parsing
        print("6. Testing response parsing...")
        mock_response = """
        Here are the differences:
        <answer>
        (5, 2): o -> .
        (10, 3):   -> o
        (7, 5): + -> -
        </answer>
        """
        
        parsed = evaluator.parse_prediction(mock_response)
        assert len(parsed) == 3, "Should parse 3 differences"
        assert parsed[0]['x'] == 5, "First x coordinate should be 5"
        assert parsed[0]['y'] == 2, "First y coordinate should be 2"
        
        print("\n✓ Full pipeline test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")
    
    evaluator = Evaluator()
    
    # Empty predictions
    result = evaluator.evaluate_sample([], [{'x': 1, 'y': 1, 'original': 'o', 'modified': '.'}])
    assert result['coordinate_precision'] == 0.0, "Empty prediction should have 0 precision"
    assert result['coordinate_recall'] == 0.0, "Empty prediction should have 0 recall"
    
    # Empty ground truth
    result = evaluator.evaluate_sample([{'x': 1, 'y': 1, 'original': 'o', 'modified': '.'}], [])
    assert result['coordinate_recall'] == 0.0, "Empty ground truth should have 0 recall"
    
    # No answer tags in response
    parsed = evaluator.parse_prediction("Some text without answer tags")
    assert len(parsed) == 0, "Should return empty list without answer tags"
    
    # Malformed coordinates
    response_malformed = """
    <answer>
    (not, valid): a -> b
    (1, 2): o -> .
    </answer>
    """
    parsed = evaluator.parse_prediction(response_malformed)
    assert len(parsed) == 1, "Should only parse valid coordinates"
    
    print("✓ Edge cases test passed!")


def test_different_pattern_sizes():
    """Test different pattern sizes"""
    print("\nTesting different pattern sizes...")
    
    sizes = [(15, 7), (19, 9), (25, 13)]
    
    for width, height in sizes:
        generator = RSAPatternGenerator(width=width, height=height)
        sample = generator.generate_sample(num_differences=5)
        
        # Check pattern dimensions
        pattern = sample['pattern1']
        assert len(pattern) == height + 2, f"Pattern height should be {height + 2}"
        assert len(pattern[1]) == width + 2, f"Pattern width should be {width + 2}"
        
        # Check all differences are within bounds
        for diff in sample['ground_truth']:
            assert 1 <= diff['x'] <= width, f"x should be in range [1, {width}]"
            assert 1 <= diff['y'] <= height, f"y should be in range [1, {height}]"
    
    print("✓ Different pattern sizes test passed!")


def test_element_consistency():
    """Test element consistency"""
    print("\nTesting element consistency...")
    
    available_elements = ['o', '+', '=', '.', ' ']
    generator = RSAPatternGenerator(
        width=19,
        height=9,
        available_elements=available_elements
    )
    
    sample = generator.generate_sample(num_differences=10)
    
    # Check that all elements in differences are from available elements
    for diff in sample['ground_truth']:
        assert diff['original'] in available_elements, \
            f"Original element '{diff['original']}' should be in available_elements"
        assert diff['modified'] in available_elements, \
            f"Modified element '{diff['modified']}' should be in available_elements"
    
    print("✓ Element consistency test passed!")


if __name__ == '__main__':
    test_full_pipeline()
    test_edge_cases()
    test_different_pattern_sizes()
    test_element_consistency()
    print("\n✓ All integration tests passed!")