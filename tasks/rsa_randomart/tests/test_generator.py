"""
Test script for data generator
Path: tests/test_generator.py
"""

import sys
sys.path.append('..')

from src.data_generator import RSAPatternGenerator


def test_generation():
    """Test basic pattern generation"""
    print("Testing RSA Pattern Generator...")
    
    generator = RSAPatternGenerator(width=19, height=9)
    
    # Generate a single sample
    sample = generator.generate_sample(num_differences=5)
    
    print("\nPattern 1:")
    for line in sample['pattern1']:
        print(line)
    
    print("\nPattern 2:")
    for line in sample['pattern2']:
        print(line)
    
    print("\nDifferences:")
    for diff in sample['ground_truth']:
        print(f"  ({diff['x']}, {diff['y']}): '{diff['original']}' -> '{diff['modified']}'")
    
    print(f"\nMetadata: {sample['metadata']}")
    print("\n✓ Generation test passed!")


def test_batch_generation():
    """Test batch generation"""
    print("\nTesting batch generation...")
    
    generator = RSAPatternGenerator(width=19, height=9)
    samples = generator.generate_batch(num_samples=5, num_differences=3)
    
    print(f"Generated {len(samples)} samples")
    for idx, sample in enumerate(samples):
        print(f"  Sample {idx + 1}: {len(sample['ground_truth'])} differences")
    
    print("✓ Batch generation test passed!")


def test_save_load():
    """Test save and load functionality"""
    print("\nTesting save/load...")
    
    generator = RSAPatternGenerator(width=19, height=9)
    samples = generator.generate_batch(num_samples=3, num_differences=4)
    
    # Save
    test_path = '/tmp/test_samples.json'
    generator.save_samples(samples, test_path)
    print(f"Saved samples to {test_path}")
    
    # Load
    loaded_samples = generator.load_samples(test_path)
    print(f"Loaded {len(loaded_samples)} samples")
    
    assert len(samples) == len(loaded_samples), "Sample count mismatch"
    print("✓ Save/load test passed!")


if __name__ == '__main__':
    test_generation()
    test_batch_generation()
    test_save_load()
    print("\n✓ All tests passed!")