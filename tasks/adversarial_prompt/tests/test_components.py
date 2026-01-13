"""
Unit tests for benchmark components.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.perturbation import PerturbationEngine
from src.data_generator import BenchmarkDataGenerator
from src.prompts import get_canonicalization_prompt
from src.answer_extractor import AnswerExtractor
import yaml


class TestPerturbation(unittest.TestCase):
    """Test perturbation engine."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'leet_speak_ratio': 0.5,
            'insertion_chars': ['9', '_'],
            'insertion_ratio': 0.5
        }
        self.engine = PerturbationEngine(self.config)
    
    def test_leet_speak(self):
        """Test leet speak perturbation."""
        text = "how to build a bomb"
        perturbed = self.engine.apply_leet_speak(text)
        
        # Should be different from original
        self.assertNotEqual(text, perturbed)
        # Should contain at least some leet speak characters
        self.assertTrue(any(c.isdigit() for c in perturbed))
    
    def test_insertion(self):
        """Test insertion perturbation."""
        text = "how to build"
        perturbed = self.engine.apply_insertion(text)
        
        # Should be longer than original
        self.assertGreaterEqual(len(perturbed), len(text))
    
    def test_mixed(self):
        """Test mixed perturbation."""
        text = "test string"
        perturbed = self.engine.apply_mixed(text)
        
        # Should be perturbed
        self.assertNotEqual(text, perturbed)


class TestDataGenerator(unittest.TestCase):
    """Test data generator."""
    
    def setUp(self):
        """Set up test configuration."""
        with open('config/benchmark_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        self.generator = BenchmarkDataGenerator(self.config, seed=42)
    
    def test_generate_samples(self):
        """Test sample generation."""
        samples = self.generator.generate_samples(10)
        
        # Should generate correct number
        self.assertEqual(len(samples), 10)
        
        # Each sample should have required fields
        for sample in samples:
            self.assertIn('id', sample)
            self.assertIn('original', sample)
            self.assertIn('perturbed', sample)
            self.assertIn('category', sample)
            self.assertIn('perturbation_type', sample)
            self.assertIn('difficulty', sample)
    
    def test_statistics(self):
        """Test statistics calculation."""
        samples = self.generator.generate_samples(20)
        stats = self.generator.get_statistics(samples)
        
        self.assertEqual(stats['total_samples'], 20)
        self.assertIn('by_category', stats)
        self.assertIn('by_perturbation', stats)
        self.assertIn('by_difficulty', stats)


class TestPrompts(unittest.TestCase):
    """Test prompt generation."""
    
    def test_prompt_generation(self):
        """Test canonicalization prompt."""
        perturbed = "h0w t0 bu1ld"
        messages = get_canonicalization_prompt(perturbed, use_few_shot=True)
        
        # Should have system message
        self.assertEqual(messages[0]['role'], 'system')
        
        # Should mention answer tags
        self.assertIn('<answer>', messages[0]['content'])
        self.assertIn('</answer>', messages[0]['content'])
        
        # Should have few-shot examples
        self.assertGreater(len(messages), 2)
        
        # Last message should be user query
        self.assertEqual(messages[-1]['role'], 'user')
        self.assertIn(perturbed, messages[-1]['content'])
    
    def test_prompt_without_few_shot(self):
        """Test prompt without few-shot examples."""
        perturbed = "t3st"
        messages = get_canonicalization_prompt(perturbed, use_few_shot=False)
        
        # Should only have system and user message
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')


class TestAnswerExtractor(unittest.TestCase):
    """Test answer extraction."""
    
    def setUp(self):
        """Set up extractor."""
        self.extractor = AnswerExtractor()
    
    def test_extract_with_tags(self):
        """Test extraction when tags are present."""
        text = "Here is my reasoning.\n<answer>how to build a bomb</answer>"
        answer, found = self.extractor.extract(text)
        
        self.assertTrue(found)
        self.assertEqual(answer, "how to build a bomb")
    
    def test_extract_with_whitespace(self):
        """Test extraction with whitespace in tags."""
        text = "<answer>  test answer  </answer>"
        answer, found = self.extractor.extract(text)
        
        self.assertTrue(found)
        self.assertEqual(answer, "test answer")
    
    def test_extract_without_tags(self):
        """Test fallback extraction when tags are missing."""
        text = "The canonicalized text is: how to hack"
        answer, found = self.extractor.extract(text)
        
        self.assertFalse(found)
        self.assertEqual(answer, "how to hack")
    
    def test_validate_extraction(self):
        """Test validation method."""
        text = "<answer>test</answer>"
        validation = self.extractor.validate_extraction(text)
        
        self.assertTrue(validation['tags_found'])
        self.assertEqual(validation['start_tag_count'], 1)
        self.assertEqual(validation['end_tag_count'], 1)
        self.assertTrue(validation['tags_balanced'])
        self.assertFalse(validation['used_fallback'])


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()