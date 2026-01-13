"""
Unit tests for cipher & decipher benchmark.
"""

import unittest
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ciphers.morse_code import MorseCode
from src.ciphers.caesar_cipher import CaesarCipher
from src.data_generation.text_generator import TextGenerator
from src.evaluation.evaluator import OutputProcessor, SimilarityCalculator, TaskType
from src.utils.prompts import PromptTemplates, PromptValidator


class TestMorseCode(unittest.TestCase):
    """Test cases for Morse code functionality."""
    
    def test_morse_encode_basic(self):
        """Test basic Morse code encoding."""
        text = "HELLO"
        expected = ".... . .-.. .-.. ---"
        result = MorseCode.encode(text)
        self.assertEqual(result, expected)
    
    def test_morse_encode_with_spaces(self):
        """Test Morse code encoding with spaces."""
        text = "HELLO WORLD"
        expected = ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."
        result = MorseCode.encode(text)
        self.assertEqual(result, expected)
    
    def test_morse_decode_basic(self):
        """Test basic Morse code decoding."""
        morse = ".... . .-.. .-.. ---"
        expected = "HELLO"
        result = MorseCode.decode(morse)
        self.assertEqual(result, expected)
    
    def test_morse_decode_with_spaces(self):
        """Test Morse code decoding with spaces."""
        morse = ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."
        expected = "HELLO WORLD"
        result = MorseCode.decode(morse)
        self.assertEqual(result, expected)
    
    def test_morse_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip."""
        original = "THE QUICK BROWN FOX"
        encoded = MorseCode.encode(original)
        decoded = MorseCode.decode(encoded)
        self.assertEqual(decoded, original)
    
    def test_morse_is_valid(self):
        """Test Morse code validation."""
        valid_morse = ".... . .-.. .-.. ---"
        invalid_morse = ".... x .-.. .-.. ---"
        
        self.assertTrue(MorseCode.is_valid_morse(valid_morse))
        self.assertFalse(MorseCode.is_valid_morse(invalid_morse))
        self.assertFalse(MorseCode.is_valid_morse(""))


class TestCaesarCipher(unittest.TestCase):
    """Test cases for Caesar cipher functionality."""
    
    def test_caesar_encode_basic(self):
        """Test basic Caesar cipher encoding."""
        text = "HELLO"
        shift = 3
        expected = "KHOOR"
        result = CaesarCipher.encode(text, shift)
        self.assertEqual(result, expected)
    
    def test_caesar_encode_wrap_around(self):
        """Test Caesar cipher encoding with wrap around."""
        text = "XYZ"
        shift = 3
        expected = "ABC"
        result = CaesarCipher.encode(text, shift)
        self.assertEqual(result, expected)
    
    def test_caesar_encode_preserve_case(self):
        """Test Caesar cipher preserves case."""
        text = "Hello World"
        shift = 1
        expected = "Ifmmp Xpsme"
        result = CaesarCipher.encode(text, shift)
        self.assertEqual(result, expected)
    
    def test_caesar_encode_non_alphabetic(self):
        """Test Caesar cipher preserves non-alphabetic characters."""
        text = "Hello, World! 123"
        shift = 5
        expected = "Mjqqt, Btwqi! 123"
        result = CaesarCipher.encode(text, shift)
        self.assertEqual(result, expected)
    
    def test_caesar_decode_basic(self):
        """Test basic Caesar cipher decoding."""
        encrypted = "KHOOR"
        shift = 3
        expected = "HELLO"
        result = CaesarCipher.decode(encrypted, shift)
        self.assertEqual(result, expected)
    
    def test_caesar_encode_decode_roundtrip(self):
        """Test Caesar encode-decode roundtrip."""
        original = "The Quick Brown Fox Jumps Over The Lazy Dog!"
        shift = 13
        encoded = CaesarCipher.encode(original, shift)
        decoded = CaesarCipher.decode(encoded, shift)
        self.assertEqual(decoded, original)
    
    def test_caesar_find_shift(self):
        """Test finding Caesar cipher shift."""
        original = "HELLO"
        encrypted = "KHOOR"
        expected_shift = 3
        result = CaesarCipher.find_shift(original, encrypted)
        self.assertEqual(result, expected_shift)
    
    def test_caesar_is_valid_shift(self):
        """Test Caesar cipher shift validation."""
        self.assertTrue(CaesarCipher.is_valid_shift(0))
        self.assertTrue(CaesarCipher.is_valid_shift(25))
        self.assertFalse(CaesarCipher.is_valid_shift(-1))
        self.assertFalse(CaesarCipher.is_valid_shift(26))


class TestTextGenerator(unittest.TestCase):
    """Test cases for text generation functionality."""
    
    def setUp(self):
        self.generator = TextGenerator()
    
    def test_generate_random_text_length(self):
        """Test random text generation respects length constraints."""
        min_len, max_len = 20, 50
        text = self.generator.generate_random_text(min_len, max_len)
        
        self.assertGreaterEqual(len(text), min_len)
        self.assertLessEqual(len(text), max_len)
    
    def test_generate_samples_count(self):
        """Test generating correct number of samples."""
        num_samples = 5
        samples = self.generator.generate_samples(num_samples)
        self.assertEqual(len(samples), num_samples)
    
    def test_generate_alphanumeric_text(self):
        """Test alphanumeric text generation."""
        length = 20
        text = self.generator.generate_alphanumeric_text(length)
        self.assertEqual(len(text), length)
        self.assertTrue(all(c.isalnum() or c.isspace() for c in text))
    
    def test_generate_punctuation_text(self):
        """Test text generation with punctuation."""
        base_text = "Hello world this is a test"
        result = self.generator.generate_punctuation_text(base_text)
        self.assertIn("Hello world this is a test", result)


class TestOutputProcessor(unittest.TestCase):
    """Test cases for output processing functionality."""
    
    def setUp(self):
        self.processor = OutputProcessor()
    
    def test_extract_answer_structured_format(self):
        """Test extracting answer from structured format."""
        output = """<think>: I need to encode this text to Morse code. </think>
<answer>: .... . .-.. .-.. --- </answer>"""
        expected = ".... . .-.. .-.. ---"
        result = self.processor.extract_answer(output, TaskType.MORSE_ENCODE)
        self.assertEqual(result, expected)
    
    def test_extract_answer_fallback(self):
        """Test fallback extraction when structured format is missing."""
        output = "The answer is: .... . .-.. .-.. ---"
        expected = ".... . .-.. .-.. ---"
        result = self.processor.extract_answer(output, TaskType.MORSE_ENCODE)
        self.assertEqual(result, expected)
    
    def test_normalize_morse_code(self):
        """Test Morse code normalization."""
        messy_morse = ".... . .-.. .-..    ---  /  .-- --- .-. .-.. -.."
        expected = ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."
        result = self.processor.normalize_morse_code(messy_morse)
        self.assertEqual(result, expected)
    
    def test_normalize_text_case_insensitive(self):
        """Test text normalization ignoring case."""
        text = "Hello World!"
        expected = "hello world!"
        result = self.processor.normalize_text(text, ignore_case=True, ignore_punctuation=False)
        self.assertEqual(result, expected)
    
    def test_normalize_text_ignore_punctuation(self):
        """Test text normalization ignoring punctuation."""
        text = "Hello, World!"
        expected = "hello world"
        result = self.processor.normalize_text(text, ignore_case=True, ignore_punctuation=True)
        self.assertEqual(result, expected)


class TestSimilarityCalculator(unittest.TestCase):
    """Test cases for similarity calculation."""
    
    def setUp(self):
        self.calc = SimilarityCalculator()
    
    def test_levenshtein_distance_identical(self):
        """Test Levenshtein distance for identical strings."""
        s1, s2 = "hello", "hello"
        result = self.calc.levenshtein_distance(s1, s2)
        self.assertEqual(result, 0)
    
    def test_levenshtein_distance_different(self):
        """Test Levenshtein distance for different strings."""
        s1, s2 = "hello", "world"
        result = self.calc.levenshtein_distance(s1, s2)
        self.assertEqual(result, 4)  # h->w, e->o, l->r, l->l, o->d
    
    def test_similarity_score_identical(self):
        """Test similarity score for identical strings."""
        s1, s2 = "hello", "hello"
        result = self.calc.similarity_score(s1, s2)
        self.assertEqual(result, 1.0)
    
    def test_similarity_score_completely_different(self):
        """Test similarity score for completely different strings."""
        s1, s2 = "abc", "xyz"
        result = self.calc.similarity_score(s1, s2)
        self.assertEqual(result, 0.0)
    
    def test_similarity_score_empty_strings(self):
        """Test similarity score for empty strings."""
        result = self.calc.similarity_score("", "")
        self.assertEqual(result, 1.0)


class TestPromptTemplates(unittest.TestCase):
    """Test cases for prompt templates."""
    
    def test_get_morse_encode_prompt(self):
        """Test Morse encode prompt generation."""
        text = "Hello World"
        prompt = PromptTemplates.get_morse_encode_prompt(text)
        
        self.assertIn(text, prompt)
        self.assertIn("<think>", prompt)
        self.assertIn("<answer>", prompt)
    
    def test_get_morse_decode_prompt(self):
        """Test Morse decode prompt generation."""
        morse = ".... . .-.. .-.. ---"
        prompt = PromptTemplates.get_morse_decode_prompt(morse)
        
        self.assertIn(morse, prompt)
        self.assertIn("<think>", prompt)
        self.assertIn("<answer>", prompt)
    
    def test_get_caesar_encode_prompt(self):
        """Test Caesar encode prompt generation."""
        text = "Hello World"
        shift = 3
        prompt = PromptTemplates.get_caesar_encode_prompt(text, shift)
        
        self.assertIn(text, prompt)
        self.assertIn(str(shift), prompt)
        self.assertIn("<think>", prompt)
        self.assertIn("<answer>", prompt)
    
    def test_get_caesar_decode_prompt(self):
        """Test Caesar decode prompt generation."""
        encrypted = "Khoor Zruog"
        shift = 3
        prompt = PromptTemplates.get_caesar_decode_prompt(encrypted, shift)
        
        self.assertIn(encrypted, prompt)
        self.assertIn(str(shift), prompt)
        self.assertIn("<think>", prompt)
        self.assertIn("<answer>", prompt)
    
    def test_get_available_styles(self):
        """Test getting available prompt styles."""
        styles = PromptTemplates.get_available_styles()
        
        self.assertIn('morse_encode', styles)
        self.assertIn('morse_decode', styles)
        self.assertIn('caesar_encode', styles)
        self.assertIn('caesar_decode', styles)
        
        # Check that each task has the expected styles
        expected_styles = ['basic', 'detailed', 'step_by_step']
        for task_styles in styles.values():
            for style in expected_styles:
                self.assertIn(style, task_styles)


class TestPromptValidator(unittest.TestCase):
    """Test cases for prompt validation."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        messy_text = '  Hello   "World"  \n\r  '
        expected = "Hello 'World'"
        result = PromptValidator.clean_text(messy_text)
        self.assertEqual(result, expected)
    
    def test_validate_shift(self):
        """Test shift validation."""
        self.assertEqual(PromptValidator.validate_shift(3), 3)
        self.assertEqual(PromptValidator.validate_shift(26), 0)  # Wraps around
        self.assertEqual(PromptValidator.validate_shift(-1), 25)  # Wraps around
        
        with self.assertRaises(ValueError):
            PromptValidator.validate_shift("not_an_int")
    
    def test_format_morse_code(self):
        """Test Morse code formatting."""
        messy_morse = ".... . .-..   .-.. ---   /   .-- --- .-. .-.. -.."
        expected = ".... . .-.. .-.. --- / .-- --- .-. .-.. -.."
        result = PromptValidator.format_morse_code(messy_morse)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMorseCode,
        TestCaesarCipher,
        TestTextGenerator,
        TestOutputProcessor,
        TestSimilarityCalculator,
        TestPromptTemplates,
        TestPromptValidator
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)