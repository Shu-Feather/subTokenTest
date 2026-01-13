import unittest
import json
from utils.parser import (
    parse_answer, 
    normalize_answer, 
    compare_answers,
    parse_coordinate,
    parse_json_answer
)


class TestParser(unittest.TestCase):
    """Test cases for answer parser utilities."""
    
    def test_parse_answer_basic(self):
        """Test basic answer parsing."""
        # Simple case
        response = "<answer>X</answer>"
        self.assertEqual(parse_answer(response), "X")
        
        # With surrounding text
        response = "Let me analyze this. <answer>O</answer> That's the answer."
        self.assertEqual(parse_answer(response), "O")
        
        # Multi-line
        response = """
        Based on the map:
        <answer>
        (3, 4)
        </answer>
        """
        self.assertEqual(parse_answer(response).strip(), "(3, 4)")
    
    def test_parse_answer_case_insensitive(self):
        """Test case-insensitive tag parsing."""
        # Uppercase tags
        response = "<ANSWER>X</ANSWER>"
        self.assertEqual(parse_answer(response), "X")
        
        # Mixed case
        response = "<Answer>test</Answer>"
        self.assertEqual(parse_answer(response), "test")
    
    def test_parse_answer_no_tags(self):
        """Test when no tags are found."""
        response = "This is just text without tags"
        self.assertIsNone(parse_answer(response))
    
    def test_parse_answer_multiple_tags(self):
        """Test with multiple answer tags (should get first)."""
        response = "<answer>first</answer> some text <answer>second</answer>"
        self.assertEqual(parse_answer(response), "first")
    
    def test_normalize_answer(self):
        """Test answer normalization."""
        # Whitespace removal
        self.assertEqual(normalize_answer("  X  "), "x")
        
        # Multiple spaces
        self.assertEqual(normalize_answer("a  b   c"), "a b c")
        
        # Lowercase conversion
        self.assertEqual(normalize_answer("ABC"), "abc")
        
        # None handling
        self.assertEqual(normalize_answer(None), "")
    
    def test_compare_answers(self):
        """Test answer comparison."""
        # Exact match (normalized)
        self.assertTrue(compare_answers("X", "x"))
        self.assertTrue(compare_answers("  X  ", "x"))
        self.assertTrue(compare_answers("ABC", "abc"))
        
        # Different answers
        self.assertFalse(compare_answers("X", "O"))
        self.assertFalse(compare_answers("123", "456"))
    
    def test_parse_coordinate(self):
        """Test coordinate parsing."""
        # Standard format
        self.assertEqual(parse_coordinate("(3, 4)"), (3, 4))
        
        # No spaces
        self.assertEqual(parse_coordinate("(3,4)"), (3, 4))
        
        # Extra spaces
        self.assertEqual(parse_coordinate("( 3 , 4 )"), (3, 4))
        
        # Without parentheses
        self.assertEqual(parse_coordinate("3, 4"), (3, 4))
        
        # Embedded in text
        self.assertEqual(parse_coordinate("The answer is (5, 2)"), (5, 2))
        
        # Invalid format
        self.assertIsNone(parse_coordinate("invalid"))
        self.assertIsNone(parse_coordinate("(a, b)"))
    
    def test_parse_coordinate_negative(self):
        """Test parsing negative coordinates."""
        self.assertEqual(parse_coordinate("(-1, -2)"), (-1, -2))
        self.assertEqual(parse_coordinate("(3, -4)"), (3, -4))
    
    def test_parse_json_answer(self):
        """Test JSON parsing from answer."""
        # Valid JSON
        json_str = '{"up": "X", "down": "_"}'
        result = parse_json_answer(json_str)
        self.assertEqual(result, {"up": "X", "down": "_"})
        
        # JSON embedded in text
        text = 'The surrounding elements are {"left": "#", "right": "O"}'
        result = parse_json_answer(text)
        self.assertEqual(result, {"left": "#", "right": "O"})
        
        # Invalid JSON
        self.assertIsNone(parse_json_answer("not json"))
        self.assertIsNone(parse_json_answer("{invalid}"))
    
    def test_parse_json_answer_complex(self):
        """Test parsing complex JSON."""
        json_str = '''
        {
            "up": "X",
            "down": "_",
            "left": "#",
            "right": "O",
            "up-left": "#",
            "up-right": "_",
            "down-left": "_",
            "down-right": "#"
        }
        '''
        result = parse_json_answer(json_str)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 8)
        self.assertEqual(result["up"], "X")


if __name__ == '__main__':
    unittest.main()