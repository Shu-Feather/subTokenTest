from typing import List, Dict, Any, Optional
import random
import string
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TestCaseLoader:
    """Load test cases from JSON files"""
    
    def __init__(self):
        pass
    
    def load_from_file(self, filepath: str, difficulty: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Load test cases from JSON file
        
        Args:
            filepath: Path to JSON file
            difficulty: Optional difficulty level ('easy', 'medium', 'hard', 'all')
                       If None, loads all available test cases
        
        Returns:
            Dictionary with 'task1' and 'task2' keys containing test cases
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Test case file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different file formats
        if 'task1' in data and 'task2' in data:
            # Check if data has difficulty levels
            if isinstance(data['task1'], dict) and 'easy' in data['task1']:
                # File has difficulty levels
                if difficulty is None or difficulty == 'all':
                    # Load all difficulties combined
                    if 'all' in data['task1']:
                        return {
                            'task1': data['task1']['all'],
                            'task2': data['task2']['all']
                        }
                    else:
                        # Combine all difficulties
                        task1_all = (data['task1'].get('easy', []) + 
                                    data['task1'].get('medium', []) + 
                                    data['task1'].get('hard', []))
                        task2_all = (data['task2'].get('easy', []) + 
                                    data['task2'].get('medium', []) + 
                                    data['task2'].get('hard', []))
                        return {
                            'task1': task1_all,
                            'task2': task2_all
                        }
                else:
                    # Load specific difficulty
                    if difficulty not in data['task1']:
                        raise ValueError(f"Difficulty '{difficulty}' not found in dataset")
                    return {
                        'task1': data['task1'][difficulty],
                        'task2': data['task2'][difficulty]
                    }
            else:
                # Simple format without difficulty levels
                return {
                    'task1': data['task1'],
                    'task2': data['task2']
                }
        else:
            raise ValueError(f"Invalid test case file format: {filepath}")
    
    def load_from_directory(self, directory: str, difficulty: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Load test cases from directory (looks for standard filenames)
        
        Args:
            directory: Directory containing test case files
            difficulty: Optional difficulty level
        
        Returns:
            Dictionary with test cases
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Try different file patterns
        if difficulty and difficulty != 'all':
            # Look for difficulty-specific file
            filepath = directory / f"test_cases_{difficulty}.json"
            if filepath.exists():
                return self.load_from_file(filepath, difficulty)
        
        # Look for combined file
        for filename in ['test_cases_all.json', 'test_cases_full.json', 'test_cases.json']:
            filepath = directory / filename
            if filepath.exists():
                return self.load_from_file(filepath, difficulty)
        
        raise FileNotFoundError(f"No test case files found in directory: {directory}")


class TestCaseGenerator:
    """Generate test cases on-the-fly (legacy support)"""
    
    def __init__(self):
        self.basic_words = [
            "hello", "world", "test", "cat", "dog", "python", "code", "ai", "ml", "data"
        ]
        
        self.complex_words = [
            "programming", "artificial", "intelligence", "machine", "learning",
            "benchmark", "evaluation", "typewriter", "simulation", "algorithm"
        ]
        
        self.special_cases = [
            "a", "I", "xyz", "test123", "hello_world", "API", "HTML", "JSON", "CPU"
        ]
    
    def generate_task1_cases(self, num_basic: int = 10, num_complex: int = 5, 
                           num_special: int = 5, num_random: int = 5) -> List[str]:
        """Generate test cases for Task 1 (Typewriter Effect)"""
        test_cases = []
        
        # Basic words
        test_cases.extend(random.sample(self.basic_words, min(num_basic, len(self.basic_words))))
        
        # Complex words
        test_cases.extend(random.sample(self.complex_words, min(num_complex, len(self.complex_words))))
        
        # Special cases
        test_cases.extend(random.sample(self.special_cases, min(num_special, len(self.special_cases))))
        
        # Random generated words
        for _ in range(num_random):
            length = random.randint(3, 8)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            test_cases.append(word)
        
        return test_cases
    
    def generate_task2_cases(self, num_each_type: int = 5) -> List[str]:
        """Generate test cases for Task 2 (Backspace Handling)"""
        test_cases = []
        
        # Basic backspace cases
        basic_cases = [
            "h e l l o ← ← k o",
            "a b c ← d",
            "← a b ←",
            "t e s t ← ← i n g",
            "p y t h o n ← ← ← c o d e"
        ]
        test_cases.extend(basic_cases[:num_each_type])
        
        # Multiple backspaces
        multi_backspace_cases = [
            "a b c d e ← ← ← f g",
            "h e l l o ← ← ← ← ← w o r l d",
            "← ← ← a b c",
            "t e x t ← ← ← ← n e w",
            "a ← b ← c ← d"
        ]
        test_cases.extend(multi_backspace_cases[:num_each_type])
        
        # Edge cases
        edge_cases = [
            "←",
            "← ←",
            "a ←",
            "← a",
            "a b ← ← ← c"
        ]
        test_cases.extend(edge_cases[:num_each_type])
        
        # Complex mixed cases
        complex_cases = self._generate_complex_backspace_cases(num_each_type)
        test_cases.extend(complex_cases)
        
        return test_cases
    
    def _generate_complex_backspace_cases(self, num_cases: int) -> List[str]:
        """Generate complex backspace test cases"""
        cases = []
        
        for _ in range(num_cases):
            # Random length sequence
            length = random.randint(8, 15)
            sequence = []
            
            for _ in range(length):
                if random.random() < 0.7:  # 70% chance for regular character
                    char = random.choice(string.ascii_lowercase)
                    sequence.append(char)
                else:  # 30% chance for backspace
                    sequence.append("←")
            
            cases.append(" ".join(sequence))
        
        return cases
    
    def generate_all_test_cases(self) -> Dict[str, List[str]]:
        """Generate all test cases for both tasks (legacy method)"""
        logger.warning("Using legacy test case generator. Consider using pre-generated datasets.")
        return {
            'task1': self.generate_task1_cases(),
            'task2': self.generate_task2_cases()
        }
    
    def save_test_cases(self, filename: str):
        """Save test cases to JSON file"""
        test_cases = self.generate_all_test_cases()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test cases saved to: {filename}")


# Convenience functions
def load_test_cases(filepath: str, difficulty: Optional[str] = None) -> Dict[str, List[str]]:
    """Convenience function to load test cases"""
    loader = TestCaseLoader()
    return loader.load_from_file(filepath, difficulty)


def load_test_cases_from_dir(directory: str, difficulty: Optional[str] = None) -> Dict[str, List[str]]:
    """Convenience function to load test cases from directory"""
    loader = TestCaseLoader()
    return loader.load_from_directory(directory, difficulty)