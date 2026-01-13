"""
Data generator for RSA-Difference Benchmark
Path: src/data_generator.py
"""

import random
import json
from typing import List, Tuple, Dict, Set
from copy import deepcopy


class RSAPatternGenerator:
    """Generator for RSA fingerprint patterns and their differences"""
    
    def __init__(
        self,
        width: int = 19,
        height: int = 9,
        available_elements: List[str] = None
    ):
        """
        Initialize the RSA pattern generator
        
        Args:
            width: Width of the RSA pattern (excluding borders)
            height: Height of the RSA pattern (excluding borders)
            available_elements: List of available elements for the pattern
        """
        self.width = width
        self.height = height
        
        if available_elements is None:
            self.available_elements = ['o', '+', '=', '.', 'E', 'S', 'B', '*', '-', ' ']
        else:
            self.available_elements = available_elements
    
    def generate_base_pattern(self, key_size: int = 2048) -> List[str]:
        """
        Generate a base RSA pattern with centered header
        
        Args:
            key_size: RSA key size to display in header
            
        Returns:
            List of strings representing each line of the pattern
        """
        pattern = []
        
        # Top border with centered RSA key size
        header_text = f" RSA {key_size} "
        header_brackets = f"[{header_text}]"
        
        # Calculate padding for centering
        # The total width for dashes is self.width (content width between the + symbols)
        header_len = len(header_brackets)
        
        # Calculate left and right dashes
        total_dashes = self.width - header_len
        left_dashes = total_dashes // 2
        right_dashes = total_dashes - left_dashes
        
        # Construct the top border
        header = "+" + "-" * left_dashes + header_brackets + "-" * right_dashes + "+"
        pattern.append(header)
        
        # Interior lines
        for _ in range(self.height):
            line = "|"
            for _ in range(self.width):
                line += random.choice(self.available_elements)
            line += "|"
            pattern.append(line)
        
        # Bottom border
        bottom = "+" + "-" * self.width + "+"
        pattern.append(bottom)
        
        return pattern
    
    def create_differences(
        self,
        pattern: List[str],
        num_differences: int
    ) -> Tuple[List[str], List[Dict]]:
        """
        Create a modified pattern with specified number of differences
        
        Args:
            pattern: Original pattern
            num_differences: Number of differences to create
            
        Returns:
            Tuple of (modified_pattern, list of difference records)
        """
        modified_pattern = [list(line) for line in pattern]
        differences = []
        used_positions = set()
        
        attempts = 0
        max_attempts = num_differences * 100
        
        while len(differences) < num_differences and attempts < max_attempts:
            attempts += 1
            
            # Random position (only interior, not borders)
            y = random.randint(1, self.height)
            x = random.randint(1, self.width)
            
            if (x, y) in used_positions:
                continue
            
            # Get original element
            original_element = pattern[y][x]
            
            # Choose a different element
            available_for_replacement = [e for e in self.available_elements if e != original_element]
            if not available_for_replacement:
                continue
            
            new_element = random.choice(available_for_replacement)
            
            # Apply modification
            modified_pattern[y][x] = new_element
            
            # Record difference
            differences.append({
                'x': x,
                'y': y,
                'original': original_element,
                'modified': new_element
            })
            
            used_positions.add((x, y))
        
        # Convert back to strings
        modified_pattern_str = [''.join(line) for line in modified_pattern]
        
        return modified_pattern_str, differences
    
    def generate_sample(
        self,
        num_differences: int = 5,
        key_size: int = 2048
    ) -> Dict:
        """
        Generate a complete sample with two patterns and their differences
        
        Args:
            num_differences: Number of differences between patterns
            key_size: RSA key size
            
        Returns:
            Dictionary containing pattern1, pattern2, and ground_truth differences
        """
        # Generate base pattern
        pattern1 = self.generate_base_pattern(key_size)
        
        # Create modified pattern
        pattern2, differences = self.create_differences(pattern1, num_differences)
        
        return {
            'pattern1': pattern1,
            'pattern2': pattern2,
            'ground_truth': differences,
            'metadata': {
                'width': self.width,
                'height': self.height,
                'key_size': key_size,
                'num_differences': len(differences)
            }
        }
    
    def generate_batch(
        self,
        num_samples: int,
        num_differences: int = 5,
        key_size: int = 2048
    ) -> List[Dict]:
        """
        Generate multiple samples
        
        Args:
            num_samples: Number of samples to generate
            num_differences: Number of differences per sample
            key_size: RSA key size
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        for _ in range(num_samples):
            sample = self.generate_sample(num_differences, key_size)
            samples.append(sample)
        
        return samples
    
    def save_samples(self, samples: List[Dict], output_path: str):
        """
        Save generated samples to a JSON file
        
        Args:
            samples: List of sample dictionaries
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
    
    def load_samples(self, input_path: str) -> List[Dict]:
        """
        Load samples from a JSON file
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            List of sample dictionaries
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        return samples