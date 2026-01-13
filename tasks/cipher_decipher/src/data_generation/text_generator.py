"""
Text generation utilities for benchmark data.
"""

import random
import string
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TextGenerator:
    """Generates text samples for cipher/decipher benchmarking."""
    
    # Sample text templates for generation
    SAMPLE_TEMPLATES = [
        "The {adjective} {noun} {verb} over the {adjective2} {noun2}",
        "In the {adjective} {noun}, we found {adjective2} {noun2}",
        "Yesterday I {verb} to the {noun} and saw a {adjective} {noun2}",
        "The {adjective} {noun} was {verb}ing near the {noun2}",
        "Every {noun} has its own {adjective} {noun2}",
        "When the {noun} {verb}, the {noun2} becomes {adjective}",
        "A {adjective} {noun} is better than a {adjective2} {noun2}",
        "The {noun} and {noun2} were {verb}ing together",
    ]
    
    # Word lists for templates
    ADJECTIVES = [
        "quick", "slow", "bright", "dark", "large", "small", "happy", "sad",
        "strong", "weak", "beautiful", "ugly", "clean", "dirty", "hot", "cold",
        "old", "new", "good", "bad", "easy", "hard", "high", "low", "long", "short"
    ]
    
    NOUNS = [
        "cat", "dog", "bird", "fish", "tree", "flower", "house", "car", "book", "table",
        "chair", "computer", "phone", "window", "door", "mountain", "river", "ocean",
        "sun", "moon", "star", "cloud", "rain", "snow", "fire", "water", "earth", "air"
    ]
    
    VERBS = [
        "run", "walk", "jump", "fly", "swim", "dance", "sing", "read", "write", "think",
        "speak", "listen", "watch", "sleep", "eat", "drink", "play", "work", "study", "learn"
    ]
    
    # Common English phrases and sentences
    COMMON_PHRASES = [
        "Hello world",
        "Good morning everyone",
        "How are you today",
        "Thank you very much",
        "Please help me",
        "I love programming",
        "The weather is nice",
        "Have a great day",
        "See you later",
        "Take care of yourself",
        "Time flies when you're having fun",
        "Practice makes perfect",
        "Actions speak louder than words",
        "Better late than never",
        "Knowledge is power",
        "Rome wasn't built in a day",
        "The early bird catches the worm",
        "All roads lead to Rome",
        "When in Rome do as the Romans do",
        "A picture is worth a thousand words"
    ]
    
    def __init__(self, llm_model=None):
        """
        Initialize TextGenerator.
        
        Args:
            llm_model: Optional LLM model for generating diverse text
        """
        self.llm_model = llm_model
    
    def generate_random_text(self, min_length: int = 10, max_length: int = 100) -> str:
        """
        Generate random text using templates and word lists.
        
        Args:
            min_length (int): Minimum text length
            max_length (int): Maximum text length
            
        Returns:
            str: Generated text
        """
        text = ""
        attempts = 0
        max_attempts = 50
        
        while len(text) < min_length and attempts < max_attempts:
            attempts += 1
            
            # Choose generation method randomly
            method = random.choice(['template', 'phrase', 'combination'])
            
            if method == 'template':
                new_text = self._generate_from_template()
            elif method == 'phrase':
                new_text = random.choice(self.COMMON_PHRASES)
            else:  # combination
                new_text = self._generate_combination()
            
            # Add to text with proper spacing
            if text and not text.endswith(' '):
                text += " "
            text += new_text
            
            # Break if we exceed max length
            if len(text) >= max_length:
                text = text[:max_length].strip()
                break
        
        # Ensure minimum length with fallback
        if len(text) < min_length:
            text = random.choice(self.COMMON_PHRASES)
        
        return text.strip()
    
    def _generate_from_template(self) -> str:
        """Generate text from a random template."""
        template = random.choice(self.SAMPLE_TEMPLATES)
        
        return template.format(
            adjective=random.choice(self.ADJECTIVES),
            adjective2=random.choice(self.ADJECTIVES),
            noun=random.choice(self.NOUNS),
            noun2=random.choice(self.NOUNS),
            verb=random.choice(self.VERBS)
        )
    
    def _generate_combination(self) -> str:
        """Generate text by combining multiple elements."""
        elements = []
        
        # Add random elements
        num_elements = random.randint(2, 4)
        for _ in range(num_elements):
            element_type = random.choice(['adjective_noun', 'phrase', 'simple_sentence'])
            
            if element_type == 'adjective_noun':
                elements.append(f"{random.choice(self.ADJECTIVES)} {random.choice(self.NOUNS)}")
            elif element_type == 'phrase':
                elements.append(random.choice(self.COMMON_PHRASES))
            else:  # simple_sentence
                elements.append(self._generate_from_template())
        
        # Join with various connectors
        connectors = [" and ", " with ", " near ", " for "]
        connector = random.choice(connectors)
        
        return connector.join(elements[:2]) if len(elements) >= 2 else elements[0]
    
    def generate_samples(self, num_samples: int, min_length: int = 10, 
                        max_length: int = 100, use_llm: bool = False) -> List[str]:
        """
        Generate multiple text samples.
        
        Args:
            num_samples (int): Number of samples to generate
            min_length (int): Minimum text length
            max_length (int): Maximum text length
            use_llm (bool): Whether to use LLM for generation
            
        Returns:
            List[str]: Generated text samples
        """
        samples = []
        
        for i in range(num_samples):
            try:
                if use_llm and self.llm_model:
                    # Try LLM generation first
                    sample = self.generate_random_text(min_length, max_length)  # Placeholder
                else:
                    sample = self.generate_random_text(min_length, max_length)
                
                # Ensure sample meets length requirements
                if len(sample) < min_length:
                    sample = random.choice(self.COMMON_PHRASES)
                
                samples.append(sample)
                
            except Exception as e:
                logger.error(f"Error generating sample {i}: {e}")
                samples.append(random.choice(self.COMMON_PHRASES))
        
        return samples
    
    def generate_alphanumeric_text(self, length: int = 20) -> str:
        """
        Generate random alphanumeric text for testing edge cases.
        
        Args:
            length (int): Length of text to generate
            
        Returns:
            str: Random alphanumeric text
        """
        chars = string.ascii_letters + string.digits + ' '
        return ''.join(random.choices(chars, k=length))
    
    def generate_punctuation_text(self, base_text: str = None) -> str:
        """
        Generate text with various punctuation marks for comprehensive testing.
        
        Args:
            base_text (str): Base text to add punctuation to
            
        Returns:
            str: Text with punctuation
        """
        if not base_text:
            base_text = self.generate_random_text()
        
        # Add random punctuation
        punctuation = ".,!?;:\"'()-"
        result = ""
        
        for i, char in enumerate(base_text):
            result += char
            if char == ' ' and random.random() < 0.1:  # 10% chance to add punctuation
                result += random.choice(punctuation)
        
        return result