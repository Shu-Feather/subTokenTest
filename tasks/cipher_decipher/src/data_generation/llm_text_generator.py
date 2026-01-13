"""
LLM-based text generator for creating diverse benchmark contexts.
Uses GPT to generate original contexts of varying difficulty levels.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import os

import openai

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Difficulty levels for benchmark contexts."""
    EASY = "easy"       # A few sentences (20-50 words)
    MEDIUM = "medium"   # Short paragraph (50-150 words)
    HARD = "hard"       # Long paragraph (150-300 words)


class LLMTextGenerator:
    """Generate diverse text contexts using GPT for cipher benchmarking."""
    
    # Word count ranges for each difficulty level
    DIFFICULTY_RANGES = {
        DifficultyLevel.EASY: (20, 50),
        DifficultyLevel.MEDIUM: (50, 150),
        DifficultyLevel.HARD: (150, 300)
    }
    
    # Content categories for diverse text generation
    CONTENT_CATEGORIES = [
        "technology and science",
        "history and culture",
        "nature and environment",
        "daily life and activities",
        "education and learning",
        "sports and entertainment",
        "food and cooking",
        "travel and geography",
        "literature and arts",
        "philosophy and ideas"
    ]
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize LLM text generator.
        
        Args:
            api_key (str): OpenAI API key
            model_name (str): Model name to use for generation
        """

        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required for LLM text generation")
        
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model_name = model_name
        
        logger.info(f"Initialized LLM text generator with model: {model_name}")
    
    def _create_generation_prompt(self, difficulty: DifficultyLevel, 
                                  category: str, count: int = 1) -> str:
        """
        Create a prompt for generating text contexts.
        
        Args:
            difficulty (DifficultyLevel): Difficulty level
            category (str): Content category
            count (int): Number of texts to generate
            
        Returns:
            str: Generation prompt
        """
        min_words, max_words = self.DIFFICULTY_RANGES[difficulty]
        
        prompt = f"""Generate {count} diverse English text sample(s) about {category}.

**Requirements**:
1. Each text should be between {min_words} and {max_words} words
2. Use clear, grammatically correct English
3. Include a variety of sentence structures
4. Use common words and phrases suitable for cipher encoding
5. Avoid special characters that cannot be encoded in Morse code or Caesar cipher
6. Make the content informative and interesting
7. Each text should be self-contained and coherent

Difficulty level: {difficulty.value}
- Easy: A few simple sentences (like a short fact or observation)
- Medium: A short paragraph (like a brief explanation or story)
- Hard: A longer paragraph (like a detailed description or narrative)

Please provide the text(s) in the following JSON format:
{{
    "texts": [
        "First text here...",
        "Second text here...",
        ...
    ]
}}

Generate exactly {count} text(s) now, and each text should have words between {min_words} and {max_words}."""
        
        return prompt
    
    async def generate_single_batch(self, difficulty: DifficultyLevel, 
                                    category: str, count: int = 5) -> List[str]:
        """
        Generate a batch of texts for a specific difficulty and category.
        
        Args:
            difficulty (DifficultyLevel): Difficulty level
            category (str): Content category
            count (int): Number of texts to generate
            
        Returns:
            List[str]: Generated texts
        """
        try:
            prompt = self._create_generation_prompt(difficulty, category, count)
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates diverse, high-quality English text samples for educational purposes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,  # Higher temperature for diversity
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                # Look for JSON in the response
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                else:
                    json_str = content
                
                parsed = json.loads(json_str)
                texts = parsed.get('texts', [])
                
                if not texts:
                    logger.warning(f"No texts found in response for {difficulty.value}/{category}")
                    return []
                
                # Validate and clean texts
                validated_texts = []
                for text in texts:
                    text = text.strip()
                    word_count = len(text.split())
                    min_words, max_words = self.DIFFICULTY_RANGES[difficulty]
                    
                    # Allow some tolerance (±20%)
                    if min_words * 0.8 <= word_count <= max_words * 1.2:
                        validated_texts.append(text)
                    else:
                        logger.debug(f"Text rejected: {word_count} words (expected {min_words}-{max_words})")
                
                logger.info(f"Generated {len(validated_texts)}/{count} texts for {difficulty.value}/{category}")
                return validated_texts
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Fallback: try to extract texts from response
                return self._fallback_extract_texts(content, count)
                
        except Exception as e:
            logger.error(f"Error generating texts: {e}")
            return []
    
    def _fallback_extract_texts(self, content: str, count: int) -> List[str]:
        """
        Fallback method to extract texts when JSON parsing fails.
        
        Args:
            content (str): Raw response content
            count (int): Expected number of texts
            
        Returns:
            List[str]: Extracted texts
        """
        # Try to split by common separators
        texts = []
        
        # Try splitting by numbers (1., 2., etc.)
        import re
        pattern = r'\d+\.\s+'
        parts = re.split(pattern, content)
        
        for part in parts:
            part = part.strip()
            if len(part) > 20:  # Minimum length check
                texts.append(part)
                if len(texts) >= count:
                    break
        
        if not texts:
            # Last resort: take the whole content if it's reasonable
            if 20 <= len(content.split()) <= 500:
                texts = [content]
        
        return texts
    
    async def generate_dataset(self, 
                          samples_per_difficulty: int = 100,
                          batch_size: int = 5,
                          difficulties: Optional[List[str]] = None,
                          save_to_file: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Generate a complete dataset with multiple difficulty levels.
        
        Args:
            samples_per_difficulty (int): Number of samples per difficulty level
            batch_size (int): Number of texts to generate per API call
            difficulties (List[str]): List of difficulty levels to generate (e.g., ['easy', 'medium'])
                                    If None, generates all difficulties
            save_to_file (str): Optional file path to save the dataset
            
        Returns:
            Dict[str, List[str]]: Dataset organized by difficulty level
        """
        # Determine which difficulties to generate
        if difficulties is None:
            difficulties_to_generate = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        else:
            # Convert string list to DifficultyLevel enum
            difficulty_map = {
                'easy': DifficultyLevel.EASY,
                'medium': DifficultyLevel.MEDIUM,
                'hard': DifficultyLevel.HARD
            }
            difficulties_to_generate = []
            for diff in difficulties:
                diff_lower = diff.lower()
                if diff_lower in difficulty_map:
                    difficulties_to_generate.append(difficulty_map[diff_lower])
                else:
                    logger.warning(f"Unknown difficulty level '{diff}', skipping")
            
            if not difficulties_to_generate:
                logger.error("No valid difficulty levels specified")
                return {'easy': [], 'medium': [], 'hard': []}
        
        logger.info(f"Starting dataset generation: {samples_per_difficulty} samples per difficulty")
        logger.info(f"Generating difficulties: {[d.value for d in difficulties_to_generate]}")
        
        dataset = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        for difficulty in difficulties_to_generate:
            logger.info(f"Generating {samples_per_difficulty} samples for {difficulty.value} difficulty")
            
            samples_generated = 0
            attempts = 0
            max_attempts = samples_per_difficulty * 3  # Prevent infinite loops
            
            while samples_generated < samples_per_difficulty and attempts < max_attempts:
                attempts += 1
                
                # Rotate through categories for diversity
                category = self.CONTENT_CATEGORIES[attempts % len(self.CONTENT_CATEGORIES)]
                
                # Calculate how many more samples needed
                remaining = samples_per_difficulty - samples_generated
                current_batch_size = min(batch_size, remaining)
                
                # Generate batch
                texts = await self.generate_single_batch(difficulty, category, current_batch_size)
                
                # Add to dataset
                for text in texts:
                    if samples_generated < samples_per_difficulty:
                        dataset[difficulty.value].append(text)
                        samples_generated += 1
                
                logger.info(f"Progress: {samples_generated}/{samples_per_difficulty} for {difficulty.value}")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(1)
            
            logger.info(f"Completed {difficulty.value}: {len(dataset[difficulty.value])} samples")
        
        # Save to file if specified
        if save_to_file:
            self._save_dataset(dataset, save_to_file)
        
        return dataset
    
    def _save_dataset(self, dataset: Dict[str, List[str]], filepath: str):
        """
        Save generated dataset to file.
        
        Args:
            dataset (Dict[str, List[str]]): Dataset to save
            filepath (str): Output file path
        """
        try:
            from pathlib import Path
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset saved to: {filepath}")
            
            # Also save statistics
            stats_file = filepath.replace('.json', '_stats.txt')
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write("Dataset Statistics\n")
                f.write("=" * 50 + "\n\n")
                
                total_samples = 0
                for difficulty, texts in dataset.items():
                    f.write(f"{difficulty.upper()}:\n")
                    f.write(f"  Total samples: {len(texts)}\n")
                    
                    if texts:
                        word_counts = [len(text.split()) for text in texts]
                        f.write(f"  Word count range: {min(word_counts)}-{max(word_counts)}\n")
                        f.write(f"  Average word count: {sum(word_counts)/len(word_counts):.1f}\n")
                    
                    f.write("\n")
                    total_samples += len(texts)
                
                f.write(f"TOTAL SAMPLES: {total_samples}\n")
            
            logger.info(f"Statistics saved to: {stats_file}")
            
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
    
    @staticmethod
    def load_dataset(filepath: str) -> Dict[str, List[str]]:
        """
        Load a previously generated dataset.
        
        Args:
            filepath (str): Path to dataset file
            
        Returns:
            Dict[str, List[str]]: Loaded dataset
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            logger.info(f"Loaded dataset from: {filepath}")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return {'easy': [], 'medium': [], 'hard': []}
    
    async def generate_sample_texts(self, difficulty: DifficultyLevel, count: int = 10) -> List[str]:
        """
        Generate a small sample of texts for testing.
        
        Args:
            difficulty (DifficultyLevel): Difficulty level
            count (int): Number of texts to generate
            
        Returns:
            List[str]: Generated texts
        """
        texts = []
        
        for i in range(0, count, 5):
            batch_size = min(5, count - i)
            category = self.CONTENT_CATEGORIES[i % len(self.CONTENT_CATEGORIES)]
            
            batch = await self.generate_single_batch(difficulty, category, batch_size)
            texts.extend(batch)
            
            if len(texts) >= count:
                break
            
            await asyncio.sleep(0.5)
        
        return texts[:count]


async def main():
    """Example usage of LLM text generator."""
    import sys
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize generator
    generator = LLMTextGenerator(api_key=api_key)
    
    print("Generating sample dataset...")
    print("=" * 60)
    
    # Generate a small sample dataset
    dataset = await generator.generate_dataset(
        samples_per_difficulty=10,
        batch_size=5,
        save_to_file="data/generated_texts.json"
    )
    
    # Display samples
    for difficulty, texts in dataset.items():
        print(f"\n{difficulty.upper()} ({len(texts)} samples):")
        print("-" * 60)
        if texts:
            print(f"Example: {texts[0][:200]}...")
            print(f"Word count: {len(texts[0].split())}")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")


if __name__ == "__main__":
    asyncio.run(main())