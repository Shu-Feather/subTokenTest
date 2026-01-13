"""
Generate typewriter benchmark datasets from wordlist with difficulty levels
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate typewriter benchmark datasets with difficulty levels"""
    
    def __init__(self, wordlist_path: str):
        self.wordlist_path = Path(wordlist_path)
        self.words = self._load_wordlist()
        
    def _load_wordlist(self) -> List[str]:
        """Load words from wordlist file"""
        if not self.wordlist_path.exists():
            raise FileNotFoundError(f"Wordlist not found: {self.wordlist_path}")
        
        words = []
        with open(self.wordlist_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and word.isalpha():  # Only alphabetic words
                    words.append(word.lower())
        
        logger.info(f"Loaded {len(words)} words from {self.wordlist_path}")
        return words
    
    def categorize_words_by_length(self) -> Dict[str, List[str]]:
        """Categorize words by difficulty based on length"""
        easy_words = []      # Length 4-5
        medium_words = []    # Length 6-9
        hard_words = []      # Length 10+
        
        for word in self.words:
            word_len = len(word)
            if 4 <= word_len <= 5:
                easy_words.append(word)
            elif 6 <= word_len <= 9:
                medium_words.append(word)
            elif word_len >= 10:
                hard_words.append(word)
        
        logger.info(f"Easy words (4-5 chars): {len(easy_words)}")
        logger.info(f"Medium words (6-9 chars): {len(medium_words)}")
        logger.info(f"Hard words (10+ chars): {len(hard_words)}")
        
        return {
            'easy': easy_words,
            'medium': medium_words,
            'hard': hard_words
        }
    
    def generate_task1_dataset(self, num_samples_per_difficulty: int = 20) -> Dict[str, List[str]]:
        """
        Generate Task 1 dataset (Typewriter Effect)
        
        Difficulty is based on word length:
        - Easy: 4-5 characters
        - Medium: 6-9 characters
        - Hard: 10+ characters
        """
        categorized_words = self.categorize_words_by_length()
        
        dataset = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        for difficulty, words in categorized_words.items():
            if len(words) == 0:
                logger.warning(f"No words available for {difficulty} difficulty")
                continue
            
            # Sample words (with replacement if needed)
            if len(words) >= num_samples_per_difficulty:
                sampled = random.sample(words, num_samples_per_difficulty)
            else:
                sampled = random.choices(words, k=num_samples_per_difficulty)
            
            dataset[difficulty] = sampled
        
        return dataset
    
    def generate_task2_dataset(self, num_samples_per_difficulty: int = 20) -> Dict[str, List[str]]:
        """
        Generate Task 2 dataset (Backspace Handling)
        
        Difficulty is based on:
        - Easy: Short sequence (5-8 tokens), 1-2 backspaces
        - Medium: Medium sequence (9-15 tokens), 3-5 backspaces
        - Hard: Long sequence (16-25 tokens), 6-10 backspaces
        """
        dataset = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        # Easy: 5-8 tokens, 1-2 backspaces
        for _ in range(num_samples_per_difficulty):
            seq_length = random.randint(5, 8)
            num_backspaces = random.randint(1, 2)
            sequence = self._generate_backspace_sequence(seq_length, num_backspaces)
            dataset['easy'].append(sequence)
        
        # Medium: 9-15 tokens, 3-5 backspaces
        for _ in range(num_samples_per_difficulty):
            seq_length = random.randint(9, 15)
            num_backspaces = random.randint(3, 5)
            sequence = self._generate_backspace_sequence(seq_length, num_backspaces)
            dataset['medium'].append(sequence)
        
        # Hard: 16-25 tokens, 6-10 backspaces
        for _ in range(num_samples_per_difficulty):
            seq_length = random.randint(16, 25)
            num_backspaces = random.randint(6, 10)
            sequence = self._generate_backspace_sequence(seq_length, num_backspaces)
            dataset['hard'].append(sequence)
        
        return dataset
    
    def _generate_backspace_sequence(self, seq_length: int, num_backspaces: int) -> str:
        """Generate a random typing sequence with backspace operations"""
        import string
        
        # Generate base character sequence
        total_length = seq_length + num_backspaces
        chars = random.choices(string.ascii_lowercase, k=total_length)
        
        # Randomly insert backspace operations
        backspace_positions = random.sample(range(1, total_length), num_backspaces)
        backspace_positions.sort(reverse=True)
        
        for pos in backspace_positions:
            chars.insert(pos, '←')
        
        # Join with spaces
        sequence = ' '.join(chars)
        return sequence
    
    def generate_full_dataset(self, 
                             num_samples_per_difficulty: int = 20,
                             output_dir: str = "datasets") -> Dict[str, Any]:
        """Generate complete dataset for both tasks with all difficulty levels"""
        
        logger.info("Generating Task 1 dataset...")
        task1_dataset = self.generate_task1_dataset(num_samples_per_difficulty)
        
        logger.info("Generating Task 2 dataset...")
        task2_dataset = self.generate_task2_dataset(num_samples_per_difficulty)
        
        # Combine all difficulties for compatibility
        task1_all = (task1_dataset['easy'] + 
                    task1_dataset['medium'] + 
                    task1_dataset['hard'])
        
        task2_all = (task2_dataset['easy'] + 
                    task2_dataset['medium'] + 
                    task2_dataset['hard'])
        
        # Shuffle combined datasets
        random.shuffle(task1_all)
        random.shuffle(task2_all)
        
        full_dataset = {
            'task1': {
                'easy': task1_dataset['easy'],
                'medium': task1_dataset['medium'],
                'hard': task1_dataset['hard'],
                'all': task1_all
            },
            'task2': {
                'easy': task2_dataset['easy'],
                'medium': task2_dataset['medium'],
                'hard': task2_dataset['hard'],
                'all': task2_all
            },
            'metadata': {
                'total_task1_samples': len(task1_all),
                'total_task2_samples': len(task2_all),
                'samples_per_difficulty': num_samples_per_difficulty,
                'task1_difficulty_distribution': {
                    'easy': f"3-5 characters, {len(task1_dataset['easy'])} samples",
                    'medium': f"6-9 characters, {len(task1_dataset['medium'])} samples",
                    'hard': f"10+ characters, {len(task1_dataset['hard'])} samples"
                },
                'task2_difficulty_distribution': {
                    'easy': f"5-8 tokens, 1-2 backspaces, {len(task2_dataset['easy'])} samples",
                    'medium': f"9-15 tokens, 3-5 backspaces, {len(task2_dataset['medium'])} samples",
                    'hard': f"16-25 tokens, 6-10 backspaces, {len(task2_dataset['hard'])} samples"
                }
            }
        }
        
        return full_dataset
    
    def save_dataset(self, dataset: Dict[str, Any], output_dir: str = "datasets"):
        """Save dataset to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full dataset
        full_path = output_path / "test_cases_full.json"
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved full dataset to: {full_path}")
        
        # Save difficulty-specific datasets
        for difficulty in ['easy', 'medium', 'hard']:
            difficulty_dataset = {
                'task1': dataset['task1'][difficulty],
                'task2': dataset['task2'][difficulty],
                'metadata': {
                    'difficulty': difficulty,
                    'task1_samples': len(dataset['task1'][difficulty]),
                    'task2_samples': len(dataset['task2'][difficulty])
                }
            }
            
            diff_path = output_path / f"test_cases_{difficulty}.json"
            with open(diff_path, 'w', encoding='utf-8') as f:
                json.dump(difficulty_dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {difficulty} dataset to: {diff_path}")
        
        # Print summary
        self._print_dataset_summary(dataset)
    
    def _print_dataset_summary(self, dataset: Dict[str, Any]):
        """Print dataset generation summary"""
        print("\n" + "="*70)
        print("DATASET GENERATION SUMMARY")
        print("="*70)
        
        print("\n--- Task 1: Typewriter Effect ---")
        for difficulty in ['easy', 'medium', 'hard']:
            samples = dataset['task1'][difficulty]
            print(f"{difficulty.capitalize()}: {len(samples)} samples")
            print(f"  Example: {samples[0] if samples else 'N/A'}")
        
        print("\n--- Task 2: Backspace Handling ---")
        for difficulty in ['easy', 'medium', 'hard']:
            samples = dataset['task2'][difficulty]
            print(f"{difficulty.capitalize()}: {len(samples)} samples")
            print(f"  Example: {samples[0] if samples else 'N/A'}")
        
        print("\n--- Metadata ---")
        print(f"Total Task 1 samples: {dataset['metadata']['total_task1_samples']}")
        print(f"Total Task 2 samples: {dataset['metadata']['total_task2_samples']}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate typewriter benchmark datasets from wordlist",
    )
    
    parser.add_argument(
        '--wordlist',
        type=str,
        default='datasets/wordlist.txt',
        help='Path to wordlist file (default: datasets/wordlist.txt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='datasets',
        help='Output directory for generated datasets (default: datasets)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=20,
        help='Number of samples per difficulty level (default: 20)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
    
    try:
        # Initialize generator
        generator = DatasetGenerator(args.wordlist)
        
        # Generate datasets
        dataset = generator.generate_full_dataset(
            num_samples_per_difficulty=args.samples,
            output_dir=args.output
        )
        
        # Save datasets
        generator.save_dataset(dataset, args.output)
        
        print(f"\n✓ Dataset generation completed successfully!")
        print(f"✓ Output directory: {args.output}")
        
    except Exception as e:
        logger.error(f"Error generating datasets: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()