"""
Script to generate test dataset
"""

import argparse
import os
from pathlib import Path
from src.data_generator import DataGenerator
from src.utils import load_config
from configs.locator import resolve_config_path

def main():
    parser = argparse.ArgumentParser(description='Generate test dataset for Context-Aware Redaction benchmark')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Total number of samples to generate')
    parser.add_argument('--difficulty', type=str, nargs='+', 
                        choices=['easy', 'medium', 'hard', 'all'],
                        default=['all'],
                        help='Difficulty levels to generate (easy/medium/hard/all). Can specify multiple.')
    parser.add_argument('--output', type=str, default='dataset.json',
                        help='Output filename')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    # Load configuration
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = resolve_config_path("context_aware_redaction", config_path)
    config = load_config(config_path)
    
    # Map difficulty names to internal difficulty levels
    difficulty_mapping = {
        'easy': 'short',
        'medium': 'medium',
        'hard': 'long'
    }
    
    # Determine which difficulties to generate
    if 'all' in args.difficulty:
        difficulties_to_generate = ['short', 'medium', 'long']
        difficulty_display = ['easy', 'medium', 'hard']
    else:
        difficulties_to_generate = [difficulty_mapping[d] for d in args.difficulty]
        difficulty_display = args.difficulty
    
    # Determine number of samples
    if args.num_samples is not None:
        # If num_samples is specified, distribute evenly across difficulties
        num_samples_per_difficulty = args.num_samples // len(difficulties_to_generate)
        remainder = args.num_samples % len(difficulties_to_generate)
    else:
        # Use config default
        num_samples_per_difficulty = config['data_generation']['samples_per_difficulty']
        remainder = 0
    
    print("="*80)
    print("Context-Aware Redaction Benchmark - Data Generation")
    print("="*80)
    print(f"Configuration: {args.config}")
    print(f"Difficulty levels: {', '.join(difficulty_display)}")
    print(f"Samples per difficulty: {num_samples_per_difficulty}" + 
          (f" (+{remainder} for first difficulty)" if remainder > 0 else ""))
    print(f"Total samples: {args.num_samples if args.num_samples else num_samples_per_difficulty * len(difficulties_to_generate)}")
    print(f"Output file: {args.output}")
    print(f"Verbose: {args.verbose}")
    print("="*80)
    
    # Initialize generator
    generator = DataGenerator(config, api_key, verbose=args.verbose)
    
    # Generate dataset with specified difficulties
    dataset = generator.generate_dataset_by_difficulty(
        difficulties=difficulties_to_generate,
        num_samples_per_difficulty=num_samples_per_difficulty,
        extra_samples=remainder
    )
    
    # Save dataset
    generator.save_dataset(dataset, args.output)
    
    print("\nData generation completed successfully!")


if __name__ == '__main__':
    main()
