"""
Script to generate LLM-based text dataset for cipher benchmark.
Run this script before running the main benchmark to create high-quality test data.
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_generation.llm_text_generator import LLMTextGenerator, DifficultyLevel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main function to generate dataset."""
    parser = argparse.ArgumentParser(
        description="Generate LLM-based text dataset for cipher benchmark",
    )
    
    parser.add_argument("--samples", type=int, default=100, 
                       help="Number of samples per difficulty level (default: 100)")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Number of texts to generate per API call (default: 5)")
    parser.add_argument("--difficulties", nargs='+', 
                       choices=['easy', 'medium', 'hard'],
                       help="Difficulty levels to generate (default: all). "
                            "Can specify one or more: easy, medium, hard")
    parser.add_argument("--output", default="datasets/generated_texts.json",
                       help="Output file path (default: datasets/generated_texts.json)")
    parser.add_argument("--model", default="gpt-3.5-turbo",
                       help="OpenAI model to use (default: gpt-3.5-turbo)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    try:
        # Initialize generator
        logger.info(f"Initializing LLM text generator with model: {args.model}")
        generator = LLMTextGenerator(api_key=api_key, model_name=args.model)
        
        # Prepare difficulty info
        if args.difficulties:
            difficulties_str = ", ".join(args.difficulties)
            logger.info(f"Generating difficulties: {difficulties_str}")
        else:
            difficulties_str = "all (easy, medium, hard)"
            logger.info(f"Generating difficulties: {difficulties_str}")
        
        # Generate dataset
        logger.info(f"Samples per difficulty: {args.samples}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Output: {args.output}")
        logger.info("-" * 60)
        
        dataset = await generator.generate_dataset(
            samples_per_difficulty=args.samples,
            batch_size=args.batch_size,
            difficulties=args.difficulties,
            save_to_file=args.output
        )
        
        # Display summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("DATASET GENERATION COMPLETE")
        logger.info("=" * 60)
        
        total_samples = 0
        for difficulty in ['easy', 'medium', 'hard']:
            texts = dataset.get(difficulty, [])
            if texts:  # Only show difficulties that were generated
                logger.info(f"{difficulty.upper()}: {len(texts)} samples")
                word_counts = [len(text.split()) for text in texts]
                logger.info(f"  Word count range: {min(word_counts)}-{max(word_counts)}")
                logger.info(f"  Average word count: {sum(word_counts)/len(word_counts):.1f}")
                total_samples += len(texts)
        
        logger.info(f"\nTOTAL SAMPLES: {total_samples}")
        logger.info(f"Dataset saved to: {args.output}")
        logger.info(f"Statistics saved to: {args.output.replace('.json', '_stats.txt')}")
        
        # Display example texts
        logger.info("\n" + "=" * 60)
        logger.info("EXAMPLE TEXTS")
        logger.info("=" * 60)
        
        for difficulty in ['easy', 'medium', 'hard']:
            if dataset.get(difficulty):
                logger.info(f"\n{difficulty.upper()} example:")
                logger.info(f"{dataset[difficulty][0]}")
        
        logger.info("\n" + "=" * 60)
        logger.info("You can now run the benchmark with this dataset using:")
        logger.info(f"python main.py --test-file {args.output}")
        
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())