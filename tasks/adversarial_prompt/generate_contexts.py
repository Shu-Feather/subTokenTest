"""
Script for generating original adversarial contexts using GPT.
"""

import argparse
import yaml
from pathlib import Path
from src.context_generator import ContextGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def format_cost(total_cost: float) -> str:
    """
    Format cost for display.
    
    Args:
        total_cost: Total cost in USD
        
    Returns:
        Formatted cost string
    """
    return f"${total_cost:.4f} USD"


def print_cost_summary(cost_info: dict):
    """
    Print detailed cost summary.
    
    Args:
        cost_info: Dictionary containing cost information
    """
    print("\n" + "="*70)
    print("COST SUMMARY")
    print("="*70)
    print(f"Total Tokens Used: {cost_info['total_tokens']:,}")
    print(f"  - Prompt Tokens: {cost_info['prompt_tokens']:,}")
    print(f"  - Completion Tokens: {cost_info['completion_tokens']:,}")
    print(f"\nTotal Cost: {format_cost(cost_info['total_cost'])}")
    print(f"  - Prompt Cost: {format_cost(cost_info['prompt_cost'])}")
    print(f"  - Completion Cost: {format_cost(cost_info['completion_cost'])}")
    print(f"\nModel: {cost_info['model']}")
    print(f"Total API Calls: {cost_info['api_calls']}")
    print(f"Average Cost per Call: {format_cost(cost_info['avg_cost_per_call'])}")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Generate original adversarial contexts using GPT"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/benchmark_config.yaml",
        help="Path to benchmark configuration file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/generated_contexts.json",
        help="Output file for generated contexts"
    )
    
    parser.add_argument(
        "--categories",
        nargs='+',
        default=None,
        help="Categories to generate (overrides config)"
    )

    parser.add_argument(
        "--difficulty_level",
        nargs='+',
        default=None,
        choices=["easy", "medium", "hard"],
        help="Difficulty levels to generate (default: all levels). "
             "Example: --difficulty_level easy medium"
    )
    
    parser.add_argument(
        "--samples_per_difficulty",
        type=int,
        default=None,
        help="Number of samples per difficulty level (overrides config)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API calls (default: 0.5, for rate limiting)"
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--cost",
        action="store_true",
        help="Display detailed cost information for API usage"
    )

    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Get context generation config
    ctx_config = config.get('context_generation', {})
    
    # Determine categories
    categories = args.categories if args.categories else ctx_config.get('categories', ["harmful_instructions"])
    
    # Determine difficulty levels
    if args.difficulty_level:
        difficulty_levels = args.difficulty_level
    else:
        difficulty_levels = ctx_config.get(
            'default_difficulties', 
            ["easy", "medium", "hard"]
        )
    
    # Determine samples per difficulty
    samples_per_difficulty = args.samples_per_difficulty if args.samples_per_difficulty \
        else ctx_config.get('samples_per_difficulty', 10)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate total contexts to generate
    total_contexts = len(categories) * len(difficulty_levels) * samples_per_difficulty

    print("\n" + "="*70)
    print("CONTEXT GENERATION CONFIGURATION")
    print("="*70)
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(f"Difficulty Levels ({len(difficulty_levels)}): {', '.join(difficulty_levels)}")
    print(f"Samples per difficulty: {samples_per_difficulty}")
    print(f"Total contexts to generate: {total_contexts}")
    print(f"  ({len(categories)} categories × {len(difficulty_levels)} difficulties × {samples_per_difficulty} samples)")
    print(f"Output file: {args.output}")
    print(f"Track costs: {args.cost}")
    print("="*70 + "\n")
    
    # Initialize generator
    print("Initializing context generator...")
    generator = ContextGenerator(
        api_key=args.api_key, 
        verbose=args.verbose,
        track_cost=args.cost,
        batch_mode="single",  # Always use single mode 
        delay=args.delay
    )

    # Generate contexts
    try:
        contexts = generator.generate_batch(
            categories=categories,
            difficulty_levels=difficulty_levels,
            samples_per_difficulty=samples_per_difficulty,
            output_file=args.output
        )
        
        print("\n" + "="*70)
        print("CONTEXT GENERATION COMPLETE")
        print("="*70)
        print(f"Generated contexts saved to: {args.output}")

        # Print cost summary if tracking enabled
        if args.cost:
            cost_info = generator.get_cost_summary()
            print_cost_summary(cost_info)
            
            # Save cost information to file
            cost_file = output_path.parent / f"{output_path.stem}_cost.json"
            import json
            with open(cost_file, 'w') as f:
                json.dump(cost_info, f, indent=2)
            print(f"Cost information saved to: {cost_file}")

        print("\nTo use these contexts in the benchmark, update your config:")
        print("  benchmark:")
        print("    use_generated_contexts: true")
        print(f"    generated_contexts_file: \"{args.output}\"")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nWARNING: Generation interrupted by user.")
        print("Partial results may have been saved.")
        raise

    except Exception as e:
        print(f"\nError during context generation: {e}")
        raise


if __name__ == "__main__":
    main()