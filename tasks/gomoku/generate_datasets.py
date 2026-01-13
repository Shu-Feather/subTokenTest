"""
Dataset Generator for Gomoku Benchmark.

This script generates test datasets for the Gomoku benchmark.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.board_generator import generate_test_cases


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate test datasets for Gomoku Benchmark with STRICT validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default datasets (9x9, 15x15, 19x19 with 100 cases each)
  python generate_datasets.py

  # Generate with custom density range (sparse boards)
  python generate_datasets.py --min-density 0.1 --max-density 0.3

  # Generate with custom density range (dense boards)
  python generate_datasets.py --min-density 0.5 --max-density 0.8

  # Generate with fixed density
  python generate_datasets.py --min-density 0.3 --max-density 0.3

  # Generate with custom outcome distribution and density
  python generate_datasets.py --white-wins 0.4 --black-wins 0.4 --no-winner 0.2 \\
    --min-density 0.3 --max-density 0.6

  # Generate only horizontal wins (STRICT - no other directions)
  python generate_datasets.py --horizontal 1.0 --vertical 0 --diagonal-down 0 --diagonal-up 0 \\
    --min-density 0.4 --max-density 0.6 --name horizontal_only

  # Generate challenging diagonal dataset
  python generate_datasets.py --white-wins 0.5 --black-wins 0.5 --no-winner 0 \\
    --horizontal 0 --vertical 0 --diagonal-down 0.5 --diagonal-up 0.5 \\
    --min-density 0.6 --max-density 0.8 --name challenging_diagonal_dense

  # Verify existing dataset
  python generate_datasets.py --verify-only --dataset-path datasets/test_data_15x15_100.json
        """
    )
    
    parser.add_argument(
        '--board-sizes',
        type=int,
        nargs='+',
        default=[9, 15, 19],
        help='Board sizes to generate (default: 9 15 19)'
    )
    
    parser.add_argument(
        '--test-counts',
        type=int,
        nargs='+',
        default=[100],
        help='Number of test cases per configuration (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets',
        help='Output directory for datasets (default: datasets)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Custom name prefix for dataset files (default: test_data)'
    )
    
    # Density control arguments
    density_group = parser.add_argument_group('Board Density Control')
    density_group.add_argument(
        '--min-density',
        type=float,
        default=0.2,
        help='Minimum board density (0.0 to 1.0, default: 0.2)'
    )
    
    density_group.add_argument(
        '--max-density',
        type=float,
        default=0.5,
        help='Maximum board density (0.0 to 1.0, default: 0.5)'
    )
    
    # Outcome distribution arguments
    outcome_group = parser.add_argument_group('Outcome Distribution')
    outcome_group.add_argument(
        '--white-wins',
        type=float,
        default=None,
        help='Proportion of WHITE_WINS cases (default: 0.333)'
    )
    
    outcome_group.add_argument(
        '--black-wins',
        type=float,
        default=None,
        help='Proportion of BLACK_WINS cases (default: 0.333)'
    )
    
    outcome_group.add_argument(
        '--no-winner',
        type=float,
        default=None,
        help='Proportion of NO_WINNER cases (default: 0.334)'
    )
    
    # Direction distribution arguments
    direction_group = parser.add_argument_group('Win Direction Distribution')
    direction_group.add_argument(
        '--horizontal',
        type=float,
        default=None,
        help='Proportion of horizontal wins (default: 0.25)'
    )
    
    direction_group.add_argument(
        '--vertical',
        type=float,
        default=None,
        help='Proportion of vertical wins (default: 0.25)'
    )
    
    direction_group.add_argument(
        '--diagonal-down',
        type=float,
        default=None,
        help='Proportion of diagonal-down (\\) wins (default: 0.25)'
    )
    
    direction_group.add_argument(
        '--diagonal-up',
        type=float,
        default=None,
        help='Proportion of diagonal-up (/) wins (default: 0.25)'
    )
    
    parser.add_argument(
        '--with-metadata',
        action='store_true',
        help='Include generation metadata in output files'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files without confirmation'
    )
    
    # Verification arguments
    verify_group = parser.add_argument_group('Verification Options')
    verify_group.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing dataset, do not generate new ones'
    )
    
    verify_group.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to dataset file for verification (use with --verify-only)'
    )
    
    return parser.parse_args()

def validate_distributions(args):
    """Validate and normalize distribution arguments"""
    
    # Build outcome distribution
    outcome_dist = {}
    if args.white_wins is not None:
        outcome_dist['WHITE_WINS'] = args.white_wins
    if args.black_wins is not None:
        outcome_dist['BLACK_WINS'] = args.black_wins
    if args.no_winner is not None:
        outcome_dist['NO_WINNER'] = args.no_winner
    
    # If no distribution specified, use defaults
    if not outcome_dist:
        outcome_dist = {
            'WHITE_WINS': 1/3,
            'BLACK_WINS': 1/3,
            'NO_WINNER': 1/3
        }
    # If partial distribution, fill in missing with equal weights
    elif len(outcome_dist) < 3:
        missing_keys = [k for k in ['WHITE_WINS', 'BLACK_WINS', 'NO_WINNER'] if k not in outcome_dist]
        remaining = 1.0 - sum(outcome_dist.values())
        if remaining < 0:
            print(f"Warning: Outcome distribution sums to {sum(outcome_dist.values())} > 1.0")
            print("Normalizing distributions...")
        else:
            for key in missing_keys:
                outcome_dist[key] = remaining / len(missing_keys)
    
    # Build direction distribution
    direction_dist = {}
    if args.horizontal is not None:
        direction_dist['HORIZONTAL'] = args.horizontal
    if args.vertical is not None:
        direction_dist['VERTICAL'] = args.vertical
    if args.diagonal_down is not None:
        direction_dist['DIAGONAL_DOWN'] = args.diagonal_down
    if args.diagonal_up is not None:
        direction_dist['DIAGONAL_UP'] = args.diagonal_up
    
    # If no distribution specified, use defaults
    if not direction_dist:
        direction_dist = {
            'HORIZONTAL': 0.25,
            'VERTICAL': 0.25,
            'DIAGONAL_DOWN': 0.25,
            'DIAGONAL_UP': 0.25
        }
    # If partial distribution, fill in missing with equal weights
    elif len(direction_dist) < 4:
        missing_keys = [k for k in ['HORIZONTAL', 'VERTICAL', 'DIAGONAL_DOWN', 'DIAGONAL_UP'] 
                       if k not in direction_dist]
        remaining = 1.0 - sum(direction_dist.values())
        if remaining < 0:
            print(f"Warning: Direction distribution sums to {sum(direction_dist.values())} > 1.0")
            print("Normalizing distributions...")
        else:
            for key in missing_keys:
                direction_dist[key] = remaining / len(missing_keys)
    
    return outcome_dist, direction_dist


def generate_filename(
    board_size: int,
    num_cases: int,
    name_prefix: Optional[str],
    outcome_dist: Dict[str, float],
    direction_dist: Dict[str, float],
    density_range: Tuple[float, float] = (0.2, 0.5)
) -> str:
    """Generate descriptive filename based on parameters"""
    
    if name_prefix:
        base = f"{name_prefix}_"
    else:
        base = ""
    
    # Check if distributions are non-standard
    is_standard_outcome = (
        abs(outcome_dist.get('WHITE_WINS', 0) - 1/3) < 0.01 and
        abs(outcome_dist.get('BLACK_WINS', 0) - 1/3) < 0.01 and
        abs(outcome_dist.get('NO_WINNER', 0) - 1/3) < 0.01
    )
    
    is_standard_direction = (
        abs(direction_dist.get('HORIZONTAL', 0) - 0.25) < 0.01 and
        abs(direction_dist.get('VERTICAL', 0) - 0.25) < 0.01 and
        abs(direction_dist.get('DIAGONAL_DOWN', 0) - 0.25) < 0.01 and
        abs(direction_dist.get('DIAGONAL_UP', 0) - 0.25) < 0.01
    )
    
    is_standard_density = (
        abs(density_range[0] - 0.2) < 0.01 and
        abs(density_range[1] - 0.5) < 0.01
    )
    
    if is_standard_outcome and is_standard_direction and is_standard_density:
        return f"{base}test_data_{board_size}x{board_size}_{num_cases}.json"
    
    # Add distribution info to filename
    parts = [base + "test_data"]
    
    if not is_standard_outcome:
        w = int(outcome_dist.get('WHITE_WINS', 0) * 100)
        b = int(outcome_dist.get('BLACK_WINS', 0) * 100)
        n = int(outcome_dist.get('NO_WINNER', 0) * 100)
        parts.append(f"w{w}b{b}n{n}")
    
    if not is_standard_direction:
        h = int(direction_dist.get('HORIZONTAL', 0) * 100)
        v = int(direction_dist.get('VERTICAL', 0) * 100)
        dd = int(direction_dist.get('DIAGONAL_DOWN', 0) * 100)
        du = int(direction_dist.get('DIAGONAL_UP', 0) * 100)
        parts.append(f"h{h}v{v}dd{dd}du{du}")
    
    if not is_standard_density:
        min_d = int(density_range[0] * 100)
        max_d = int(density_range[1] * 100)
        parts.append(f"d{min_d}-{max_d}")
    
    parts.append(f"{board_size}x{board_size}_{num_cases}.json")
    
    return "_".join(parts)


def generate_dataset(
    board_size: int,
    num_cases: int,
    output_dir: str,
    name_prefix: Optional[str] = None,
    outcome_distribution: Optional[Dict[str, float]] = None,
    direction_distribution: Optional[Dict[str, float]] = None,
    density_range: Tuple[float, float] = (0.2, 0.5),
    with_metadata: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    overwrite: bool = False,
    verify_after_generation: bool = True  # NEW PARAMETER
) -> str:
    """
    Generate a single dataset file with optional post-generation verification.
    
    Args:
        board_size: Size of the board (e.g., 9 for 9x9)
        num_cases: Number of test cases to generate
        output_dir: Output directory path
        name_prefix: Custom name prefix for the file
        outcome_distribution: Distribution of game outcomes
        direction_distribution: Distribution of win directions
        density_range: Tuple of (min_density, max_density)
        with_metadata: Whether to include metadata
        seed: Random seed
        verbose: Enable verbose output
        overwrite: Overwrite existing files
        verify_after_generation: Verify dataset after generation (default: True)
        
    Returns:
        Path to the generated file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = generate_filename(
        board_size, num_cases, name_prefix,
        outcome_distribution or {},
        direction_distribution or {},
        density_range
    )
    filepath = os.path.join(output_dir, filename)
    
    # Check if file exists
    if os.path.exists(filepath) and not overwrite:
        response = input(f"File {filepath} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print(f"Skipping {filename}")
            return filepath
    
    if verbose:
        print(f"Generating {num_cases} test cases for {board_size}x{board_size} board...")
        if outcome_distribution:
            print(f"  Outcome distribution: {outcome_distribution}")
        if direction_distribution:
            print(f"  Direction distribution: {direction_distribution}")
        print(f"  Density range: {density_range[0]:.2f} - {density_range[1]:.2f}")
    
    # Generate test cases
    test_cases = generate_test_cases(
        board_size=board_size,
        num_cases=num_cases,
        seed=seed,
        outcome_distribution=outcome_distribution,
        direction_distribution=direction_distribution,
        density_range=density_range,
        verbose=verbose
    )
    
    # Prepare data
    data_to_save = []
    for idx, (board, expected, direction, density) in enumerate(test_cases):
        case = {
            'id': idx,
            'board': board,
            'expected': expected,
            'board_size': board_size,
            'density': round(density, 3)
        }
        if direction:
            case['win_direction'] = direction
        data_to_save.append(case)
    
    # Add metadata if requested
    if with_metadata:
        output = {
            'metadata': {
                'board_size': board_size,
                'num_cases': num_cases,
                'outcome_distribution': outcome_distribution,
                'direction_distribution': direction_distribution,
                'density_range': {
                    'min': density_range[0],
                    'max': density_range[1]
                },
                'generated_at': datetime.now().isoformat(),
                'seed': seed,
                'version': '1.0',
                'name_prefix': name_prefix,
                'strict_verification': True
            },
            'test_cases': data_to_save
        }
    else:
        output = data_to_save
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"✓ Saved {num_cases} cases to {filepath}")
    
    # POST-GENERATION VERIFICATION
    if verify_after_generation:
        verification_passed = verify_generated_dataset(filepath, verbose=verbose)
        
        if not verification_passed:
            print(f"⚠️  WARNING: Dataset {filename} failed post-generation verification!")
            print(f"   Consider regenerating with different parameters or seed.")
    
    return filepath

def verify_generated_dataset(filepath: str, verbose: bool = False):
    """
    Verify a generated dataset file immediately after creation
    
    Args:
        filepath: Path to the generated dataset file
        verbose: Print detailed verification information
    """
    from src.board_generator import verify_dataset_file
    
    print(f"\n{'='*80}")
    print(f"Post-Generation Verification: {os.path.basename(filepath)}")
    print(f"{'='*80}")
    
    try:
        stats = verify_dataset_file(filepath, verbose=verbose)
        
        # Return True if all tests passed
        if stats['passed'] == stats['total']:
            print(f"✅ All {stats['total']} test cases PASSED strict verification!\n")
            return True
        else:
            print(f"⚠️  {stats['failed']} out of {stats['total']} test cases FAILED!\n")
            return False
    except Exception as e:
        print(f"❌ Verification error: {e}\n")
        return False


def generate_all_datasets(args, outcome_dist, direction_dist):
    """Generate all requested datasets with strict verification"""
    print("=" * 80)
    print("Gomoku STRICT Dataset Generator")
    print("=" * 80)
    print(f"Board sizes: {args.board_sizes}")
    print(f"Test counts: {args.test_counts}")
    print(f"Output directory: {args.output_dir}")
    if args.name:
        print(f"Name prefix: {args.name}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    
    print(f"\nOutcome Distribution:")
    for outcome, prob in outcome_dist.items():
        print(f"  {outcome}: {prob:.2%}")
    
    print(f"\nWin Direction Distribution:")
    for direction, prob in direction_dist.items():
        print(f"  {direction}: {prob:.2%}")
    
    print(f"\nBoard Density Range:")
    print(f"  Min: {args.min_density:.2%}")
    print(f"  Max: {args.max_density:.2%}")
    
    print(f"\n⚠️  STRICT MODE ENABLED:")
    print(f"  - Each winning board has EXACTLY ONE winner")
    print(f"  - Each winning board wins in EXACTLY ONE direction")
    print(f"  - Opponent has NO winning lines")
    print(f"  - No invalid boards (both players winning)")
    
    print("=" * 80)
    print()
    
    generated_files = []
    verification_results = []
    total_datasets = len(args.board_sizes) * len(args.test_counts)
    current = 0
    
    density_range = (args.min_density, args.max_density)
    
    for board_size in args.board_sizes:
        for num_cases in args.test_counts:
            current += 1
            print(f"[{current}/{total_datasets}] Generating {board_size}x{board_size}, {num_cases} cases...")
            
            try:
                filepath = generate_dataset(
                    board_size=board_size,
                    num_cases=num_cases,
                    output_dir=args.output_dir,
                    name_prefix=args.name,
                    outcome_distribution=outcome_dist,
                    direction_distribution=direction_dist,
                    density_range=density_range,
                    with_metadata=args.with_metadata,
                    seed=args.seed,
                    verbose=args.verbose,
                    overwrite=args.overwrite,
                    verify_after_generation=True  # Always verify
                )
                generated_files.append(filepath)
            except Exception as e:
                print(f"✗ Error generating dataset: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                continue
    
    print()
    print("=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print(f"Total datasets generated: {len(generated_files)}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("Generated files:")
    for filepath in generated_files:
        file_size = os.path.getsize(filepath)
        size_kb = file_size / 1024
        print(f"  - {os.path.basename(filepath)} ({size_kb:.2f} KB)")
    print()
    
    return generated_files


def print_dataset_summary(output_dir: str):
    """Print summary of all datasets in the directory"""
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist")
        return
    
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No datasets found in {output_dir}")
        return
    
    print()
    print("=" * 130)
    print(f"Dataset Summary: {output_dir}")
    print("=" * 130)
    print(f"{'Filename':<50} {'Board':<10} {'Cases':<8} {'W/B/N':<15} {'H/V/DD/DU':<20} {'Density':<15} {'Size':<10}")
    print("-" * 130)
    
    total_cases = 0
    total_size = 0
    
    for filename in sorted(json_files):
        filepath = os.path.join(output_dir, filename)
        file_size = os.path.getsize(filepath)
        total_size += file_size
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check format
            if isinstance(data, dict) and 'test_cases' in data:
                metadata = data.get('metadata', {})
                board_size = metadata.get('board_size', 'N/A')
                num_cases = len(data['test_cases'])
                test_cases = data['test_cases']
                
                # Get distributions from metadata
                outcome_dist = metadata.get('outcome_distribution', {})
                direction_dist = metadata.get('direction_distribution', {})
                density_info = metadata.get('density_range', {})
            else:
                # Try to infer from data
                if data and len(data) > 0:
                    board_size = data[0].get('board_size', 'N/A')
                    num_cases = len(data)
                    test_cases = data
                    outcome_dist = {}
                    direction_dist = {}
                    density_info = {}
                else:
                    board_size = 'N/A'
                    num_cases = 0
                    test_cases = []
                    outcome_dist = {}
                    direction_dist = {}
                    density_info = {}
            
            # Calculate actual distributions from test cases
            if test_cases:
                outcome_counts = {'WHITE_WINS': 0, 'BLACK_WINS': 0, 'NO_WINNER': 0}
                direction_counts = {'HORIZONTAL': 0, 'VERTICAL': 0, 'DIAGONAL_DOWN': 0, 'DIAGONAL_UP': 0}
                densities = []
                
                for case in test_cases:
                    expected = case.get('expected', '')
                    if expected in outcome_counts:
                        outcome_counts[expected] += 1
                    
                    direction = case.get('win_direction')
                    if direction and direction in direction_counts:
                        direction_counts[direction] += 1
                    
                    density = case.get('density')
                    if density is not None:
                        densities.append(density)
                
                # Format distributions
                w = outcome_counts['WHITE_WINS']
                b = outcome_counts['BLACK_WINS']
                n = outcome_counts['NO_WINNER']
                outcome_str = f"{w}/{b}/{n}"
                
                total_wins = w + b
                if total_wins > 0:
                    h = direction_counts['HORIZONTAL']
                    v = direction_counts['VERTICAL']
                    dd = direction_counts['DIAGONAL_DOWN']
                    du = direction_counts['DIAGONAL_UP']
                    direction_str = f"{h}/{v}/{dd}/{du}"
                else:
                    direction_str = "N/A"
                
                # Format density
                if densities:
                    min_density = min(densities)
                    max_density = max(densities)
                    avg_density = sum(densities) / len(densities)
                    density_str = f"{min_density:.2f}-{max_density:.2f} (avg:{avg_density:.2f})"
                else:
                    density_str = "N/A"
            else:
                outcome_str = "N/A"
                direction_str = "N/A"
                density_str = "N/A"
            
            total_cases += num_cases
            size_kb = file_size / 1024
            
            board_str = f"{board_size}x{board_size}" if board_size != 'N/A' else 'N/A'
            print(f"{filename:<50} {board_str:<10} {num_cases:<8} {outcome_str:<15} {direction_str:<20} {density_str:<15} {size_kb:<10.2f} KB")
        
        except Exception as e:
            size_kb = file_size / 1024
            print(f"{filename:<50} {'Error':<10} {'N/A':<8} {'N/A':<15} {'N/A':<20} {'N/A':<15} {size_kb:<10.2f} KB")
    
    print("-" * 130)
    print(f"Total: {len(json_files)} files, {total_cases} test cases, {total_size/1024:.2f} KB")
    print("=" * 130)
    print()

def print_distribution_examples():
    """Print examples of distribution configurations"""
    print("\n" + "=" * 80)
    print("Distribution Configuration Examples")
    print("=" * 80)
    
    print("\n1. Sparse boards (10-30% filled):")
    print("   --min-density 0.1 --max-density 0.3")
    
    print("\n2. Dense boards (50-80% filled):")
    print("   --min-density 0.5 --max-density 0.8")
    
    print("\n3. Fixed density (exactly 40% filled):")
    print("   --min-density 0.4 --max-density 0.4")
    
    print("\n4. Only winning cases with sparse boards:")
    print("   --white-wins 0.5 --black-wins 0.5 --no-winner 0")
    print("   --min-density 0.15 --max-density 0.25")
    
    print("\n5. Only horizontal wins with dense boards:")
    print("   --horizontal 1.0 --vertical 0 --diagonal-down 0 --diagonal-up 0")
    print("   --min-density 0.6 --max-density 0.8")
    
    print("\n6. Only diagonal wins with medium density:")
    print("   --horizontal 0 --vertical 0 --diagonal-down 0.5 --diagonal-up 0.5")
    print("   --min-density 0.3 --max-density 0.5")
    
    print("\n7. Challenging dataset (dense diagonal-only wins):")
    print("   --white-wins 0.5 --black-wins 0.5 --no-winner 0")
    print("   --horizontal 0 --vertical 0 --diagonal-down 0.5 --diagonal-up 0.5")
    print("   --min-density 0.6 --max-density 0.8")
    print("   --name challenging_diagonal_dense")
    
    print("\n8. Easy dataset (sparse horizontal wins):")
    print("   --white-wins 0.5 --black-wins 0.5 --no-winner 0")
    print("   --horizontal 1.0 --vertical 0 --diagonal-down 0 --diagonal-up 0")
    print("   --min-density 0.1 --max-density 0.2")
    print("   --name easy_horizontal_sparse")
    
    print("=" * 80 + "\n")


def main():
    """Main function"""
    args = parse_arguments()
    
    try:
        # Verification-only mode
        if args.verify_only:
            if args.dataset_path:
                from src.board_generator import verify_dataset_file
                print(f"\n{'='*80}")
                print(f"STRICT Verification Mode")
                print(f"{'='*80}\n")
                verify_dataset_file(args.dataset_path, verbose=args.verbose)
            else:
                # Verify all datasets in output directory
                from src.board_generator import verify_dataset_file
                import glob
                
                dataset_files = glob.glob(os.path.join(args.output_dir, '*.json'))
                
                if not dataset_files:
                    print(f"No dataset files found in {args.output_dir}")
                    return
                
                print(f"\n{'='*80}")
                print(f"STRICT Batch Verification Mode")
                print(f"Found {len(dataset_files)} dataset files")
                print(f"{'='*80}\n")
                
                all_passed = True
                for filepath in sorted(dataset_files):
                    result = verify_dataset_file(filepath, verbose=False)
                    if result['failed'] > 0:
                        all_passed = False
                
                if all_passed:
                    print(f"\n✅ All datasets passed STRICT verification!")
                else:
                    print(f"\n⚠️  Some datasets failed STRICT verification!")
            
            return
        
        # Normal generation mode
        # Validate and normalize distributions
        outcome_dist, direction_dist = validate_distributions(args)
        
        # Set random seed if provided
        if args.seed is not None:
            import random
            import numpy as np
            random.seed(args.seed)
            np.random.seed(args.seed)
            if args.verbose:
                print(f"Random seed set to {args.seed}")
        
        # Generate datasets
        generated_files = generate_all_datasets(args, outcome_dist, direction_dist)
        
        # Print summary
        print_dataset_summary(args.output_dir)
        
        print("✅ All done!")
        
        # Show examples if verbose
        if args.verbose:
            print_distribution_examples()
        
    except KeyboardInterrupt:
        print("\n\nDataset generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()