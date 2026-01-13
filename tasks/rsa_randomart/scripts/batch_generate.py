"""
Batch data generation script
Path: scripts/batch_generate.py
"""

import argparse
import sys
import os
sys.path.append('..')

from src.data_generator import RSAPatternGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Batch generate RSA-difference benchmark data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--num-datasets',
        type=int,
        default=5,
        help='Number of different datasets to generate'
    )
    parser.add_argument(
        '--samples-per-dataset',
        type=int,
        default=100,
        help='Number of samples per dataset'
    )
    parser.add_argument(
        '--min-differences',
        type=int,
        default=3,
        help='Minimum number of differences'
    )
    parser.add_argument(
        '--max-differences',
        type=int,
        default=10,
        help='Maximum number of differences'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    generator = RSAPatternGenerator(width=19, height=9)
    
    # Generate datasets with varying difficulty
    for i in range(args.num_datasets):
        num_diffs = args.min_differences + \
                   (args.max_differences - args.min_differences) * i // (args.num_datasets - 1)
        
        print(f"\nGenerating dataset {i+1}/{args.num_datasets} "
              f"with {num_diffs} differences...")
        
        samples = generator.generate_batch(
            num_samples=args.samples_per_dataset,
            num_differences=num_diffs
        )
        
        output_path = os.path.join(
            args.output_dir,
            f'dataset_diff{num_diffs}_n{args.samples_per_dataset}.json'
        )
        
        generator.save_samples(samples, output_path)
        print(f"Saved to {output_path}")
    
    print(f"\n✓ Generated {args.num_datasets} datasets in {args.output_dir}/")


if __name__ == '__main__':
    main()