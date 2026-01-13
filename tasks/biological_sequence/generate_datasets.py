"""
Dataset generation script for Biological Sequence Manipulation Benchmark.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import random
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import BiologicalSequenceGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate test datasets for Biological Sequence Benchmark"
    )
    
    # Dataset size parameters
    parser.add_argument(
        '--num-cases',
        type=int,
        default=100,
        help='Number of test cases to generate for each task (default: 100)'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=8,
        help='Minimum sequence length (default: 8)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=15,
        help='Maximum sequence length (default: 15)'
    )
    
    # Task selection
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=['dna_complement', 'rna_complement', 'protein_three_to_one', 'protein_one_to_three', 'all'],
        default=['all'],
        help='Specific tasks to generate (default: all)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets',
        help='Directory to save the generated datasets (default: datasets)'
    )
    
    parser.add_argument(
        '--filename',
        type=str,
        default=None,
        help='Custom filename for the dataset (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Save each task as a separate file instead of one combined file'
    )
    
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Save JSON with pretty formatting (indented)'
    )
    
    parser.add_argument(
        '--include-metadata',
        action='store_true',
        default=True,
        help='Include metadata (generation time, parameters) in the output'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible generation'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    return parser.parse_args()


def get_task_list(tasks_arg):
    """Convert task argument to list of tasks."""
    all_tasks = ['dna_complement', 'rna_complement', 'protein_three_to_one', 'protein_one_to_three']
    
    if 'all' in tasks_arg:
        return all_tasks
    return tasks_arg


def generate_datasets(args):
    """Generate datasets based on arguments."""
    # Setup random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        if args.verbose:
            print(f"Random seed set to: {args.seed}")
    
    # Initialize generator
    generator = BiologicalSequenceGenerator()
    
    # Get task list
    tasks = get_task_list(args.tasks)
    
    if not args.quiet:
        print("="*60)
        print("Biological Sequence Dataset Generator")
        print("="*60)
        print(f"Generating datasets for {len(tasks)} task(s):")
        for task in tasks:
            print(f"  - {task}")
        print(f"Number of cases per task: {args.num_cases}")
        print(f"Sequence length range: [{args.min_length}, {args.max_length}]")
        print()
    
    # Generate datasets
    all_datasets = {}
    
    for task in tasks:
        if args.verbose:
            print(f"Generating {task}...")
        
        # Determine sequence length range for this task
        min_len = args.min_length
        max_len = args.max_length
        
        # Generate test cases
        test_cases = generator.generate_test_cases(
            task_type=task,
            num_cases=args.num_cases,
            sequence_length_range=(min_len, max_len)
        )
        
        all_datasets[task] = test_cases
        
        if args.verbose:
            print(f"  ✓ Generated {len(test_cases)} cases")
            # Show a sample
            if test_cases:
                sample = test_cases[0]
                print(f"    Sample: {sample['input'][:30]}... → {sample['expected_output'][:30]}...")
    
    if not args.quiet:
        print(f"\n✓ Total cases generated: {sum(len(cases) for cases in all_datasets.values())}")
    
    return all_datasets


def create_metadata(args, datasets):
    """Create metadata for the dataset."""
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'parameters': {
            'num_cases': args.num_cases,
            'min_length': args.min_length,
            'max_length': args.max_length,
            'tasks': get_task_list(args.tasks)
        },
        'statistics': {}
    }
    
    # Add seed if used
    if args.seed is not None:
        metadata['parameters']['seed'] = args.seed
    
    # Calculate statistics
    for task, cases in datasets.items():
        if cases:
            lengths = [case['sequence_length'] for case in cases]
            metadata['statistics'][task] = {
                'num_cases': len(cases),
                'avg_length': sum(lengths) / len(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths)
            }
    
    return metadata


def save_datasets(datasets, args):
    """Save datasets to files."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine filename
    if args.filename:
        base_filename = args.filename
        if not base_filename.endswith('.json'):
            base_filename += '.json'
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f'bio_seq_dataset_{timestamp}.json'
    
    # Prepare JSON formatting options
    json_kwargs = {
        'ensure_ascii': False
    }
    if args.pretty:
        json_kwargs['indent'] = 2
    
    # Save datasets
    saved_files = []
    
    if args.split:
        # Save each task as a separate file
        for task, cases in datasets.items():
            task_filename = base_filename.replace('.json', f'_{task}.json')
            filepath = output_dir / task_filename
            
            data = {'data': cases}
            if args.include_metadata:
                data['metadata'] = create_metadata(args, {task: cases})
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, **json_kwargs)
            
            saved_files.append(filepath)
            
            if args.verbose:
                print(f"Saved {task}: {filepath} ({len(cases)} cases)")
    
    else:
        # Save all tasks in one file
        filepath = output_dir / base_filename
        
        data = {'datasets': datasets}
        if args.include_metadata:
            data['metadata'] = create_metadata(args, datasets)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, **json_kwargs)
        
        saved_files.append(filepath)
        
        if not args.quiet:
            print(f"\n✓ Dataset saved: {filepath}")
    
    return saved_files


def display_summary(datasets, saved_files, args):
    """Display generation summary."""
    if args.quiet:
        return
    
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    
    # Task summary
    print("\nTasks Generated:")
    total_cases = 0
    for task, cases in datasets.items():
        print(f"  {task:30s}: {len(cases):4d} cases")
        total_cases += len(cases)
    print(f"  {'TOTAL':30s}: {total_cases:4d} cases")
    
    # File summary
    print(f"\nFiles Saved ({len(saved_files)}):")
    for filepath in saved_files:
        size_kb = filepath.stat().st_size / 1024
        print(f"  {filepath.name:40s}: {size_kb:6.1f} KB")
    
    # Quick stats
    if datasets:
        sample_task = list(datasets.keys())[0]
        sample_case = datasets[sample_task][0]
        
        print("\nSample Data:")
        print(f"  Task: {sample_case['task_type']}")
        print(f"  Input: {sample_case['input']}")
        print(f"  Output: {sample_case['expected_output']}")
    
    print("\n" + "="*60)


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Validate lengths
    if args.min_length < 1:
        errors.append("Minimum length must be at least 1")
    
    if args.max_length < args.min_length:
        errors.append("Maximum length must be greater than or equal to minimum length")
    
    if args.num_cases < 1:
        errors.append("Number of cases must be at least 1")
    
    # Check for conflicting options
    if args.quiet and args.verbose:
        errors.append("Cannot use both --quiet and --verbose")
    
    return errors


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    errors = validate_arguments(args)
    if errors:
        print("Error: Invalid arguments:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Generate datasets
        datasets = generate_datasets(args)
        
        # Save datasets
        saved_files = save_datasets(datasets, args)
        
        # Display summary
        display_summary(datasets, saved_files, args)
        
        if not args.quiet:
            print("\n✓ Dataset generation completed successfully!")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user", file=sys.stderr)
        return 1
    
    except Exception as e:
        print(f"\n✗ Error during generation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())