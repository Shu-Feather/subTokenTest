"""
Utility script for analyzing benchmark results.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(results_path: str):
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_failures(results: list, threshold: float = 0.9):
    """
    Analyze failed predictions.
    
    Args:
        results: List of result dictionaries
        threshold: Similarity threshold for considering a prediction as failure
    """
    failures = [r for r in results if r['similarity_score'] < threshold]
    
    print(f"\n{'='*70}")
    print(f"FAILURE ANALYSIS (similarity < {threshold})")
    print(f"{'='*70}")
    print(f"Total failures: {len(failures)} / {len(results)} "
          f"({len(failures)/len(results)*100:.1f}%)")
    
    # Group by difficulty
    by_difficulty = defaultdict(list)
    for failure in failures:
        by_difficulty[failure.get('difficulty', 'unknown')].append(failure)
    
    print(f"\nFailures by difficulty:")
    for diff, fails in sorted(by_difficulty.items()):
        print(f"  {diff}: {len(fails)}")
    
    # Group by category
    by_category = defaultdict(list)
    for failure in failures:
        by_category[failure['category']].append(failure)
    
    print(f"\nFailures by category:")
    for cat, fails in sorted(by_category.items()):
        print(f"  {cat}: {len(fails)}")
    
    # Group by perturbation type
    by_perturbation = defaultdict(list)
    for failure in failures:
        by_perturbation[failure['perturbation_type']].append(failure)
    
    print(f"\nFailures by perturbation type:")
    for pert, fails in sorted(by_perturbation.items()):
        print(f"  {pert}: {len(fails)}")
    
    # Analyze answer tag usage
    no_tags = [f for f in failures if not f.get('tags_found', False)]
    print(f"\nFailures without answer tags: {len(no_tags)} / {len(failures)} "
          f"({len(no_tags)/len(failures)*100:.1f}%)")
    
    # Show worst cases
    print(f"\n{'='*70}")
    print("TOP 10 WORST PREDICTIONS")
    print(f"{'='*70}")
    
    sorted_failures = sorted(failures, key=lambda x: x['similarity_score'])[:10]
    
    for i, failure in enumerate(sorted_failures, 1):
        print(f"\n{i}. Similarity: {failure['similarity_score']:.3f}")
        print(f"   Difficulty: {failure.get('difficulty', 'unknown')}")
        print(f"   Original:   {failure['original']}")
        print(f"   Perturbed:  {failure['perturbed']}")
        print(f"   Prediction: {failure['prediction']}")
        print(f"   Tags Found: {failure.get('tags_found', False)}")
        print(f"   Category: {failure['category']}")
        print(f"   Perturbation: {failure['perturbation_type']}")


def analyze_difficulty_progression(results: list):
    """
    Analyze performance across difficulty levels.
    
    Args:
        results: List of result dictionaries
    """
    print(f"\n{'='*70}")
    print("DIFFICULTY PROGRESSION ANALYSIS")
    print(f"{'='*70}")
    
    by_difficulty = defaultdict(list)
    for result in results:
        diff = result.get('difficulty', 'unknown')
        by_difficulty[diff].append(result)
    
    difficulty_order = ['easy', 'medium', 'hard', 'unknown']
    
    for diff in difficulty_order:
        if diff not in by_difficulty:
            continue
        
        diff_results = by_difficulty[diff]
        total = len(diff_results)
        exact_match = sum(1 for r in diff_results if r['exact_match'])
        avg_similarity = sum(r['similarity_score'] for r in diff_results) / total
        tags_found = sum(1 for r in diff_results if r.get('tags_found', False))
        
        print(f"\n{diff.upper()}:")
        print(f"  Samples: {total}")
        print(f"  Exact Match Rate: {exact_match/total:.2%}")
        print(f"  Avg Similarity: {avg_similarity:.4f}")
        print(f"  Tags Found Rate: {tags_found/total:.2%}")


def analyze_tag_usage(results: list):
    """
    Analyze answer tag usage patterns.
    
    Args:
        results: List of result dictionaries
    """
    print(f"\n{'='*70}")
    print("ANSWER TAG USAGE ANALYSIS")
    print(f"{'='*70}")
    
    total = len(results)
    tags_found = sum(1 for r in results if r.get('tags_found', False))
    
    print(f"\nOverall tag usage: {tags_found} / {total} ({tags_found/total:.2%})")
    
    # Analyze correlation between tag usage and accuracy
    with_tags = [r for r in results if r.get('tags_found', False)]
    without_tags = [r for r in results if not r.get('tags_found', False)]
    
    if with_tags:
        with_tags_accuracy = sum(1 for r in with_tags if r['exact_match']) / len(with_tags)
        with_tags_similarity = sum(r['similarity_score'] for r in with_tags) / len(with_tags)
        print(f"\nWith tags ({len(with_tags)} samples):")
        print(f"  Exact Match Rate: {with_tags_accuracy:.2%}")
        print(f"  Avg Similarity: {with_tags_similarity:.4f}")
    
    if without_tags:
        without_tags_accuracy = sum(1 for r in without_tags if r['exact_match']) / len(without_tags)
        without_tags_similarity = sum(r['similarity_score'] for r in without_tags) / len(without_tags)
        print(f"\nWithout tags ({len(without_tags)} samples):")
        print(f"  Exact Match Rate: {without_tags_accuracy:.2%}")
        print(f"  Avg Similarity: {without_tags_similarity:.4f}")
    
    # Analyze by difficulty
    print(f"\nTag usage by difficulty:")
    by_difficulty = defaultdict(lambda: {'total': 0, 'with_tags': 0})
    for result in results:
        diff = result.get('difficulty', 'unknown')
        by_difficulty[diff]['total'] += 1
        if result.get('tags_found', False):
            by_difficulty[diff]['with_tags'] += 1
    
    for diff, stats in sorted(by_difficulty.items()):
        rate = stats['with_tags'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {diff}: {stats['with_tags']} / {stats['total']} ({rate:.2%})")


def compare_models(result_files: list):
    """
    Compare results from multiple models.
    
    Args:
        result_files: List of paths to result JSON files
    """
    all_results = []
    model_names = []
    
    for file_path in result_files:
        results = load_results(file_path)
        model_name = Path(file_path).parent.name
        all_results.append(results)
        model_names.append(model_name)
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}")
    
    for model_name, results in zip(model_names, all_results):
        exact_match = sum(1 for r in results if r['exact_match']) / len(results)
        avg_similarity = sum(r['similarity_score'] for r in results) / len(results)
        tags_found = sum(1 for r in results if r.get('tags_found', False)) / len(results)
        
        print(f"\n{model_name}:")
        print(f"  Exact Match Rate: {exact_match:.2%}")
        print(f"  Avg Similarity: {avg_similarity:.4f}")
        print(f"  Tags Found Rate: {tags_found:.2%}")
        
        # Breakdown by difficulty
        by_difficulty = defaultdict(list)
        for r in results:
            by_difficulty[r.get('difficulty', 'unknown')].append(r)
        
        print(f"  By difficulty:")
        for diff in ['easy', 'medium', 'hard']:
            if diff in by_difficulty:
                diff_results = by_difficulty[diff]
                diff_accuracy = sum(1 for r in diff_results if r['exact_match']) / len(diff_results)
                print(f"    {diff}: {diff_accuracy:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results.json file"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for failure analysis"
    )
    
    parser.add_argument(
        "--compare",
        nargs='+',
        help="Compare multiple result files"
    )
    
    parser.add_argument(
        "--analyze_difficulty",
        action="store_true",
        help="Analyze difficulty progression"
    )
    
    parser.add_argument(
        "--analyze_tags",
        action="store_true",
        help="Analyze answer tag usage"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare)
    else:
        results = load_results(args.results)
        analyze_failures(results, args.threshold)
        
        if args.analyze_difficulty:
            analyze_difficulty_progression(results)
        
        if args.analyze_tags:
            analyze_tag_usage(results)


if __name__ == "__main__":
    main()