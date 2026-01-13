"""
Compare results from multiple evaluation runs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from pathlib import Path
import pandas as pd

def load_result_file(filepath):
    """Load a result JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_metrics(results):
    """Extract key metrics from results"""
    if 'model_info' in results:
        # Single model result
        return {
            'model': results['model_info']['name'],
            'prompt_type': results.get('prompt_type', 'unknown'),
            'overall_accuracy': results['metrics']['overall_accuracy'],
            'overall_score': results['metrics']['overall_score'],
            'task1_accuracy': results['metrics']['task1_metrics']['accuracy'],
            'task2_accuracy': results['metrics']['task2_metrics']['accuracy'],
        }
    else:
        # Multi-model result
        metrics_list = []
        for model_name, model_data in results.items():
            if 'error' in model_data:
                continue
            for prompt_type, result in model_data.items():
                metrics_list.append({
                    'model': model_name,
                    'prompt_type': prompt_type,
                    'overall_accuracy': result['metrics']['overall_accuracy'],
                    'overall_score': result['metrics']['overall_score'],
                    'task1_accuracy': result['metrics']['task1_metrics']['accuracy'],
                    'task2_accuracy': result['metrics']['task2_metrics']['accuracy'],
                })
        return metrics_list

def compare_results(result_files):
    """Compare multiple result files"""
    all_metrics = []
    
    for filepath in result_files:
        print(f"Loading: {filepath}")
        results = load_result_file(filepath)
        metrics = extract_metrics(results)
        
        if isinstance(metrics, list):
            all_metrics.extend(metrics)
        else:
            all_metrics.append(metrics)
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\nBest Overall Accuracy:")
    best_overall = df.loc[df['overall_accuracy'].idxmax()]
    print(f"  {best_overall['model']} ({best_overall['prompt_type']}): {best_overall['overall_accuracy']:.3f}")
    
    print("\nBest Task 1 Accuracy:")
    best_task1 = df.loc[df['task1_accuracy'].idxmax()]
    print(f"  {best_task1['model']} ({best_task1['prompt_type']}): {best_task1['task1_accuracy']:.3f}")
    
    print("\nBest Task 2 Accuracy:")
    best_task2 = df.loc[df['task2_accuracy'].idxmax()]
    print(f"  {best_task2['model']} ({best_task2['prompt_type']}): {best_task2['task2_accuracy']:.3f}")
    
    # Model ranking
    print("\n" + "="*80)
    print("MODEL RANKING (by overall accuracy)")
    print("="*80)
    ranked = df.sort_values('overall_accuracy', ascending=False)
    for idx, row in ranked.iterrows():
        print(f"{row['model']:20s} ({row['prompt_type']:10s}): {row['overall_accuracy']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument('files', nargs='+', help='Result JSON files to compare')
    
    args = parser.parse_args()
    
    # Verify files exist
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
    
    compare_results(args.files)

if __name__ == "__main__":
    main()