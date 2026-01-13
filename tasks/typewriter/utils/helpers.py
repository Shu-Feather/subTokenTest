import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

def save_results(results: Dict[str, Any], output_dir: str = "results"):
    """Save evaluation results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if 'model_info' in results:
        # Single model results
        model_name = results['model_info']['name'].replace('/', '_')
        filename = f"{model_name}_{timestamp}.json"
    else:
        # Multiple model results
        filename = f"benchmark_results_{timestamp}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {filepath}")
    return filepath

def load_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_summary_report(results: Dict[str, Any]) -> str:
    """Create a human-readable summary report with usage info"""
    if 'model_info' in results:
        # Single model results
        return _create_single_model_report(results)
    else:
        # Multiple model results
        return _create_multi_model_report(results)

def _format_usage_info(usage: Dict[str, Any]) -> str:
    """Format usage info for display"""
    if not usage or usage.get('total_samples', 0) == 0:
        return "No usage data available"
    
    report = []
    report.append(f"Total Tokens: {usage.get('total_tokens', 0):,}")
    report.append(f"Input Tokens: {usage.get('input_tokens', 0):,}")
    report.append(f"Output Tokens: {usage.get('output_tokens', 0):,}")
    
    reasoning = usage.get('reasoning_tokens', 0)
    if reasoning > 0:
        report.append(f"Reasoning Tokens: {reasoning:,}")
    
    samples_with_usage = usage.get('samples_with_usage', 0)
    total_samples = usage.get('total_samples', 0)
    report.append(f"Samples with Usage Info: {samples_with_usage}/{total_samples}")
    
    if 'avg_tokens_per_sample' in usage:
        report.append(f"Avg Tokens/Sample: {usage['avg_tokens_per_sample']:.1f}")
        report.append(f"Avg Input/Sample: {usage['avg_input_per_sample']:.1f}")
        report.append(f"Avg Output/Sample: {usage['avg_output_per_sample']:.1f}")
    
    return "\n".join(report)

def _create_single_model_report(results: Dict[str, Any]) -> str:
    """Create summary for single model results with usage info"""
    model_info = results['model_info']
    metrics = results['metrics']
    
    report = f"""
=== TYPEWRITER BENCHMARK RESULTS ===

Model: {model_info['name']} ({model_info['type']})
Evaluation Time: {datetime.fromtimestamp(results['evaluation_timestamp'])}
Prompt Type: {results['prompt_type']}
Supports Usage Tracking: {results.get('supports_usage_tracking', False)}

=== OVERALL PERFORMANCE ===
Overall Accuracy: {metrics['overall_accuracy']:.3f}
Overall Score: {metrics['overall_score']:.3f}

=== TASK 1: TYPEWRITER EFFECT ===
Accuracy: {metrics['task1_metrics']['accuracy']:.3f}
Average Score: {metrics['task1_metrics']['average_score']:.3f}
Total Samples: {metrics['task1_metrics']['total_samples']}

=== TASK 2: BACKSPACE HANDLING ===
Accuracy: {metrics['task2_metrics']['accuracy']:.3f}
Average Score: {metrics['task2_metrics']['average_score']:.3f}
Total Samples: {metrics['task2_metrics']['total_samples']}

=== DETAILED STATISTICS ===
Task 1 - Min/Max Score: {metrics['task1_metrics']['min_score']:.3f} / {metrics['task1_metrics']['max_score']:.3f}
Task 1 - Std Deviation: {metrics['task1_metrics']['std_score']:.3f}
Task 2 - Min/Max Score: {metrics['task2_metrics']['min_score']:.3f} / {metrics['task2_metrics']['max_score']:.3f}
Task 2 - Std Deviation: {metrics['task2_metrics']['std_score']:.3f}
"""
    
    # Add usage information if available
    if 'total_usage' in results:
        usage = results['total_usage']
        if usage.get('samples_with_usage', 0) > 0:
            report += f"""
=== TOKEN USAGE STATISTICS ===
{_format_usage_info(usage)}
"""
    
    return report

def _create_multi_model_report(results: Dict[str, Any]) -> str:
    """Create summary for multiple model results with usage info"""
    report = "\n=== TYPEWRITER BENCHMARK COMPARISON ===\n\n"
    
    # Collect results for comparison
    model_scores = []
    
    for model_name, model_results in results.items():
        if 'error' in model_results:
            report += f"Model {model_name}: ERROR - {model_results['error']}\n"
            continue
        
        for prompt_type, result in model_results.items():
            metrics = result['metrics']
            usage = result.get('total_usage', {})
            
            model_scores.append({
                'model': model_name,
                'prompt_type': prompt_type,
                'overall_accuracy': metrics['overall_accuracy'],
                'overall_score': metrics['overall_score'],
                'task1_accuracy': metrics['task1_metrics']['accuracy'],
                'task2_accuracy': metrics['task2_metrics']['accuracy'],
                'total_tokens': usage.get('total_tokens', 0),
                'avg_tokens': usage.get('avg_tokens_per_sample', 0),
            })
    
    if model_scores:
        # Create comparison table
        df = pd.DataFrame(model_scores)
        report += "=== PERFORMANCE COMPARISON ===\n"
        
        # Performance table
        perf_df = df[['model', 'prompt_type', 'overall_accuracy', 'overall_score', 
                     'task1_accuracy', 'task2_accuracy']]
        report += perf_df.to_string(index=False, float_format='%.3f')
        report += "\n\n"
        
        # Usage table (if available)
        if df['total_tokens'].sum() > 0:
            report += "=== TOKEN USAGE COMPARISON ===\n"
            usage_df = df[['model', 'prompt_type', 'total_tokens', 'avg_tokens']]
            usage_df = usage_df[usage_df['total_tokens'] > 0]  # Only show models with usage data
            if not usage_df.empty:
                report += usage_df.to_string(index=False, float_format='%.1f')
                report += "\n\n"
        
        # Best performers
        best_overall = df.loc[df['overall_accuracy'].idxmax()]
        best_task1 = df.loc[df['task1_accuracy'].idxmax()]
        best_task2 = df.loc[df['task2_accuracy'].idxmax()]
        
        report += f"=== BEST PERFORMERS ===\n"
        report += f"Overall Best: {best_overall['model']} ({best_overall['prompt_type']}) - {best_overall['overall_accuracy']:.3f}\n"
        report += f"Task 1 Best: {best_task1['model']} ({best_task1['prompt_type']}) - {best_task1['task1_accuracy']:.3f}\n"
        report += f"Task 2 Best: {best_task2['model']} ({best_task2['prompt_type']}) - {best_task2['task2_accuracy']:.3f}\n"
        
        # Most efficient (if usage data available)
        if df['total_tokens'].sum() > 0:
            efficient_df = df[df['total_tokens'] > 0].copy()
            if not efficient_df.empty:
                # Calculate efficiency: accuracy per 1000 tokens
                efficient_df['efficiency'] = efficient_df['overall_accuracy'] / (efficient_df['total_tokens'] / 1000)
                most_efficient = efficient_df.loc[efficient_df['efficiency'].idxmax()]
                report += f"Most Efficient: {most_efficient['model']} ({most_efficient['prompt_type']}) - "
                report += f"{most_efficient['overall_accuracy']:.3f} accuracy with {most_efficient['total_tokens']:,} tokens\n"
    
    return report

def setup_environment():
    """Set up environment and check dependencies"""
    required_dirs = ['results', 'data', 'datasets']
    
    for dir_name in required_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Check if .env file exists for API keys
    if not os.path.exists('.env'):
        print("Warning: .env file not found. Please create one with your API keys:")
        print("OPENAI_API_KEY=your_openai_key")
        print("DEEPSEEK_API_KEY=your_deepseek_key")

def format_error_message(error: Exception) -> str:
    """Format error message for logging"""
    return f"{type(error).__name__}: {str(error)}"

def print_usage_summary(results: Dict[str, Any]):
    """Print a quick usage summary"""
    if 'total_usage' not in results:
        print("No usage data available")
        return
    
    usage = results['total_usage']
    if usage.get('samples_with_usage', 0) == 0:
        print("No usage data available")
        return
    
    print("\n" + "="*60)
    print("TOKEN USAGE SUMMARY")
    print("="*60)
    print(f"Total Tokens Used: {usage.get('total_tokens', 0):,}")
    print(f"  - Input Tokens: {usage.get('input_tokens', 0):,}")
    print(f"  - Output Tokens: {usage.get('output_tokens', 0):,}")
    
    reasoning = usage.get('reasoning_tokens', 0)
    if reasoning > 0:
        print(f"  - Reasoning Tokens: {reasoning:,}")
    
    if 'avg_tokens_per_sample' in usage:
        print(f"\nAverage per Sample: {usage['avg_tokens_per_sample']:.1f} tokens")
    
    print("="*60 + "\n")