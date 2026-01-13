"""
Main benchmark implementation
"""

import os
from typing import Dict, Any, List
from tqdm import tqdm

from .utils import load_json, save_json, extract_answer
from .models import BaseModel
from .evaluator import Evaluator


class ContextAwareRedactionBenchmark:
    """
    Context-Aware Redaction Benchmark
    """
    
    def __init__(self, config: Dict[str, Any], model: BaseModel, verbose: bool = False):
        """
        Initialize benchmark
        
        Args:
            config: Configuration dictionary
            model: Model instance to evaluate
            verbose: Whether to print verbose output
        """
        self.config = config
        self.model = model
        self.verbose = verbose
        self.evaluator = Evaluator(verbose=verbose)
        self.restricted_reasoning = self.config.get('prompt', {}).get('restricted_reasoning', False)
    
    def create_prompt(self, original_text: str) -> str:
        """
        Create prompt for the model
        
        Args:
            original_text: Original text containing sensitive information
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a data privacy assistant. Your task is to redact (mask) sensitive information in the given text according to specific rules.

**Redaction Rules:**

1. **18-digit ID Card Number**: Keep the first 6 digits and last 2 digits, mask the middle 10 digits with asterisks (*)
   - Example: 123456789012345678 → 123456**********78

2. **Phone Number** (format: +[area code] [11-digit number]): Keep the '+' sign and area code, keep the first 3 digits and last 4 digits of the phone number, mask the middle 4 digits with asterisks (*)
   - Example: +12 12345678901 → +12 123****8901

3. **Credit Card Number**: Keep the first 6 digits and last 4 digits, mask all middle digits with asterisks (*)
   - Example: 1234 5678 9012 3456 → 123456******3456

**Instructions:**
- Carefully identify all sensitive information in the text (phone numbers, ID card numbers, credit card numbers)
- Apply the appropriate masking rule for each type of sensitive information
- Keep all other text exactly the same
- Place your final redacted text between <answer> and </answer> tags

**Text to redact:**
{original_text}

**Your redacted text (place between <answer></answer> tags):**"""

        if self.restricted_reasoning:
            prompt += (
                "\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
            )

        return prompt
    
    def run_single(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run benchmark on a single sample
        
        Args:
            sample: Test sample
            
        Returns:
            Result dictionary
        """
        # Use original_context from the dataset
        original_text = sample['original_context']
        ground_truth = sample['redacted_context']
        
        # Create prompt
        prompt = self.create_prompt(original_text)
        
        # Get model prediction with usage information
        model_output, usage_info = self.model.generate_with_usage(prompt)
        
        # Extract answer from tags
        prediction = extract_answer(model_output)
        
        # If no answer tags found, use the whole output
        if not prediction:
            prediction = model_output
            if self.verbose:
                print("Warning: No <answer> tags found in model output, using full output")
        
        # Evaluate
        eval_result = self.evaluator.evaluate_single(prediction, ground_truth, original_text)
        
        result = {
            'sample_id': sample['id'],
            'difficulty': sample['difficulty'],
            'original_text': original_text,
            'ground_truth': ground_truth,
            'model_output': model_output,
            'prediction': prediction,
            'evaluation': eval_result,
            'token_usage': usage_info  # Add token usage information
        }
        
        return result
    
    def run(self, dataset_path: str, num_samples: int = None) -> Dict[str, Any]:
        """
        Run benchmark on dataset
        
        Args:
            dataset_path: Path to dataset JSON file
            num_samples: Number of samples to test (None for all)
            
        Returns:
            Results dictionary
        """
        print(f"\nLoading dataset from {dataset_path}...")
        dataset = load_json(dataset_path)
        
        if num_samples:
            dataset = dataset[:num_samples]
        
        print(f"Running benchmark on {len(dataset)} samples...")
        print(f"Model: {self.model.model_name}")
        
        results = []
        
        for sample in tqdm(dataset, desc="Processing samples"):
            try:
                result = self.run_single(sample)
                results.append(result)
            except Exception as e:
                print(f"Error when processing sample: {e}")
        
        # Aggregate results - FIXED: removed incorrect get() calls
        all_predictions = [r['prediction'] for r in results]
        all_ground_truths = [r['ground_truth'] for r in results]
        all_original_texts = [r['original_text'] for r in results]
        
        overall_metrics = self.evaluator.evaluate_batch(
            all_predictions, all_ground_truths, all_original_texts
        )
        
        # Results by difficulty - FIXED: removed incorrect get() call
        difficulties = [r['difficulty'] for r in results]
        difficulty_metrics = self.evaluator.evaluate_by_difficulty(
            overall_metrics['individual_results'], difficulties
        )
        
        # Aggregate token usage statistics
        token_stats = self._aggregate_token_usage(results)
        
        final_results = {
            'model': self.model.model_name,
            'total_samples': len(dataset),
            'overall_metrics': {
                'exact_match_rate': overall_metrics['exact_match_rate'],
                'avg_precision': overall_metrics['avg_precision'],
                'avg_recall': overall_metrics['avg_recall'],
                'avg_f1': overall_metrics['avg_f1'],
                'leakage_rate': overall_metrics['leakage_rate'],
                'avg_leakage_ratio': overall_metrics['avg_leakage_ratio']
            },
            'difficulty_metrics': difficulty_metrics,
            'token_usage_stats': token_stats,  # Add aggregated token usage
            'detailed_results': results
        }
        
        return final_results
    
    def _aggregate_token_usage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate token usage statistics from all results
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Dictionary with aggregated token usage statistics
        """
        total_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_reasoning_tokens = 0
        
        usage_by_difficulty = {}
        
        for result in results:
            usage = result.get('token_usage', {})
            total_tokens += usage.get('total_tokens', 0)
            total_input_tokens += usage.get('input_tokens', 0)
            total_output_tokens += usage.get('output_tokens', 0)
            total_reasoning_tokens += usage.get('reasoning_tokens', 0)
            
            # Aggregate by difficulty
            difficulty = result['difficulty']
            if difficulty not in usage_by_difficulty:
                usage_by_difficulty[difficulty] = {
                    'total_tokens': 0,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'reasoning_tokens': 0,
                    'count': 0
                }
            
            usage_by_difficulty[difficulty]['total_tokens'] += usage.get('total_tokens', 0)
            usage_by_difficulty[difficulty]['input_tokens'] += usage.get('input_tokens', 0)
            usage_by_difficulty[difficulty]['output_tokens'] += usage.get('output_tokens', 0)
            usage_by_difficulty[difficulty]['reasoning_tokens'] += usage.get('reasoning_tokens', 0)
            usage_by_difficulty[difficulty]['count'] += 1
        
        # Calculate averages by difficulty
        for difficulty in usage_by_difficulty:
            count = usage_by_difficulty[difficulty]['count']
            if count > 0:
                usage_by_difficulty[difficulty]['avg_total_tokens'] = \
                    usage_by_difficulty[difficulty]['total_tokens'] / count
                usage_by_difficulty[difficulty]['avg_input_tokens'] = \
                    usage_by_difficulty[difficulty]['input_tokens'] / count
                usage_by_difficulty[difficulty]['avg_output_tokens'] = \
                    usage_by_difficulty[difficulty]['output_tokens'] / count
                usage_by_difficulty[difficulty]['avg_reasoning_tokens'] = \
                    usage_by_difficulty[difficulty]['reasoning_tokens'] / count
        
        num_samples = len(results)
        
        return {
            'total_tokens': total_tokens,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_reasoning_tokens': total_reasoning_tokens,
            'avg_tokens_per_sample': total_tokens / num_samples if num_samples > 0 else 0,
            'avg_input_tokens_per_sample': total_input_tokens / num_samples if num_samples > 0 else 0,
            'avg_output_tokens_per_sample': total_output_tokens / num_samples if num_samples > 0 else 0,
            'avg_reasoning_tokens_per_sample': total_reasoning_tokens / num_samples if num_samples > 0 else 0,
            'by_difficulty': usage_by_difficulty
        }
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print summary of results
        
        Args:
            results: Results dictionary
        """
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {results['model']}")
        print(f"Total Samples: {results['total_samples']}")
        
        # Print token usage statistics if available
        token_stats = results.get('token_usage_stats', {})
        if token_stats and token_stats.get('total_tokens', 0) > 0:
            print(f"\n{'Token Usage Statistics':^80}")
            print(f"{'-'*80}")
            print(f"Total Tokens:           {token_stats['total_tokens']:,}")
            print(f"  Input Tokens:         {token_stats['total_input_tokens']:,}")
            print(f"  Output Tokens:        {token_stats['total_output_tokens']:,}")
            if token_stats['total_reasoning_tokens'] > 0:
                print(f"  Reasoning Tokens:     {token_stats['total_reasoning_tokens']:,}")
            print(f"\nAverage per Sample:")
            print(f"  Total Tokens:         {token_stats['avg_tokens_per_sample']:.1f}")
            print(f"  Input Tokens:         {token_stats['avg_input_tokens_per_sample']:.1f}")
            print(f"  Output Tokens:        {token_stats['avg_output_tokens_per_sample']:.1f}")
            if token_stats['avg_reasoning_tokens_per_sample'] > 0:
                print(f"  Reasoning Tokens:     {token_stats['avg_reasoning_tokens_per_sample']:.1f}")
        
        print(f"\n{'Overall Metrics (Number-Level Binary Classification)':^80}")
        print(f"{'-'*80}")
        
        overall = results['overall_metrics']
        # Get detailed results for additional stats
        detailed = results.get('detailed_results', [])
        if detailed:
            total_numbers = sum(r.get('total_sensitive_items', 0) for r in detailed)
            total_tp = sum(r.get('true_positives', 0) for r in detailed)
            total_fn = sum(r.get('false_negatives', 0) for r in detailed)
            total_fp = sum(r.get('false_positives', 0) for r in detailed)
            
            print(f"Total Sensitive Numbers: {total_numbers}")
            print(f"  True Positives (TP):   {total_tp} (correctly redacted)")
            print(f"  False Negatives (FN):  {total_fn} (missed/incorrect)")
            print(f"  False Positives (FP):  {total_fp}")
            print(f"\nMetrics:")
        
        print(f"Exact Match Rate:    {overall['exact_match_rate']:.2%}")
        print(f"Precision:           {overall['avg_precision']:.4f}")
        print(f"Recall:              {overall['avg_recall']:.4f}")
        print(f"F1 Score:            {overall['avg_f1']:.4f}")
        print(f"\nLeakage Statistics:")
        print(f"Leakage Rate:        {overall['leakage_rate']:.2%} (samples with leakage)")
        print(f"Avg Leakage Ratio:   {overall['avg_leakage_ratio']:.4f} (avg leaked numbers per sample)")
        
        print(f"\n{'Metrics by Difficulty':^80}")
        print(f"{'-'*80}")
        
        for diff, metrics in sorted(results['difficulty_metrics'].items()):
            print(f"\n{diff.upper()} ({metrics['total_samples']} samples, "
                  f"{metrics['total_sensitive_numbers']} numbers):")
            print(f"  Binary Classification:")
            print(f"    TP: {metrics['total_true_positives']}, "
                  f"FN: {metrics['total_false_negatives']}, "
                  f"FP: {metrics['total_false_positives']}")
            print(f"  Exact Match Rate:  {metrics['exact_match_rate']:.2%}")
            print(f"  Precision:         {metrics['avg_precision']:.4f}")
            print(f"  Recall:            {metrics['avg_recall']:.4f}")
            print(f"  F1 Score:          {metrics['avg_f1']:.4f}")
            print(f"  Leakage Rate:      {metrics['leakage_rate']:.2%}")
            print(f"  Leaked Numbers:    {metrics['total_leaked_numbers']}/{metrics['total_sensitive_numbers']}")
            
            # Print token usage by difficulty if available
            if token_stats and 'by_difficulty' in token_stats:
                diff_usage = token_stats['by_difficulty'].get(diff, {})
                if diff_usage.get('total_tokens', 0) > 0:
                    print(f"  Avg Tokens:        {diff_usage.get('avg_total_tokens', 0):.1f}")
                    print(f"    Input:           {diff_usage.get('avg_input_tokens', 0):.1f}")
                    print(f"    Output:          {diff_usage.get('avg_output_tokens', 0):.1f}")
                    if diff_usage.get('avg_reasoning_tokens', 0) > 0:
                        print(f"    Reasoning:       {diff_usage.get('avg_reasoning_tokens', 0):.1f}")
        
        print(f"\n{'='*80}\n")
