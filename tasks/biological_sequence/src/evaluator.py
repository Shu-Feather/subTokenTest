"""
Evaluator for comparing model outputs with expected results.
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    """Container for evaluation results of a single test case."""
    task_type: str
    input_sequence: str
    expected_output: str
    model_output: str
    raw_response: str
    is_correct: bool
    confidence_score: float
    error_details: str = ""
    
    # Token usage information
    usage_info: Dict[str, Any] = field(default_factory=dict)


class SequenceEvaluator:
    """Evaluator for biological sequence manipulation tasks."""
    
    def __init__(self):
        self.evaluation_functions = {
            'dna_complement': self._evaluate_exact_match,
            'rna_complement': self._evaluate_exact_match,
            'protein_three_to_one': self._evaluate_exact_match,
            'protein_one_to_three': self._evaluate_exact_match
        }
    
    def _normalize_sequence(self, sequence: str, task_type: str) -> str:
        """
        Normalize sequence for comparison.
        
        Args:
            sequence: Raw sequence string
            task_type: Type of task for specific normalization
            
        Returns:
            Normalized sequence string
        """
        # Remove common formatting and whitespace
        sequence = sequence.strip().upper()
        
        # Remove quotes if present
        sequence = sequence.strip('"\'')
        
        # For protein sequences, handle different formatting
        if 'protein' in task_type:
            if 'three_to_one' in task_type:
                # Input should be three-letter (with hyphens), output should be one-letter (no hyphens)
                if '-' in sequence:
                    # This is likely the input format (three-letter with hyphens)
                    sequence = re.sub(r'\s*-\s*', '-', sequence)
                else:
                    # This might be the output format (one-letter without separators)
                    # Remove any spaces
                    sequence = re.sub(r'\s+', '', sequence)
            elif 'one_to_three' in task_type:
                # Input should be one-letter (no separators), output should be three-letter (with hyphens)
                if '-' in sequence:
                    # This is likely the output format (three-letter with hyphens)
                    sequence = re.sub(r'\s*-\s*', '-', sequence)
                else:
                    # This is likely the input format (one-letter without separators)
                    sequence = re.sub(r'\s+', '', sequence)
            else:
                # General protein sequence normalization
                if '-' in sequence:
                    sequence = re.sub(r'\s*-\s*', '-', sequence)
                else:
                    # Handle cases where spaces might be used instead of no separators
                    # Only remove spaces if it results in valid single amino acid letters
                    no_space_seq = re.sub(r'\s+', '', sequence)
                    if len(no_space_seq) > 0 and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in no_space_seq):
                        sequence = no_space_seq
        
        # For nucleic acid sequences, remove spaces and other non-base characters
        elif 'dna' in task_type or 'rna' in task_type:
            sequence = re.sub(r'[^ATCGU]', '', sequence)
        
        return sequence
    
    def _evaluate_exact_match(self, expected: str, actual: str, task_type: str) -> Tuple[bool, float, str]:
        """
        Evaluate using exact string matching.
        
        Args:
            expected: Expected output sequence
            actual: Model's output sequence
            task_type: Type of task
            
        Returns:
            Tuple of (is_correct, confidence_score, error_details)
        """
        expected_norm = self._normalize_sequence(expected, task_type)
        actual_norm = self._normalize_sequence(actual, task_type)
        
        is_correct = expected_norm == actual_norm
        confidence_score = 1.0 if is_correct else self._calculate_similarity_score(expected_norm, actual_norm)
        
        error_details = ""
        if not is_correct:
            error_details = f"Expected: '{expected_norm}', Got: '{actual_norm}'"
            
        return is_correct, confidence_score, error_details
    
    def _calculate_similarity_score(self, expected: str, actual: str) -> float:
        """
        Calculate similarity score between expected and actual sequences.
        
        Args:
            expected: Expected sequence
            actual: Actual sequence
            
        Returns:
            Similarity score between 0 and 1
        """
        if not expected and not actual:
            return 1.0  # Both empty sequences are identical
        
        if not expected or not actual:
            return 0.0  # One empty, one non-empty
        
        # Calculate character-level accuracy
        max_len = max(len(expected), len(actual))
        if max_len == 0:
            return 1.0
        
        matches = sum(1 for i in range(min(len(expected), len(actual))) 
                     if expected[i] == actual[i])
        
        # Penalize length differences
        length_penalty = abs(len(expected) - len(actual)) / max_len
        similarity = matches / max_len - length_penalty
        
        return max(0.0, similarity)
    
    def evaluate_single_case(self, test_case: Dict, model_output: str, raw_response: str,
                           usage_info: Dict[str, Any] = None) -> EvaluationResult:
        """
        Evaluate a single test case.
        
        Args:
            test_case: Test case dictionary with expected output
            model_output: Model's cleaned output
            raw_response: Model's raw response
            usage_info: Optional token usage information
            
        Returns:
            EvaluationResult object with usage information
        """
        task_type = test_case['task_type']
        expected_output = test_case['expected_output']
        input_sequence = test_case['input']
        
        # Get evaluation function for this task type
        eval_func = self.evaluation_functions.get(task_type, self._evaluate_exact_match)
        
        # Evaluate the output
        is_correct, confidence_score, error_details = eval_func(
            expected_output, model_output, task_type
        )
        
        return EvaluationResult(
            task_type=task_type,
            input_sequence=input_sequence,
            expected_output=expected_output,
            model_output=model_output,
            raw_response=raw_response,
            is_correct=is_correct,
            confidence_score=confidence_score,
            error_details=error_details,
            usage_info=usage_info or {}
        )
    
    def evaluate_batch(self, test_cases: List[Dict], model_outputs: List[str], 
                      raw_responses: List[str], usage_infos: List[Dict[str, Any]] = None) -> List[EvaluationResult]:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of test case dictionaries
            model_outputs: List of model outputs
            raw_responses: List of raw model responses
            usage_infos: Optional list of token usage information
            
        Returns:
            List of EvaluationResult objects
        """
        if len(test_cases) != len(model_outputs) != len(raw_responses):
            raise ValueError("Length mismatch between test cases, outputs, and responses")
        
        if usage_infos is None:
            usage_infos = [{}] * len(test_cases)
        
        results = []
        for test_case, output, raw_response, usage_info in zip(test_cases, model_outputs, raw_responses, usage_infos):
            result = self.evaluate_single_case(test_case, output, raw_response, usage_info)
            results.append(result)
        
        return results
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Calculate various evaluation metrics from results, including token usage statistics.
        
        Args:
            results: List of EvaluationResult objects
            
        Returns:
            Dictionary containing various metrics including token usage
        """
        if not results:
            return {}
        
        # Overall metrics
        total_cases = len(results)
        correct_cases = sum(1 for r in results if r.is_correct)
        accuracy = correct_cases / total_cases
        
        # Calculate average confidence score
        avg_confidence = np.mean([r.confidence_score for r in results])
        
        # Aggregate token usage statistics
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        output_tokens = 0
        requests_with_usage = 0
        
        for r in results:
            if r.usage_info and isinstance(r.usage_info, dict):
                total_tokens += r.usage_info.get('total_tokens', 0)
                prompt_tokens += r.usage_info.get('prompt_tokens', 0)
                completion_tokens += r.usage_info.get('completion_tokens', 0)
                reasoning_tokens += r.usage_info.get('reasoning_tokens', 0)
                output_tokens += r.usage_info.get('output_tokens', 0)
                if r.usage_info.get('total_tokens', 0) > 0:
                    requests_with_usage += 1
        
        # Calculate averages
        usage_stats = {
            'total_tokens': total_tokens,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'reasoning_tokens': reasoning_tokens,
            'output_tokens': output_tokens,
            'num_requests_with_usage': requests_with_usage
        }
        
        if requests_with_usage > 0:
            usage_stats['avg_total_tokens'] = total_tokens / requests_with_usage
            usage_stats['avg_prompt_tokens'] = prompt_tokens / requests_with_usage
            usage_stats['avg_completion_tokens'] = completion_tokens / requests_with_usage
            usage_stats['avg_reasoning_tokens'] = reasoning_tokens / requests_with_usage
            usage_stats['avg_output_tokens'] = output_tokens / requests_with_usage
            
            if total_tokens > 0:
                usage_stats['reasoning_ratio'] = reasoning_tokens / total_tokens
            else:
                usage_stats['reasoning_ratio'] = 0.0
        
        # Per-task metrics
        task_metrics = {}
        tasks = set(r.task_type for r in results)
        
        for task in tasks:
            task_results = [r for r in results if r.task_type == task]
            task_correct = sum(1 for r in task_results if r.is_correct)
            task_total = len(task_results)
            task_accuracy = task_correct / task_total if task_total > 0 else 0
            task_avg_confidence = np.mean([r.confidence_score for r in task_results])
            
            # Task-level token usage
            task_total_tokens = sum(r.usage_info.get('total_tokens', 0) for r in task_results if r.usage_info)
            task_requests_with_usage = sum(1 for r in task_results if r.usage_info and r.usage_info.get('total_tokens', 0) > 0)
            
            task_usage = {
                'total_tokens': task_total_tokens,
                'avg_tokens': task_total_tokens / task_requests_with_usage if task_requests_with_usage > 0 else 0
            }
            
            task_metrics[task] = {
                'accuracy': task_accuracy,
                'correct': task_correct,
                'total': task_total,
                'avg_confidence': task_avg_confidence,
                'usage': task_usage
            }
        
        return {
            'overall': {
                'accuracy': accuracy,
                'correct': correct_cases,
                'total': total_cases,
                'avg_confidence': avg_confidence
            },
            'usage_statistics': usage_stats,
            'by_task': task_metrics
        }
    
    def generate_detailed_report(self, results: List[EvaluationResult]) -> str:
        """
        Generate a detailed text report from evaluation results, including token usage.
        
        Args:
            results: List of EvaluationResult objects
            
        Returns:
            Formatted report string
        """
        if not results:
            return "No results to report."
        
        metrics = self.calculate_metrics(results)
        
        report = []
        report.append("="*60)
        report.append("BIOLOGICAL SEQUENCE MANIPULATION BENCHMARK RESULTS")
        report.append("="*60)
        
        # Overall results
        overall = metrics['overall']
        report.append(f"\nOVERALL PERFORMANCE:")
        report.append(f"Accuracy: {overall['accuracy']:.2%} ({overall['correct']}/{overall['total']})")
        report.append(f"Average Confidence: {overall['avg_confidence']:.3f}")
        
        # Token usage statistics
        if 'usage_statistics' in metrics:
            usage = metrics['usage_statistics']
            if usage.get('num_requests_with_usage', 0) > 0:
                report.append(f"\nTOKEN USAGE STATISTICS:")
                report.append(f"Total Tokens: {usage['total_tokens']:,}")
                report.append(f"  Prompt Tokens: {usage['prompt_tokens']:,}")
                report.append(f"  Completion Tokens: {usage['completion_tokens']:,}")
                report.append(f"  Reasoning Tokens: {usage['reasoning_tokens']:,}")
                report.append(f"  Output Tokens: {usage['output_tokens']:,}")
                report.append(f"\nAverage per Request:")
                report.append(f"  Total: {usage.get('avg_total_tokens', 0):.2f}")
                report.append(f"  Prompt: {usage.get('avg_prompt_tokens', 0):.2f}")
                report.append(f"  Completion: {usage.get('avg_completion_tokens', 0):.2f}")
                report.append(f"  Reasoning: {usage.get('avg_reasoning_tokens', 0):.2f}")
                report.append(f"  Reasoning Ratio: {usage.get('reasoning_ratio', 0):.2%}")
        
        # Per-task results
        report.append(f"\nPER-TASK BREAKDOWN:")
        for task_type, task_metrics in metrics['by_task'].items():
            task_name = task_type.replace('_', ' ').title()
            report.append(f"\n{task_name}:")
            report.append(f"  Accuracy: {task_metrics['accuracy']:.2%} ({task_metrics['correct']}/{task_metrics['total']})")
            report.append(f"  Avg Confidence: {task_metrics['avg_confidence']:.3f}")
            
            if 'usage' in task_metrics and task_metrics['usage']['total_tokens'] > 0:
                report.append(f"  Total Tokens: {task_metrics['usage']['total_tokens']:,}")
                report.append(f"  Avg Tokens/Request: {task_metrics['usage']['avg_tokens']:.2f}")
        
        # Error analysis
        errors = [r for r in results if not r.is_correct]
        if errors:
            report.append(f"\nERROR ANALYSIS ({len(errors)} errors):")
            error_by_task = {}
            for error in errors:
                task = error.task_type
                if task not in error_by_task:
                    error_by_task[task] = []
                error_by_task[task].append(error)
            
            for task_type, task_errors in error_by_task.items():
                task_name = task_type.replace('_', ' ').title()
                report.append(f"\n{task_name} Errors ({len(task_errors)}):")
                for i, error in enumerate(task_errors[:3]):  # Show first 3 errors
                    report.append(f"  Example {i+1}:")
                    report.append(f"    Input: {error.input_sequence}")
                    report.append(f"    {error.error_details}")
                    
                    # Show token usage for this error case if available
                    if error.usage_info and error.usage_info.get('total_tokens', 0) > 0:
                        report.append(f"    Tokens used: {error.usage_info['total_tokens']}")
                
                if len(task_errors) > 3:
                    report.append(f"    ... and {len(task_errors) - 3} more errors")
        
        return "\n".join(report)
    
    def save_results_to_csv(self, results: List[EvaluationResult], filepath: str):
        """
        Save evaluation results to CSV file, including token usage information.
        
        Args:
            results: List of EvaluationResult objects
            filepath: Path to save CSV file
        """
        data = []
        for result in results:
            row = {
                'task_type': result.task_type,
                'input_sequence': result.input_sequence,
                'expected_output': result.expected_output,
                'model_output': result.model_output,
                'is_correct': result.is_correct,
                'confidence_score': result.confidence_score,
                'error_details': result.error_details
            }
            
            # Add token usage columns if available
            if result.usage_info and isinstance(result.usage_info, dict):
                row['total_tokens'] = result.usage_info.get('total_tokens', 0)
                row['prompt_tokens'] = result.usage_info.get('prompt_tokens', 0)
                row['completion_tokens'] = result.usage_info.get('completion_tokens', 0)
                row['reasoning_tokens'] = result.usage_info.get('reasoning_tokens', 0)
                row['output_tokens'] = result.usage_info.get('output_tokens', 0)
            else:
                row['total_tokens'] = 0
                row['prompt_tokens'] = 0
                row['completion_tokens'] = 0
                row['reasoning_tokens'] = 0
                row['output_tokens'] = 0
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)


def main():
    """Example usage of the evaluator."""
    evaluator = SequenceEvaluator()
    
    # Example test cases
    test_cases = [
        {
            'task_type': 'dna_complement',
            'input': 'ATCG',
            'expected_output': 'TAGC'
        },
        {
            'task_type': 'protein_three_to_one',
            'input': 'GLY-ARG-PHE',
            'expected_output': 'GRF'
        }
    ]
    
    # Example model outputs (simulating correct and incorrect responses)
    model_outputs = ['TAGC', 'GRG']  # Second one is wrong
    raw_responses = [
        '<ANSWER>TAGC</ANSWER>',
        'The answer is GRG'
    ]
    
    # Example usage information (simulating API responses)
    usage_infos = [
        {
            'total_tokens': 150,
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'reasoning_tokens': 0,
            'output_tokens': 50
        },
        {
            'total_tokens': 200,
            'prompt_tokens': 120,
            'completion_tokens': 80,
            'reasoning_tokens': 30,
            'output_tokens': 50
        }
    ]
    
    # Evaluate
    results = evaluator.evaluate_batch(test_cases, model_outputs, raw_responses, usage_infos)
    
    # Generate report
    report = evaluator.generate_detailed_report(results)
    print(report)
    
    # Show metrics
    metrics = evaluator.calculate_metrics(results)
    print("\n" + "="*60)
    print("METRICS:")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    import json
    main()