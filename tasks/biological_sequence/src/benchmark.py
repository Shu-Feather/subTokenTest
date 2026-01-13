"""
Main benchmark runner that orchestrates the entire evaluation process.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from .data_generator import BiologicalSequenceGenerator
from .prompt_templates import PromptTemplates
from .model_interface import ModelFactory, BaseModelInterface
from .evaluator import SequenceEvaluator, EvaluationResult


class BiologicalSequenceBenchmark:
    """Main benchmark class for biological sequence manipulation tasks."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize benchmark with configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.generator = BiologicalSequenceGenerator()
        self.prompt_templates = PromptTemplates()
        self.evaluator = SequenceEvaluator()
        self.verbose = False  # Verbose mode flag
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()

        # Ensure prompt settings defaults exist
        benchmark_settings = self.config.setdefault("benchmark_settings", {})
        benchmark_settings.setdefault("restricted_reasoning", False)
        self.restricted_reasoning = benchmark_settings.get("restricted_reasoning", False)
    
    def _get_default_config(self) -> Dict:
        """Get default benchmark configuration."""
        return {
            "benchmark_settings": {
                "results_dir": "results",
                "save_raw_responses": True,
                "save_detailed_results": True,
                "restricted_reasoning": False
            },
            "tasks": {
                "dna_complement": {
                    "enabled": True,
                    "num_cases": 50,
                    "sequence_length_range": [8, 15]
                },
                "rna_complement": {
                    "enabled": True,
                    "num_cases": 50,
                    "sequence_length_range": [8, 15]
                },
                "protein_three_to_one": {
                    "enabled": True,
                    "num_cases": 50,
                    "sequence_length_range": [5, 10]
                },
                "protein_one_to_three": {
                    "enabled": True,
                    "num_cases": 50,
                    "sequence_length_range": [5, 10]
                }
            },
            "models": {
                "gpt35": {
                    "provider": "openai",
                    "model_name": "gpt-3.5-turbo",
                    "parameters": {
                        "temperature": 0.1,
                        "max_tokens": 1024
                    }
                }
            }
        }
    
    def generate_test_data(self, task_filter: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Generate test data for all enabled tasks.
        
        Args:
            task_filter: Optional list of specific tasks to generate data for
            
        Returns:
            Dictionary mapping task types to test cases
        """
        test_data = {}
        tasks_config = self.config.get("tasks", {})
        
        for task_type, task_config in tasks_config.items():
            if not task_config.get("enabled", True):
                continue
            
            if task_filter and task_type not in task_filter:
                continue
            
            num_cases = task_config.get("num_cases", 50)
            length_range = tuple(task_config.get("sequence_length_range", [8, 15]))
            
            test_cases = self.generator.generate_test_cases(
                task_type, num_cases, length_range
            )
            test_data[task_type] = test_cases
        
        return test_data
    
    def run_single_model(self, model_interface: BaseModelInterface,
                        test_data: Dict[str, List[Dict]],
                        verbose: bool = True) -> Dict[str, List[EvaluationResult]]:
        """
        Run benchmark on a single model.
        
        Args:
            model_interface: Model interface to test
            test_data: Test data dictionary
            verbose: Whether to show progress bars
            
        Returns:
            Dictionary mapping task types to evaluation results
        """
        all_results = {}
        
        for task_type, test_cases in test_data.items():
            if verbose:
                print(f"\nRunning {task_type} on {model_interface}...")
                pbar = tqdm(test_cases, desc=f"{task_type}")
            else:
                pbar = test_cases
            
            task_results = []
            
            for idx, test_case in enumerate(pbar):
                try:
                    # Generate prompt
                    prompt = self.prompt_templates.get_prompt_for_task(
                        task_type, test_case['input'],
                        restricted_reasoning=self.restricted_reasoning
                    )
                    
                    # Verbose logging: Print prompt
                    if self.verbose:
                        print("\n" + "="*80)
                        print(f"TEST CASE #{idx + 1} - {task_type}")
                        print("="*80)
                        print(f"Input Sequence: {test_case['input']}")
                        print(f"Expected Output: {test_case['expected_output']}")
                        print("\n" + "-"*80)
                        print("PROMPT TO MODEL:")
                        print("-"*80)
                        print(prompt)
                        print("-"*80)
                    
                    # Get model response WITH usage information
                    raw_response, usage_info = model_interface.generate_response(
                        prompt, 
                        verbose=self.verbose
                    )
                    
                    # Verbose logging: Print raw response
                    if self.verbose:
                        print("\nRAW MODEL RESPONSE:")
                        print("-"*80)
                        print(raw_response)
                        print("-"*80)
                        
                        # Print token usage
                        if usage_info:
                            print("\nTOKEN USAGE:")
                            print("-"*80)
                            print(f"  Total Tokens: {usage_info.get('total_tokens', 'N/A')}")
                            print(f"  Prompt Tokens: {usage_info.get('prompt_tokens', 'N/A')}")
                            print(f"  Completion Tokens: {usage_info.get('completion_tokens', 'N/A')}")
                            print(f"  Reasoning Tokens: {usage_info.get('reasoning_tokens', 'N/A')}")
                            print(f"  Output Tokens: {usage_info.get('output_tokens', 'N/A')}")
                            if usage_info.get('total_tokens', 0) > 0:
                                reasoning_ratio = usage_info.get('reasoning_tokens', 0) / usage_info['total_tokens']
                                print(f"  Reasoning Ratio: {reasoning_ratio:.2%}")
                            print("-"*80)
                    
                    # Extract answer from response
                    model_output = self.prompt_templates.extract_answer_from_response(raw_response)
                    
                    # Verbose logging: Print extracted answer
                    if self.verbose:
                        print(f"\nEXTRACTED ANSWER: {model_output}")
                    
                    # Evaluate result
                    result = self.evaluator.evaluate_single_case(
                        test_case, model_output, raw_response
                    )
                    
                    # Add usage information to the result
                    if hasattr(result, 'usage_info'):
                        result.usage_info = usage_info or {}
                    else:
                        # If EvaluationResult doesn't have usage_info attribute, add it dynamically
                        result.usage_info = usage_info or {}
                    
                    # Verbose logging: Print evaluation result
                    if self.verbose:
                        print(f"\nEVALUATION RESULT:")
                        print(f"  Correct: {result.is_correct}")
                        print(f"  Confidence Score: {result.confidence_score:.3f}")
                        if not result.is_correct:
                            print(f"  Error: {result.error_details}")
                        print("="*80 + "\n")
                    
                    task_results.append(result)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    if verbose:
                        print(f"Error processing case: {e}")
                    if self.verbose:
                        print(f"\nERROR: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        print("="*80 + "\n")
                    
                    # Create error result
                    error_result = EvaluationResult(
                        task_type=task_type,
                        input_sequence=test_case['input'],
                        expected_output=test_case['expected_output'],
                        model_output="ERROR",
                        raw_response=str(e),
                        is_correct=False,
                        confidence_score=0.0,
                        error_details=f"Processing error: {e}"
                    )
                    # Add empty usage info for error cases
                    error_result.usage_info = {}
                    task_results.append(error_result)
            
            all_results[task_type] = task_results
        
        return all_results
    
    def aggregate_usage_stats(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Aggregate token usage statistics from evaluation results.
        
        Args:
            results: List of EvaluationResult objects
            
        Returns:
            Dictionary with aggregated usage statistics
        """
        total_stats = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "num_requests": 0,
            "num_successful_requests": 0
        }
        
        for result in results:
            usage_info = getattr(result, 'usage_info', None)
            if usage_info and isinstance(usage_info, dict):
                total_stats["total_tokens"] += usage_info.get("total_tokens", 0)
                total_stats["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
                total_stats["completion_tokens"] += usage_info.get("completion_tokens", 0)
                total_stats["reasoning_tokens"] += usage_info.get("reasoning_tokens", 0)
                total_stats["output_tokens"] += usage_info.get("output_tokens", 0)
                total_stats["num_requests"] += 1
                
                # Count successful requests (with actual token usage)
                if usage_info.get("total_tokens", 0) > 0:
                    total_stats["num_successful_requests"] += 1
        
        # Calculate averages
        if total_stats["num_successful_requests"] > 0:
            divisor = total_stats["num_successful_requests"]
            total_stats["avg_total_tokens"] = total_stats["total_tokens"] / divisor
            total_stats["avg_prompt_tokens"] = total_stats["prompt_tokens"] / divisor
            total_stats["avg_completion_tokens"] = total_stats["completion_tokens"] / divisor
            total_stats["avg_reasoning_tokens"] = total_stats["reasoning_tokens"] / divisor
            total_stats["avg_output_tokens"] = total_stats["output_tokens"] / divisor
            
            # Calculate reasoning ratio
            if total_stats["total_tokens"] > 0:
                total_stats["reasoning_ratio"] = total_stats["reasoning_tokens"] / total_stats["total_tokens"]
            else:
                total_stats["reasoning_ratio"] = 0.0
        else:
            total_stats["avg_total_tokens"] = 0
            total_stats["avg_prompt_tokens"] = 0
            total_stats["avg_completion_tokens"] = 0
            total_stats["avg_reasoning_tokens"] = 0
            total_stats["avg_output_tokens"] = 0
            total_stats["reasoning_ratio"] = 0.0
        
        return total_stats
    
    def run_multiple_models(self, model_configs: Dict[str, Dict],
                           test_data: Optional[Dict[str, List[Dict]]] = None,
                           verbose: bool = True) -> Dict[str, Dict[str, List[EvaluationResult]]]:
        """
        Run benchmark on multiple models.
        
        Args:
            model_configs: Dictionary of model configurations
            test_data: Pre-generated test data (optional)
            verbose: Whether to show progress
            
        Returns:
            Dictionary mapping model names to their results
        """
        if test_data is None:
            if verbose:
                print("Generating test data...")
            test_data = self.generate_test_data()
        
        all_model_results = {}
        
        for model_name, model_config in model_configs.items():
            if verbose:
                print(f"\n{'='*50}")
                print(f"Testing model: {model_name}")
                print(f"{'='*50}")
            
            try:
                # Create model interface
                model_interface = ModelFactory.create_model_interface(
                    model_config['provider'],
                    model_config['model_name'],
                    **model_config.get('parameters', {})
                )
                
                # Run benchmark
                model_results = self.run_single_model(
                    model_interface, test_data, verbose
                )
                
                all_model_results[model_name] = model_results
                
            except Exception as e:
                if verbose:
                    print(f"Failed to test model {model_name}: {e}")
                all_model_results[model_name] = {"error": str(e)}
        
        return all_model_results
    
    def save_results(self, results: Dict[str, Dict[str, List[EvaluationResult]]],
                    timestamp: Optional[str] = None):
        """
        Save benchmark results to files with token usage statistics.
        
        Args:
            results: Benchmark results
            timestamp: Optional timestamp for file naming
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = self.config.get("benchmark_settings", {}).get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Calculate usage statistics for all models
        all_usage_stats = {}
        
        for model_name, model_results in results.items():
            if "error" in model_results:
                continue
            
            model_dir = os.path.join(results_dir, f"{model_name}_{timestamp}")
            os.makedirs(model_dir, exist_ok=True)
            
            # Collect all results for this model
            all_results = []
            task_usage_stats = {}
            
            # Save detailed results CSV for each task
            for task_type, task_results in model_results.items():
                csv_path = os.path.join(model_dir, f"{task_type}_results.csv")
                self.evaluator.save_results_to_csv(task_results, csv_path)
                
                all_results.extend(task_results)
                
                # Calculate usage stats per task
                task_usage_stats[task_type] = self.aggregate_usage_stats(task_results)
            
            # Calculate overall usage stats for this model
            overall_usage_stats = self.aggregate_usage_stats(all_results)
            all_usage_stats[model_name] = {
                "overall": overall_usage_stats,
                "per_task": task_usage_stats
            }
            
            # Save summary report with usage statistics
            report = self.evaluator.generate_detailed_report(all_results)
            report_path = os.path.join(model_dir, "benchmark_report.txt")
            
            with open(report_path, 'w') as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Configuration: {json.dumps(self.config, indent=2)}\n\n")
                f.write("="*80 + "\n")
                f.write("TOKEN USAGE STATISTICS\n")
                f.write("="*80 + "\n\n")
                
                # Overall usage statistics
                f.write("Overall Statistics:\n")
                f.write("-"*80 + "\n")
                f.write(f"  Total Requests: {overall_usage_stats['num_requests']}\n")
                f.write(f"  Successful Requests: {overall_usage_stats['num_successful_requests']}\n")
                f.write(f"  Total Tokens: {overall_usage_stats['total_tokens']:,}\n")
                f.write(f"  Prompt Tokens: {overall_usage_stats['prompt_tokens']:,}\n")
                f.write(f"  Completion Tokens: {overall_usage_stats['completion_tokens']:,}\n")
                f.write(f"  Reasoning Tokens: {overall_usage_stats['reasoning_tokens']:,}\n")
                f.write(f"  Output Tokens: {overall_usage_stats['output_tokens']:,}\n")
                
                if overall_usage_stats['num_successful_requests'] > 0:
                    f.write(f"\nAverage Per Request:\n")
                    f.write(f"  Avg Total Tokens: {overall_usage_stats['avg_total_tokens']:.2f}\n")
                    f.write(f"  Avg Prompt Tokens: {overall_usage_stats['avg_prompt_tokens']:.2f}\n")
                    f.write(f"  Avg Completion Tokens: {overall_usage_stats['avg_completion_tokens']:.2f}\n")
                    f.write(f"  Avg Reasoning Tokens: {overall_usage_stats['avg_reasoning_tokens']:.2f}\n")
                    f.write(f"  Avg Output Tokens: {overall_usage_stats['avg_output_tokens']:.2f}\n")
                    f.write(f"  Reasoning Ratio: {overall_usage_stats['reasoning_ratio']:.2%}\n")
                
                # Per-task usage statistics
                f.write("\n" + "="*80 + "\n")
                f.write("PER-TASK TOKEN USAGE\n")
                f.write("="*80 + "\n\n")
                
                for task_type, task_stats in task_usage_stats.items():
                    f.write(f"{task_type}:\n")
                    f.write("-"*80 + "\n")
                    f.write(f"  Requests: {task_stats['num_successful_requests']}/{task_stats['num_requests']}\n")
                    f.write(f"  Total Tokens: {task_stats['total_tokens']:,}\n")
                    f.write(f"  Prompt Tokens: {task_stats['prompt_tokens']:,}\n")
                    f.write(f"  Completion Tokens: {task_stats['completion_tokens']:,}\n")
                    f.write(f"  Reasoning Tokens: {task_stats['reasoning_tokens']:,}\n")
                    f.write(f"  Output Tokens: {task_stats['output_tokens']:,}\n")
                    
                    if task_stats['num_successful_requests'] > 0:
                        f.write(f"  Avg Total: {task_stats['avg_total_tokens']:.2f}\n")
                        f.write(f"  Reasoning Ratio: {task_stats['reasoning_ratio']:.2%}\n")
                    f.write("\n")
                
                # Original performance report
                f.write("="*80 + "\n")
                f.write("PERFORMANCE REPORT\n")
                f.write("="*80 + "\n\n")
                f.write(report)
            
            # Save metrics as JSON (including usage stats)
            metrics = self.evaluator.calculate_metrics(all_results)
            metrics['usage_statistics'] = all_usage_stats[model_name]
            
            metrics_path = os.path.join(model_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            # Save detailed results with usage info as JSON
            detailed_results_path = os.path.join(model_dir, "detailed_results.json")
            detailed_results = []
            
            for result in all_results:
                result_dict = {
                    "task_type": result.task_type,
                    "input_sequence": result.input_sequence,
                    "expected_output": result.expected_output,
                    "model_output": result.model_output,
                    "raw_response": result.raw_response if self.config.get("benchmark_settings", {}).get("save_raw_responses", True) else None,
                    "is_correct": result.is_correct,
                    "confidence_score": result.confidence_score,
                    "error_details": result.error_details,
                    "usage_info": getattr(result, 'usage_info', {})
                }
                detailed_results.append(result_dict)
            
            with open(detailed_results_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Save cross-model comparison with usage statistics
        if len(results) > 1:
            comparison_path = os.path.join(results_dir, f"model_comparison_{timestamp}.txt")
            with open(comparison_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write("MODEL COMPARISON REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Models Tested: {', '.join([k for k in results.keys() if 'error' not in results[k]])}\n\n")
                
                # Performance comparison
                f.write("="*80 + "\n")
                f.write("ACCURACY COMPARISON\n")
                f.write("="*80 + "\n\n")
                
                for model_name, model_results in results.items():
                    if "error" in model_results:
                        f.write(f"{model_name}: ERROR - {model_results['error']}\n\n")
                        continue
                    
                    all_results = []
                    for task_results in model_results.values():
                        all_results.extend(task_results)
                    
                    total = len(all_results)
                    correct = sum(1 for r in all_results if r.is_correct)
                    accuracy = correct / total if total > 0 else 0
                    
                    f.write(f"{model_name}:\n")
                    f.write(f"  Overall Accuracy: {accuracy:.2%} ({correct}/{total})\n")
                    
                    # Per-task accuracy
                    for task_type, task_results in model_results.items():
                        task_total = len(task_results)
                        task_correct = sum(1 for r in task_results if r.is_correct)
                        task_accuracy = task_correct / task_total if task_total > 0 else 0
                        f.write(f"  {task_type}: {task_accuracy:.2%} ({task_correct}/{task_total})\n")
                    f.write("\n")
                
                # Token usage comparison
                f.write("="*80 + "\n")
                f.write("TOKEN USAGE COMPARISON\n")
                f.write("="*80 + "\n\n")
                
                for model_name, usage_stats in all_usage_stats.items():
                    overall_stats = usage_stats['overall']
                    f.write(f"{model_name}:\n")
                    f.write(f"  Total Tokens: {overall_stats['total_tokens']:,}\n")
                    f.write(f"  Avg Tokens/Request: {overall_stats.get('avg_total_tokens', 0):.2f}\n")
                    f.write(f"  Reasoning Tokens: {overall_stats['reasoning_tokens']:,}\n")
                    f.write(f"  Reasoning Ratio: {overall_stats.get('reasoning_ratio', 0):.2%}\n")
                    f.write("\n")
                
                # Cost estimation (if applicable)
                f.write("="*80 + "\n")
                f.write("EFFICIENCY METRICS\n")
                f.write("="*80 + "\n\n")
                
                for model_name, usage_stats in all_usage_stats.items():
                    if "error" in results[model_name]:
                        continue
                    
                    all_results = []
                    for task_results in results[model_name].values():
                        all_results.extend(task_results)
                    
                    total = len(all_results)
                    correct = sum(1 for r in all_results if r.is_correct)
                    accuracy = correct / total if total > 0 else 0
                    
                    overall_stats = usage_stats['overall']
                    avg_tokens = overall_stats.get('avg_total_tokens', 0)
                    
                    # Tokens per correct answer
                    if correct > 0:
                        tokens_per_correct = overall_stats['total_tokens'] / correct
                    else:
                        tokens_per_correct = float('inf')
                    
                    f.write(f"{model_name}:\n")
                    f.write(f"  Accuracy: {accuracy:.2%}\n")
                    f.write(f"  Avg Tokens/Request: {avg_tokens:.2f}\n")
                    f.write(f"  Tokens/Correct Answer: {tokens_per_correct:.2f}\n")
                    f.write(f"  Efficiency Score: {accuracy / (avg_tokens / 1000) if avg_tokens > 0 else 0:.4f}\n")
                    f.write("\n")
            
            print(f"\nModel comparison saved to: {comparison_path}")
    
    def run_full_benchmark(self, model_configs: Optional[Dict[str, Dict]] = None,
                          save_results: bool = True, verbose: bool = True) -> Dict:
        """
        Run complete benchmark with all configurations.
        
        Args:
            model_configs: Optional model configurations (uses config file if None)
            save_results: Whether to save results to files
            verbose: Whether to show progress
            
        Returns:
            Complete benchmark results including usage statistics
        """
        if model_configs is None:
            model_configs = self.config.get("models", {})
        
        if not model_configs:
            raise ValueError("No models configured for testing")
        
        # Generate test data
        if verbose:
            print("Generating test data...")
        test_data = self.generate_test_data()
        
        # Show test data summary
        if verbose:
            total_cases = sum(len(cases) for cases in test_data.values())
            print(f"Generated {total_cases} test cases across {len(test_data)} tasks")
            for task_type, cases in test_data.items():
                print(f"  {task_type}: {len(cases)} cases")
        
        # Run benchmark on all models
        results = self.run_multiple_models(model_configs, test_data, verbose)
        
        # Calculate usage statistics
        usage_statistics = {}
        for model_name, model_results in results.items():
            if "error" not in model_results:
                all_results = []
                for task_results in model_results.values():
                    all_results.extend(task_results)
                usage_statistics[model_name] = self.aggregate_usage_stats(all_results)
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_results(results, timestamp)
            if verbose:
                print(f"\nResults saved to: {self.config.get('benchmark_settings', {}).get('results_dir', 'results')}")
        
        return {
            "results": results,
            "test_data": test_data,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "usage_statistics": usage_statistics
        }


def main():
    """Example usage of the benchmark."""
    benchmark = BiologicalSequenceBenchmark()
    
    # Example model configuration
    model_configs = {
        "test_model": {
            "provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "parameters": {
                "temperature": 0.1,
                "max_tokens": 512
            }
        }
    }
    
    try:
        results = benchmark.run_full_benchmark(model_configs, verbose=True)
        print("\nBenchmark completed successfully!")
        
        # Print usage summary
        if "usage_statistics" in results:
            print("\n" + "="*60)
            print("TOKEN USAGE SUMMARY")
            print("="*60)
            for model_name, stats in results["usage_statistics"].items():
                print(f"\n{model_name}:")
                print(f"  Total Tokens: {stats['total_tokens']:,}")
                print(f"  Avg Tokens/Request: {stats.get('avg_total_tokens', 0):.2f}")
                if stats.get('reasoning_tokens', 0) > 0:
                    print(f"  Reasoning Tokens: {stats['reasoning_tokens']:,}")
                    print(f"  Reasoning Ratio: {stats.get('reasoning_ratio', 0):.2%}")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
