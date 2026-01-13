import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import random
from typing import Optional

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.tree.config import Config
from data.data_manager import DataManager
from models.model_interface import ModelInterface
from evaluation.evaluator import Evaluator


def resolve_path(path_str: str) -> str:
    """Resolve path relative to CWD or repository root."""
    path = Path(path_str)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
        relative = Path(*path.parts[1:]) if len(path.parts) > 1 else Path(path.name)
    else:
        relative = path

    candidates.extend([
        Path(__file__).resolve().parent / relative,
        Path(__file__).resolve().parent / "datasets" / relative.name,
        PROJECT_ROOT / relative,
        PROJECT_ROOT / "test" / "datasets" / relative.name,
        relative,
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Binary Tree Structure Benchmark")
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=["openai", "deepseek", "vllm"], 
                       help="Type of model to use")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Name/path of the model")
    parser.add_argument("--task_type", type=str, default="both",
                       choices=["task1", "task2", "both"],
                       help="Type of task to test: task1 (structure questions), task2 (structure completion), or both")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of test samples to generate")
    parser.add_argument("--max_depth", type=int, default=4,
                       help="Maximum depth of generated trees")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed conversation with LLM")
    parser.add_argument("--api_key", type=str, default=None,
                       help="API key for closed-source models")
    parser.add_argument("--base_url", type=str, default=None,
                       help="Base URL for API models")
    parser.add_argument("--test_file", type=str, default=None,
                       help="Load test cases from JSON file instead of generating")
    parser.add_argument("--restricted_reasoning", action="store_true",
                       help="Use restricted thinking prompts that encourage minimal reasoning")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None,
                       help="GPU memory fraction for vLLM models (0-1, overrides default)")
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                       help="Tensor parallel size for vLLM models (>=1, overrides default)")
    parser.add_argument("--enforce_eager", dest="enforce_eager", action="store_true",
                       help="Force vLLM eager mode (overrides default)")
    parser.add_argument("--no_enforce_eager", dest="enforce_eager", action="store_false",
                       help="Disable vLLM eager mode (overrides default)")
    parser.set_defaults(enforce_eager=None)
    
    args = parser.parse_args()
    if args.test_file:
        args.test_file = resolve_path(args.test_file)
    args.output_dir = resolve_path(args.output_dir)
    
    # Initialize config
    config = Config(
        model_type=args.model_type,
        model_name=args.model_name,
        task_type=args.task_type,
        num_samples=args.num_samples,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        verbose=args.verbose,
        api_key=args.api_key,
        base_url=args.base_url,
        restricted_reasoning=args.restricted_reasoning
    )
    if args.gpu_memory_utilization is not None:
        if not 0 < args.gpu_memory_utilization <= 1:
            raise ValueError("--gpu_memory_utilization must be between 0 and 1")
        config.vllm_gpu_memory_utilization = args.gpu_memory_utilization
    if args.tensor_parallel_size is not None:
        if args.tensor_parallel_size < 1:
            raise ValueError("--tensor_parallel_size must be >= 1")
        config.vllm_tensor_parallel_size = args.tensor_parallel_size
    if args.enforce_eager is not None:
        config.vllm_enforce_eager = args.enforce_eager
    
    if args.verbose:
        print(f"Starting Binary Tree Benchmark with config: {config.__dict__}")
        print(f"Testing task type: {args.task_type}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    # data_manager = DataManager(config)
    model_interface = ModelInterface(config)
    evaluator = Evaluator(config)
    
    # Load or generate test data
    if args.test_file:
        test_data = load_test_data_from_file(args.test_file, args.task_type, args.num_samples, args.verbose)
        test_data_path = args.test_file
    else:
        data_manager = DataManager(config)
        if args.verbose:
            print("Generating test data...")
        test_data = data_manager.generate_test_data()
        
        # Save test data
        test_data_path = os.path.join(args.output_dir, f"test_data_{args.task_type}.json")
        data_manager.save_test_data(test_data, test_data_path)
    
    if args.verbose:
        print(f"Test data loaded from: {test_data_path}")
        print(f"Total available samples: {len(test_data)}")
        print(f"Using {len(test_data)} samples for evaluation")
    
    # Run evaluation
    if args.verbose:
        print("Starting evaluation...")
    
    results = {
        "task1_results": [],
        "task2_results": [],
        "task1_accuracy": 0.0,
        "task2_accuracy": 0.0,
        "task2_avg_similarity": 0.0,
        "total_samples": len(test_data),
        "task_type_tested": args.task_type,
        "test_data_source": args.test_file if args.test_file else "generated",
        "token_usage_summary": {
            "total_tokens": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_tokens": 0,
            "supports_token_tracking": args.model_type in ["openai", "deepseek"]
        }
    }
    
    task1_correct = 0
    task2_correct = 0
    task2_similarity_scores = []
    total_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0
    
    for i, sample in enumerate(test_data):
        if args.verbose:
            print(f"\nProcessing sample {i+1}/{len(test_data)}")
            print(f"Task type: {sample['task_type']}")
            if 'difficulty' in sample:
                print(f"Difficulty: {sample['difficulty']}")

        # Get model response with token usage
        response, usage_info = get_model_response_with_usage(model_interface, sample["prompt"], args.verbose)
        
        if args.verbose:
            print("=" * 80)
            print("PROMPT:")
            print(sample["prompt"])
            print("\n" + "-" * 40)
            print("EXPECTED RESPONSE:")
            print(sample["expected_answer"])
            print("\n" + "-" * 40)
            print("MODEL RESPONSE:")
            print(response)
            
            # Print token usage if available
            if usage_info:
                print("\n" + "-" * 40)
                print("TOKEN USAGE:")
                print(f"Input tokens: {usage_info.get('input_tokens', 'N/A')}")
                print(f"Output tokens: {usage_info.get('output_tokens', 'N/A')}")
                print(f"Total tokens: {usage_info.get('total_tokens', 'N/A')}")
                if usage_info.get('reasoning_tokens', 0) > 0:
                    print(f"Reasoning tokens: {usage_info['reasoning_tokens']}")
            
            print("=" * 80)
        
        # Evaluate response with similarity
        try:
            is_correct, similarity = evaluator.evaluate_response_with_similarity(
                sample["task_type"], 
                response, 
                sample["expected_answer"],
                sample.get("tree_structure")
            )
        except:
            is_correct = None
            similarity = None
        
        result = {
            "sample_id": i,
            "task_type": sample["task_type"],
            "prompt": sample["prompt"],
            "expected_answer": sample["expected_answer"],
            "model_response": response,
            "is_correct": is_correct,
            "similarity": similarity
        }

        # Add token usage information if available
        if usage_info:
            result["token_usage"] = usage_info
            
            # Update totals
            total_tokens += usage_info.get('total_tokens', 0)
            total_input_tokens += usage_info.get('input_tokens', 0)
            total_output_tokens += usage_info.get('output_tokens', 0)
            total_reasoning_tokens += usage_info.get('reasoning_tokens', 0)
        else:
            result["token_usage"] = None

        # Include additional metadata if available
        if 'difficulty' in sample:
            result["difficulty"] = sample["difficulty"]
        if 'question_type' in sample:
            result["question_type"] = sample["question_type"]
        if 'tree_depth' in sample:
            result["tree_depth"] = sample["tree_depth"]
        
        if sample["task_type"] == "task1":
            results["task1_results"].append(result)
            if is_correct:
                task1_correct += 1
        else:
            results["task2_results"].append(result)
            if similarity is not None:
                task2_similarity_scores.append(similarity)
            if is_correct:
                task2_correct += 1
        
        if args.verbose:
            print(f"Evaluation result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            if sample["task_type"] == "task2":
                if similarity is not None:
                    print(f"Similarity score: {similarity:.3f}")
                else:
                    print(f"Similarity score: N/A (evaluation failed)")
            if not is_correct:
                # Show more detailed comparison for incorrect answers
                extracted_answer = evaluator._extract_answer_for_verbose(response, sample["task_type"])
                print(f"Extracted from model:\n{extracted_answer}")
                print(f"Expected:\n{sample['expected_answer']}")
    
    # Update token usage summary
    results["token_usage_summary"].update({
        "total_tokens": total_tokens,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "average_tokens_per_sample": total_tokens / len(test_data) if test_data else 0,
        "average_input_tokens_per_sample": total_input_tokens / len(test_data) if test_data else 0,
        "average_output_tokens_per_sample": total_output_tokens / len(test_data) if test_data else 0,
    })
    
    # Calculate accuracies
    task1_total = len(results["task1_results"])
    task2_total = len(results["task2_results"])
    
    if task1_total > 0:
        results["task1_accuracy"] = task1_correct / task1_total
    if task2_total > 0:
        results["task2_accuracy"] = task2_correct / task2_total
        if task2_similarity_scores:  
            results["task2_avg_similarity"] = sum(task2_similarity_scores) / len(task2_similarity_scores)
        else:
            results["task2_avg_similarity"] = 0.0

    # Calculate difficulty-based accuracies for task1 if available
    if task1_total > 0 and any('difficulty' in r for r in results["task1_results"]):
        difficulty_stats = calculate_difficulty_stats(results["task1_results"])
        results["task1_difficulty_stats"] = difficulty_stats
    
    # Calculate token usage by task type and difficulty
    if total_tokens > 0:
        token_stats = calculate_token_usage_stats(results["task1_results"] + results["task2_results"])
        results["token_usage_stats"] = token_stats
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_suffix = "loaded" if args.test_file else "generated"
    results_path = os.path.join(args.output_dir, f"results_{args.task_type}_{source_suffix}_{timestamp}.json")

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\nEvaluation completed!")
    print(f"Task type tested: {args.task_type}")
    print(f"Test data source: {'Loaded from file' if args.test_file else 'Generated'}")
    
    if args.task_type in ["task1", "both"] and task1_total > 0:
        print(f"Task 1 (Structure Questions) Accuracy: {results['task1_accuracy']:.3f} ({task1_correct}/{task1_total})")
        
        # Print difficulty breakdown if available
        if 'task1_difficulty_stats' in results:
            print("Task 1 Accuracy by Difficulty:")
            for difficulty, stats in results['task1_difficulty_stats'].items():
                print(f"  {difficulty}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
    
    if args.task_type in ["task2", "both"] and task2_total > 0:
        print(f"Task 2 (Path Analysis) Accuracy: {results['task2_accuracy']:.3f} ({task2_correct}/{task2_total})")
        print(f"Task 2 Average Similarity: {results['task2_avg_similarity']:.3f}")
    
    if args.task_type == "both":
        overall_correct = task1_correct + task2_correct
        overall_total = task1_total + task2_total
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        print(f"Overall Accuracy: {overall_accuracy:.3f} ({overall_correct}/{overall_total})")
    
    # Print token usage summary
    if total_tokens > 0:
        print(f"\nToken Usage Summary:")
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Total input tokens: {total_input_tokens:,}")
        print(f"Total output tokens: {total_output_tokens:,}")
        if total_reasoning_tokens > 0:
            print(f"Total reasoning tokens: {total_reasoning_tokens:,}")
        print(f"Average tokens per sample: {total_tokens / len(test_data):.1f}")
        print(f"Average input tokens per sample: {total_input_tokens / len(test_data):.1f}")
        print(f"Average output tokens per sample: {total_output_tokens / len(test_data):.1f}")
    
    print(f"Results saved to: {results_path}")

def get_model_response_with_usage(model_interface, prompt: str, verbose: bool = False):
    """Get model response and usage information"""
    if getattr(model_interface.config, "restricted_reasoning", False):
        prompt = (
            f"{prompt}\n\nAnswer directly after the "
            "'### My answer is:' tag without extended thinking."
        )
    # Check if this is an API model that supports token tracking
    if model_interface.config.model_type in ["openai", "deepseek"]:
        try:
            response, usage_info = model_interface._get_api_response(prompt)
            return response, usage_info
        except Exception as e:
            if verbose:
                print(f"Error getting response with usage info: {e}")
            # Fallback to regular response
            response = model_interface.get_response(prompt)
            return response, None
    else:
        # For VLLM or other models, just get the regular response
        response = model_interface.get_response(prompt)
        return response, None

def load_test_data_from_file(file_path: str, task_type: str, num_samples: int, verbose: bool = False):
    """Load test data from JSON file with filtering and sampling"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract samples from the file structure
    if 'samples' in data:
        # Generated by generate_tasks1.py format
        all_samples = data['samples']
        if verbose:
            print(f"Loaded data from generate_tasks1.py format")
            if 'generation_info' in data:
                gen_info = data['generation_info']
                print(f"Original difficulty: {gen_info.get('difficulty', 'unknown')}")
                print(f"Original sample count: {gen_info.get('total_samples', 'unknown')}")
    elif isinstance(data, list):
        # Direct list of samples
        all_samples = data
        if verbose:
            print(f"Loaded data from direct list format")
    else:
        # Assume it's the data_manager format
        all_samples = data
        if verbose:
            print(f"Loaded data from data_manager format")
    
    # Filter by task type
    filtered_samples = []
    if task_type == "both":
        filtered_samples = all_samples
    else:
        filtered_samples = [sample for sample in all_samples if sample.get('task_type') == task_type]
    
    if verbose:
        print(f"After filtering by task_type '{task_type}': {len(filtered_samples)} samples")
    
    # Apply num_samples limit
    if len(filtered_samples) > num_samples:
        # Randomly sample to get the desired number
        filtered_samples = random.sample(filtered_samples, num_samples)
        if verbose:
            print(f"Randomly sampled {num_samples} samples from {len(filtered_samples)} available")
    elif len(filtered_samples) < num_samples:
        if verbose:
            print(f"Warning: Requested {num_samples} samples but only {len(filtered_samples)} available")
    
    return filtered_samples

def calculate_difficulty_stats(task1_results):
    """Calculate accuracy statistics by difficulty level"""
    difficulty_stats = {}
    
    for result in task1_results:
        if 'difficulty' not in result:
            continue
        
        difficulty = result['difficulty']
        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {'correct': 0, 'total': 0}
        
        difficulty_stats[difficulty]['total'] += 1
        if result['is_correct']:
            difficulty_stats[difficulty]['correct'] += 1
    
    # Calculate accuracy for each difficulty
    for difficulty in difficulty_stats:
        stats = difficulty_stats[difficulty]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    return difficulty_stats

def calculate_token_usage_stats(all_results):
    """Calculate token usage statistics by task type and difficulty"""
    token_stats = {
        "by_task_type": {},
        "by_difficulty": {},
        "by_correctness": {
            "correct": {"count": 0, "total_tokens": 0, "avg_tokens": 0},
            "incorrect": {"count": 0, "total_tokens": 0, "avg_tokens": 0}
        }
    }
    
    # Group by task type
    task_type_stats = {}
    difficulty_stats = {}
    correctness_stats = {"correct": [], "incorrect": []}
    
    for result in all_results:
        if 'token_usage' not in result or result['token_usage'] is None:
            continue
        
        token_usage = result['token_usage']
        total_tokens = token_usage.get('total_tokens', 0)
        
        if total_tokens == 0:
            continue
        
        task_type = result.get('task_type', 'unknown')
        difficulty = result.get('difficulty', 'unknown')
        is_correct = result.get('is_correct', False)
        
        # By task type
        if task_type not in task_type_stats:
            task_type_stats[task_type] = []
        task_type_stats[task_type].append(total_tokens)
        
        # By difficulty
        if difficulty != 'unknown':
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = []
            difficulty_stats[difficulty].append(total_tokens)
        
        # By correctness
        if is_correct:
            correctness_stats["correct"].append(total_tokens)
        else:
            correctness_stats["incorrect"].append(total_tokens)
    
    # Calculate statistics for task types
    for task_type, tokens_list in task_type_stats.items():
        if tokens_list:
            token_stats["by_task_type"][task_type] = {
                "count": len(tokens_list),
                "total_tokens": sum(tokens_list),
                "avg_tokens": sum(tokens_list) / len(tokens_list),
                "min_tokens": min(tokens_list),
                "max_tokens": max(tokens_list)
            }
    
    # Calculate statistics for difficulties
    for difficulty, tokens_list in difficulty_stats.items():
        if tokens_list:
            token_stats["by_difficulty"][difficulty] = {
                "count": len(tokens_list),
                "total_tokens": sum(tokens_list),
                "avg_tokens": sum(tokens_list) / len(tokens_list),
                "min_tokens": min(tokens_list),
                "max_tokens": max(tokens_list)
            }
    
    # Calculate statistics for correctness
    for correctness, tokens_list in correctness_stats.items():
        if tokens_list:
            token_stats["by_correctness"][correctness] = {
                "count": len(tokens_list),
                "total_tokens": sum(tokens_list),
                "avg_tokens": sum(tokens_list) / len(tokens_list),
                "min_tokens": min(tokens_list),
                "max_tokens": max(tokens_list)
            }
    
    return token_stats

if __name__ == "__main__":
    main()
