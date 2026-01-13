"""
Evaluation module for measuring canonicalization performance.
"""

import Levenshtein
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
from src.models.base_model import BaseModel
from src.prompts import get_canonicalization_prompt
from src.answer_extractor import AnswerExtractor


class Evaluator:
    """Evaluator for canonicalization task."""
    
    def __init__(self, model: BaseModel, config: Dict, verbose: bool = False):
        """
        Initialize evaluator.
        
        Args:
            model: LLM model instance
            config: Configuration dictionary
            verbose: Whether to print verbose output
        """
        self.model = model
        self.config = config
        self.verbose = verbose
        
        # Initialize answer extractor
        answer_config = config.get('evaluation', {}).get('answer_tags', {})
        start_tag = answer_config.get('start', '<answer>')
        end_tag = answer_config.get('end', '</answer>')
        self.answer_extractor = AnswerExtractor(start_tag, end_tag)

        # Get prompt configuration
        self.prompt_config = config.get('prompt', {})
        self.restricted_reasoning = self.prompt_config.get('restricted_reasoning', False)

        if self.verbose:
            print(f"\n{'='*70}")
            print("EVALUATOR CONFIGURATION")
            print(f"{'='*70}")
            print(f"Use few-shot examples: {self.prompt_config['use_few_shot']}")
            print(f"Number of few-shot examples: {self.prompt_config['num_few_shot_examples']}")
            print(f"Answer tags: {start_tag} ... {end_tag}")
            print(f"{'='*70}\n")
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """
        Evaluate a single sample.
        
        Args:
            sample: Sample dictionary with 'original' and 'perturbed' keys
            
        Returns:
            Evaluation results dictionary including usage_info
        """
        # Get canonicalization prompt with config
        messages = get_canonicalization_prompt(
            perturbed_text=sample['perturbed'],
            use_few_shot=self.prompt_config['use_few_shot'],
            num_examples=self.prompt_config['num_few_shot_examples'],
            restricted_reasoning=self.restricted_reasoning
        )
        
        # Generate raw response with usage info
        raw_response, usage_info = self.model.generate(messages)
        
        # Extract answer using tags
        prediction, tags_found = self.answer_extractor.extract(raw_response)
        
        # Validate extraction
        validation_info = self.answer_extractor.validate_extraction(raw_response)
        
        # Clean prediction and ground truth
        prediction = prediction.strip().lower()
        ground_truth = sample['original'].strip().lower()
        
        # Calculate metrics
        exact_match = prediction == ground_truth
        levenshtein_dist = Levenshtein.distance(prediction, ground_truth)
        
        # Normalized Levenshtein similarity (0-1 scale)
        max_len = max(len(prediction), len(ground_truth))
        if max_len == 0:
            normalized_similarity = 1.0
        else:
            normalized_similarity = 1.0 - (levenshtein_dist / max_len)
        
        result = {
            'sample_id': sample['id'],
            'category': sample['category'],
            'difficulty': sample.get('difficulty', 'unknown'),
            'perturbation_type': sample['perturbation_type'],
            'original': ground_truth,
            'perturbed': sample['perturbed'],
            'raw_response': raw_response,
            'prediction': prediction,
            'exact_match': exact_match,
            'levenshtein_distance': levenshtein_dist,
            'similarity_score': normalized_similarity,
            'tags_found': tags_found,
            'answer_extraction': validation_info,
            'usage_info': usage_info  # Add token usage information
        }
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Sample ID: {sample['id']}")
            print(f"Category: {sample['category']}")
            print(f"Difficulty: {sample.get('difficulty', 'unknown')}")
            print(f"Perturbation: {sample['perturbation_type']}")
            print(f"{'='*70}")
            print(f"Original:   {ground_truth}")
            print(f"Perturbed:  {sample['perturbed']}")
            print(f"\nRaw Response:")
            print("-" * 70)
            print(raw_response)
            print("-" * 70)
            print(f"\nExtracted Answer: {prediction}")
            print(f"Tags Found: {tags_found}")
            print(f"Tag Validation: {validation_info}")
            print(f"\nMetrics:")
            print(f"  Exact Match: {exact_match}")
            print(f"  Similarity: {normalized_similarity:.3f}")
            print(f"  Levenshtein Distance: {levenshtein_dist}")
            
            # Print token usage info if available
            if usage_info:
                print(f"\nToken Usage:")
                print(f"  Total: {usage_info.get('total_tokens', 0)}")
                print(f"  Prompt: {usage_info.get('prompt_tokens', 0)}")
                print(f"  Completion: {usage_info.get('completion_tokens', 0)}")
                print(f"  Reasoning: {usage_info.get('reasoning_tokens', 0)}")
                print(f"  Output: {usage_info.get('output_tokens', 0)}")
                if usage_info.get('total_tokens', 0) > 0:
                    reasoning_ratio = usage_info.get('reasoning_tokens', 0) / usage_info['total_tokens']
                    print(f"  Reasoning Ratio: {reasoning_ratio:.2%}")
            
            print(f"{'='*70}\n")
        
        return result
    
    def evaluate_batch(self, samples: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Evaluate a batch of samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Tuple of (results list, aggregated metrics dict)
        """
        results = []
        
        print(f"\nEvaluating {len(samples)} samples...")

        # If the model supports batching, process samples in chunks
        batch_size = self.config.get('models', {}).get('vllm', {}).get('batch_size', 1)
        supports_batch = hasattr(self.model, "generate_batch") and batch_size and batch_size > 1

        if supports_batch:
            batch_size = int(batch_size)
            for idx in range(0, len(samples), batch_size):
                batch = samples[idx: idx + batch_size]
                # Build prompts once to reuse formatting logic in evaluate_sample
                messages_list = [
                    get_canonicalization_prompt(
                        perturbed_text=sample['perturbed'],
                        use_few_shot=self.prompt_config['use_few_shot'],
                        num_examples=self.prompt_config['num_few_shot_examples'],
                        restricted_reasoning=self.restricted_reasoning
                    )
                    for sample in batch
                ]
                batch_outputs = self.model.generate_batch(messages_list)
                for sample, (raw_response, usage_info) in zip(batch, batch_outputs):
                    prediction, tags_found = self.answer_extractor.extract(raw_response)
                    validation_info = self.answer_extractor.validate_extraction(raw_response)
                    prediction = prediction.strip().lower()
                    ground_truth = sample['original'].strip().lower()
                    exact_match = prediction == ground_truth
                    levenshtein_dist = Levenshtein.distance(prediction, ground_truth)
                    max_len = max(len(prediction), len(ground_truth))
                    normalized_similarity = 1.0 if max_len == 0 else 1.0 - (levenshtein_dist / max_len)
                    results.append({
                        'sample_id': sample['id'],
                        'category': sample['category'],
                        'difficulty': sample.get('difficulty', 'unknown'),
                        'perturbation_type': sample['perturbation_type'],
                        'original': ground_truth,
                        'perturbed': sample['perturbed'],
                        'raw_response': raw_response,
                        'prediction': prediction,
                        'exact_match': exact_match,
                        'levenshtein_distance': levenshtein_dist,
                        'similarity_score': normalized_similarity,
                        'tags_found': tags_found,
                        'answer_extraction': validation_info,
                        'usage_info': usage_info
                    })
        else:
            for sample in tqdm(samples, disable=self.verbose):
                result = self.evaluate_sample(sample)
                results.append(result)
        
        # Calculate aggregated metrics
        metrics = self._calculate_metrics(results)
        
        return results, metrics
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate aggregated metrics from results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Aggregated metrics dictionary
        """
        total = len(results)
        
        if total == 0:
            return {}
        
        exact_matches = sum(1 for r in results if r['exact_match'])
        avg_similarity = sum(r['similarity_score'] for r in results) / total
        avg_levenshtein = sum(r['levenshtein_distance'] for r in results) / total
        tags_found_count = sum(1 for r in results if r['tags_found'])
        
        # Metrics by category
        by_category = {}
        for result in results:
            cat = result['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result)
        
        category_metrics = {}
        for cat, cat_results in by_category.items():
            cat_total = len(cat_results)
            cat_exact = sum(1 for r in cat_results if r['exact_match'])
            cat_similarity = sum(r['similarity_score'] for r in cat_results) / cat_total
            cat_tags_found = sum(1 for r in cat_results if r['tags_found'])
            
            category_metrics[cat] = {
                'exact_match_rate': cat_exact / cat_total,
                'avg_similarity': cat_similarity,
                'tags_found_rate': cat_tags_found / cat_total,
                'count': cat_total
            }
        
        # Metrics by perturbation type
        by_perturbation = {}
        for result in results:
            pert = result['perturbation_type']
            if pert not in by_perturbation:
                by_perturbation[pert] = []
            by_perturbation[pert].append(result)
        
        perturbation_metrics = {}
        for pert, pert_results in by_perturbation.items():
            pert_total = len(pert_results)
            pert_exact = sum(1 for r in pert_results if r['exact_match'])
            pert_similarity = sum(r['similarity_score'] for r in pert_results) / pert_total
            pert_tags_found = sum(1 for r in pert_results if r['tags_found'])
            
            perturbation_metrics[pert] = {
                'exact_match_rate': pert_exact / pert_total,
                'avg_similarity': pert_similarity,
                'tags_found_rate': pert_tags_found / pert_total,
                'count': pert_total
            }
        
        # Metrics by difficulty
        by_difficulty = {}
        for result in results:
            diff = result.get('difficulty', 'unknown')
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(result)
        
        difficulty_metrics = {}
        for diff, diff_results in by_difficulty.items():
            diff_total = len(diff_results)
            diff_exact = sum(1 for r in diff_results if r['exact_match'])
            diff_similarity = sum(r['similarity_score'] for r in diff_results) / diff_total
            diff_tags_found = sum(1 for r in diff_results if r['tags_found'])
            
            difficulty_metrics[diff] = {
                'exact_match_rate': diff_exact / diff_total,
                'avg_similarity': diff_similarity,
                'tags_found_rate': diff_tags_found / diff_total,
                'count': diff_total
            }
        
        metrics = {
            'total_samples': total,
            'exact_match_rate': exact_matches / total,
            'avg_similarity_score': avg_similarity,
            'avg_levenshtein_distance': avg_levenshtein,
            'tags_found_rate': tags_found_count / total,
            'by_category': category_metrics,
            'by_perturbation': perturbation_metrics,
            'by_difficulty': difficulty_metrics
        }
        
        return metrics
