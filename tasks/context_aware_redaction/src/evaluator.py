"""
Evaluator for Context-Aware Redaction benchmark
Binary classification at the number level
"""

import re
from typing import Dict, Any, List, Tuple, Set
from .utils import extract_answer, normalize_text, parse_sensitive_info, apply_redaction_rules


class Evaluator:
    """
    Evaluate model performance on redaction task
    Each sensitive number is treated as a binary classification sample
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize evaluator
        
        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
    
    def normalize_redacted_value(self, value: str) -> str:
        """
        Normalize a redacted value for comparison (remove extra whitespace)
        
        Args:
            value: Redacted value
            
        Returns:
            Normalized value
        """
        return re.sub(r'\s+', ' ', value).strip()
    
    def get_expected_redaction(self, original_value: str, info_type: str) -> str:
        """
        Get the expected redaction for a sensitive value
        
        Args:
            original_value: Original sensitive value
            info_type: Type of sensitive information
            
        Returns:
            Expected redacted value
        """
        return self.normalize_redacted_value(
            apply_redaction_rules("", info_type, original_value)
        )
    
    def find_redaction_in_text(self, text: str, expected_redaction: str, 
                               info_type: str, context_window: int = 50) -> bool:
        """
        Check if the expected redaction appears in the text
        
        Args:
            text: Text to search in
            expected_redaction: Expected redacted value
            info_type: Type of information
            context_window: Characters to look around for context matching
            
        Returns:
            True if found, False otherwise
        """
        # Normalize text for comparison
        text_norm = self.normalize_redacted_value(text)
        
        # Direct exact match
        if expected_redaction in text_norm:
            return True
        
        # For phone numbers, also check without spaces between area code and number
        if info_type == 'phone':
            # Remove space between area code and number for matching
            expected_no_space = expected_redaction.replace(' ', '')
            text_no_space = text_norm.replace(' ', '')
            if expected_no_space in text_no_space:
                return True
        
        return False
    
    def check_information_leakage(self, original_value: str, prediction: str) -> bool:
        """
        Check if original sensitive information appears in prediction (not redacted)
        
        Args:
            original_value: Original sensitive value
            prediction: Model's prediction
            
        Returns:
            True if leaked (found in prediction), False otherwise
        """
        # Normalize for comparison
        pred_norm = self.normalize_redacted_value(prediction)
        orig_norm = self.normalize_redacted_value(original_value)
        
        return orig_norm in pred_norm
    
    def compute_number_level_metrics(self, prediction: str, ground_truth: str,
                                    original_text: str) -> Dict[str, Any]:
        """
        Compute precision, recall, and F1 at the number level
        Each sensitive number is a binary classification sample:
        - TP: Correctly redacted (expected redaction found in prediction)
        - FN: Should be redacted but not/incorrectly redacted
        - FP: Not applicable in this strict matching (we only check if expected redactions exist)
        
        Args:
            prediction: Predicted redacted text
            ground_truth: Ground truth redacted text
            original_text: Original text with sensitive information
            
        Returns:
            Dictionary with precision, recall, F1, and detailed counts
        """
        # Get all sensitive information from original text
        original_sensitive_info = parse_sensitive_info(original_text)
        
        true_positives = 0  # Correctly redacted numbers
        false_negatives = 0  # Numbers that should be redacted but weren't/incorrectly redacted
        false_positives = 0  # Currently not used in strict matching
        
        leaked_items = []
        correctly_redacted_items = []
        incorrectly_redacted_items = []
        
        # For each sensitive item, check if it's correctly redacted
        for orig_info in original_sensitive_info:
            orig_value = orig_info['value']
            info_type = orig_info['type']
            
            # Get expected redaction
            expected_redaction = self.get_expected_redaction(orig_value, info_type)
            
            # Check if correctly redacted in prediction
            is_correctly_redacted = self.find_redaction_in_text(
                prediction, expected_redaction, info_type
            )
            
            # Check for information leakage
            is_leaked = self.check_information_leakage(orig_value, prediction)
            
            if is_correctly_redacted and not is_leaked:
                # Correctly redacted: TP
                true_positives += 1
                correctly_redacted_items.append({
                    'type': info_type,
                    'original': orig_value,
                    'expected': expected_redaction,
                    'status': 'correct'
                })
            else:
                # Not correctly redacted or leaked: FN
                false_negatives += 1
                status = 'leaked' if is_leaked else 'incorrect_or_missing'
                incorrectly_redacted_items.append({
                    'type': info_type,
                    'original': orig_value,
                    'expected': expected_redaction,
                    'status': status
                })
                
                if is_leaked:
                    leaked_items.append({
                        'type': info_type,
                        'value': orig_value
                    })
        
        # Calculate metrics
        total_numbers = len(original_sensitive_info)
        
        # Precision: TP / (TP + FP)
        # In our case, FP = 0 in strict matching, so Precision = TP / TP = 1.0 if TP > 0
        # But more meaningfully: Precision = TP / total_predictions if we count predictions
        # For simplicity in binary classification per number: Precision = accuracy of positive predictions
        precision = true_positives / total_numbers if total_numbers > 0 else 1.0
        
        # Recall: TP / (TP + FN) = TP / total_numbers
        recall = true_positives / total_numbers if total_numbers > 0 else 1.0
        
        # F1: harmonic mean of precision and recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Special case: if no sensitive info exists
        if total_numbers == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,  # 0 in current implementation
            'false_negatives': false_negatives,
            'total_sensitive_numbers': total_numbers,
            'correctly_redacted_items': correctly_redacted_items if self.verbose else [],
            'incorrectly_redacted_items': incorrectly_redacted_items if self.verbose else [],
            'leaked_items': leaked_items
        }
    
    def evaluate_single(self, prediction: str, ground_truth: str, 
                       original_text: str) -> Dict[str, Any]:
        """
        Evaluate a single prediction
        
        Args:
            prediction: Model's predicted redacted text
            ground_truth: Ground truth redacted text
            original_text: Original text before redaction
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Normalize texts
        pred_norm = normalize_text(prediction)
        gold_norm = normalize_text(ground_truth)
        
        # Exact match
        exact_match = pred_norm == gold_norm
        
        # Number-level metrics (binary classification per number)
        number_metrics = self.compute_number_level_metrics(
            prediction, ground_truth, original_text
        )
        
        # Extract leakage information
        leaked_items = number_metrics['leaked_items']
        has_leakage = len(leaked_items) > 0
        leakage_count = len(leaked_items)
        total_sensitive = number_metrics['total_sensitive_numbers']
        leakage_ratio = leakage_count / total_sensitive if total_sensitive > 0 else 0.0
        
        results = {
            'exact_match': exact_match,
            'precision': number_metrics['precision'],
            'recall': number_metrics['recall'],
            'f1': number_metrics['f1'],
            'true_positives': number_metrics['true_positives'],
            'false_positives': number_metrics['false_positives'],
            'false_negatives': number_metrics['false_negatives'],
            'has_leakage': has_leakage,
            'leakage_count': leakage_count,
            'leakage_ratio': leakage_ratio,
            'total_sensitive_items': total_sensitive,
            'leaked_items': leaked_items if self.verbose else []
        }
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("Evaluation Result:")
            print(f"{'='*80}")
            print(f"Exact Match: {exact_match}")
            print(f"\nBinary Classification Metrics (per number):")
            print(f"  Total Sensitive Numbers: {total_sensitive}")
            print(f"  True Positives (TP):     {number_metrics['true_positives']} (correctly redacted)")
            print(f"  False Negatives (FN):    {number_metrics['false_negatives']} (missed/incorrect)")
            print(f"  False Positives (FP):    {number_metrics['false_positives']}")
            print(f"\n  Precision: {number_metrics['precision']:.4f}")
            print(f"  Recall:    {number_metrics['recall']:.4f}")
            print(f"  F1 Score:  {number_metrics['f1']:.4f}")
            
            print(f"\nInformation Leakage:")
            print(f"  Has Leakage: {has_leakage}")
            print(f"  Leaked Count: {leakage_count}/{total_sensitive}")
            print(f"  Leakage Ratio: {leakage_ratio:.4f}")
            
            if leaked_items:
                print(f"\n  Leaked Items:")
                for item in leaked_items:
                    print(f"    - {item['type']}: {item['value']}")
            
            if number_metrics.get('correctly_redacted_items'):
                print(f"\n  Correctly Redacted ({len(number_metrics['correctly_redacted_items'])}):")
                for item in number_metrics['correctly_redacted_items']:
                    print(f"    ✓ {item['type']}: {item['original']} → {item['expected']}")
            
            if number_metrics.get('incorrectly_redacted_items'):
                print(f"\n  Incorrectly Redacted/Missing ({len(number_metrics['incorrectly_redacted_items'])}):")
                for item in number_metrics['incorrectly_redacted_items']:
                    print(f"    ✗ {item['type']}: {item['original']} → {item['expected']} [{item['status']}]")
            
            print(f"{'='*80}\n")
        
        return results
    
    def evaluate_batch(self, predictions: List[str], ground_truths: List[str],
                      original_texts: List[str]) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions
        Aggregate metrics across all sensitive numbers (not per sample)
        
        Args:
            predictions: List of predicted redacted texts
            ground_truths: List of ground truth redacted texts
            original_texts: List of original texts
            
        Returns:
            Dictionary with aggregated metrics
        """
        results = []
        
        for pred, gold, orig in zip(predictions, ground_truths, original_texts):
            result = self.evaluate_single(pred, gold, orig)
            results.append(result)
        
        # Aggregate across all numbers (not samples)
        total_samples = len(results)
        total_tp = sum(r['true_positives'] for r in results)
        total_fp = sum(r['false_positives'] for r in results)
        total_fn = sum(r['false_negatives'] for r in results)
        total_numbers = sum(r['total_sensitive_items'] for r in results)
        
        # Calculate overall precision, recall, F1
        # Precision = TP / (TP + FP)
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        
        # Recall = TP / (TP + FN) = TP / total_numbers
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        # F1 = 2 * (P * R) / (P + R)
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
                     if (overall_precision + overall_recall) > 0 else 0.0
        
        # Special case: no sensitive numbers at all
        if total_numbers == 0:
            overall_precision = 1.0
            overall_recall = 1.0
            overall_f1 = 1.0
        
        aggregated = {
            'total_samples': total_samples,
            'total_sensitive_numbers': total_numbers,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'exact_match_rate': sum(r['exact_match'] for r in results) / total_samples,
            'avg_precision': overall_precision,
            'avg_recall': overall_recall,
            'avg_f1': overall_f1,
            'leakage_rate': sum(r['has_leakage'] for r in results) / total_samples,
            'avg_leakage_ratio': sum(r['leakage_ratio'] for r in results) / total_samples,
            'total_leaked_numbers': sum(r['leakage_count'] for r in results),
            'individual_results': results
        }
        
        return aggregated
    
    def evaluate_by_difficulty(self, results: List[Dict[str, Any]], 
                               difficulties: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results by difficulty level
        
        Args:
            results: List of individual evaluation results
            difficulties: List of difficulty levels corresponding to results
            
        Returns:
            Dictionary mapping difficulty to metrics
        """
        difficulty_results = {}
        
        for diff in set(difficulties):
            diff_indices = [i for i, d in enumerate(difficulties) if d == diff]
            diff_results = [results[i] for i in diff_indices]
            
            if not diff_results:
                continue
            
            total_samples = len(diff_results)
            total_tp = sum(r['true_positives'] for r in diff_results)
            total_fp = sum(r['false_positives'] for r in diff_results)
            total_fn = sum(r['false_negatives'] for r in diff_results)
            total_numbers = sum(r['total_sensitive_items'] for r in diff_results)
            
            # Calculate metrics from aggregated counts
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Special case: no sensitive numbers
            if total_numbers == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            
            difficulty_results[diff] = {
                'total_samples': total_samples,
                'total_sensitive_numbers': total_numbers,
                'total_true_positives': total_tp,
                'total_false_positives': total_fp,
                'total_false_negatives': total_fn,
                'exact_match_rate': sum(r['exact_match'] for r in diff_results) / total_samples,
                'avg_precision': precision,
                'avg_recall': recall,
                'avg_f1': f1,
                'leakage_rate': sum(r['has_leakage'] for r in diff_results) / total_samples,
                'avg_leakage_ratio': sum(r['leakage_ratio'] for r in diff_results) / total_samples,
                'total_leaked_numbers': sum(r['leakage_count'] for r in diff_results),
            }
        
        return difficulty_results
