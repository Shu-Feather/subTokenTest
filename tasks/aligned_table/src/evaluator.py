"""
Evaluator for scoring LLM responses
Location: src/evaluator.py
"""

from typing import Dict, List, Tuple
from .utils import (
    extract_table_content,
    check_alignment,
    parse_answer,
    normalize_table_data
)


class Evaluator:
    """Evaluate LLM responses for the aligned-table benchmark."""
    
    def __init__(self, config: Dict):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config.get('evaluation', {})
        self.content_weight = self.eval_config.get('content_weight', 0.6)
        self.alignment_weight = self.eval_config.get('alignment_weight', 0.4)
    
    def evaluate(self, response: str, ground_truth: List[List[str]], 
                 table_format: str, verbose: bool = False) -> Dict:
        """
        Evaluate a single response.
        
        Args:
            response: LLM's response string
            ground_truth: Ground truth table data (2D list)
            table_format: Expected table format ('latex', 'text', 'markdown')
            verbose: Whether to print detailed evaluation info
            
        Returns:
            Dictionary containing evaluation scores and metrics
        """
        # Extract answer from response
        answer = parse_answer(response)
        
        if answer is None:
            if verbose:
                print("WARNING: No answer tags found in response")
            return {
                'content_score': 0.0,
                'alignment_score': 0.0,
                'is_aligned': False,
                'total_score': 0.0,
                'error': 'No answer tags found'
            }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Extracted Answer:")
            print(f"{'='*60}")
            print(answer)
            print(f"{'='*60}\n")
        
        # Normalize ground truth to ensure consistent columns
        normalized_ground_truth = normalize_table_data(ground_truth)
        
        # Evaluate content accuracy
        content_score, content_details = self._evaluate_content(
            answer, normalized_ground_truth, table_format, verbose
        )
        
        # Evaluate alignment
        is_aligned, alignment_score = check_alignment(answer, table_format)
        
        if verbose:
            print(f"\nContent Score: {content_score:.4f}")
            print(f"Alignment Score: {alignment_score:.4f}")
            print(f"Is Aligned: {is_aligned}")
        
        # Calculate total score
        total_score = (
            self.content_weight * content_score +
            self.alignment_weight * alignment_score
        )
        
        return {
            'content_score': content_score,
            'alignment_score': alignment_score,
            'is_aligned': is_aligned,
            'total_score': total_score,
            'content_details': content_details
        }
    
    def _evaluate_content(self, answer: str, ground_truth: List[List[str]], 
                         table_format: str, verbose: bool = False) -> Tuple[float, Dict]:
        """
        Evaluate content accuracy.
        
        Args:
            answer: Extracted answer string
            ground_truth: Ground truth table data (should be normalized)
            table_format: Table format
            verbose: Whether to print details
            
        Returns:
            Tuple of (content_score, details_dict)
        """
        try:
            # Extract content from the answer
            extracted_content = extract_table_content(answer, table_format)
            
            # Normalize extracted content as well
            extracted_content = normalize_table_data(extracted_content)
            
            if verbose:
                print(f"\nExtracted Content:")
                for row in extracted_content:
                    print(f"  {row}")
                print(f"\nGround Truth:")
                for row in ground_truth:
                    print(f"  {row}")
            
            # Check dimensions
            if len(extracted_content) != len(ground_truth):
                row_score = 0.0
                if verbose:
                    print(f"\nRow count mismatch: {len(extracted_content)} vs {len(ground_truth)}")
            else:
                row_score = 1.0
            
            # Check if all rows have correct number of columns
            expected_cols = len(ground_truth[0]) if ground_truth else 0
            col_score = 0.0
            if extracted_content:
                correct_col_rows = sum(
                    1 for row in extracted_content if len(row) == expected_cols
                )
                col_score = correct_col_rows / len(extracted_content) if extracted_content else 0.0
                
                if verbose and col_score < 1.0:
                    print(f"\nColumn count issues:")
                    for i, row in enumerate(extracted_content):
                        if len(row) != expected_cols:
                            print(f"  Row {i}: {len(row)} cols (expected {expected_cols})")
            
            # Calculate cell-level accuracy
            total_cells = 0
            correct_cells = 0
            
            # Use the minimum row count to avoid index errors
            min_rows = min(len(extracted_content), len(ground_truth))
            
            for i in range(min_rows):
                pred_row = extracted_content[i]
                true_row = ground_truth[i]
                
                # Use minimum column count
                min_cols = min(len(pred_row), len(true_row))
                
                for j in range(min_cols):
                    total_cells += 1
                    pred_cell = str(pred_row[j]).strip().lower()
                    true_cell = str(true_row[j]).strip().lower()
                    
                    # Normalize and compare
                    if pred_cell == true_cell:
                        correct_cells += 1
                    elif verbose:
                        print(f"  Mismatch at [{i},{j}]: '{pred_row[j]}' vs '{true_row[j]}'")
                
                # Count missing cells as errors
                if len(true_row) > len(pred_row):
                    missing = len(true_row) - len(pred_row)
                    total_cells += missing
                    if verbose:
                        print(f"  Row {i}: {missing} missing cells")
            
            # Count missing rows
            if len(ground_truth) > len(extracted_content):
                missing_rows = len(ground_truth) - len(extracted_content)
                missing_cells = missing_rows * expected_cols
                total_cells += missing_cells
                if verbose:
                    print(f"\n{missing_rows} missing rows ({missing_cells} cells)")
            
            cell_accuracy = correct_cells / total_cells if total_cells > 0 else 0.0
            
            # Combined content score
            # content_score = (
            #     0.3 * row_score +
            #     0.3 * col_score +
            #     0.4 * cell_accuracy
            # )

            content_score = (
                1.0 * cell_accuracy
            )
            
            details = {
                'row_score': row_score,
                'col_score': col_score,
                'cell_accuracy': cell_accuracy,
                'correct_cells': correct_cells,
                'total_cells': total_cells,
                'extracted_rows': len(extracted_content),
                'expected_rows': len(ground_truth),
                'extracted_cols': len(extracted_content[0]) if extracted_content else 0,
                'expected_cols': expected_cols
            }
            
            return content_score, details
            
        except Exception as e:
            if verbose:
                print(f"Error evaluating content: {e}")
                import traceback
                traceback.print_exc()
            return 0.0, {'error': str(e)}
    
    def evaluate_batch(self, responses: List[str], ground_truths: List[List[List[str]]], 
                      table_formats: List[str], verbose: bool = False) -> List[Dict]:
        """
        Evaluate multiple responses.
        
        Args:
            responses: List of LLM responses
            ground_truths: List of ground truth table data
            table_formats: List of table formats
            verbose: Whether to print detailed info
            
        Returns:
            List of evaluation result dictionaries
        """
        results = []
        
        for i, (response, gt, fmt) in enumerate(zip(responses, ground_truths, table_formats)):
            if verbose:
                print(f"\n{'#'*60}")
                print(f"Evaluating Test Case {i+1}")
                print(f"{'#'*60}")
            
            result = self.evaluate(response, gt, fmt, verbose)
            result['test_id'] = i
            results.append(result)
        
        return results
    
    def compute_summary_statistics(self, results: List[Dict]) -> Dict:
        """
        Compute summary statistics from evaluation results.
        
        Args:
            results: List of evaluation result dictionaries
            
        Returns:
            Dictionary of summary statistics
        """
        if not results:
            return {}
        
        # Filter out results with errors
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results to summarize'}
        
        # Calculate averages
        avg_content = sum(r['content_score'] for r in valid_results) / len(valid_results)
        avg_alignment = sum(r['alignment_score'] for r in valid_results) / len(valid_results)
        avg_total = sum(r['total_score'] for r in valid_results) / len(valid_results)
        
        # Calculate alignment rate
        aligned_count = sum(1 for r in valid_results if r['is_aligned'])
        alignment_rate = aligned_count / len(valid_results)
        
        # Calculate perfect scores
        perfect_content = sum(1 for r in valid_results if r['content_score'] == 1.0)
        perfect_alignment = sum(1 for r in valid_results if r['alignment_score'] == 1.0)
        perfect_total = sum(1 for r in valid_results if r['total_score'] == 1.0)
        
        summary = {
            'total_cases': len(results),
            'valid_cases': len(valid_results),
            'avg_content_score': avg_content,
            'avg_alignment_score': avg_alignment,
            'avg_total_score': avg_total,
            'alignment_rate': alignment_rate,
            'perfect_content_count': perfect_content,
            'perfect_alignment_count': perfect_alignment,
            'perfect_total_count': perfect_total,
            'perfect_content_rate': perfect_content / len(valid_results),
            'perfect_alignment_rate': perfect_alignment / len(valid_results),
            'perfect_total_rate': perfect_total / len(valid_results)
        }
        
        return summary
    
    def compute_format_statistics(self, results: List[Dict], 
                                  test_cases: List[Dict]) -> Dict:
        """
        Compute statistics broken down by table format.
        
        Args:
            results: List of evaluation results
            test_cases: List of test cases
            
        Returns:
            Dictionary of format-specific statistics
        """
        format_results = {}
        
        for result, test_case in zip(results, test_cases):
            fmt = test_case['table_format']
            
            if fmt not in format_results:
                format_results[fmt] = []
            
            format_results[fmt].append(result)
        
        # Compute statistics for each format
        format_stats = {}
        for fmt, fmt_results in format_results.items():
            format_stats[fmt] = self.compute_summary_statistics(fmt_results)
        
        return format_stats