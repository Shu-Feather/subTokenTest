"""
Response evaluator for the Gomoku benchmark.
"""

import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from collections import Counter

@dataclass
class BenchmarkResults:
    """Class to hold benchmark evaluation results"""
    total_cases: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    white_wins_precision: float = 0.0
    white_wins_recall: float = 0.0
    white_wins_f1: float = 0.0
    black_wins_precision: float = 0.0
    black_wins_recall: float = 0.0
    black_wins_f1: float = 0.0
    no_winner_precision: float = 0.0
    no_winner_recall: float = 0.0
    no_winner_f1: float = 0.0
    confusion_matrix: Dict[str, Dict[str, int]] = None
    error_cases: int = 0
    parse_failures: int = 0
    
    def __post_init__(self):
        if self.confusion_matrix is None:
            self.confusion_matrix = {
                "WHITE_WINS": {"WHITE_WINS": 0, "BLACK_WINS": 0, "NO_WINNER": 0, "ERROR": 0, "PARSE_FAIL": 0},
                "BLACK_WINS": {"WHITE_WINS": 0, "BLACK_WINS": 0, "NO_WINNER": 0, "ERROR": 0, "PARSE_FAIL": 0},
                "NO_WINNER": {"WHITE_WINS": 0, "BLACK_WINS": 0, "NO_WINNER": 0, "ERROR": 0, "PARSE_FAIL": 0}
            }

class ResponseParser:
    """Parser for extracting game state from model responses"""
    
    VALID_RESPONSES = {"WHITE_WINS", "BLACK_WINS", "NO_WINNER"}
    
    @classmethod
    def parse_response(cls, response: str) -> str:
        """
        Parse model response to extract game state
        Returns: One of VALID_RESPONSES, "ERROR", or "PARSE_FAIL"
        """
        if not response or response == "ERROR":
            return "ERROR"
        
        # Clean the response
        cleaned_response = response.strip().upper()
        
        # Direct match
        if cleaned_response in cls.VALID_RESPONSES:
            return cleaned_response
        
        # Pattern matching for various response formats
        patterns = [
            # Exact matches with quotes or punctuation
            r'^["\']?(WHITE_WINS|BLACK_WINS|NO_WINNER)["\']?[.!]?$',
            
            # Match with explanatory text
            r'(?:ANSWER|RESULT|STATE|WINNER)?\s*:?\s*(WHITE_WINS|BLACK_WINS|NO_WINNER)',
            
            # Match in sentences
            r'\b(WHITE_WINS|BLACK_WINS|NO_WINNER)\b',
            
            # Alternative phrasings
            r'\b(WHITE\s+WINS?|BLACK\s+WINS?|NO\s+WINNER)\b',
            
            # Match "white wins", "black wins", "no winner"
            r'\b(WHITE|BLACK)\s+(WIN|WINS)\b',
            r'\bNO\s+(WINNER|WIN)\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned_response)
            if match:
                matched_text = match.group(1).replace(' ', '_').upper()
                
                # Normalize variations
                if matched_text in ['WHITE_WIN', 'WHITE_WINS']:
                    return 'WHITE_WINS'
                elif matched_text in ['BLACK_WIN', 'BLACK_WINS']:
                    return 'BLACK_WINS'
                elif matched_text in ['NO_WINNER', 'NO_WIN']:
                    return 'NO_WINNER'
                elif matched_text in cls.VALID_RESPONSES:
                    return matched_text
        
        # Try to match individual words
        words = cleaned_response.split()
        for word in words:
            if word in cls.VALID_RESPONSES:
                return word
        
        # Check for semantic matches
        if any(phrase in cleaned_response for phrase in ['WHITE WIN', 'WHITE VICTORY', 'WHITE HAS WON']):
            return 'WHITE_WINS'
        elif any(phrase in cleaned_response for phrase in ['BLACK WIN', 'BLACK VICTORY', 'BLACK HAS WON']):
            return 'BLACK_WINS'
        elif any(phrase in cleaned_response for phrase in ['NO WIN', 'NO VICTOR', 'DRAW', 'TIE', 'NOBODY WINS']):
            return 'NO_WINNER'
        
        return "PARSE_FAIL"

def calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def evaluate_responses(predictions: List[Tuple[str, str]]) -> BenchmarkResults:
    """
    Evaluate model predictions against ground truth
    
    Args:
        predictions: List of (expected, predicted) tuples
    
    Returns:
        BenchmarkResults object with detailed metrics
    """
    results = BenchmarkResults()
    results.total_cases = len(predictions)
    
    if results.total_cases == 0:
        return results
    
    # Parse all predictions
    parsed_predictions = []
    for expected, predicted in predictions:
        parsed_predicted = ResponseParser.parse_response(predicted)
        parsed_predictions.append((expected, parsed_predicted))
        
        # Count errors and parse failures
        if parsed_predicted == "ERROR":
            results.error_cases += 1
        elif parsed_predicted == "PARSE_FAIL":
            results.parse_failures += 1
    
    # Build confusion matrix and count correct predictions
    for expected, predicted in parsed_predictions:
        if expected in results.confusion_matrix:
            if predicted in results.confusion_matrix[expected]:
                results.confusion_matrix[expected][predicted] += 1
            else:
                results.confusion_matrix[expected]["PARSE_FAIL"] += 1
        
        if expected == predicted:
            results.correct_predictions += 1
    
    # Calculate overall accuracy
    results.accuracy = results.correct_predictions / results.total_cases
    
    # Calculate per-class metrics
    class_names = ["WHITE_WINS", "BLACK_WINS", "NO_WINNER"]
    
    for class_name in class_names:
        # True positives: correctly predicted as this class
        tp = results.confusion_matrix[class_name][class_name]
        
        # False positives: incorrectly predicted as this class
        fp = sum(results.confusion_matrix[other_class][class_name] 
                for other_class in class_names if other_class != class_name)
        
        # False negatives: this class predicted as something else
        fn = sum(results.confusion_matrix[class_name][other_class] 
                for other_class in results.confusion_matrix[class_name]
                if other_class != class_name)
        
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        
        if class_name == "WHITE_WINS":
            results.white_wins_precision = precision
            results.white_wins_recall = recall
            results.white_wins_f1 = f1
        elif class_name == "BLACK_WINS":
            results.black_wins_precision = precision
            results.black_wins_recall = recall
            results.black_wins_f1 = f1
        elif class_name == "NO_WINNER":
            results.no_winner_precision = precision
            results.no_winner_recall = recall
            results.no_winner_f1 = f1
    
    return results

def print_evaluation_summary(results: BenchmarkResults, model_name: str = ""):
    """Print a formatted summary of evaluation results"""
    if model_name:
        print(f"\n=== Evaluation Results for {model_name} ===")
    else:
        print(f"\n=== Evaluation Results ===")
    
    print(f"Total cases: {results.total_cases}")
    print(f"Correct predictions: {results.correct_predictions}")
    print(f"Overall accuracy: {results.accuracy:.2%}")
    print(f"Error cases: {results.error_cases}")
    print(f"Parse failures: {results.parse_failures}")
    
    print(f"\n--- Per-class Metrics ---")
    print(f"WHITE_WINS - Precision: {results.white_wins_precision:.2%}, "
          f"Recall: {results.white_wins_recall:.2%}, F1: {results.white_wins_f1:.2%}")
    print(f"BLACK_WINS - Precision: {results.black_wins_precision:.2%}, "
          f"Recall: {results.black_wins_recall:.2%}, F1: {results.black_wins_f1:.2%}")
    print(f"NO_WINNER  - Precision: {results.no_winner_precision:.2%}, "
          f"Recall: {results.no_winner_recall:.2%}, F1: {results.no_winner_f1:.2%}")
    
    print(f"\n--- Confusion Matrix ---")
    print("{:<15} {:<12} {:<12} {:<12} {:<8} {:<12}".format("Actual\\Predicted", "WHITE_WINS", "BLACK_WINS", "NO_WINNER", "ERROR", "PARSE_FAIL"))
    print("-" * 80)
    
    for actual_class in ["WHITE_WINS", "BLACK_WINS", "NO_WINNER"]:
        row = f"{actual_class:<15}"
        for predicted_class in ["WHITE_WINS", "BLACK_WINS", "NO_WINNER", "ERROR", "PARSE_FAIL"]:
            count = results.confusion_matrix[actual_class][predicted_class]
            row += f" {count:<11}"
        print(row)