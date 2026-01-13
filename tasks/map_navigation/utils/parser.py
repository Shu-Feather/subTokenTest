# utils/parser.py

import re
import json
from typing import Optional


def parse_answer(response: str) -> Optional[str]:
    """
    Parse the answer from LLM response.
    Looks for content between <answer> and </answer> tags.
    
    Args:
        response: The full response from the LLM
        
    Returns:
        Extracted answer or None if not found
    """
    # Try to find answer between tags
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if match:
        answer = match.group(1).strip()
        return answer
    
    # If no tags found, return None
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    
    Args:
        answer: Raw answer string
        
    Returns:
        Normalized answer
    """
    if answer is None:
        return ""
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    # Convert to lowercase for case-insensitive comparison
    answer = answer.lower()
    
    return answer


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """
    Compare predicted answer with ground truth using exact match.
    
    Args:
        predicted: Answer from the model
        ground_truth: Correct answer
        
    Returns:
        True if answers match, False otherwise
    """
    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    # Exact match
    return pred_norm == gt_norm


def parse_coordinate(answer: str) -> Optional[tuple]:
    """
    Parse coordinate from answer string.
    Supports both positive and negative integers.
    
    Args:
        answer: Answer string like "(3, 4)" or "3, 4" or "(-1, -2)"
        
    Returns:
        Tuple of (x, y) or None if parsing fails
    """
    # Updated pattern to support negative numbers
    # Matches: optional '(', optional '-', digits, comma, optional '-', digits, optional ')'
    pattern = r'\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?'
    match = re.search(pattern, answer)
    
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return (x, y)
    
    return None


def parse_json_answer(answer: str) -> Optional[dict]:
    """
    Parse JSON from answer string.
    
    Args:
        answer: Answer string containing JSON
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    try:
        # Try to parse as JSON
        return json.loads(answer)
    except json.JSONDecodeError:
        # Try to find JSON object in the string
        pattern = r'\{[^}]+\}'
        match = re.search(pattern, answer)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    
    return None