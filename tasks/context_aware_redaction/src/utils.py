"""
Utility functions for the benchmark
"""

import re
import yaml
import json
import os
from typing import Dict, Any, List, Tuple


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def extract_answer(text: str) -> str:
    """
    Extract answer from text between <answer> and </answer> tags
    
    Args:
        text: Text containing answer tags
        
    Returns:
        Extracted answer or empty string if not found
    """
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Return the last match (in case there are multiple)
        return matches[-1].strip()
    return ""


def apply_redaction_rules(text: str, info_type: str, value: str) -> str:
    """
    Apply redaction rules to sensitive information
    
    Args:
        text: Original text
        info_type: Type of sensitive information (phone/id_card/credit_card)
        value: The sensitive value to redact
        
    Returns:
        Redacted value
    """
    if info_type == "id_card":
        # 18-digit ID: keep first 6 and last 2, mask middle 10
        if len(value) == 18:
            return value[:6] + "*" * 10 + value[-2:]
        return value
        
    elif info_type == "phone":
        # Format: +86 18355132086
        # Keep + and area code, keep first 3 and last 4 of phone number, mask middle 4
        match = re.match(r'(\+\d+)\s*(\d{3})(\d{4})(\d{4})', value.replace(' ', ''))
        if match:
            area, first3, middle4, last4 = match.groups()
            return f"{area} {first3}****{last4}"
        return value
        
    elif info_type == "credit_card":
        # Keep first 6 and last 4, mask middle
        digits_only = re.sub(r'\s+', '', value)
        if len(digits_only) >= 10:
            middle_len = len(digits_only) - 10
            return digits_only[:6] + "*" * middle_len + digits_only[-4:]
        return value
    
    return value


def parse_sensitive_info(text: str) -> List[Dict[str, str]]:
    """
    Parse sensitive information from text
    
    Args:
        text: Text to parse
        
    Returns:
        List of dictionaries containing type and value of sensitive info
    """
    sensitive_info = []
    used_ranges = []  # List of (start, end) tuples for used positions
    
    def overlaps_with_used(start: int, end: int) -> bool:
        """Check if range overlaps with any used range"""
        for used_start, used_end in used_ranges:
            if not (end <= used_start or start >= used_end):
                return True
        return False
    
    def add_used_range(start: int, end: int):
        """Add a range to used ranges"""
        used_ranges.append((start, end))
    
    # Priority 1: Phone numbers (most specific pattern)
    # Format: +XX XXXXXXXXXXX (+ sign, 1-3 digits, space, 11 digits)
    phone_pattern = r'\+\d{1,3}\s+\d{11}'
    for match in re.finditer(phone_pattern, text):
        start, end = match.start(), match.end()
        if not overlaps_with_used(start, end):
            sensitive_info.append({
                "type": "phone",
                "value": match.group(),
                "start": start,
                "end": end
            })
            add_used_range(start, end)
    
    # Priority 2: Credit cards with formatting (spaces or dashes)
    # Format: XXXX XXXX XXXX XXXX or XXXX-XXXX-XXXX-XXXX
    # This pattern REQUIRES spaces or dashes
    card_pattern = r'\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4,7}'
    for match in re.finditer(card_pattern, text):
        start, end = match.start(), match.end()
        if not overlaps_with_used(start, end):
            value = match.group()
            digits_only = re.sub(r'[\s-]', '', value)
            
            # Validate: 13-19 digits total
            if 13 <= len(digits_only) <= 19:
                sensitive_info.append({
                    "type": "credit_card",
                    "value": value,
                    "start": start,
                    "end": end
                })
                add_used_range(start, end)
    
    # Priority 3: ID cards (exactly 18 consecutive digits, no formatting)
    # Find all sequences of digits
    digit_pattern = r'\d+'
    for match in re.finditer(digit_pattern, text):
        start, end = match.start(), match.end()
        value = match.group()
        
        # Check if exactly 18 digits
        if len(value) == 18:
            if not overlaps_with_used(start, end):
                # Additional check: make sure it's not part of credit card formatting
                # (already handled by checking overlaps)
                sensitive_info.append({
                    "type": "id_card",
                    "value": value,
                    "start": start,
                    "end": end
                })
                add_used_range(start, end)
    
    # Sort by position
    sensitive_info.sort(key=lambda x: x["start"])
    
    return sensitive_info


def create_ground_truth(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Create ground truth redacted text and metadata
    
    Args:
        text: Original text
        
    Returns:
        Tuple of (redacted_text, sensitive_info_list)
    """
    sensitive_info = parse_sensitive_info(text)
    redacted_text = text
    
    # Apply redactions in reverse order to maintain positions
    for info in reversed(sensitive_info):
        redacted_value = apply_redaction_rules(text, info["type"], info["value"])
        redacted_text = (
            redacted_text[:info["start"]] + 
            redacted_value + 
            redacted_text[info["end"]:])
    
    return redacted_text, sensitive_info


def save_json(data: Any, filepath: str) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Path to output file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to input file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()