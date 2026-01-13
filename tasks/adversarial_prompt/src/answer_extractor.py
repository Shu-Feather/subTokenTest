"""
Answer extraction utilities for parsing LLM outputs.
"""

import re
from typing import Optional, Tuple


class AnswerExtractor:
    """Extract answers from LLM outputs using tag-based parsing."""
    
    def __init__(self, start_tag: str = "<answer>", end_tag: str = "</answer>"):
        """
        Initialize answer extractor.
        
        Args:
            start_tag: Opening tag for answer
            end_tag: Closing tag for answer
        """
        self.start_tag = start_tag
        self.end_tag = end_tag
        
        # Create regex pattern for extraction
        # This handles potential whitespace and is case-insensitive
        escaped_start = re.escape(start_tag)
        escaped_end = re.escape(end_tag)
        self.pattern = re.compile(
            f"{escaped_start}\\s*(.*?)\\s*{escaped_end}",
            re.IGNORECASE | re.DOTALL
        )
    
    def extract(self, text: str) -> Tuple[Optional[str], bool]:
        """
        Extract answer from text using tags.
        
        Args:
            text: LLM output text
            
        Returns:
            Tuple of (extracted_answer, success_flag)
            If tags are found, returns (answer, True)
            If tags are not found, returns (cleaned_text, False)
        """
        # Try to find answer between tags
        match = self.pattern.search(text)
        
        if match:
            # Successfully found answer between tags
            answer = match.group(1).strip()
            return answer, True
        
        # Tags not found - fallback to cleaning the entire response
        # Remove common prefixes and clean up
        cleaned = self._fallback_clean(text)
        return cleaned, False
    
    def _fallback_clean(self, text: str) -> str:
        """
        Fallback cleaning when tags are not found.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove common response patterns
        text = text.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "the canonicalized text is:",
            "canonicalized text:",
            "answer:",
            "result:",
            "output:",
            "here is the canonicalized text:",
            "the answer is:",
        ]
        
        text_lower = text.lower()
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # Remove quotes if the entire text is quoted
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
        
        # Take only the first line if multiple lines
        lines = text.split('\n')
        if lines:
            text = lines[0].strip()
        
        return text.lower()
    
    def validate_extraction(self, text: str) -> dict:
        """
        Validate and analyze the extraction.
        
        Args:
            text: LLM output text
            
        Returns:
            Dictionary with validation information
        """
        answer, success = self.extract(text)
        
        # Count tag occurrences
        start_count = text.lower().count(self.start_tag.lower())
        end_count = text.lower().count(self.end_tag.lower())
        
        validation = {
            "extracted_answer": answer,
            "tags_found": success,
            "start_tag_count": start_count,
            "end_tag_count": end_count,
            "tags_balanced": start_count == end_count,
            "used_fallback": not success
        }
        
        return validation