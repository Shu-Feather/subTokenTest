"""
Base model interface for LLM providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any


class BaseModel(ABC):
    """Abstract base class for LLM models."""
    
    def __init__(self, config: Dict, verbose: bool = False):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
            verbose: Whether to print verbose output
        """
        self.config = config
        self.verbose = verbose
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate response from messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (response_text, usage_info)
            where usage_info contains:
                - total_tokens: total tokens used
                - prompt_tokens: tokens in prompt/input
                - completion_tokens: tokens in completion/output
                - reasoning_tokens: tokens used for reasoning (if applicable)
                - output_tokens: visible output tokens
                - raw_usage: original usage object for debugging
        """
        pass
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)