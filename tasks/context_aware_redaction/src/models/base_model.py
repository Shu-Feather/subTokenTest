"""
Base model interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


class BaseModel(ABC):
    """
    Abstract base class for all model interfaces
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize base model
        
        Args:
            model_name: Name or path of the model
            config: Configuration dictionary
            verbose: Whether to print verbose output
        """
        self.model_name = model_name
        self.config = config
        self.verbose = verbose
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from model
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        pass
    
    def generate_with_usage(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """
        Generate response with usage information (for API models)
        Default implementation returns empty usage for non-API models
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_text, usage_info)
        """
        result = self.generate(prompt, **kwargs)
        empty_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0
        }
        return result, empty_usage
    
    def batch_generate_with_usage(self, prompts: List[str], **kwargs) -> Tuple[List[str], List[Dict[str, int]]]:
        """
        Generate responses for multiple prompts with usage information
        Default implementation returns empty usage for non-API models
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (list of generated texts, list of usage info)
        """
        results = self.batch_generate(prompts, **kwargs)
        empty_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0
        }
        usage_infos = [empty_usage.copy() for _ in results]
        return results, usage_infos
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"