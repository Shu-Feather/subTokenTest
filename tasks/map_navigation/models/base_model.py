from abc import ABC, abstractmethod
from typing import Dict, Tuple


class BaseModel(ABC):
    """Base class for all model interfaces."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the model.
        
        Args:
            model_name: Name/path of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.kwargs = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, token_usage_dict)
        """
        pass
    
    @abstractmethod
    def generate_with_system(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        **kwargs
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate response with separate system and user prompts.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, token_usage_dict)
        """
        pass