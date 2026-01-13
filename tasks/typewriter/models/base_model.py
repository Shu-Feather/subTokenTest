from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from configs.typewriter.model_config import ModelConfig

class BaseModel(ABC):
    """Abstract base class for all model implementations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Generate response from the model
        
        Returns:
            Either:
            - str: generated text (for backward compatibility)
            - Tuple[str, Dict]: (generated text, usage_info) for models that support usage tracking
        """
        pass
    
    def generate_with_usage(self, prompt: str, **kwargs) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate response and return usage info if available
        
        Returns:
            Tuple of (generated_text, usage_info or None)
        """
        result = self.generate(prompt, **kwargs)
        
        # Check if model returns usage info
        if isinstance(result, tuple) and len(result) == 2:
            return result
        else:
            # Model doesn't support usage tracking
            return result, None
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and properly configured"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.name,
            "type": self.config.model_type,
            "model_id": self.config.model_id
        }
    
    def supports_usage_tracking(self) -> bool:
        """Check if model supports usage tracking"""
        # Check if generate returns tuple
        try:
            # This is a heuristic - models that return tuples support usage tracking
            return self.config.model_type in ['openai', 'deepseek']
        except:
            return False