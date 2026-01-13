from .base_model import BaseModel
from .openai_model import OpenAIModel
from .deepseek_model import DeepSeekModel
from .huggingface_model import HuggingFaceModel
from .vllm_model import VLLMModel

__all__ = ['BaseModel', 'OpenAIModel', 'DeepSeekModel', 'HuggingFaceModel', 'VLLMModel']

def create_model(model_config: dict) -> BaseModel:
    """
    Factory function to create model instances based on configuration.
    
    Args:
        model_config (dict): Model configuration dictionary
        
    Returns:
        BaseModel: Instantiated model
        
    Raises:
        ValueError: If model type is unsupported or not available
    """
    model_type = model_config.get('type', '').lower()
    
    if model_type == 'vllm':
        return VLLMModel(model_config)
    
    elif model_type == 'openai':
        return OpenAIModel(model_config)
    
    elif model_type == 'deepseek':
        return DeepSeekModel(model_config)
    
    elif model_type == 'huggingface':
        return HuggingFaceModel(model_config)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")