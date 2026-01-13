from .base_model import BaseModel
from .openai_model import OpenAIModel
from .deepseek_model import DeepSeekModel
from .huggingface_model import HuggingFaceModel
from .local_model import LocalModel
from configs.typewriter.model_config import ModelConfig, get_model_config

# Try to import vLLM model
try:
    from .vllm_model import VLLMModel, VLLM_AVAILABLE
except ImportError:
    VLLM_AVAILABLE = False
    VLLMModel = None

def create_model(model_name: str, use_vllm: bool = False) -> BaseModel:
    """
    Factory function to create model instances
    
    Args:
        model_name: Name of the model configuration
        use_vllm: If True and model is local, use vLLM for inference
    
    Returns:
        Model instance
    """
    config = get_model_config(model_name)
    
    if config.model_type == "openai":
        return OpenAIModel(config)
    elif config.model_type == "deepseek":
        return DeepSeekModel(config)
    elif config.model_type == "local":
        # Use vLLM if requested and available
        if use_vllm:
            if VLLM_AVAILABLE and VLLMModel is not None:
                return VLLMModel(config)
            else:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("vLLM requested but not available, falling back to HuggingFace")
                return LocalModel(config)
        else:
            return LocalModel(config)
    elif config.model_type == "huggingface":
        return HuggingFaceModel(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

__all__ = [
    "BaseModel", 
    "OpenAIModel", 
    "DeepSeekModel", 
    "HuggingFaceModel", 
    "LocalModel",
    "VLLMModel",
    "create_model",
    "VLLM_AVAILABLE"
]