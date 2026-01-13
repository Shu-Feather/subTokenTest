import logging
from typing import Optional, Dict, Any, Tuple
from typewriter.models.base_model import BaseModel
from configs.typewriter.model_config import ModelConfig
from .shared_loader import load_shared_models

_shared = load_shared_models()
SharedDeepSeekModel = _shared.DeepSeekModel
ensure_usage_aliases = _shared.ensure_usage_aliases

logger = logging.getLogger(__name__)

class DeepSeekModel(BaseModel):
    """DeepSeek API model implementation with usage tracking"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("DeepSeek API key is required")
        
        self.api_key = config.api_key
        self.base_url = config.base_url or "https://api.deepseek.com"
        self.is_reasoning_model = (
            config.additional_params and config.additional_params.get("enable_reasoning", False)
        )

        shared_cfg = {
            "model_name": config.model_id,
            "api_key": config.api_key,
            "base_url": self.base_url,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            "reasoning_effort": config.additional_params.get("reasoning_effort", "medium")
            if config.additional_params
            else "medium",
        }
        self.shared_model = SharedDeepSeekModel(model_name=config.model_id, config=shared_cfg)

        logger.info(f"Initialized DeepSeek model: {config.model_id}")
        if self.is_reasoning_model:
            logger.info("Reasoning mode enabled")
        
    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Generate response using DeepSeek API with usage tracking"""
        try:
            response, usage = self.shared_model.generate(
                [{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                reasoning_effort=kwargs.get("reasoning_effort", "medium"),
                timeout=kwargs.get("timeout", self.config.timeout),
            )
            usage_info = ensure_usage_aliases(usage)
            mapped_usage = {
                "total_tokens": usage_info["total_tokens"],
                "input_tokens": usage_info["input_tokens"],
                "output_tokens": usage_info["output_tokens"],
                "reasoning_tokens": usage_info.get("reasoning_tokens", 0),
            }
            return response, mapped_usage
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            raise Exception(f"DeepSeek API error: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if DeepSeek API is available"""
        try:
            self.shared_model.generate([{"role": "user", "content": "Hi"}], max_tokens=5, timeout=10)
            return True
        except Exception as e:
            logger.error(f"DeepSeek availability check failed: {str(e)}")
            return False
