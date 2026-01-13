from typing import Optional, Dict, Any, Tuple

from typewriter.models.base_model import BaseModel
from configs.typewriter.model_config import ModelConfig
from .shared_loader import load_shared_models

_shared = load_shared_models()
SharedOpenAIModel = _shared.OpenAIModel
ensure_usage_aliases = _shared.ensure_usage_aliases

class OpenAIModel(BaseModel):
    """OpenAI API model implementation with usage tracking"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("OpenAI API key is required")

        shared_cfg = {
            "model_name": config.model_id,
            "api_key": config.api_key,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout": config.timeout,
            "reasoning_effort": getattr(config, "reasoning_effort", "medium"),
        }
        self.shared_model = SharedOpenAIModel(model_name=config.model_id, config=shared_cfg)
    
    def _is_o_series_model(self) -> bool:
        """Check if model is o-series"""
        model_id_lower = self.config.model_id.lower()
        return any(x in model_id_lower for x in ['o1', 'o3', 'o4'])

    def _extract_responses_text(self, response) -> str:
        """
        Extract text output from Responses API response.
        Prefer response.output_text if available, otherwise fall back to assembling text parts.
        """
        try:
            if hasattr(response, "output_text") and response.output_text:
                return (response.output_text or "").strip()
            
            text_parts = []
            output = getattr(response, "output", None)
            if output:
                for item in output:
                    content = getattr(item, "content", None)
                    if not content and isinstance(item, dict):
                        content = item.get("content", [])
                    if content:
                        for c in content:
                            t = getattr(c, "text", None)
                            if not t and isinstance(c, dict):
                                t = c.get("text")
                            if t:
                                text_parts.append(t)
            if text_parts:
                return "".join(text_parts).strip()
        except Exception:
            pass
        return ""
    
    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_text, usage_info)
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            result, usage = self.shared_model.generate(
                messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                timeout=kwargs.get("timeout", self.config.timeout),
            )
            usage_info = ensure_usage_aliases(usage)
            return result, {
                "total_tokens": usage_info["total_tokens"],
                "input_tokens": usage_info["input_tokens"],
                "output_tokens": usage_info["output_tokens"],
                "reasoning_tokens": usage_info.get("reasoning_tokens", 0),
            }
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available"""
        try:
            self.shared_model.generate([{"role": "user", "content": "Hello"}], max_tokens=1)
            return True
        except Exception:
            return False
