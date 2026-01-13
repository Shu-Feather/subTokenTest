import os
import logging
from typing import Dict, Any, Optional, List
from typewriter.models.base_model import BaseModel
from configs.typewriter.model_config import ModelConfig
from .shared_loader import load_shared_models

try:
    _shared = load_shared_models()
    SharedVLLMModel = _shared.VLLMModel
    ensure_usage_aliases = _shared.ensure_usage_aliases
    VLLM_AVAILABLE = True
except Exception:
    SharedVLLMModel = None
    ensure_usage_aliases = None
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class VLLMModel(BaseModel):
    """vLLM model implementation for high-performance local model inference"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        if not VLLM_AVAILABLE or SharedVLLMModel is None:
            raise ImportError(
                "vLLM is not installed. Please install it with:\n"
                "pip install vllm\n"
                "or for CUDA 12.1: pip install vllm-cuda121"
            )

        # Validate local path
        if not config.local_path or not os.path.exists(config.local_path):
            raise ValueError(
                f"Local model path does not exist: {config.local_path}\n"
                f"Please set the correct path in environment variables or config."
            )

        vllm_cfg = {
            "model_name": config.local_path,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "batch_size": config.batch_size,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size or 1,
            "enforce_eager": config.enforce_eager,
            "gpu_memory_utilization": config.gpu_memory_utilization,
        }
        if config.additional_params:
            vllm_cfg.update(config.additional_params)

        self.model = SharedVLLMModel(model=config.local_path, config=vllm_cfg)
        self.batch_size = config.batch_size

    def _format_prompt_llama3(self, prompt: str) -> str:
        """Format prompt for Llama 3 models"""
        formatted = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted
    
    def _format_prompt_qwen(self, prompt: str) -> str:
        """Format prompt for Qwen models"""
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return formatted
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt based on model type"""
        model_id_lower = self.config.model_id.lower()
        
        if "llama-3" in model_id_lower or "llama3" in model_id_lower:
            return self._format_prompt_llama3(prompt)
        elif "qwen" in model_id_lower:
            return self._format_prompt_qwen(prompt)
        else:
            # Default: no special formatting
            return prompt
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using vLLM"""
        try:
            formatted_prompt = self._format_prompt(prompt)
            response, usage = self.model.generate(
                formatted_prompt,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
            return response
        except Exception as e:
            logger.error(f"vLLM generation error: {str(e)}")
            raise Exception(f"vLLM generation error: {str(e)}")
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts in batch (vLLM's strength)
        
        Args:
            prompts: List of prompts to generate responses for
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated responses
        """
        try:
            formatted_prompts = [self._format_prompt(p) for p in prompts]
            results = self.model.generate_batch(
                formatted_prompts,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", self.config.top_p),
            )
            return [resp for resp, _ in results]
        except Exception as e:
            logger.error(f"vLLM batch generation error: {str(e)}")
            raise Exception(f"vLLM batch generation error: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if vLLM model is available"""
        return self.model is not None and VLLM_AVAILABLE
