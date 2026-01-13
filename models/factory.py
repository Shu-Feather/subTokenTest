"""
Factory helpers for creating shared model instances.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .api import APIModel, DeepSeekModel, OpenAIModel
from .vllm import VLLMModel


def create_model(
    model_type: str,
    model_name: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Create a model instance based on a simple type string.

    Args:
        model_type: One of ['api', 'openai', 'deepseek', 'vllm'].
        model_name: Model identifier or path.
        config: Optional configuration dictionary.
        **kwargs: Extra parameters forwarded to the underlying model.
    """
    model_type = (model_type or "").lower()
    if model_type in {"api", "openai"}:
        return OpenAIModel(model_name=model_name, config=config, **kwargs)
    if model_type == "deepseek":
        return DeepSeekModel(model_name=model_name, config=config, **kwargs)
    if model_type in {"vllm", "vllm_chat"}:
        return VLLMModel(model=model_name, config=config, **kwargs)
    raise ValueError(f"Unsupported model type: {model_type}")
