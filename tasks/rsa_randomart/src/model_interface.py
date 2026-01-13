"""
Model interface wrappers that delegate to the shared /models package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import VLLMModel as SharedVLLMModel, OpenAIModel as SharedOpenAIModel, ensure_usage_aliases  # type: ignore


class ModelInterface:
    """Abstract base class for model interfaces"""

    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict]:
        raise NotImplementedError

    def cleanup(self):
        pass


class VLLMInterface(ModelInterface):
    """Interface for VLLM models using the shared wrapper."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        max_tokens: int = 32768,
        temperature: float = 0.6,
        top_p: float = 0.95,
        batch_size: int = 1,
        enforce_eager: bool = True,
        **kwargs,
    ):
        config = {
            "model_name": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "batch_size": batch_size,
            "enforce_eager": enforce_eager,
        }
        self.model = SharedVLLMModel(model=model_path, config=config)

    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict]:
        response, usage = self.model.generate(prompt, **kwargs)
        return response, ensure_usage_aliases(usage)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[Tuple[str, Dict]]:
        return self.model.generate_batch(prompts, **kwargs)

    def cleanup(self):
        if hasattr(self.model, "shutdown"):
            try:
                self.model.shutdown()
            except Exception:
                pass


class OpenAIInterface(ModelInterface):
    """Interface for OpenAI API using the shared wrapper."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        timeout: int = 60,
        max_tokens: int = 6000,
        reasoning_effort: str = "medium",
        **kwargs,
    ):
        config = {
            "model_name": model_name,
            "api_key": api_key,
            "base_url": api_base,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
        self.model = SharedOpenAIModel(model_name=model_name, config=config)

    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict]:
        response, usage = self.model.generate(prompt, **kwargs)
        return response, ensure_usage_aliases(usage)

    def cleanup(self):
        pass


def create_model_interface(
    model_type: str,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    config: Optional[Dict] = None,
    **kwargs,
):
    """
    Factory helper to create a model interface based on type.

    Args:
        model_type: 'vllm', 'openai', or 'deepseek'
        model_path: Path/ID for vLLM models
        model_name: Model name for API models
        config: Optional configuration dictionary (expects a 'models' section)
        kwargs: Additional parameters forwarded to the interface
    """
    model_type = (model_type or "").lower()
    cfg_models = (config or {}).get("models", {}) if isinstance(config, dict) else {}

    if model_type == "vllm":
        vllm_cfg = cfg_models.get("vllm", {})
        path = model_path or vllm_cfg.get("model_path")
        if not path:
            raise ValueError("model_path is required for vLLM models")
        return VLLMInterface(
            model_path=path,
            tensor_parallel_size=vllm_cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.9),
            max_model_len=vllm_cfg.get("max_model_len", 4096),
            max_tokens=vllm_cfg.get("max_tokens", 32768),
            temperature=vllm_cfg.get("temperature", 0.6),
            top_p=vllm_cfg.get("top_p", 0.95),
            batch_size=vllm_cfg.get("batch_size", 1),
            enforce_eager=vllm_cfg.get("enforce_eager", True),
            **kwargs,
        )

    if model_type == "openai":
        openai_cfg = cfg_models.get("openai", {})
        return OpenAIInterface(
            model_name=model_name or openai_cfg.get("model_name", "gpt-4"),
            api_key=openai_cfg.get("api_key") or kwargs.get("api_key"),
            api_base=openai_cfg.get("api_base", "https://api.openai.com/v1"),
            timeout=openai_cfg.get("timeout", 300),
            max_tokens=openai_cfg.get("max_tokens", 6000),
            reasoning_effort=kwargs.get("reasoning_effort", openai_cfg.get("reasoning_effort", "medium")),
        )

    if model_type == "deepseek":
        deepseek_cfg = cfg_models.get("deepseek", {})
        return OpenAIInterface(
            model_name=model_name or deepseek_cfg.get("model_name", "deepseek-chat"),
            api_key=deepseek_cfg.get("api_key") or kwargs.get("api_key"),
            api_base=deepseek_cfg.get("api_base", "https://api.deepseek.com/v1"),
            timeout=deepseek_cfg.get("timeout", 300),
            max_tokens=deepseek_cfg.get("max_tokens", 6000),
            reasoning_effort=kwargs.get("reasoning_effort", deepseek_cfg.get("reasoning_effort", "medium")),
        )

    raise ValueError(f"Unsupported model type: {model_type}")


__all__ = [
    "ModelInterface",
    "VLLMInterface",
    "OpenAIInterface",
    "create_model_interface",
]
