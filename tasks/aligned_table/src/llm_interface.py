"""
Lightweight LLM interface that delegates to the shared /models package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import VLLMModel, OpenAIModel, DeepSeekModel, ensure_usage_aliases  # type: ignore


class BaseLLM:
    """Base class for LLM interfaces."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_with_usage(self, prompt: str) -> Tuple[str, Dict]:
        raise NotImplementedError

    def batch_generate(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError

    def batch_generate_with_usage(self, prompts: List[str]) -> List[Tuple[str, Dict]]:
        raise NotImplementedError


class VLLMInterface(BaseLLM):
    """Interface for local models using vLLM."""

    def __init__(self, model_path: str, config: Dict):
        vllm_cfg = (config or {}).get("models", {}).get("vllm", {})
        # Ensure the requested model is respected
        vllm_cfg = {**vllm_cfg, "model_name": model_path}
        self.model = VLLMModel(model=model_path, config=vllm_cfg)

    def generate(self, prompt: str) -> str:
        response, _ = self.generate_with_usage(prompt)
        return response

    def generate_with_usage(self, prompt: str) -> Tuple[str, Dict]:
        return self.model.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [resp for resp, _ in self.batch_generate_with_usage(prompts)]

    def batch_generate_with_usage(self, prompts: List[str]) -> List[Tuple[str, Dict]]:
        return self.model.generate_batch(prompts)


class OpenAIInterface(BaseLLM):
    """Interface for OpenAI models via shared APIModel."""

    def __init__(self, model_name: str, config: Dict, api_key: Optional[str] = None):
        api_cfg = (config or {}).get("models", {}).get("openai", {})
        api_cfg = {**api_cfg, "model_name": model_name}
        if api_key:
            api_cfg["api_key"] = api_key
        self.model = OpenAIModel(model_name=model_name, config=api_cfg)

    def generate(self, prompt: str) -> str:
        response, _ = self.generate_with_usage(prompt)
        return response

    def generate_with_usage(self, prompt: str) -> Tuple[str, Dict]:
        return self.model.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [resp for resp, _ in self.batch_generate_with_usage(prompts)]

    def batch_generate_with_usage(self, prompts: List[str]) -> List[Tuple[str, Dict]]:
        return self.model.generate_batch(prompts)


class DeepSeekInterface(BaseLLM):
    """Interface for DeepSeek models via shared APIModel."""

    def __init__(self, model_name: str, config: Dict, api_key: Optional[str] = None):
        api_cfg = (config or {}).get("models", {}).get("deepseek", {})
        api_cfg = {**api_cfg, "model_name": model_name}
        if api_key:
            api_cfg["api_key"] = api_key
        self.model = DeepSeekModel(model_name=model_name, config=api_cfg)

    def generate(self, prompt: str) -> str:
        response, _ = self.generate_with_usage(prompt)
        return response

    def generate_with_usage(self, prompt: str) -> Tuple[str, Dict]:
        return self.model.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [resp for resp, _ in self.batch_generate_with_usage(prompts)]

    def batch_generate_with_usage(self, prompts: List[str]) -> List[Tuple[str, Dict]]:
        return self.model.generate_batch(prompts)


class LLMInterface:
    """Factory class for creating LLM interfaces."""

    @staticmethod
    def create(model_type: str, model_name: str, config: Dict, api_key: Optional[str] = None) -> BaseLLM:
        model_config = config.get("models", {})

        if model_type.lower() == "vllm":
            return VLLMInterface(model_name, {"models": model_config})
        if model_type.lower() == "openai":
            return OpenAIInterface(model_name, {"models": model_config}, api_key)
        if model_type.lower() == "deepseek":
            return DeepSeekInterface(model_name, {"models": model_config}, api_key)
        raise ValueError(f"Unsupported model type: {model_type}")
