"""
vLLM model wrapper that delegates to the shared /models package.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import VLLMModel as SharedVLLMModel, ensure_usage_aliases  # type: ignore

from .base_model import BaseModel


class VLLMModel(BaseModel):
    """Async-friendly wrapper around the shared vLLM model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_path = config.get("model_path") or config.get("model_name")
        if not model_path:
            raise ValueError("model_path must be provided for vLLM models")
        self.model = SharedVLLMModel(model=model_path, config=config)
        self.batch_size = config.get("batch_size", 1)

    async def generate_response(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        response, usage = await loop.run_in_executor(None, lambda: self.model.generate(prompt, **kwargs))
        usage = ensure_usage_aliases(usage)
        mapped_usage = {
            "total_tokens": usage["total_tokens"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
        }
        return response, mapped_usage

    def generate_batch(
        self,
        prompts: List[str],
        use_chat_template: bool = None,
        **kwargs,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        results: List[Tuple[str, Dict[str, Any]]] = []
        bs = max(1, int(self.batch_size) if self.batch_size else 1)
        for start in range(0, len(prompts), bs):
            chunk = prompts[start : start + bs]
            chunk_results = self.model.generate_batch(chunk, **kwargs)
            for response, usage in chunk_results:
                usage = ensure_usage_aliases(usage)
                mapped_usage = {
                    "total_tokens": usage["total_tokens"],
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                }
                results.append((response, mapped_usage))
        return results

    def is_available(self) -> bool:
        return True
