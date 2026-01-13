"""
DeepSeek model wrapper that delegates to the shared /models package.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DeepSeekModel as SharedDeepSeekModel, ensure_usage_aliases  # type: ignore

from .base_model import BaseModel


class DeepSeekModel(BaseModel):
    """Async-compatible wrapper around the shared DeepSeek API model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = SharedDeepSeekModel(
            model_name=config.get("model_name", "deepseek-chat"),
            config=config,
            base_url=config.get("base_url", "https://api.deepseek.com"),
        )

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

    def is_available(self) -> bool:
        return True
