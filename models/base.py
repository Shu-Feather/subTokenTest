"""
Shared base utilities for tokenbench model interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

Message = Dict[str, str]
PromptLike = Union[str, List[Message]]


def ensure_usage_aliases(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize usage dictionaries so callers can rely on common keys.
    Adds aliases for prompt/input and completion/output tokens and fills totals when missing.
    """
    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0) or 0)
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0) or 0)
    usage["prompt_tokens"] = prompt_tokens
    usage["input_tokens"] = usage.get("input_tokens", prompt_tokens)
    usage["completion_tokens"] = completion_tokens
    usage["output_tokens"] = usage.get("output_tokens", completion_tokens)
    usage.setdefault("reasoning_tokens", 0)
    usage.setdefault("raw_usage", {})
    if "total_tokens" not in usage:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage


def normalize_messages(prompt_or_messages: PromptLike) -> List[Message]:
    """Accept either a prompt string or a list of role/content dicts and return a normalized message list."""
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]
    return prompt_or_messages


class BaseModel(ABC):
    """Abstract base class for all shared model interfaces."""

    def __init__(self, model_name: str, verbose: bool = False, batch_size: int = 1):
        self.model_name = model_name
        self.verbose = verbose
        self.batch_size = max(1, int(batch_size or 1))

    def _log(self, message: str):
        if self.verbose:
            print(message)

    @abstractmethod
    def generate(self, prompt_or_messages: PromptLike, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Generate a single response."""
        raise NotImplementedError

    def generate_with_system(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate using separate system and user prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate(messages, **kwargs)

    def generate_batch(
        self, prompts_or_messages: List[PromptLike], **kwargs
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Batch generation fallback that iterates sequentially."""
        return [self.generate(item, **kwargs) for item in prompts_or_messages]

    def _format_messages_as_prompt(self, messages: List[Message]) -> str:
        """
        Convert a chat-style message list to a single prompt string.
        Used by vLLM when no tokenizer chat template is available.
        """
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)
