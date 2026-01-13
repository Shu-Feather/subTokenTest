"""
OpenAI-compatible API models shared across tasks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .base import BaseModel, ensure_usage_aliases, normalize_messages


class APIModel(BaseModel):
    """Unified interface for OpenAI-compatible APIs (OpenAI, DeepSeek, etc.)."""

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        **kwargs,
    ):
        cfg = dict(config or {})
        cfg.update(kwargs)

        model_id = cfg.get("model_name", model_name)
        super().__init__(model_id, verbose=cfg.get("verbose", False), batch_size=cfg.get("batch_size", 1))

        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 512)
        self.timeout = cfg.get("timeout", 300)
        self.reasoning_effort = cfg.get("reasoning_effort", "medium")

        # API key / base URL resolution
        self.api_key_env = api_key_env or cfg.get("api_key_env") or "OPENAI_API_KEY"
        resolved_key = api_key or cfg.get("api_key") or os.getenv(self.api_key_env)
        if not resolved_key:
            raise ValueError(f"API key not found in environment variable {self.api_key_env}")

        resolved_base_url = base_url or cfg.get("api_base") or cfg.get("base_url")
        client_kwargs: Dict[str, Any] = {"api_key": resolved_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**client_kwargs)

        self._log(
            f"Initialized API model {self.model_name} "
            f"(base_url={resolved_base_url or 'https://api.openai.com'})"
        )

    # Heuristics for o-series / Responses API routing
    def _is_o_series_model(self) -> bool:
        name = (self.model_name or "").lower()
        return name in {"o1", "o3", "o3-mini", "o4", "o4-mini"} or name.startswith(("o1-", "o3-", "o4-"))

    def _extract_responses_text(self, response) -> str:
        """Extract text output from Responses API response."""
        try:
            if hasattr(response, "output_text") and response.output_text:
                return (response.output_text or "").strip()

            text_parts: List[str] = []
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

    def _extract_usage_info(self, response) -> Dict[str, Any]:
        """
        Extract token usage information across Chat Completions and Responses APIs.
        Returns dict with prompt/completion/reasoning tokens plus aliases.
        """
        usage_info: Dict[str, Any] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "raw_usage": {},
        }

        try:
            usage = None
            if hasattr(response, "usage"):
                usage = response.usage
            elif isinstance(response, dict) and "usage" in response:
                usage = response["usage"]

            if not usage:
                return ensure_usage_aliases(usage_info)

            # Convert to a dictionary representation when needed
            if isinstance(usage, dict):
                usage_dict = usage
            elif hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()
            elif hasattr(usage, "__dict__"):
                usage_dict = usage.__dict__
            else:
                usage_dict = {}

            usage_info["raw_usage"] = usage_dict

            # Responses API format (input_tokens/output_tokens)
            if "input_tokens" in usage_dict or "output_tokens" in usage_dict:
                input_tokens = usage_dict.get("input_tokens", 0) or 0
                output_tokens = usage_dict.get("output_tokens", 0) or 0
                usage_info["prompt_tokens"] = input_tokens
                usage_info["completion_tokens"] = output_tokens
                usage_info["total_tokens"] = input_tokens + output_tokens

                output_details = usage_dict.get("output_tokens_details", {}) or {}
                if not isinstance(output_details, dict) and hasattr(output_details, "model_dump"):
                    output_details = output_details.model_dump()
                elif not isinstance(output_details, dict) and hasattr(output_details, "__dict__"):
                    output_details = output_details.__dict__

                reasoning_tokens = output_details.get("reasoning_tokens", 0) or 0
                audio_tokens = output_details.get("audio_tokens", 0) or 0
                usage_info["reasoning_tokens"] = reasoning_tokens
                usage_info["output_tokens"] = output_tokens - reasoning_tokens - audio_tokens

            # Chat Completions API format (prompt_tokens/completion_tokens)
            elif "prompt_tokens" in usage_dict or "completion_tokens" in usage_dict:
                usage_info["total_tokens"] = usage_dict.get("total_tokens", 0) or 0
                usage_info["prompt_tokens"] = usage_dict.get("prompt_tokens", 0) or 0
                usage_info["completion_tokens"] = usage_dict.get("completion_tokens", 0) or 0

                completion_details = usage_dict.get("completion_tokens_details", {}) or {}
                if not isinstance(completion_details, dict) and hasattr(completion_details, "model_dump"):
                    completion_details = completion_details.model_dump()
                elif not isinstance(completion_details, dict) and hasattr(completion_details, "__dict__"):
                    completion_details = completion_details.__dict__

                reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
                audio_tokens = completion_details.get("audio_tokens", 0) or 0
                accepted_prediction_tokens = completion_details.get("accepted_prediction_tokens", 0) or 0
                usage_info["reasoning_tokens"] = reasoning_tokens
                usage_info["output_tokens"] = (
                    usage_info["completion_tokens"]
                    - reasoning_tokens
                    - audio_tokens
                    - accepted_prediction_tokens
                )

        except Exception as e:
            self._log(f"Warning: failed to parse usage info: {e}")

        return ensure_usage_aliases(usage_info)

    def generate(self, prompt_or_messages, **kwargs) -> Tuple[str, Dict[str, Any]]:
        messages = normalize_messages(prompt_or_messages)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)

        try:
            response = None
            if self._is_o_series_model():
                params: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": messages,
                    "max_output_tokens": max_tokens,
                }
                if reasoning_effort:
                    params["reasoning"] = {"effort": reasoning_effort}
                response = self.client.responses.create(**params, timeout=self.timeout)
                result = self._extract_responses_text(response)
            else:
                chat_kwargs: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "timeout": self.timeout,
                }
                if self.model_name.startswith("gpt-5"):
                    chat_kwargs["max_completion_tokens"] = max_tokens
                else:
                    chat_kwargs["max_completion_tokens"] = max_tokens
                    chat_kwargs["temperature"] = temperature
                response = self.client.chat.completions.create(**chat_kwargs)
                result = response.choices[0].message.content.strip()

            usage_info = self._extract_usage_info(response)
            return result, usage_info
        except Exception as e:
            self._log(f"Error during API call: {e}")
            return "", ensure_usage_aliases({})

    # Simple alias for callers that use this naming
    def generate_with_usage(self, prompt_or_messages, **kwargs) -> Tuple[str, Dict[str, Any]]:
        return self.generate(prompt_or_messages, **kwargs)

    def generate_batch(self, prompts_or_messages: List[Any], **kwargs) -> List[Tuple[str, Dict[str, Any]]]:
        results: List[Tuple[str, Dict[str, Any]]] = []
        for item in prompts_or_messages:
            results.append(self.generate(item, **kwargs))
        return results


class OpenAIModel(APIModel):
    """Explicit alias for OpenAI models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(model_name=model_name, config=config, **kwargs)


class DeepSeekModel(APIModel):
    """DeepSeek API model with default base URL and env var."""

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config,
            base_url=base_url,
            api_key_env=api_key_env,
            **kwargs,
        )
