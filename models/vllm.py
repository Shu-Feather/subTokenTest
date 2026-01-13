"""
vLLM-based local model interface shared across tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import BaseModel, ensure_usage_aliases, normalize_messages

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - optional dependency
    LLM = None
    SamplingParams = None


class VLLMModel(BaseModel):
    """Unified vLLM wrapper with token usage extraction and optional chat templates."""

    def __init__(self, model: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        cfg = dict(config or {})
        cfg.update(kwargs)

        model_id = cfg.get("model_name") or cfg.get("model_path") or model
        super().__init__(model_id, verbose=cfg.get("verbose", False), batch_size=cfg.get("batch_size", 1))

        if LLM is None or SamplingParams is None:
            raise ImportError(
                "vLLM is not installed. Install with `pip install vllm` (or the appropriate CUDA build)."
            )

        self.tensor_parallel_size = cfg.get("tensor_parallel_size", 1)
        self.gpu_memory_utilization = cfg.get("gpu_memory_utilization", 0.7)
        self.max_model_len = cfg.get("max_model_len")
        self.trust_remote_code = cfg.get("trust_remote_code", True)
        self.enforce_eager = cfg.get("enforce_eager", True)
        self.dtype = cfg.get("dtype", "auto")
        self.quantization = cfg.get("quantization")
        self.swap_space = cfg.get("swap_space", 4)
        self.use_chat_template = cfg.get("use_chat_template", False)

        self.temperature = cfg.get("temperature", 0.6)
        self.top_p = cfg.get("top_p", 0.95)
        self.top_k = cfg.get("top_k", -1)
        self.max_tokens = cfg.get("max_tokens", 32768)
        self.stop = cfg.get("stop_tokens") or cfg.get("stop")
        self.repetition_penalty = cfg.get("repetition_penalty", 1.0)

        vllm_kwargs: Dict[str, Any] = {
            "model": model_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "enforce_eager": self.enforce_eager,
            "swap_space": self.swap_space,
        }
        if self.max_model_len is not None:
            vllm_kwargs["max_model_len"] = self.max_model_len
        if self.quantization is not None:
            vllm_kwargs["quantization"] = self.quantization

        self._log(
            f"Initializing vLLM model {model_id} "
            f"(tp={self.tensor_parallel_size}, gpu_mem={self.gpu_memory_utilization})"
        )
        self.llm = LLM(**vllm_kwargs)
        self.tokenizer = None
        try:
            self.tokenizer = self.llm.get_tokenizer()
        except Exception:
            self.tokenizer = None

        self.default_sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=self.stop,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k if self.top_k and self.top_k > 0 else -1,
        )

    def _build_sampling_params(self, **overrides) -> SamplingParams:
        """Create sampling parameters with optional overrides."""
        params = {
            "temperature": overrides.get("temperature", self.temperature),
            "top_p": overrides.get("top_p", self.top_p),
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
            "stop": overrides.get("stop", self.stop),
            "repetition_penalty": overrides.get("repetition_penalty", self.repetition_penalty),
            "top_k": overrides.get("top_k", self.top_k if self.top_k and self.top_k > 0 else -1),
        }
        return SamplingParams(**params)

    def _format_prompt(self, prompt_or_messages) -> str:
        """Format either a raw prompt or chat messages for generation."""
        if isinstance(prompt_or_messages, str):
            return prompt_or_messages

        messages = normalize_messages(prompt_or_messages)
        if self.use_chat_template and self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback to manual formatting if template application fails
                pass
        return self._format_messages_as_prompt(messages)

    def _extract_usage_info(self, output) -> Dict[str, Any]:
        """Extract token counts from vLLM RequestOutput."""
        usage_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,  # vLLM does not expose reasoning tokens
            "output_tokens": 0,
            "total_tokens": 0,
            "raw_usage": {},
        }
        try:
            if hasattr(output, "prompt_token_ids"):
                usage_info["prompt_tokens"] = len(output.prompt_token_ids)
            if hasattr(output, "outputs") and output.outputs:
                token_ids = getattr(output.outputs[0], "token_ids", None)
                if token_ids is not None:
                    usage_info["completion_tokens"] = len(token_ids)
                    usage_info["output_tokens"] = len(token_ids)
            usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"]
            usage_info["raw_usage"] = {
                "prompt_token_count": usage_info["prompt_tokens"],
                "completion_token_count": usage_info["completion_tokens"],
                "finish_reason": (
                    output.outputs[0].finish_reason if hasattr(output, "outputs") and output.outputs else None
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            usage_info["raw_usage"] = {"warning": f"Failed to parse vLLM usage: {exc}"}
        return ensure_usage_aliases(usage_info)

    def generate(self, prompt_or_messages, **kwargs) -> Tuple[str, Dict[str, Any]]:
        prompt = self._format_prompt(prompt_or_messages)
        sampling_params = self._build_sampling_params(**kwargs)
        try:
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0]
            response = output.outputs[0].text.strip() if output.outputs else ""
            usage_info = self._extract_usage_info(output)
            return response, usage_info
        except Exception as e:
            self._log(f"Error during vLLM generation: {e}")
            return "", ensure_usage_aliases({})

    def generate_batch(
        self, prompts_or_messages: List[Any], **kwargs
    ) -> List[Tuple[str, Dict[str, Any]]]:
        formatted_prompts = [self._format_prompt(p) for p in prompts_or_messages]
        sampling_params = self._build_sampling_params(**kwargs)
        results: List[Tuple[str, Dict[str, Any]]] = []
        step = self.batch_size
        for start in range(0, len(formatted_prompts), step):
            chunk = formatted_prompts[start : start + step]
            try:
                outputs = self.llm.generate(chunk, sampling_params)
                for output in outputs:
                    response = output.outputs[0].text.strip() if output.outputs else ""
                    usage_info = self._extract_usage_info(output)
                    results.append((response, usage_info))
            except Exception as e:
                self._log(f"Error during vLLM batch generation: {e}")
                results.extend([("", ensure_usage_aliases({}))] * len(chunk))
        return results

    def shutdown(self):
        """Gracefully tear down vLLM resources."""
        try:
            if hasattr(self.llm, "shutdown"):
                self.llm.shutdown()
            elif hasattr(self.llm, "llm_engine") and hasattr(self.llm.llm_engine, "shutdown"):
                self.llm.llm_engine.shutdown()
        except Exception:
            pass
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
