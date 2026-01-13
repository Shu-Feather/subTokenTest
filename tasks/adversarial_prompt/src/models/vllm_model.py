"""
VLLM-based model implementation for local models.
"""

from typing import List, Dict, Tuple, Optional, Any
from vllm import LLM, SamplingParams
from src.models.base_model import BaseModel


class VLLMModel(BaseModel):
    """VLLM model implementation for local LLMs."""
    
    def __init__(self, model_path: str, config: Dict, verbose: bool = False):
        """
        Initialize VLLM model.
        
        Args:
            model_path: Path to model (local or HuggingFace)
            config: Model configuration
            verbose: Whether to print verbose output
        """
        super().__init__(config, verbose)
        self.model_path = model_path
        self.gpu_memory_utilization = config.get('gpu_memory_utilization', 0.7)
        self.tensor_parallel_size = config.get('tensor_parallel_size', 1)
        self.enforce_eager = config.get('enforce_eager', True)
        self.batch_size = config.get('batch_size', 1)
        
        # Validate resource settings
        if self.gpu_memory_utilization is not None:
            if not 0 < self.gpu_memory_utilization <= 1:
                raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if not isinstance(self.tensor_parallel_size, int) or self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be an integer >= 1")
        if self.enforce_eager is not None and not isinstance(self.enforce_eager, bool):
            raise ValueError("enforce_eager must be a boolean or None")
        
        self._log(f"\n{'='*60}")
        self._log(f"Initializing VLLM model: {model_path}")
        self._log(f"GPU memory utilization: {self.gpu_memory_utilization}")
        self._log(f"Tensor parallel size: {self.tensor_parallel_size}")
        self._log(f"Enforce eager: {self.enforce_eager}")
        self._log(f"{'='*60}\n")
        
        # Initialize VLLM
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=self.tensor_parallel_size,
            enforce_eager=self.enforce_eager,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        
        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            temperature=config.get('temperature', 0.6),
            max_tokens=config.get('max_tokens', 32768),
            top_p=config.get('top_p', 0.95),
        )
        
        self._log("VLLM model initialized successfully\n")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into a single prompt string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        # Format for chat models
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}\n"
        
        formatted += "<|assistant|>\n"
        return formatted
    
    def _extract_usage_info(self, prompt: str, output) -> Dict[str, Any]:
        """
        Extract token usage information from VLLM output.
        
        Args:
            prompt: Input prompt string
            output: VLLM RequestOutput object
            
        Returns:
            Dict with:
                - total_tokens: total tokens used
                - prompt_tokens: tokens in prompt/input
                - completion_tokens: tokens in completion/output
                - reasoning_tokens: 0 (not applicable for VLLM)
                - output_tokens: same as completion_tokens
                - raw_usage: original usage info for debugging
        """
        usage_info = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,  # VLLM doesn't distinguish reasoning tokens
            "output_tokens": 0,
            "raw_usage": {}
        }
        
        try:
            # Extract token counts from VLLM output
            # VLLM RequestOutput has prompt_token_ids and outputs[0].token_ids
            
            # Count prompt tokens
            if hasattr(output, 'prompt_token_ids'):
                prompt_tokens = len(output.prompt_token_ids)
                usage_info["prompt_tokens"] = prompt_tokens
            
            # Count completion tokens
            if hasattr(output, 'outputs') and len(output.outputs) > 0:
                completion_tokens = len(output.outputs[0].token_ids)
                usage_info["completion_tokens"] = completion_tokens
                usage_info["output_tokens"] = completion_tokens
            
            # Calculate total
            usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"]
            
            # Store raw info for debugging
            usage_info["raw_usage"] = {
                "prompt_token_ids_length": usage_info["prompt_tokens"],
                "output_token_ids_length": usage_info["completion_tokens"],
                "finish_reason": output.outputs[0].finish_reason if hasattr(output, 'outputs') and len(output.outputs) > 0 else None,
            }
            
            self._log(f"  Prompt tokens: {usage_info['prompt_tokens']}")
            self._log(f"  Completion tokens: {usage_info['completion_tokens']}")
            self._log(f"  Total tokens: {usage_info['total_tokens']}")
            
        except Exception as e:
            self._log(f"Warning: Could not extract usage info: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        return usage_info
    
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate response using VLLM.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (response_text, usage_info)
            where usage_info contains:
                - total_tokens: total tokens used
                - prompt_tokens: tokens in prompt/input
                - completion_tokens: tokens in completion/output
                - reasoning_tokens: 0 (not applicable for VLLM)
                - output_tokens: same as completion_tokens
                - raw_usage: original usage info for debugging
        """
        # Format messages
        prompt = self._format_messages(messages)
        
        if self.verbose:
            self._log(f"\n{'='*60}")
            self._log("VLLM INPUT PROMPT:")
            self._log(f"{'='*60}")
            self._log(prompt)
            self._log(f"{'='*60}\n")
        
        # Generate
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            output = outputs[0]
            response = output.outputs[0].text.strip()
            
            # Extract usage information
            usage_info = self._extract_usage_info(prompt, output)
            
            if self.verbose:
                self._log(f"\n{'='*60}")
                self._log("VLLM OUTPUT:")
                self._log(f"{'='*60}")
                self._log(response)
                self._log(f"\n{'='*60}")
                self._log("TOKEN USAGE:")
                self._log(f"{'='*60}")
                self._log(f"  Total: {usage_info['total_tokens']}")
                self._log(f"  Prompt: {usage_info['prompt_tokens']}")
                self._log(f"  Completion: {usage_info['completion_tokens']}")
                self._log(f"  Finish Reason: {usage_info['raw_usage'].get('finish_reason', 'N/A')}")
                self._log(f"{'='*60}\n")
            
            return response, usage_info
            
        except Exception as e:
            self._log(f"Error during VLLM generation: {e}")
            print(e)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return "", None

    def shutdown(self):
        """Gracefully shut down vLLM and torch distributed (when initialized)."""
        try:
            if hasattr(self.llm, "shutdown"):
                self.llm.shutdown()
            elif hasattr(self.llm, "llm_engine") and hasattr(self.llm.llm_engine, "shutdown"):
                self.llm.llm_engine.shutdown()
        except Exception as e:
            self._log(f"Warning: vLLM shutdown encountered an issue: {e}")
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            # Silently ignore teardown errors
            pass

    def generate_batch(self, batch_messages: List[List[Dict[str, str]]]) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """
        Generate responses for a batch of message lists in a single or few vLLM calls.
        Uses the configured batch_size to chunk requests to avoid OOM.
        """
        prompts = [self._format_messages(msgs) for msgs in batch_messages]
        results: List[Tuple[str, Optional[Dict[str, Any]]]] = []

        # Chunk to respect batch size
        chunk_size = max(1, int(self.batch_size) if self.batch_size else 1)
        for start in range(0, len(prompts), chunk_size):
            chunk = prompts[start:start + chunk_size]
            try:
                outputs = self.llm.generate(chunk, self.sampling_params)
                for output in outputs:
                    response = output.outputs[0].text.strip()
                    usage_info = self._extract_usage_info("", output)
                    results.append((response, usage_info))
            except Exception as e:
                self._log(f"Error during VLLM batch generation: {e}")
                results.extend([("", None)] * len(chunk))
        return results
