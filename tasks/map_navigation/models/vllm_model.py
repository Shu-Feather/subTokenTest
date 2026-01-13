from typing import Dict, Tuple, List
from vllm import LLM, SamplingParams
from .base_model import BaseModel


class VLLMModel(BaseModel):
    """Model interface using vLLM for local inference."""
    
    def __init__(
        self, 
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ):
        """
        Initialize vLLM model.
        
        Args:
            model_name: HuggingFace model name or local path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            **kwargs: Additional vLLM parameters
        """
        # Remove non-LLM kwargs before passing to LLM
        batch_size = kwargs.pop('batch_size', 1)
        # sampling params are handled separately; avoid forwarding to LLM
        for k in ['verbose', 'temperature', 'top_p', 'max_tokens', 'reasoning_effort', 'api_key', 'api_base']:
            kwargs.pop(k, None)

        super().__init__(model_name, **kwargs)
        self.batch_size = batch_size
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs
        )
        
        # Get tokenizer for token counting
        self.tokenizer = self.llm.get_tokenizer()
    
    def _create_sampling_params(self, **kwargs) -> SamplingParams:
        """Create sampling parameters for generation."""
        return SamplingParams(
            temperature=kwargs.get('temperature', 0.6),
            top_p=kwargs.get('top_p', 0.95),
            max_tokens=kwargs.get('max_tokens', 32768),
            stop=kwargs.get('stop', None),
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def _format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format chat prompt for the model.
        Different models use different chat templates.
        """
        # Try to use the model's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass
        
        # Fallback to simple format
        return f"{system_prompt}\n\n{user_prompt}"
    
    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, token_usage_dict)
        """
        sampling_params = self._create_sampling_params(**kwargs)
        
        # Count input tokens
        prompt_tokens = self._count_tokens(prompt)
        
        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text
        
        # Count output tokens
        completion_tokens = self._count_tokens(response)
        
        token_usage = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens
        }
        
        return response, token_usage

    def generate_batch(self, prompts: List[str], **kwargs) -> List[Tuple[str, Dict[str, int]]]:
        """Generate responses for multiple prompts using batched vLLM calls."""
        sampling_params = self._create_sampling_params(**kwargs)
        bs = max(1, int(self.batch_size) if self.batch_size else 1)
        results: List[Tuple[str, Dict[str, int]]] = []
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            outputs = self.llm.generate(chunk, sampling_params)
            for prompt, output in zip(chunk, outputs):
                response = output.outputs[0].text
                completion_tokens = self._count_tokens(response)
                prompt_tokens = self._count_tokens(prompt)
                token_usage = {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                }
                results.append((response, token_usage))
        return results

    def generate_batch_with_system(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        **kwargs
    ) -> List[Tuple[str, Dict[str, int]]]:
        """Generate responses with system/user prompts batched together."""
        formatted = [
            self._format_chat_prompt(sys, usr)
            for sys, usr in zip(system_prompts, user_prompts)
        ]
        return self.generate_batch(formatted, **kwargs)
    
    def generate_with_system(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        **kwargs
    ) -> Tuple[str, Dict[str, int]]:
        """
        Generate response with separate system and user prompts.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, token_usage_dict)
        """
        # Format as chat prompt
        full_prompt = self._format_chat_prompt(system_prompt, user_prompt)
        
        return self.generate(full_prompt, **kwargs)
