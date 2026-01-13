"""
VLLM model interface for local models
"""

from typing import Dict, Any, List, Tuple
from vllm import LLM, SamplingParams

from .base_model import BaseModel


class VLLMModel(BaseModel):
    """
    Interface for models running with VLLM
    Supports local models like Llama, Qwen, etc.
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any], verbose: bool = False):
        """
        Initialize VLLM model
        
        Args:
            model_name: Path to model or HuggingFace model name
            config: Configuration dictionary
            verbose: Whether to print verbose output
        """
        super().__init__(model_name, config, verbose)
        
        vllm_config = config.get('models', {}).get('vllm', {})
        enforce_eager = vllm_config.get('enforce_eager', True)
        self.batch_size = vllm_config.get('batch_size', 1)
        temperature = vllm_config.get('temperature', 0.6)
        top_p = vllm_config.get('top_p', 0.95)
        max_tokens = vllm_config.get('max_tokens', 32768)
        
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=vllm_config.get('gpu_memory_utilization', 0.9),
            tensor_parallel_size=vllm_config.get('tensor_parallel_size', 1),
            max_model_len=vllm_config.get('max_model_len', 4096),
            enforce_eager=enforce_eager,
            trust_remote_code=True
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        if self.verbose:
            print(f"Initialized VLLM model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from VLLM model
        
        Args:
            prompt: Input prompt
            **kwargs: Additional sampling parameters
            
        Returns:
            Generated text
        """
        result, _ = self.generate_with_usage(prompt, **kwargs)
        return result
    
    def generate_with_usage(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """
        Generate response from VLLM model with usage information
        Note: VLLM doesn't track tokens the same way as API models,
        so usage info will be empty for local models
        
        Args:
            prompt: Input prompt
            **kwargs: Additional sampling parameters
            
        Returns:
            Tuple of (generated_text, usage_info)
        """
        # Update sampling params with kwargs
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', self.sampling_params.temperature),
            top_p=kwargs.get('top_p', self.sampling_params.top_p),
            max_tokens=kwargs.get('max_tokens', self.sampling_params.max_tokens),
        )
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("VLLM Input Prompt:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")
        
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("VLLM Output:")
            print(f"{'='*80}")
            print(response)
            print(f"{'='*80}\n")
        
        # VLLM doesn't provide detailed token usage like API models
        # Return empty usage info
        empty_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0
        }
        
        return response, empty_usage
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated texts
        """
        results, _ = self.batch_generate_with_usage(prompts, **kwargs)
        return results
    
    def batch_generate_with_usage(self, prompts: List[str], **kwargs) -> Tuple[List[str], List[Dict[str, int]]]:
        """
        Generate responses for multiple prompts with usage information
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional sampling parameters
            
        Returns:
            Tuple of (list of generated texts, list of usage info)
        """
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', self.sampling_params.temperature),
            top_p=kwargs.get('top_p', self.sampling_params.top_p),
            max_tokens=kwargs.get('max_tokens', self.sampling_params.max_tokens),
        )
        
        if self.verbose:
            print(f"\nGenerating {len(prompts)} responses with VLLM...")
        
        empty_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0
        }
        batch_size = max(1, int(self.batch_size) if self.batch_size else 1)
        responses: List[str] = []
        outputs_usage: List[Dict[str, int]] = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            outputs = self.llm.generate(chunk, sampling_params)
            for output in outputs:
                responses.append(output.outputs[0].text)
                outputs_usage.append(empty_usage.copy())
        
        if self.verbose:
            for i, (prompt, response) in enumerate(zip(prompts, responses)):
                print(f"\n{'='*80}")
                print(f"Batch Item {i+1}/{len(prompts)}")
                print(f"{'='*80}")
                print("Prompt:", prompt[:200] + "..." if len(prompt) > 200 else prompt)
                print("\nResponse:", response[:200] + "..." if len(response) > 200 else response)
                print(f"{'='*80}\n")
        
        # VLLM doesn't provide token usage
        return responses, outputs_usage
