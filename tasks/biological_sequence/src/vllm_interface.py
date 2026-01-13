"""
vLLM interface for high-performance inference of open-source models.
"""

import os
from typing import Dict, List, Any, Tuple
from .model_interface import BaseModelInterface
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def _build_usage_info(output: Any) -> Dict[str, Any]:
    """Build token usage statistics from a vLLM RequestOutput."""
    usage_info = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "output_tokens": 0,
        "raw_usage": {}
    }

    try:
        prompt_ids = getattr(output, "prompt_token_ids", None) or []
        prompt_tokens = len(prompt_ids)

        completion_tokens = 0
        generated_outputs = getattr(output, "outputs", None)
        if generated_outputs:
            first_output = generated_outputs[0]
            token_ids = getattr(first_output, "token_ids", None)
            if token_ids is not None:
                completion_tokens = len(token_ids)

        usage_info.update({
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "output_tokens": completion_tokens,
            "raw_usage": {
                "prompt_token_count": prompt_tokens,
                "completion_token_count": completion_tokens,
                "source": "vLLM RequestOutput counts"
            }
        })
    except Exception as exc:
        usage_info["raw_usage"] = {"warning": f"Failed to parse vLLM usage: {exc}"}

    return usage_info


class VLLMInterface(BaseModelInterface):
    """
    Interface for vLLM-based model inference.
    
    vLLM provides high-throughput and memory-efficient inference for LLMs.
    Suitable for: Llama, Qwen, Mistral, and other HuggingFace-compatible models.
    Supports both HuggingFace model IDs and local model paths.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize vLLM interface.
        
        Args:
            model_name: Path to local model or HuggingFace model ID
                       Examples:
                       - "/path/to/Llama-3-8B"
                       - "meta-llama/Meta-Llama-3-8B-Instruct"
                       - "/home/user/models/Qwen2-7B-Instruct"
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Check if model_name is a local path
        self.is_local_path = os.path.exists(model_name) and os.path.isdir(model_name)
        
        # Extract vLLM-specific parameters
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        self.max_model_len = kwargs.get('max_model_len', None)
        self.trust_remote_code = kwargs.get('trust_remote_code', True)
        self.dtype = kwargs.get('dtype', 'auto')
        self.quantization = kwargs.get('quantization', None)  # e.g., 'awq', 'gptq'
        self.enforce_eager = kwargs.get('enforce_eager', True)
        self.swap_space = kwargs.get('swap_space', 4)  # CPU swap space in GB
        self.batch_size = kwargs.get('batch_size', 1)
        
        # Sampling parameters
        self.temperature = kwargs.get('temperature', 0.6)
        self.top_p = kwargs.get('top_p', 0.95)
        self.top_k = kwargs.get('top_k', -1)
        self.max_tokens = kwargs.get('max_tokens', 32768)
        self.stop_tokens = kwargs.get('stop_tokens', None)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        
        print(f"Loading vLLM model: {model_name}")
        if self.is_local_path:
            print(f"  Loading from local path")
        print(f"  Tensor parallel size: {self.tensor_parallel_size}")
        print(f"  GPU memory utilization: {self.gpu_memory_utilization}")
        if self.quantization:
            print(f"  Quantization: {self.quantization}")
        
        # Initialize vLLM engine
        try:
            vllm_kwargs = {
                'model': model_name,
                'tensor_parallel_size': self.tensor_parallel_size,
                'gpu_memory_utilization': self.gpu_memory_utilization,
                'trust_remote_code': self.trust_remote_code,
                'dtype': self.dtype,
                'enforce_eager': self.enforce_eager,
                'swap_space': self.swap_space
            }
            
            # Add optional parameters
            if self.max_model_len is not None:
                vllm_kwargs['max_model_len'] = self.max_model_len
            if self.quantization is not None:
                vllm_kwargs['quantization'] = self.quantization
            
            self.llm = LLM(**vllm_kwargs)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load vLLM model: {e}")
        
        # Create sampling parameters
        sampling_kwargs = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens,
            'repetition_penalty': self.repetition_penalty
        }
        
        if self.top_k > 0:
            sampling_kwargs['top_k'] = self.top_k
        if self.stop_tokens is not None:
            sampling_kwargs['stop'] = self.stop_tokens
        
        self.sampling_params = SamplingParams(**sampling_kwargs)
    
    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response using vLLM.
        
        Args:
            prompt: Input prompt string
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of generated response string and usage information
        """
        try:
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - vLLM Request")
                print("="*80)
                print(f"Model: {self.model_name}")
                print(f"\nPrompt:\n{prompt}")
                print("="*80)
            
            # Generate with vLLM
            outputs = self.llm.generate([prompt], self.sampling_params)
            
            # Extract the generated text
            if outputs and len(outputs) > 0:
                request_output = outputs[0]
                generated_text = ""
                if request_output.outputs:
                    generated_text = request_output.outputs[0].text or ""
                generated_text = generated_text.strip()

                usage_info = _build_usage_info(request_output)

                if verbose:
                    print(f"\nModel Response:\n{generated_text}")
                    print("\nToken Usage (counts from vLLM):")
                    print(f"  Total: {usage_info['total_tokens']}")
                    print(f"  Prompt: {usage_info['prompt_tokens']}")
                    print(f"  Completion: {usage_info['completion_tokens']}")
                    print("="*80 + "\n")

                return generated_text, usage_info
            else:
                raise RuntimeError("vLLM returned empty output")
                
        except Exception as e:
            raise RuntimeError(f"vLLM generation error: {e}")
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Generate responses for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            List of generated responses
        """
        try:
            responses = []
            bs = max(1, int(self.batch_size) if self.batch_size else 1)
            for i in range(0, len(prompts), bs):
                chunk = prompts[i:i + bs]
                outputs = self.llm.generate(chunk, self.sampling_params)
                for output in outputs:
                    if output.outputs:
                        responses.append(output.outputs[0].text.strip())
                    else:
                        responses.append("")
            return responses
        except Exception as e:
            raise RuntimeError(f"vLLM batch generation error: {e}")
    
    def __str__(self):
        model_identifier = os.path.basename(self.model_name) if self.is_local_path else self.model_name
        return f"VLLMInterface({model_identifier})"


class VLLMChatInterface(BaseModelInterface):
    """
    Interface for vLLM with chat template support.
    
    This interface applies the model's chat template for better instruction following.
    Supports both HuggingFace model IDs and local model paths.
    Ideal for: Llama-3-Instruct, Qwen2-Instruct, and other instruction-tuned models.
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize vLLM chat interface.
        
        Args:
            model_name: Path to local model or HuggingFace model ID
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        # Check if model_name is a local path
        self.is_local_path = os.path.exists(model_name) and os.path.isdir(model_name)
        
        # Extract parameters
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        self.max_model_len = kwargs.get('max_model_len', None)
        self.trust_remote_code = kwargs.get('trust_remote_code', True)
        self.dtype = kwargs.get('dtype', 'auto')
        self.quantization = kwargs.get('quantization', None)
        self.enforce_eager = kwargs.get('enforce_eager', True)
        self.swap_space = kwargs.get('swap_space', 4)
        self.batch_size = kwargs.get('batch_size', 1)
        
        # Sampling parameters
        self.temperature = kwargs.get('temperature', 0.6)
        self.top_p = kwargs.get('top_p', 0.95)
        self.top_k = kwargs.get('top_k', -1)
        self.max_tokens = kwargs.get('max_tokens', 32768)
        self.stop_tokens = kwargs.get('stop_tokens', None)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        
        # Chat template settings
        self.system_message = kwargs.get('system_message', None)
        self.use_default_system_message = kwargs.get('use_default_system_message', True)
        
        print(f"Loading vLLM chat model: {model_name}")
        if self.is_local_path:
            print(f"  Loading from local path: {model_name}")
        
        # Load tokenizer for chat template
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.trust_remote_code
            )
            print(f"✓ Tokenizer loaded")
            
            # Check if model has chat template
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                print(f"✓ Chat template detected")
            else:
                print(f"⚠ No chat template found, will use simple formatting")
                
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
        
        # Initialize vLLM engine
        try:
            vllm_kwargs = {
                'model': model_name,
                'tensor_parallel_size': self.tensor_parallel_size,
                'gpu_memory_utilization': self.gpu_memory_utilization,
                'trust_remote_code': self.trust_remote_code,
                'dtype': self.dtype,
                'enforce_eager': self.enforce_eager,
                'swap_space': self.swap_space
            }
            
            if self.max_model_len is not None:
                vllm_kwargs['max_model_len'] = self.max_model_len
            if self.quantization is not None:
                vllm_kwargs['quantization'] = self.quantization
            
            self.llm = LLM(**vllm_kwargs)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load vLLM model: {e}")
        
        # Create sampling parameters
        sampling_kwargs = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_tokens': self.max_tokens,
            'repetition_penalty': self.repetition_penalty
        }
        
        if self.top_k > 0:
            sampling_kwargs['top_k'] = self.top_k
        if self.stop_tokens is not None:
            sampling_kwargs['stop'] = self.stop_tokens
        
        self.sampling_params = SamplingParams(**sampling_kwargs)
    
    def _apply_chat_template(self, prompt: str) -> str:
        """
        Apply chat template to the prompt.
        
        Args:
            prompt: User message
            
        Returns:
            Formatted prompt with chat template
        """
        if self.tokenizer is None or not hasattr(self.tokenizer, 'chat_template'):
            # Fallback to simple formatting
            if self.system_message:
                return f"System: {self.system_message}\n\nUser: {prompt}\n\nAssistant:"
            return f"User: {prompt}\n\nAssistant:"
        
        try:
            messages = []
            
            # Add system message if provided or use default
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            elif self.use_default_system_message:
                # Default system message for biology tasks
                default_msg = "You are a helpful and knowledgeable biology assistant. Provide accurate and precise answers."
                messages.append({"role": "system", "content": default_msg})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Apply template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            return formatted_prompt
            
        except Exception as e:
            print(f"Warning: Failed to apply chat template: {e}")
            # Fallback
            if self.system_message:
                return f"System: {self.system_message}\n\nUser: {prompt}\n\nAssistant:"
            return f"User: {prompt}\n\nAssistant:"
    
    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response using vLLM with chat template.
        
        Args:
            prompt: Input prompt string
            verbose: Whether to print detailed logs
            
        Returns:
            Tuple of generated response string and usage information
        """
        try:
            # Apply chat template
            formatted_prompt = self._apply_chat_template(prompt)

            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - vLLM Chat Request")
                print("="*80)
                print(f"Model: {self.model_name}")
                print(f"\nFormatted Prompt:\n{formatted_prompt}")
                print("="*80)
            
            # Generate with vLLM
            outputs = self.llm.generate([formatted_prompt], self.sampling_params)
            
            # Extract the generated text
            if outputs and len(outputs) > 0:
                request_output = outputs[0]
                generated_text = ""
                if request_output.outputs:
                    generated_text = request_output.outputs[0].text or ""
                generated_text = generated_text.strip()

                usage_info = _build_usage_info(request_output)

                if verbose:
                    print(f"\nModel Response:\n{generated_text}")
                    print("\nToken Usage (counts from vLLM):")
                    print(f"  Total: {usage_info['total_tokens']}")
                    print(f"  Prompt: {usage_info['prompt_tokens']}")
                    print(f"  Completion: {usage_info['completion_tokens']}")
                    print("="*80 + "\n")

                return generated_text, usage_info
            else:
                raise RuntimeError("vLLM returned empty output")
                
        except Exception as e:
            raise RuntimeError(f"vLLM generation error: {e}")
    
    def __str__(self):
        model_identifier = os.path.basename(self.model_name) if self.is_local_path else self.model_name
        return f"VLLMChatInterface({model_identifier})"

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts using chat template in batches."""
        bs = max(1, int(self.batch_size) if self.batch_size else 1)
        formatted_prompts = [self._apply_chat_template(p) for p in prompts]
        responses: List[str] = []
        try:
            for i in range(0, len(formatted_prompts), bs):
                chunk = formatted_prompts[i:i + bs]
                outputs = self.llm.generate(chunk, self.sampling_params)
                for output in outputs:
                    if output.outputs:
                        responses.append(output.outputs[0].text.strip())
                    else:
                        responses.append("")
        except Exception as e:
            raise RuntimeError(f"vLLM chat batch generation error: {e}")
        return responses

def main():
    """Example usage of vLLM interfaces."""
    print("vLLM Interface Example")
    print("=" * 50)
    
    # Example 1: Basic vLLM interface
    try:
        print("\n1. Testing basic vLLM interface...")
        model = VLLMInterface(
            "meta-llama/Llama-3-8b-instruct",
            temperature=0.1,
            max_tokens=100
        )
        
        response, usage = model.generate_response("What is DNA?")
        print(f"Response: {response}")
        print(f"Usage: {usage}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Chat interface
    try:
        print("\n2. Testing vLLM chat interface...")
        chat_model = VLLMChatInterface(
            "meta-llama/Llama-3-8b-instruct",
            temperature=0.1,
            max_tokens=100,
            system_message="You are a helpful biology assistant."
        )
        
        response, usage = chat_model.generate_response("What is DNA?")
        print(f"Response: {response}")
        print(f"Usage: {usage}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
