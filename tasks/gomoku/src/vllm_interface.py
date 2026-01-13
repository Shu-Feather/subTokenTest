"""
vLLM interface for local model inference.

This module provides high-performance inference using vLLM for local models.
"""

import openai
import torch
from vllm import LLM, SamplingParams
import os
from typing import Optional, List, Dict, Any
from configs.gomoku.config import ModelConfig, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

class VLLMInterface:
    """Interface for vLLM-based local models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.llm = None
        self.sampling_params = None
        self._initialize_vllm()
    
    def _initialize_vllm(self):
        """Initialize vLLM engine"""
        try:
            # Create sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                top_k=50
            )
            
            # Initialize vLLM engine
            # Determine tensor parallel size based on available GPUs
            detected_tp = torch.cuda.device_count() if torch.cuda.is_available() else 1
            tensor_parallel_size = self.config.tensor_parallel_size or detected_tp
            
            vllm_kwargs = {
                "model": self.config.model_name,
                "tensor_parallel_size": min(tensor_parallel_size, 4),  # Max 4 GPUs
                "trust_remote_code": True,
                "dtype": "auto",
                "enforce_eager": self.config.enforce_eager,  # Disable CUDA graphs to avoid compilation
                "disable_log_stats": True  # Reduce logging overhead
            }
            
            # Add GPU memory utilization if specified
            if self.config.gpu_memory_utilization is not None:
                vllm_kwargs["gpu_memory_utilization"] = self.config.gpu_memory_utilization
            else:
                vllm_kwargs["gpu_memory_utilization"] = 0.9
            
            self.llm = LLM(**vllm_kwargs)
            
        except ImportError:
            raise ImportError(
                "vLLM is required for VLLMInterface. "
                "Install it with: pip install vllm"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM: {e}")
    
    def format_prompt(self, board_representation: str, board_size: int) -> str:
        """Format the user prompt with board information"""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            board_size=board_size,
            board_representation=board_representation
        )
        if getattr(self.config, "restricted_reasoning", False):
            user_prompt = (
                f"{user_prompt}\n\nAnswer directly after <answer> tags without thinking or reasoning. Begin your answer now: <answer>"
            )
        return user_prompt
    
    def _build_chat_prompt(self, user_prompt: str) -> str:
        """Build a chat-formatted prompt"""
        # Try to use the model's chat template if available
        try:
            if hasattr(self.llm, 'get_tokenizer'):
                tokenizer = self.llm.get_tokenizer()
                
                if hasattr(tokenizer, 'apply_chat_template'):
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    return prompt
        except Exception:
            pass
        
        # Fallback to manual formatting
        return f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
    
    def generate_response(
        self, 
        board_representation: str, 
        board_size: int,
        verbose: bool = False
    ) -> Optional[str]:
        """Generate response for a given board using vLLM"""
        try:
            user_prompt = self.format_prompt(board_representation, board_size)
            full_prompt = self._build_chat_prompt(user_prompt)
            
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - vLLM Request")
                print("="*80)
                print(f"Model: {self.config.model_name}")
                print(f"Board Size: {board_size}x{board_size}")
                print(f"\nFull Prompt:\n{full_prompt}")
                print("="*80)
            
            # Generate response using vLLM
            outputs = self.llm.generate([full_prompt], self.sampling_params)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text.strip()
                
                if verbose:
                    print(f"\nModel Response:\n{generated_text}")
                    print("="*80 + "\n")
                
                return generated_text
            else:
                return None
                
        except Exception as e:
            print(f"Error with vLLM generation: {e}")
            return None
    
    def batch_generate_responses(
        self,
        prompts: List[str],
        verbose: bool = False
    ) -> List[Optional[str]]:
        """Generate responses for multiple prompts in batch"""
        try:
            if verbose:
                print(f"\n[vLLM Batch] Processing {len(prompts)} prompts...")
            bs = max(1, int(getattr(self.config, "batch_size", 1) or 1))
            responses: List[Optional[str]] = []
            for i in range(0, len(prompts), bs):
                chunk = prompts[i:i + bs]
                outputs = self.llm.generate(chunk, self.sampling_params)
                for output in outputs:
                    if output.outputs:
                        responses.append(output.outputs[0].text.strip())
                    else:
                        responses.append(None)
            return responses
        except Exception as e:
            print(f"Error with vLLM batch generation: {e}")
            return [None] * len(prompts)


class VLLMServerInterface:
    """Interface for vLLM OpenAI-compatible server"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:8000"
        self.api_key = config.api_key or "EMPTY"
    
    def format_prompt(self, board_representation: str, board_size: int) -> str:
        """Format the user prompt with board information"""
        return USER_PROMPT_TEMPLATE.format(
            board_size=board_size,
            board_representation=board_representation
        )
    
    def generate_response(
        self, 
        board_representation: str, 
        board_size: int,
        verbose: bool = False
    ) -> Optional[str]:
        """Generate response using vLLM OpenAI-compatible server"""
        try:
            # Configure OpenAI client for vLLM server
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=f"{self.base_url}/v1"
            )
            
            user_prompt = self.format_prompt(board_representation, board_size)
            
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - vLLM Server Request")
                print("="*80)
                print(f"Model: {self.config.model_name}")
                print(f"Server: {self.base_url}")
                print(f"Board Size: {board_size}x{board_size}")
                print(f"\nSystem Prompt:\n{SYSTEM_PROMPT}")
                print(f"\nUser Prompt:\n{user_prompt}")
                print("="*80)
            
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            if verbose:
                print(f"\nModel Response:\n{generated_text}")
                print("="*80 + "\n")
            
            return generated_text
            
        except Exception as e:
            print(f"Error with vLLM server: {e}")
            return None
