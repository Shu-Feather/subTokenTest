import os
from tkinter import NO
from typing import Optional, Dict, Any, Tuple
import requests
import json

from openai import OpenAI
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

class ModelInterface:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.sampling_params = None
        self.client = None
        
        if config.model_type == "vllm":
            self._init_vllm()
        elif config.model_type in ["openai", "deepseek"]:
            self._init_api_client()
    
    def _init_vllm(self):
        """Initialize VLLM model"""

        if LLM is None or SamplingParams is None:
            raise ImportError("vllm is required for model_type='vllm'. Please install vllm or choose an API model.")
        
        if self.config.verbose:
            print(f"Loading VLLM model: {self.config.model_name}")
        
        self.model = LLM(
            model=self.config.model_name,
            tensor_parallel_size=self.config.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
            max_model_len=self.config.vllm_max_model_len,
            trust_remote_code=True,
            enforce_eager=self.config.vllm_enforce_eager,
        )
        
        self.sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            stop=["</s>", "<|im_end|>", "<|endoftext|>"]
        )
        
        if self.config.verbose:
            print("VLLM model loaded successfully")
    
    def _init_api_client(self):
        """Initialize API client for OpenAI or DeepSeek"""
        
        if self.config.model_type == "openai":
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
        elif self.config.model_type == "deepseek":
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url="https://api.deepseek.com/v1"
            )
        
        if self.config.verbose:
            print(f"API client initialized for {self.config.model_type}")
    
    def get_response(self, prompt: str) -> str:
        """Get response from model (backward compatibility method)"""
        if self.config.model_type == "vllm":
            return self._get_vllm_response(prompt)
        elif self.config.model_type in ["openai", "deepseek"]:
            response, _ = self._get_api_response(prompt)  # Ignore usage info for backward compatibility
            return response if response else ""
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _get_vllm_response(self, prompt: str) -> str:
        """Get response from VLLM model"""
        try:
            outputs = self.model.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            return response
        except Exception as e:
            if self.config.verbose:
                print(f"Error getting VLLM response: {e}")
            return None

    def _get_vllm_batch_responses(self, prompts: list) -> list:
        """Get responses from VLLM model for a list of prompts"""
        responses = []
        try:
            batch_size = max(1, int(getattr(self.config, "vllm_batch_size", 1) or 1))
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i:i + batch_size]
                outputs = self.model.generate(chunk, self.sampling_params)
                for output in outputs:
                    responses.append(output.outputs[0].text.strip() if output.outputs else "")
        except Exception as e:
            if self.config.verbose:
                print(f"Error getting VLLM batch responses: {e}")
            responses.extend([None] * (len(prompts) - len(responses)))
        return responses
    
    def _is_o_series_model(self) -> bool:
        """Return True if model is one of the o-series models we want to route via Responses API."""
        name = (self.config.model_name or "").lower()

        return name in {"o1", "o3", "o3-mini", "o4", "o4-mini"} or name.startswith(("o1-", "o3-", "o4-"))
    
    def _extract_responses_text(self, response) -> str:
        """
        Extract text output from Responses API response.
        Prefer response.output_text if available, otherwise fall back to assembling text parts.
        """
        try:
            if hasattr(response, "output_text") and response.output_text:
                return (response.output_text or "").strip()
            
            text_parts = []
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
    
    def _get_api_response(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response using OpenAI API
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (generated_text, usage_info)
        """
        try:
            response = None
            messages=[{"role": "user", "content": prompt}]

            if self.config.model_type == "deepseek":
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_new_tokens,
                    top_p=self.config.top_p
                )
                
                result = response.choices[0].message.content.strip()
            
            elif self.config.model_type == "openai":
                if self._is_o_series_model():
                    kwargs = {
                        "model": self.config.model_name,
                        "input": messages,
                        "max_output_tokens": self.config.max_new_tokens,
                    }
                    kwargs["reasoning"] = {"effort": self.config.reasoning_effort}

                    response = self.client.responses.create(**kwargs)
                    result = self._extract_responses_text(response)
                
                else:
                    if self.config.model_name.startswith("gpt-5"):
                        response = self.client.chat.completions.create(
                            model=self.config.model_name,
                            messages=messages,
                            timeout=300,
                            max_completion_tokens=self.config.max_new_tokens,
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.config.model_name,
                            messages=messages,
                            temperature=0.0,
                            max_completion_tokens=self.config.max_new_tokens,
                            timeout=300
                        )
                    
                    result = response.choices[0].message.content.strip()
            
            if response == None:
                print(f"Do not support this API model:{self.config.model_type}.")
                return None, None

            # Extract usage information
            usage = response.usage
            usage_info = {
                'total_tokens': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'reasoning_tokens': 0,
            }
            
            if usage:
                usage_info['total_tokens'] = getattr(usage, 'total_tokens', 0)
                usage_info['input_tokens'] = getattr(usage, 'prompt_tokens', 0)
                usage_info['output_tokens'] = getattr(usage, 'completion_tokens', 0)
                
                # Handle reasoning tokens
                # Different API versions may have different structures
                try:
                    completion_details = getattr(usage, 'completion_tokens_details', None)
                    if completion_details:
                        # Try as object attribute first
                        reasoning = getattr(completion_details, 'reasoning_tokens', None)
                        if reasoning is not None:
                            usage_info['reasoning_tokens'] = reasoning
                        # Try as dict access (for some API versions)
                        elif hasattr(completion_details, '__getitem__'):
                            reasoning = completion_details.get('reasoning_tokens', 0)
                            if reasoning is not None:
                                usage_info['reasoning_tokens'] = reasoning
                except (AttributeError, TypeError, KeyError):
                    # If any error occurs, keep reasoning_tokens as 0
                    pass
            
            return result, usage_info

        except Exception as e:
            if self.config.verbose:
                print(f"Error getting API response: {e}")
            return None, None
