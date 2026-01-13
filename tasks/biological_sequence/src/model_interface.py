"""
Interface for interacting with different LLM models (open source and closed source).
"""

from email import message
import os
import json
from pyexpat.errors import messages
import time
import requests
from abc import ABC, abstractmethod
import openai
from typing import Dict, List, Optional, Any, Tuple
try:
    from vllm import LLM, SamplingParams
except ImportError:  # vLLM is optional for API-only runs
    LLM = None
    SamplingParams = None
from transformers import AutoTokenizer
from pathlib import Path
import sys

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    OpenAIModel as SharedOpenAIModel,
    DeepSeekModel as SharedDeepSeekModel,
    VLLMModel as SharedVLLMModel,
    ensure_usage_aliases,
)


class BaseModelInterface(ABC):
    """Abstract base class for model interfaces."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt string
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (response_text, usage_info)
            where usage_info contains token usage details:
                - total_tokens: total tokens used
                - prompt_tokens: tokens in prompt/input
                - completion_tokens: tokens in completion/output
                - reasoning_tokens: tokens used for reasoning (if applicable)
                - output_tokens: visible output tokens
                - raw_usage: original usage object for debugging
        """
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.model_name})"


class OpenAIInterface(BaseModelInterface):
    """Interface for OpenAI models (GPT-3.5, GPT-4, etc.)."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = openai.OpenAI(
            api_key=kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
        )
        self.max_tokens = kwargs.get('max_tokens', 8192)
        self.temperature = kwargs.get('temperature', 0.1)
        self.timeout = kwargs.get('timeout', 60)
        self.reasoning_effort = kwargs.get('reasoning_effort', 'medium')  # 'low' | 'medium' | 'high'
    
    def _is_o_series_model(self) -> bool:
        """Return True if model is one of the o-series models we want to route via Responses API."""
        name = (self.model_name or "").lower()
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
    
    def _extract_usage_info(self, response, verbose: bool = False) -> Dict[str, Any]:
        """
        Extract token usage information from API response.
        Handles both Chat Completions API and Responses API formats.
        
        Chat Completions API format:
            - prompt_tokens
            - completion_tokens
            - total_tokens
            - completion_tokens_details.reasoning_tokens
        
        Responses API format (o-series):
            - input_tokens
            - output_tokens
            - output_tokens_details.reasoning_tokens
        
        Returns a dict with:
            - total_tokens: total tokens used
            - prompt_tokens: tokens in prompt/input
            - completion_tokens: tokens in completion/output (including reasoning)
            - reasoning_tokens: tokens used for thinking/reasoning
            - output_tokens: visible output tokens (completion - reasoning - other)
            - raw_usage: original usage object for debugging
        """
        usage_info = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "raw_usage": {}
        }
        
        try:
            usage = None
            
            # Try to get usage from response object
            if hasattr(response, "usage"):
                usage = response.usage
            elif isinstance(response, dict) and "usage" in response:
                usage = response["usage"]
            
            if not usage:
                if verbose:
                    print("Warning: No usage information found in response")
                return usage_info
            
            # Convert to dict if it's an object
            if not isinstance(usage, dict):
                if hasattr(usage, "model_dump"):
                    usage_dict = usage.model_dump()
                elif hasattr(usage, "__dict__"):
                    usage_dict = usage.__dict__
                else:
                    usage_dict = {}
            else:
                usage_dict = usage
            
            # Store raw usage for debugging
            usage_info["raw_usage"] = usage_dict
            
            # Handle Responses API format (o-series: input_tokens, output_tokens)
            if "input_tokens" in usage_dict or "output_tokens" in usage_dict:
                if verbose:
                    print("Detected Responses API format (o-series)")
                
                input_tokens = usage_dict.get("input_tokens", 0)
                output_tokens = usage_dict.get("output_tokens", 0)
                
                usage_info["prompt_tokens"] = input_tokens
                usage_info["completion_tokens"] = output_tokens
                usage_info["total_tokens"] = input_tokens + output_tokens
                
                # Extract reasoning tokens from output_tokens_details
                output_details = usage_dict.get("output_tokens_details", {})
                if not isinstance(output_details, dict) and hasattr(output_details, "model_dump"):
                    output_details = output_details.model_dump()
                elif not isinstance(output_details, dict) and hasattr(output_details, "__dict__"):
                    output_details = output_details.__dict__
                
                reasoning_tokens = output_details.get("reasoning_tokens", 0)
                audio_tokens = output_details.get("audio_tokens", 0)
                
                usage_info["reasoning_tokens"] = reasoning_tokens
                
                # Calculate visible output tokens
                usage_info["output_tokens"] = output_tokens - reasoning_tokens - audio_tokens
                
                if verbose:
                    print(f"  Input tokens: {input_tokens}")
                    print(f"  Output tokens: {output_tokens}")
                    print(f"  Reasoning tokens: {reasoning_tokens}")
                    print(f"  Visible output tokens: {usage_info['output_tokens']}")
            
            # Handle Chat Completions API format (prompt_tokens, completion_tokens)
            elif "prompt_tokens" in usage_dict or "completion_tokens" in usage_dict:
                if verbose:
                    print("Detected Chat Completions API format")
                
                usage_info["total_tokens"] = usage_dict.get("total_tokens", 0)
                usage_info["prompt_tokens"] = usage_dict.get("prompt_tokens", 0)
                usage_info["completion_tokens"] = usage_dict.get("completion_tokens", 0)
                
                # Extract reasoning tokens from completion_tokens_details
                completion_details = usage_dict.get("completion_tokens_details", {})
                if not isinstance(completion_details, dict) and hasattr(completion_details, "model_dump"):
                    completion_details = completion_details.model_dump()
                elif not isinstance(completion_details, dict) and hasattr(completion_details, "__dict__"):
                    completion_details = completion_details.__dict__
                
                reasoning_tokens = completion_details.get("reasoning_tokens", 0)
                audio_tokens = completion_details.get("audio_tokens", 0)
                accepted_prediction_tokens = completion_details.get("accepted_prediction_tokens", 0)
                
                usage_info["reasoning_tokens"] = reasoning_tokens
                
                # Calculate visible output tokens
                usage_info["output_tokens"] = (
                    usage_info["completion_tokens"] 
                    - reasoning_tokens 
                    - audio_tokens 
                    - accepted_prediction_tokens
                )
                
                if verbose:
                    print(f"  Prompt tokens: {usage_info['prompt_tokens']}")
                    print(f"  Completion tokens: {usage_info['completion_tokens']}")
                    print(f"  Reasoning tokens: {reasoning_tokens}")
                    print(f"  Visible output tokens: {usage_info['output_tokens']}")
            
            else:
                if verbose:
                    print("Warning: Unknown usage format")
                    print(f"Available keys: {list(usage_dict.keys())}")
        
        except Exception as e:
            print(f"Warning: Could not extract usage info: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
        
        return usage_info
    
    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate response using OpenAI API."""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - OpenAI API Request")
                print("="*80)
                print(f"Model: {self.model_name}")
                print(f"\nPrompt:\n{prompt}")
                print("="*80)
            
            response = None
            
            if self._is_o_series_model():
                if verbose:
                    print("Using Responses API for o-series model")
                
                kwargs = {
                    "model": self.model_name,
                    "input": messages,
                    "max_output_tokens": self.max_tokens,
                }
                # reasoning effort for o-series
                kwargs["reasoning"] = {"effort": self.reasoning_effort}

                response = self.client.responses.create(**kwargs)
                result = self._extract_responses_text(response)
            else:
                if verbose:
                    print("Using Chat Completions API")
                
                if self.model_name.startswith("gpt-5"):
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        timeout=self.timeout,
                        max_completion_tokens=self.max_tokens
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_completion_tokens=self.max_tokens,
                        timeout=self.timeout
                    )
                
                result = response.choices[0].message.content.strip()
            
            # Extract usage information
            usage_info = self._extract_usage_info(response, verbose=verbose)
            
            if verbose:
                print(f"\nModel Response:\n{result}")
                print(f"\nToken Usage Summary:")
                print(f"  Total: {usage_info['total_tokens']}")
                print(f"  Prompt/Input: {usage_info['prompt_tokens']}")
                print(f"  Completion/Output: {usage_info['completion_tokens']}")
                print(f"  Reasoning: {usage_info['reasoning_tokens']}")
                print(f"  Visible Output: {usage_info['output_tokens']}")
                
                if usage_info['total_tokens'] > 0:
                    thinking_ratio = usage_info['reasoning_tokens'] / usage_info['total_tokens']
                    print(f"  Thinking Ratio: {thinking_ratio:.2%}")
                
                print("="*80 + "\n")
            
            return result, usage_info
        
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return "", None


class DeepSeekInterface(BaseModelInterface):
    """Interface for DeepSeek models."""
    
    def __init__(self, model_name: str = "deepseek-chat", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get('api_key') or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = kwargs.get('base_url', 'https://api.deepseek.com/v1')
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.temperature = kwargs.get('temperature', 0.1)
        self.timeout = kwargs.get('timeout', 300)

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def _extract_usage_info(self, response, verbose: bool = False) -> Dict[str, Any]:
        """Extract token usage information from DeepSeek API response"""
        usage_info = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "output_tokens": 0,
            "raw_usage": {}
        }
        
        try:
            usage = None
            
            # Try to get usage from response object
            if hasattr(response, "usage"):
                usage = response.usage
            elif isinstance(response, dict) and "usage" in response:
                usage = response["usage"]
            
            if not usage:
                if verbose:
                    print("Warning: No usage information found in response")
                return usage_info
            
            # Convert to dict if it's an object
            if not isinstance(usage, dict):
                if hasattr(usage, "model_dump"):
                    usage_dict = usage.model_dump()
                elif hasattr(usage, "__dict__"):
                    usage_dict = usage.__dict__
                else:
                    usage_dict = {}
            else:
                usage_dict = usage
            
            usage_info["raw_usage"] = usage_dict
            usage_info["total_tokens"] = usage_dict.get("total_tokens", 0)
            usage_info["prompt_tokens"] = usage_dict.get("prompt_tokens", 0)
            usage_info["completion_tokens"] = usage_dict.get("completion_tokens", 0)
            
            # DeepSeek may also have reasoning tokens
            completion_details = usage_dict.get("completion_tokens_details", {})
            if not isinstance(completion_details, dict) and hasattr(completion_details, "model_dump"):
                completion_details = completion_details.model_dump()
            elif not isinstance(completion_details, dict) and hasattr(completion_details, "__dict__"):
                completion_details = completion_details.__dict__
            
            reasoning_tokens = completion_details.get("reasoning_tokens", 0)
            usage_info["reasoning_tokens"] = reasoning_tokens
            usage_info["output_tokens"] = usage_info["completion_tokens"] - reasoning_tokens
            
            if verbose:
                print(f"  Prompt tokens: {usage_info['prompt_tokens']}")
                print(f"  Completion tokens: {usage_info['completion_tokens']}")
                print(f"  Reasoning tokens: {reasoning_tokens}")
                print(f"  Output tokens: {usage_info['output_tokens']}")
        
        except Exception as e:
            print(f"Warning: Could not extract usage info: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
        
        return usage_info
    
    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate response using DeepSeek API."""
        try:
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - DeepSeek API Request")
                print("="*80)
                print(f"Model: {self.model_name}")
                print(f"\nPrompt:\n{prompt}")
                print("="*80)
            
            messages = [{'role': 'user', 'content': prompt}]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )

            result = response.choices[0].message.content.strip()
            
            # Extract usage information
            usage_info = self._extract_usage_info(response, verbose=verbose)
            
            if verbose:
                print(f"\nModel Response:\n{result}")
                print(f"\nToken Usage:")
                print(f"  Total: {usage_info['total_tokens']}")
                print(f"  Prompt: {usage_info['prompt_tokens']}")
                print(f"  Completion: {usage_info['completion_tokens']}")
                print(f"  Reasoning: {usage_info['reasoning_tokens']}")
                print(f"  Output: {usage_info['output_tokens']}")
                print("="*80 + "\n")
            
            return result, usage_info
        
        except Exception as e:
            print(f"Error during DeepSeek API call: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return "", None
        
    def __str__(self):
        return f"DeepSeekInterface({self.model_name})"

class DeepSeekVLLMInterface(BaseModelInterface):
    """
    Interface for running DeepSeek models locally with vLLM.
    
    This allows you to run DeepSeek models downloaded to your local machine
    using vLLM for high-performance inference.
    """
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize local DeepSeek model with vLLM.
        
        Args:
            model_path: Path to local DeepSeek model directory
            **kwargs: vLLM configuration parameters
        """
        super().__init__(model_path, **kwargs)
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # vLLM parameters
        self.tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.9)
        self.max_model_len = kwargs.get('max_model_len', None)
        self.trust_remote_code = kwargs.get('trust_remote_code', True)
        self.dtype = kwargs.get('dtype', 'auto')
        
        # Sampling parameters
        self.temperature = kwargs.get('temperature', 0.1)
        self.top_p = kwargs.get('top_p', 1.0)
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        
        print(f"Loading local DeepSeek model with vLLM: {model_path}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
        
        # Load model with vLLM
        try:
            vllm_kwargs = {
                'model': model_path,
                'tensor_parallel_size': self.tensor_parallel_size,
                'gpu_memory_utilization': self.gpu_memory_utilization,
                'trust_remote_code': self.trust_remote_code,
                'dtype': self.dtype
            }
            
            if self.max_model_len:
                vllm_kwargs['max_model_len'] = self.max_model_len
            
            self.llm = LLM(**vllm_kwargs)
            print("✓ Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load DeepSeek model with vLLM: {e}")
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty
        )
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for DeepSeek models."""
        if self.tokenizer and hasattr(self.tokenizer, 'chat_template'):
            try:
                messages = [{'role': 'user', 'content': prompt}]
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                pass
        
        # Fallback formatting
        return f"User: {prompt}\n\nAssistant:"
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Rough estimation: ~4 chars per token
        return len(text) // 4
    
    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate response using local DeepSeek model.
        
        Args:
            prompt: Input prompt string
            verbose: Whether to print detailed information
            
        Returns:
            Tuple of (response_text, usage_info)
        """
        try:
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - DeepSeek vLLM Request")
                print("="*80)
                print(f"Model: {os.path.basename(self.model_name)}")
                print(f"\nPrompt:\n{prompt}")
                print("="*80)
            
            formatted_prompt = self._format_prompt(prompt)
            
            # Estimate prompt tokens
            prompt_tokens = self._estimate_tokens(formatted_prompt)
            
            outputs = self.llm.generate([formatted_prompt], self.sampling_params)
            
            if outputs and len(outputs) > 0:
                result = outputs[0].outputs[0].text.strip()
                
                # Estimate completion tokens
                completion_tokens = self._estimate_tokens(result)
                
                # Create usage info (vLLM doesn't provide exact token counts by default)
                usage_info = {
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "reasoning_tokens": 0,
                    "output_tokens": completion_tokens,
                    "raw_usage": {
                        "note": "Token counts are estimates for vLLM local models"
                    }
                }
                
                if verbose:
                    print(f"\nModel Response:\n{result}")
                    print(f"\nToken Usage (Estimated):")
                    print(f"  Total: {usage_info['total_tokens']}")
                    print(f"  Prompt: {usage_info['prompt_tokens']}")
                    print(f"  Completion: {usage_info['completion_tokens']}")
                    print("="*80 + "\n")
                
                return result, usage_info
            else:
                raise RuntimeError("vLLM returned empty output")
        
        except Exception as e:
            print(f"DeepSeek vLLM generation error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return "", None
    
    def __str__(self):
        return f"DeepSeekVLLMInterface({os.path.basename(self.model_name)})"

class HuggingFaceInterface(BaseModelInterface):
    """Interface for HuggingFace models (Llama, Qwen, etc.)."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("transformers and torch are required for HuggingFace models")
        
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
        self.temperature = kwargs.get('temperature', 0.1)
        self.do_sample = kwargs.get('do_sample', True)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate response using HuggingFace model."""
        try:
            import torch
            
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - HuggingFace Model Request")
                print("="*80)
                print(f"Model: {self.model_name}")
                print(f"\nPrompt:\n{prompt}")
                print("="*80)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            prompt_tokens = inputs.shape[1]
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response (remove input prompt)
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            result = response.strip()
            
            # Calculate token usage
            completion_tokens = outputs.shape[1] - prompt_tokens
            usage_info = {
                "total_tokens": outputs.shape[1],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning_tokens": 0,
                "output_tokens": completion_tokens,
                "raw_usage": {}
            }
            
            if verbose:
                print(f"\nModel Response:\n{result}")
                print(f"\nToken Usage:")
                print(f"  Total: {usage_info['total_tokens']}")
                print(f"  Prompt: {usage_info['prompt_tokens']}")
                print(f"  Completion: {usage_info['completion_tokens']}")
                print("="*80 + "\n")
            
            return result, usage_info
        
        except Exception as e:
            print(f"HuggingFace model error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return "", None


class OllamaInterface(BaseModelInterface):
    """Interface for Ollama models (local deployment)."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = kwargs.get('base_url', 'http://localhost:11434')
        self.temperature = kwargs.get('temperature', 0.1)
        self.num_predict = kwargs.get('num_predict', 512)
    
    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate response using Ollama API."""
        try:
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - Ollama API Request")
                print("="*80)
                print(f"Model: {self.model_name}")
                print(f"\nPrompt:\n{prompt}")
                print("="*80)
            
            data = {
                'model': self.model_name,
                'prompt': prompt,
                'options': {
                    'temperature': self.temperature,
                    'num_predict': self.num_predict
                }
            }
            
            response = requests.post(
                f'{self.base_url}/api/generate',
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            # Ollama returns streaming response, need to parse
            result = ""
            for line in response.text.strip().split('\n'):
                if line:
                    json_response = json.loads(line)
                    result += json_response.get('response', '')
                    if json_response.get('done', False):
                        break
            
            # Ollama doesn't provide detailed token usage, create placeholder
            usage_info = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "raw_usage": {
                    "note": "Ollama does not provide token usage information"
                }
            }
            
            if verbose:
                print(f"\nModel Response:\n{result}")
                print("\nNote: Token usage not available for Ollama models")
                print("="*80 + "\n")
            
            return result, usage_info
        
        except Exception as e:
            print(f"Ollama API error: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return "", None


class SharedAPIInterface(BaseModelInterface):
    """Wrapper that delegates API calls to the shared /models package."""

    def __init__(self, provider: str, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        config = {"model_name": model_name, **kwargs}
        if provider == "openai":
            self.model = SharedOpenAIModel(model_name=model_name, config=config)
        else:
            self.model = SharedDeepSeekModel(model_name=model_name, config=config)

    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        response, usage = self.model.generate(prompt)
        return response, ensure_usage_aliases(usage)


class SharedVLLMInterface(BaseModelInterface):
    """Wrapper that delegates vLLM calls to the shared /models package."""

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        config = {"model_name": model_name, **kwargs}
        self.model = SharedVLLMModel(model=model_name, config=config)

    def generate_response(self, prompt: str, verbose: bool = False) -> Tuple[str, Optional[Dict[str, Any]]]:
        response, usage = self.model.generate(prompt)
        return response, ensure_usage_aliases(usage)


class ModelFactory:
    """Factory class for creating model interfaces."""
    
    @staticmethod
    def create_model_interface(provider: str, model_name: str, **kwargs) -> BaseModelInterface:
        """
        Create a model interface based on provider type.
        
        Args:
            provider: Model provider ('openai', 'deepseek', 'huggingface', 'ollama')
            model_name: Name of the model
            **kwargs: Additional configuration parameters
            
        Returns:
            Model interface instance
        """
        provider = provider.lower()
        
        if provider == 'openai':
            return SharedAPIInterface('openai', model_name, **kwargs)
        elif provider == 'deepseek':
            return SharedAPIInterface('deepseek', model_name, **kwargs)
        elif provider == 'huggingface':
            return HuggingFaceInterface(model_name, **kwargs)
        elif provider == 'ollama':
            return OllamaInterface(model_name, **kwargs)
        elif provider == 'vllm':
            return SharedVLLMInterface(model_name, **kwargs)
        elif provider == 'vllm_chat':
            return SharedVLLMInterface(model_name, use_chat_template=True, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def load_model_from_config(config_path: str, model_key: str) -> BaseModelInterface:
        """
        Load model interface from configuration file.
        
        Args:
            config_path: Path to configuration JSON file
            model_key: Key of the model in configuration
            
        Returns:
            Model interface instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if model_key not in config['models']:
            raise KeyError(f"Model {model_key} not found in configuration")
        
        model_config = config['models'][model_key]
        provider = model_config['provider']
        model_name = model_config['model_name']
        params = model_config.get('parameters', {})
        
        return ModelFactory.create_model_interface(provider, model_name, **params)


def main():
    """Example usage of model interfaces."""
    # Example with OpenAI (requires API key)
    try:
        openai_model = ModelFactory.create_model_interface(
            'openai', 
            'gpt-3.5-turbo',
            temperature=0.1
        )
        print(f"Created model: {openai_model}")
        
        # Test generation with usage info
        response, usage = openai_model.generate_response(
            "What is DNA?", 
            verbose=True
        )
        print(f"\nResponse: {response}")
        print(f"Usage: {usage}")
    except Exception as e:
        print(f"Failed to create OpenAI model: {e}")


if __name__ == "__main__":
    main()
