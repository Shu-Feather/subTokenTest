"""
Model interface for different LLM providers.
"""

from email import message
try:
    from openai import OpenAI
except ImportError:  # Optional: old interfaces are not used in the refactored path
    OpenAI = None
import os
import requests
import json
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import sys
from configs.gomoku.config import ModelConfig, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cli.py").exists():
            return parent
    return Path(__file__).resolve().parents[-1]


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import OpenAIModel as SharedOpenAIModel, DeepSeekModel as SharedDeepSeekModel, VLLMModel as SharedVLLMModel, ensure_usage_aliases  # type: ignore

class ModelInterface(ABC):
    """Abstract base class for model interfaces"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    @abstractmethod
    def generate_response(
        self, 
        board_representation: str, 
        board_size: int,
        verbose: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Generate response for a given board
        
        Returns:
            Tuple of (response_text, usage_info)
            where usage_info contains token usage details
        """
        pass
    
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


class SharedAPIModelInterface(ModelInterface):
    """Wrapper that routes API calls through the shared /models package."""

    def __init__(self, config: ModelConfig, provider: str):
        super().__init__(config)
        api_config = {
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "reasoning_effort": config.reasoning_effort,
            "timeout": config.timeout,
            "api_key": config.api_key,
            "base_url": config.base_url,
        }
        if provider == "openai":
            self.model = SharedOpenAIModel(model_name=config.model_name, config=api_config)
        else:
            self.model = SharedDeepSeekModel(
                model_name=config.model_name,
                config=api_config,
                base_url=config.base_url or "https://api.deepseek.com",
                api_key_env="DEEPSEEK_API_KEY",
            )

    def generate_response(
        self, board_representation: str, board_size: int, verbose: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        user_prompt = self.format_prompt(board_representation, board_size)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response, usage = self.model.generate(messages)
        return response, ensure_usage_aliases(usage)


class SharedVLLMInterface(ModelInterface):
    """Wrapper that routes vLLM calls through the shared /models package."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        vllm_cfg = {
            "model_name": config.model_name,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "batch_size": config.batch_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "tensor_parallel_size": config.tensor_parallel_size,
            "enforce_eager": config.enforce_eager,
            "use_chat_template": True,
        }
        self.model = SharedVLLMModel(model=config.model_name, config=vllm_cfg)

    def generate_response(
        self, board_representation: str, board_size: int, verbose: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        user_prompt = self.format_prompt(board_representation, board_size)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response, usage = self.model.generate(messages)
        return response, ensure_usage_aliases(usage)

class OpenAIInterface(ModelInterface):
    """Interface for OpenAI models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        api_key = getattr(config, "api_key", None) or os.getenv("OPENAI_API_KEY")
        base_url = getattr(config, "base_url", None) or os.getenv("OPENAI_API_BASE")  
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.config = config
    
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
    
    def _extract_usage_info(self, response, verbose: bool = False) -> Dict[str, Any]:
        """
        Extract token usage information from API response
        Handles both Chat Completions API and Responses API formats
        
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
            
            # if verbose:
            #     print("\n" + "="*60)
            #     print("DEBUG: Raw Usage Data")
            #     print("="*60)
            #     print(json.dumps(usage_dict, indent=2, default=str))
            #     print("="*60 + "\n")
            
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
    
    def generate_response(
        self, 
        board_representation: str, 
        board_size: int,
        verbose: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            user_prompt = self.format_prompt(board_representation, board_size)
            
            messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - OpenAI API Request")
                print("="*80)
                print(f"Model: {self.config.model_name}")
                print(f"Board Size: {board_size}x{board_size}")
                print(f"\nSystem Prompt:\n{SYSTEM_PROMPT}")
                print(f"\nUser Prompt:\n{user_prompt}")
                print("="*80)
            
            response = None
            if self._is_o_series_model():
                if verbose:
                    print("Using Responses API for o-series model")
                
                kwargs = {
                    "model": self.config.model_name,
                    "input": messages,
                    "max_output_tokens": self.config.max_tokens,
                }
                # reasoning effort for o-series
                kwargs["reasoning"] = {"effort": self.config.reasoning_effort}

                response = self.client.responses.create(**kwargs)
                result = self._extract_responses_text(response)
            
            else:
                if verbose:
                    print("Using Chat Completions API")
                
                if self.config.model_name.startswith("gpt-5"):
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        timeout=self.config.timeout,
                        max_completion_tokens=self.config.max_tokens
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=messages,
                        temperature=self.config.temperature,
                        max_completion_tokens=self.config.max_tokens,
                        timeout=self.config.timeout
                    )
                
                result = response.choices[0].message.content.strip()
            
            # Extract usage information with verbose flag
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
            print(f"Error with OpenAI API: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return None, None

class DeepSeekInterface(ModelInterface):
    """Interface for DeepSeek models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    def _extract_usage_info(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
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
            usage = response_json.get("usage", {})
            usage_info["raw_usage"] = usage
            usage_info["total_tokens"] = usage.get("total_tokens", 0)
            usage_info["prompt_tokens"] = usage.get("prompt_tokens", 0)
            usage_info["completion_tokens"] = usage.get("completion_tokens", 0)
            
            # DeepSeek may also have reasoning tokens
            completion_details = usage.get("completion_tokens_details", {})
            reasoning_tokens = completion_details.get("reasoning_tokens", 0)
            usage_info["reasoning_tokens"] = reasoning_tokens
            usage_info["output_tokens"] = usage_info["completion_tokens"] - reasoning_tokens
        except Exception as e:
            print(f"Warning: Could not extract usage info: {e}")
        
        return usage_info
    
    def generate_response(
        self, 
        board_representation: str, 
        board_size: int,
        verbose: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            user_prompt = self.format_prompt(board_representation, board_size)
            
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - DeepSeek API Request")
                print("="*80)
                print(f"Model: {self.config.model_name}")
                print(f"Board Size: {board_size}x{board_size}")
                print(f"\nSystem Prompt:\n{SYSTEM_PROMPT}")
                print(f"\nUser Prompt:\n{user_prompt}")
                print("="*80)
            
            payload = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            response = requests.post(
                f"{self.config.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result_json = response.json()
            generated_text = result_json["choices"][0]["message"]["content"].strip()
            
            # Extract usage information
            usage_info = self._extract_usage_info(result_json)
            
            if verbose:
                print(f"\nModel Response:\n{generated_text}")
                print(f"\nToken Usage:")
                print(f"  Total: {usage_info['total_tokens']}")
                print(f"  Prompt: {usage_info['prompt_tokens']}")
                print(f"  Completion: {usage_info['completion_tokens']}")
                print(f"  Reasoning: {usage_info['reasoning_tokens']}")
                print(f"  Output: {usage_info['output_tokens']}")
                print("="*80 + "\n")
            
            return generated_text, usage_info
            
        except Exception as e:
            print(f"Error with DeepSeek API: {e}")
            return None, None

class OllamaInterface(ModelInterface):
    """Interface for Ollama models (local) - DEPRECATED, use VLLMInterface instead"""
    
    def generate_response(
        self, 
        board_representation: str, 
        board_size: int,
        verbose: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            user_prompt = self.format_prompt(board_representation, board_size)
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
            
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - Ollama API Request")
                print("="*80)
                print(f"Model: {self.config.model_name}")
                print(f"Board Size: {board_size}x{board_size}")
                print(f"\nFull Prompt:\n{full_prompt}")
                print("="*80)
            
            payload = {
                "model": self.config.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
            
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["response"].strip()
            
            # Ollama doesn't provide detailed token usage, create placeholder
            usage_info = {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "output_tokens": 0,
                "raw_usage": {}
            }
            
            if verbose:
                print(f"\nModel Response:\n{generated_text}")
                print("="*80 + "\n")
            
            return generated_text, usage_info
            
        except Exception as e:
            print(f"Error with Ollama API: {e}")
            return None, None

class TransformersInterface(ModelInterface):
    """Interface for Hugging Face Transformers models - DEPRECATED, use VLLMInterface instead"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            raise ImportError("transformers and torch are required for TransformersInterface")
    
    def generate_response(
        self, 
        board_representation: str, 
        board_size: int,
        verbose: bool = False
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            user_prompt = self.format_prompt(board_representation, board_size)
            
            if verbose:
                print("\n" + "="*80)
                print("VERBOSE MODE - Transformers Model Request")
                print("="*80)
                print(f"Model: {self.config.model_name}")
                print(f"Board Size: {board_size}x{board_size}")
                print(f"\nSystem Prompt:\n{SYSTEM_PROMPT}")
                print(f"\nUser Prompt:\n{user_prompt}")
                print("="*80)
            
            # Format as chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            prompt_tokens = inputs['input_ids'].shape[1]
            
            if hasattr(self.model, 'cuda') and self.model.device.type == 'cuda':
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):], 
                skip_special_tokens=True
            )
            
            generated_text = response.strip()
            
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
                print(f"\nModel Response:\n{generated_text}")
                print(f"\nToken Usage:")
                print(f"  Total: {usage_info['total_tokens']}")
                print(f"  Prompt: {usage_info['prompt_tokens']}")
                print(f"  Completion: {usage_info['completion_tokens']}")
                print("="*80 + "\n")
            
            return generated_text, usage_info
            
        except Exception as e:
            print(f"Error with Transformers model: {e}")
            return None, None

def create_model_interface(config: ModelConfig) -> ModelInterface:
    """Factory function to create appropriate model interface"""
    if config.model_type == "openai":
        return SharedAPIModelInterface(config, provider="openai")
    elif config.model_type == "deepseek":
        return SharedAPIModelInterface(config, provider="deepseek")
    elif config.model_type == "vllm":
        return SharedVLLMInterface(config)
    elif config.model_type == "vllm-server":
        from .vllm_interface import VLLMServerInterface
        return VLLMServerInterface(config)
    elif config.model_type == "ollama":
        return OllamaInterface(config)
    elif config.model_type == "transformers":
        return TransformersInterface(config)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")
