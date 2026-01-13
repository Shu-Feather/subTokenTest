"""
API-based model implementations (OpenAI, DeepSeek, etc.).
"""

import os
from typing import List, Dict, Tuple, Optional, Any
from openai import OpenAI
from src.models.base_model import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI API model implementation."""
    
    def __init__(self, config: Dict, verbose: bool = False):
        """
        Initialize OpenAI model.
        
        Args:
            config: Model configuration
            verbose: Whether to print verbose output
        """
        super().__init__(config, verbose)
        
        api_key_env = config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = config.get('model_name', 'gpt-4')
        self.temperature = config.get('temperature', 0.0)
        self.max_tokens = config.get('max_tokens', 512)
        self.timeout = config.get('timeout', 300)
        
        # reasoning effort parameter for o-series
        self.reasoning_effort = config.get('reasoning_effort', 'medium')  # 'low' | 'medium' | 'high'
    
        self._log(f"\n{'='*60}")
        self._log(f"Initialized OpenAI model: {self.model_name}")
        self._log(f"temperature={self.temperature}, max_tokens={self.max_tokens}")
        self._log(f"{'='*60}\n")
    
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

    def _extract_usage_info(self, response) -> Dict[str, Any]:
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
        
        Returns:
            Dict with:
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
                self._log("Warning: No usage information found in response")
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
                self._log("Detected Responses API format (o-series)")
                
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
                
                self._log(f"  Input tokens: {input_tokens}")
                self._log(f"  Output tokens: {output_tokens}")
                self._log(f"  Reasoning tokens: {reasoning_tokens}")
                self._log(f"  Visible output tokens: {usage_info['output_tokens']}")
            
            # Handle Chat Completions API format (prompt_tokens, completion_tokens)
            elif "prompt_tokens" in usage_dict or "completion_tokens" in usage_dict:
                self._log("Detected Chat Completions API format")
                
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
                
                self._log(f"  Prompt tokens: {usage_info['prompt_tokens']}")
                self._log(f"  Completion tokens: {usage_info['completion_tokens']}")
                self._log(f"  Reasoning tokens: {reasoning_tokens}")
                self._log(f"  Visible output tokens: {usage_info['output_tokens']}")
            
            else:
                self._log("Warning: Unknown usage format")
                self._log(f"Available keys: {list(usage_dict.keys())}")
        
        except Exception as e:
            self._log(f"Warning: Could not extract usage info: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        return usage_info

    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate response using OpenAI API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (response_text, usage_info)
            where usage_info contains:
                - total_tokens: total tokens used
                - prompt_tokens: tokens in prompt/input
                - completion_tokens: tokens in completion/output
                - reasoning_tokens: tokens used for reasoning (if applicable)
                - output_tokens: visible output tokens
                - raw_usage: original usage object for debugging
        """
        if self.verbose:
            self._log(f"\n{'='*60}")
            self._log("OPENAI INPUT MESSAGES:")
            self._log(f"{'='*60}")
            for msg in messages:
                self._log(f"[{msg['role'].upper()}]")
                self._log(msg['content'])
                self._log("-" * 60)
            self._log(f"{'='*60}\n")

        try:
            response = None
            
            if self._is_o_series_model():
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
            usage_info = self._extract_usage_info(response)
            
            if self.verbose:
                self._log(f"\n{'='*60}")
                self._log("OPENAI OUTPUT:")
                self._log(f"{'='*60}")
                self._log(result)
                self._log(f"\n{'='*60}")
                self._log("TOKEN USAGE:")
                self._log(f"{'='*60}")
                self._log(f"  Total: {usage_info['total_tokens']}")
                self._log(f"  Prompt/Input: {usage_info['prompt_tokens']}")
                self._log(f"  Completion/Output: {usage_info['completion_tokens']}")
                self._log(f"  Reasoning: {usage_info['reasoning_tokens']}")
                self._log(f"  Visible Output: {usage_info['output_tokens']}")
                
                if usage_info['total_tokens'] > 0:
                    thinking_ratio = usage_info['reasoning_tokens'] / usage_info['total_tokens']
                    self._log(f"  Thinking Ratio: {thinking_ratio:.2%}")
                
                self._log(f"{'='*60}\n")
            
            return result, usage_info
            
        except Exception as e:
            self._log(f"Error during OpenAI API call: {e}")
            print(e)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return "", None


class DeepSeekModel(BaseModel):
    """DeepSeek API model implementation."""
    
    def __init__(self, config: Dict, verbose: bool = False):
        """
        Initialize DeepSeek model.
        
        Args:
            config: Model configuration
            verbose: Whether to print verbose output
        """
        super().__init__(config, verbose)
        
        api_key_env = config.get('api_key_env', 'DEEPSEEK_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")
        
        base_url = config.get('base_url', 'https://api.deepseek.com')
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = config.get('model_name', 'deepseek-chat')
        self.temperature = config.get('temperature', 0.0)
        self.max_tokens = config.get('max_tokens', 512)
        self.timeout = config.get('timeout', 300)
        
        self._log(f"\n{'='*60}")
        self._log(f"Initialized DeepSeek model: {self.model_name}")
        self._log(f"temperature={self.temperature}, max_tokens={self.max_tokens}")
        self._log(f"{'='*60}\n")
    
    def _extract_usage_info(self, response) -> Dict[str, Any]:
        """
        Extract token usage information from DeepSeek API response.
        
        Returns:
            Dict with:
                - total_tokens: total tokens used
                - prompt_tokens: tokens in prompt/input
                - completion_tokens: tokens in completion/output
                - reasoning_tokens: tokens used for reasoning (if applicable)
                - output_tokens: visible output tokens
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
                self._log("Warning: No usage information found in response")
                return usage_info
            
            # Convert to dict if it's an object
            if isinstance(usage, dict):
                usage_dict = usage
            elif hasattr(usage, "model_dump"):
                usage_dict = usage.model_dump()
            elif hasattr(usage, "__dict__"):
                usage_dict = usage.__dict__
            else:
                usage_dict = {}
            
            usage_info["raw_usage"] = usage_dict
            
            usage_info["total_tokens"] = usage_dict.get("total_tokens", 0) or 0
            usage_info["prompt_tokens"] = usage_dict.get("prompt_tokens", 0) or 0
            usage_info["completion_tokens"] = usage_dict.get("completion_tokens", 0) or 0
            
            # DeepSeek may also have reasoning tokens
            completion_details = usage_dict.get("completion_tokens_details") or {}
            
            # Convert to dict if it's an object
            if not isinstance(completion_details, dict):
                if hasattr(completion_details, "model_dump"):
                    completion_details = completion_details.model_dump() or {}
                elif hasattr(completion_details, "__dict__"):
                    completion_details = completion_details.__dict__ or {}
                else:
                    completion_details = {}
            
            reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
            audio_tokens = completion_details.get("audio_tokens", 0) or 0
            accepted_prediction_tokens = completion_details.get("accepted_prediction_tokens", 0) or 0
            
            usage_info["reasoning_tokens"] = reasoning_tokens
            
            # Calculate visible output tokens
            visible_output_tokens = (
                usage_info["completion_tokens"]
                - reasoning_tokens
                - audio_tokens
                - accepted_prediction_tokens
            )
            if visible_output_tokens < 0:
                visible_output_tokens = usage_info["completion_tokens"]
            
            usage_info["output_tokens"] = visible_output_tokens
            
            self._log(f"  Prompt tokens: {usage_info['prompt_tokens']}")
            self._log(f"  Completion tokens: {usage_info['completion_tokens']}")
            self._log(f"  Reasoning tokens: {reasoning_tokens}")
            self._log(f"  Visible output tokens: {usage_info['output_tokens']}")
        
        except Exception as e:
            self._log(f"Warning: Could not extract usage info: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        return usage_info

    
    def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate response using DeepSeek API.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (response_text, usage_info)
            where usage_info contains:
                - total_tokens: total tokens used
                - prompt_tokens: tokens in prompt/input
                - completion_tokens: tokens in completion/output
                - reasoning_tokens: tokens used for reasoning (if applicable)
                - output_tokens: visible output tokens
                - raw_usage: original usage object for debugging
        """
        if self.verbose:
            self._log(f"\n{'='*60}")
            self._log("DEEPSEEK INPUT MESSAGES:")
            self._log(f"{'='*60}")
            for msg in messages:
                self._log(f"[{msg['role'].upper()}]")
                self._log(msg['content'])
                self._log("-" * 60)
            self._log(f"{'='*60}\n")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )
            
            result = response.choices[0].message.content.strip()
            
            # Extract usage information
            usage_info = self._extract_usage_info(response)
            
            if self.verbose:
                self._log(f"\n{'='*60}")
                self._log("DEEPSEEK OUTPUT:")
                self._log(f"{'='*60}")
                self._log(result)
                self._log(f"\n{'='*60}")
                self._log("TOKEN USAGE:")
                self._log(f"{'='*60}")
                self._log(f"  Total: {usage_info['total_tokens']}")
                self._log(f"  Prompt: {usage_info['prompt_tokens']}")
                self._log(f"  Completion: {usage_info['completion_tokens']}")
                self._log(f"  Reasoning: {usage_info['reasoning_tokens']}")
                self._log(f"  Visible Output: {usage_info['output_tokens']}")
                
                if usage_info['total_tokens'] > 0:
                    thinking_ratio = usage_info['reasoning_tokens'] / usage_info['total_tokens']
                    self._log(f"  Thinking Ratio: {thinking_ratio:.2%}")
                
                self._log(f"{'='*60}\n")
            
            return result, usage_info
            
        except Exception as e:
            self._log(f"Error during DeepSeek API call: {e}")
            print(e)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return "", None