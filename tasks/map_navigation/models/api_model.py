import os
from typing import Dict, Tuple, List, Any, Optional
import openai
from .base_model import BaseModel


class APIModel(BaseModel):
    """Model interface for API-based models (OpenAI, DeepSeek, etc.)."""
    
    def __init__(
        self, 
        model_name: str,
        api_key: str = None,
        api_base: str = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize API model.
        
        Args:
            model_name: Model name for the API
            api_key: API key (if None, will try to get from environment)
            api_base: API base URL (for custom endpoints)
            verbose: Whether to print verbose output
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)
        
        self.verbose = verbose
        
        # Set API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment
            self.api_key = os.getenv('OPENAI_API_KEY') or os.getenv('DEEPSEEK_API_KEY')
        
        if not self.api_key:
            raise ValueError("API key not found. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY environment variable.")
        
        # Set API base
        self.api_base = api_base
        
        # Initialize OpenAI client
        if self.api_base:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )
        else:
            self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model-specific parameters
        self.temperature = kwargs.get('temperature', 0.0)
        self.max_tokens = kwargs.get('max_tokens', 6000)
        self.timeout = kwargs.get('timeout', 300)
        self.reasoning_effort = kwargs.get('reasoning_effort', 'medium')  # for o-series models
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Initialized API model: {self.model_name}")
            print(f"Temperature: {self.temperature}, Max tokens: {self.max_tokens}")
            if self.api_base:
                print(f"API Base: {self.api_base}")
            print(f"{'='*60}\n")
    
    def _is_o_series_model(self) -> bool:
        """
        Check if the model is an o-series model (o1, o3, o4).
        These models use the Responses API instead of Chat Completions API.
        
        Returns:
            True if model is o-series, False otherwise
        """
        model_lower = self.model_name.lower()
        return (
            model_lower in {"o1", "o3", "o3-mini", "o4", "o4-mini"} or 
            model_lower.startswith(("o1-", "o3-", "o4-"))
        )
    
    def _is_deepseek_model(self) -> bool:
        """
        Check if the model is a DeepSeek model.
        
        Returns:
            True if model is DeepSeek, False otherwise
        """
        return "deepseek" in self.model_name.lower()
    
    def _extract_responses_text(self, response) -> str:
        """
        Extract text output from Responses API response (for o-series models).
        
        Args:
            response: API response object
            
        Returns:
            Extracted text string
        """
        try:
            # Try to get output_text directly
            if hasattr(response, "output_text") and response.output_text:
                return (response.output_text or "").strip()
            
            # Fall back to assembling from output parts
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
        except Exception as e:
            if self.verbose:
                print(f"Warning: Error extracting text from Responses API: {e}")
        
        return ""
    
    def _extract_usage_info(self, response) -> Dict[str, Any]:
        """
        Extract token usage information from API response.
        Handles both Chat Completions API and Responses API formats.
        
        Chat Completions API format (GPT-4, GPT-5, DeepSeek):
            - prompt_tokens
            - completion_tokens
            - total_tokens
            - completion_tokens_details.reasoning_tokens (optional)
        
        Responses API format (o-series):
            - input_tokens
            - output_tokens
            - output_tokens_details.reasoning_tokens (optional)
        
        Args:
            response: API response object
            
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
                if self.verbose:
                    print("Warning: No usage information found in response")
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
            
            # Store raw usage for debugging
            usage_info["raw_usage"] = usage_dict
            
            # Handle Responses API format (o-series: input_tokens, output_tokens)
            if "input_tokens" in usage_dict or "output_tokens" in usage_dict:
                if self.verbose:
                    print("Detected Responses API format (o-series)")
                
                input_tokens = usage_dict.get("input_tokens", 0) or 0
                output_tokens = usage_dict.get("output_tokens", 0) or 0
                
                usage_info["prompt_tokens"] = input_tokens
                usage_info["completion_tokens"] = output_tokens
                usage_info["total_tokens"] = input_tokens + output_tokens
                
                # Extract reasoning tokens from output_tokens_details
                output_details = usage_dict.get("output_tokens_details", {})
                if not isinstance(output_details, dict):
                    if hasattr(output_details, "model_dump"):
                        output_details = output_details.model_dump() or {}
                    elif hasattr(output_details, "__dict__"):
                        output_details = output_details.__dict__ or {}
                    else:
                        output_details = {}
                
                reasoning_tokens = output_details.get("reasoning_tokens", 0) or 0
                audio_tokens = output_details.get("audio_tokens", 0) or 0
                
                usage_info["reasoning_tokens"] = reasoning_tokens
                
                # Calculate visible output tokens
                visible_output = output_tokens - reasoning_tokens - audio_tokens
                usage_info["output_tokens"] = max(0, visible_output)
                
                if self.verbose:
                    print(f"  Input tokens: {input_tokens}")
                    print(f"  Output tokens: {output_tokens}")
                    print(f"  Reasoning tokens: {reasoning_tokens}")
                    print(f"  Visible output tokens: {usage_info['output_tokens']}")
            
            # Handle Chat Completions API format (prompt_tokens, completion_tokens)
            elif "prompt_tokens" in usage_dict or "completion_tokens" in usage_dict:
                if self.verbose:
                    print("Detected Chat Completions API format")
                
                usage_info["total_tokens"] = usage_dict.get("total_tokens", 0) or 0
                usage_info["prompt_tokens"] = usage_dict.get("prompt_tokens", 0) or 0
                usage_info["completion_tokens"] = usage_dict.get("completion_tokens", 0) or 0
                
                # Extract reasoning tokens from completion_tokens_details
                completion_details = usage_dict.get("completion_tokens_details", {})
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
                visible_output = (
                    usage_info["completion_tokens"] 
                    - reasoning_tokens 
                    - audio_tokens 
                    - accepted_prediction_tokens
                )
                usage_info["output_tokens"] = max(0, visible_output)
                
                if self.verbose:
                    print(f"  Prompt tokens: {usage_info['prompt_tokens']}")
                    print(f"  Completion tokens: {usage_info['completion_tokens']}")
                    print(f"  Reasoning tokens: {reasoning_tokens}")
                    print(f"  Visible output tokens: {usage_info['output_tokens']}")
            
            else:
                if self.verbose:
                    print(f"Warning: Unknown usage format. Available keys: {list(usage_dict.keys())}")
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not extract usage info: {e}")
                import traceback
                traceback.print_exc()
        
        return usage_info
    
    def _prepare_messages(
        self, 
        system_prompt: str, 
        user_prompt: str
    ) -> List[Dict[str, str]]:
        """
        Prepare messages for API call.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # For o-series models, combine system and user prompts
        # as they don't support separate system messages via Responses API
        if self._is_o_series_model():
            combined_prompt = system_prompt + "\n\n" + user_prompt if system_prompt else user_prompt
            messages.append({"role": "user", "content": combined_prompt})
        else:
            # Standard format for other models
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response from the model.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, token_usage_dict)
        """
        # For single prompt, use it as user message
        return self.generate_with_system("", prompt, **kwargs)
    
    def generate_with_system(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response with separate system and user prompts.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (response_text, token_usage_dict)
            where token_usage_dict contains:
                - total_tokens: total tokens used
                - prompt_tokens: tokens in prompt/input
                - completion_tokens: tokens in completion/output
                - reasoning_tokens: tokens used for reasoning (if applicable)
                - output_tokens: visible output tokens
                - raw_usage: original usage object for debugging
        """
        # Prepare messages
        messages = self._prepare_messages(system_prompt, user_prompt)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"INPUT MESSAGES ({self.model_name}):")
            print(f"{'='*60}")
            for msg in messages:
                print(f"[{msg['role'].upper()}]")
                print(msg['content'])
                print("-" * 60)
            print(f"{'='*60}\n")
        
        # Override parameters with kwargs
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        timeout = kwargs.get('timeout', self.timeout)
        
        try:
            response = None
            result = ""
            
            # Use Responses API for o-series models
            if self._is_o_series_model():
                if self.verbose:
                    print(f"Using Responses API for {self.model_name}")
                
                api_params = {
                    "model": self.model_name,
                    "input": messages,
                    "max_output_tokens": max_tokens,
                }
                
                # Add reasoning effort for o-series
                api_params["reasoning"] = {"effort": self.reasoning_effort}
                
                response = self.client.responses.create(**api_params)
                result = self._extract_responses_text(response)
            
            # Use Chat Completions API for other models
            else:
                if self.verbose:
                    print(f"Using Chat Completions API for {self.model_name}")
                
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "timeout": timeout,
                }
                
                # GPT-5 uses max_completion_tokens
                if self.model_name.startswith("gpt-5"):
                    api_params["max_completion_tokens"] = max_tokens
                else:
                    api_params["temperature"] = temperature
                    api_params["max_completion_tokens"] = max_tokens
                
                response = self.client.chat.completions.create(**api_params)
                result = response.choices[0].message.content.strip()
            
            # Extract token usage
            usage_info = self._extract_usage_info(response)
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"OUTPUT ({self.model_name}):")
                print(f"{'='*60}")
                print(result)
                print(f"\n{'='*60}")
                print("TOKEN USAGE:")
                print(f"{'='*60}")
                print(f"  Total: {usage_info['total_tokens']}")
                print(f"  Prompt/Input: {usage_info['prompt_tokens']}")
                print(f"  Completion/Output: {usage_info['completion_tokens']}")
                print(f"  Reasoning: {usage_info['reasoning_tokens']}")
                print(f"  Visible Output: {usage_info['output_tokens']}")
                
                if usage_info['total_tokens'] > 0 and usage_info['reasoning_tokens'] > 0:
                    thinking_ratio = usage_info['reasoning_tokens'] / usage_info['total_tokens']
                    print(f"  Thinking Ratio: {thinking_ratio:.2%}")
                
                print(f"{'='*60}\n")
            
            return result, usage_info
            
        except Exception as e:
            error_msg = f"API call failed for {self.model_name}: {str(e)}"
            if self.verbose:
                print(f"Error: {error_msg}")
                import traceback
                traceback.print_exc()
            raise RuntimeError(error_msg)
