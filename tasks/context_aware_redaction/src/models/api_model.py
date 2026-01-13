"""
API model interface for closed-source models
"""

from typing import Dict, Any, List, Tuple
import time
from openai import OpenAI
import re

from .base_model import BaseModel


class APIModel(BaseModel):
    """
    Interface for API-based models (OpenAI, DeepSeek, etc.)
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any], 
                 api_key: str, base_url: str = None, verbose: bool = False):
        """
        Initialize API model
        
        Args:
            model_name: Model name (e.g., 'gpt-4', 'deepseek-chat')
            config: Configuration dictionary
            api_key: API key
            base_url: Base URL for API (optional, for custom endpoints)
            verbose: Whether to print verbose output
        """
        super().__init__(model_name, config, verbose)
        
        self.api_config = config.get('models', {}).get('api', {})
        self.timeout = self.api_config.get('timeout', 60)
        self.max_retries = self.api_config.get('max_retries', 3)
        
        # Initialize OpenAI client (compatible with OpenAI-like APIs)
        client_kwargs = {'api_key': api_key}

        if base_url:
            client_kwargs['base_url'] = base_url
        else:
            if 'base_url' in self.api_config:
                client_kwargs['base_url'] = self.api_config.get('base_url')
        
        self.client = OpenAI(**client_kwargs)
        
        if self.verbose:
            print(f"Initialized API model: {model_name}")
            if base_url or 'base_url' in client_kwargs:
                print(f"Base URL: {client_kwargs.get('base_url')}")
    
    def _use_responses_api(self) -> bool:
        """
        Decide whether to use the Responses API (for models like gpt-5, o4-mini, o3).
        """
        name = (self.model_name or "").lower()
        # Common new models that prefer/require the Responses API
        if name.startswith("gpt-5"):
            return True
        if name.startswith("o3") or name.startswith("o4"):
            return True
        # Some vendors mirror OpenAI naming; fallback heuristic:
        # Any "oX" family, e.g., "o1", "o2", "o3.*", "o4.*"
        if re.match(r"^o\d", name):
            return True
        return False
    
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
        
        # Last-resort stringification
        return str(response)
    
    def _extract_usage_info(self, response) -> Dict[str, int]:
        """
        Extract token usage information from API response
        
        Args:
            response: API response object
            
        Returns:
            Dictionary with token usage information
        """
        usage_info = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0,
        }
        
        try:
            usage = getattr(response, 'usage', None)
            
            if usage:
                if self._use_responses_api():
                    # Standard fields
                    usage_info['total_tokens'] = getattr(usage, 'total_tokens', 0)
                    usage_info['input_tokens'] = getattr(usage, 'input_tokens', 0)
                    usage_info['output_tokens'] = getattr(usage, 'output_tokens', 0)
                    
                    # Handle reasoning tokens (for o-series models)
                    try:
                        completion_details = getattr(usage, 'output_tokens_details', None)
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
                else:
                    # Standard fields
                    usage_info['total_tokens'] = getattr(usage, 'total_tokens', 0)
                    usage_info['input_tokens'] = getattr(usage, 'prompt_tokens', 0)
                    usage_info['output_tokens'] = getattr(usage, 'completion_tokens', 0)
                    
                    # Handle reasoning tokens (for o-series models)
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
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not extract usage information: {e}")
        
        return usage_info
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from API model
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        result, _ = self.generate_with_usage(prompt, **kwargs)
        return result
    
    def generate_with_usage(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """
        Generate response from API model and return usage information
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_text, usage_info)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("API Input Prompt:")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")
        
        temperature = self.api_config.get('temperature', 0.7)
        max_tokens = self.api_config.get('max_tokens', 6000)
        reasoning_effort = self.api_config.get('reasoning_effort', 'medium')

        for attempt in range(self.max_retries):
            try:
                if self._use_responses_api():
                    # Use Responses API
                    params: Dict[str, Any] = {
                        "model": self.model_name,
                        "input": prompt,
                        # Responses API uses max_output_tokens
                        "max_output_tokens": max_tokens
                    }
                    # Optional: reasoning settings for o family
                    name = (self.model_name or "").lower()
                    if ("o" in name or name.startswith("o")) and reasoning_effort:
                        params["reasoning"] = {"effort": reasoning_effort}
                    
                    response = self.client.responses.create(
                        **params,
                        timeout=self.timeout
                    )
                    result = self._extract_responses_text(response)
                else:
                    # Use Chat Completions API
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=self.timeout
                    )
                
                    result = response.choices[0].message.content
                
                # Extract usage information
                usage_info = self._extract_usage_info(response)
                
                if self.verbose:
                    print(f"\n{'='*80}")
                    print("API Output:")
                    print(f"{'='*80}")
                    print(result)
                    print(f"{'='*80}")
                    print("\nToken Usage:")
                    print(f"  Input Tokens:     {usage_info['input_tokens']}")
                    print(f"  Output Tokens:    {usage_info['output_tokens']}")
                    print(f"  Reasoning Tokens: {usage_info['reasoning_tokens']}")
                    print(f"  Total Tokens:     {usage_info['total_tokens']}")
                    print(f"{'='*80}\n")
                
                return result, usage_info
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"API call failed after {self.max_retries} attempts: {e}")
                    raise
        
        return "", {'total_tokens': 0, 'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0}
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        responses = []
        
        if self.verbose:
            print(f"\nGenerating {len(prompts)} responses with API model...")
        
        for i, prompt in enumerate(prompts):
            if self.verbose:
                print(f"\nProcessing prompt {i+1}/{len(prompts)}...")
            
            response = self.generate(prompt, **kwargs)
            responses.append(response)
            
            # Small delay to avoid rate limiting
            if i < len(prompts) - 1:
                time.sleep(0.5)
        
        return responses
    
    def batch_generate_with_usage(self, prompts: List[str], **kwargs) -> Tuple[List[str], List[Dict[str, int]]]:
        """
        Generate responses for multiple prompts with usage information
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (list of generated texts, list of usage info)
        """
        responses = []
        usage_infos = []
        
        if self.verbose:
            print(f"\nGenerating {len(prompts)} responses with API model...")
        
        for i, prompt in enumerate(prompts):
            if self.verbose:
                print(f"\nProcessing prompt {i+1}/{len(prompts)}...")
            
            response, usage_info = self.generate_with_usage(prompt, **kwargs)
            responses.append(response)
            usage_infos.append(usage_info)
            
            # Small delay to avoid rate limiting
            if i < len(prompts) - 1:
                time.sleep(0.5)
        
        return responses, usage_infos