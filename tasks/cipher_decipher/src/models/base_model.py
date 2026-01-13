"""
Base model interface for LLM benchmarking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for LLM models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration.
        
        Args:
            config (Dict[str, Any]): Model configuration
        """
        self.config = config
        self.model_name = config.get('model_name', 'unknown')
        self.max_tokens = config.get('max_tokens', 32768)
        self.temperature = config.get('temperature', 0.6)
        self.timeout = config.get('timeout', 30)
        
        # Statistics tracking (now includes token usage)
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_response_time': 0.0,
            'avg_response_time': 0.0,
            'total_tokens': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_reasoning_tokens': 0,
            'avg_tokens_per_query': 0.0
        }
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response from the model.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Tuple[str, Dict[str, Any]]: Generated response and token usage information
                Token usage dict should contain:
                - total_tokens (int): Total tokens used
                - input_tokens (int): Input/prompt tokens
                - output_tokens (int): Output/completion tokens
                - reasoning_tokens (int): Reasoning tokens (for o-series models)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the model is available and properly configured.
        
        Returns:
            bool: True if available, False otherwise
        """
        pass
    
    async def query_with_timeout(self, prompt: str, **kwargs) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Query the model with timeout handling.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Tuple[Optional[str], Dict[str, Any]]: Response and token usage info, or (None, empty_dict) if failed
        """
        start_time = time.time()
        self.stats['total_queries'] += 1
        
        # Default empty usage info
        empty_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'reasoning_tokens': 0
        }
        
        try:
            response, usage_info = await self.generate_response(prompt, **kwargs)
            
            # Update statistics
            response_time = time.time() - start_time
            self.stats['successful_queries'] += 1
            self.stats['total_response_time'] += response_time
            self.stats['avg_response_time'] = (
                self.stats['total_response_time'] / self.stats['successful_queries']
            )
            
            # Update token statistics
            if usage_info:
                self.stats['total_tokens'] += usage_info.get('total_tokens', 0)
                self.stats['total_input_tokens'] += usage_info.get('input_tokens', 0)
                self.stats['total_output_tokens'] += usage_info.get('output_tokens', 0)
                self.stats['total_reasoning_tokens'] += usage_info.get('reasoning_tokens', 0)
                
                if self.stats['successful_queries'] > 0:
                    self.stats['avg_tokens_per_query'] = (
                        self.stats['total_tokens'] / self.stats['successful_queries']
                    )
            
            logger.debug(f"Model {self.model_name} responded in {response_time:.2f}s, "
                        f"used {usage_info.get('total_tokens', 0)} tokens")
            
            return response.strip() if response else None, usage_info or empty_usage
            
        except Exception as e:
            self.stats['failed_queries'] += 1
            logger.error(f"Model {self.model_name} query failed: {e}")
            return None, empty_usage
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get model performance statistics including token usage.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        success_rate = (
            self.stats['successful_queries'] / max(self.stats['total_queries'], 1)
        ) * 100
        
        return {
            **self.stats,
            'model_name': self.model_name,
            'success_rate': success_rate,
            'token_statistics': {
                'total_tokens_used': self.stats['total_tokens'],
                'total_input_tokens': self.stats['total_input_tokens'],
                'total_output_tokens': self.stats['total_output_tokens'],
                'total_reasoning_tokens': self.stats['total_reasoning_tokens'],
                'avg_tokens_per_query': self.stats['avg_tokens_per_query']
            }
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_response_time': 0.0,
            'avg_response_time': 0.0,
            'total_tokens': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_reasoning_tokens': 0,
            'avg_tokens_per_query': 0.0
        }
    
    def validate_config(self) -> bool:
        """
        Validate model configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        required_fields = ['model_name']
        for field in required_fields:
            if field not in self.config:
                logger.error(f"Missing required config field: {field}")
                return False
        return True
    
    def get_token_usage_summary(self) -> Dict[str, Any]:
        """
        Get a summary of token usage statistics.
        
        Returns:
            Dict[str, Any]: Token usage summary
        """
        return {
            'total_tokens': self.stats['total_tokens'],
            'input_tokens': self.stats['total_input_tokens'],
            'output_tokens': self.stats['total_output_tokens'],
            'reasoning_tokens': self.stats['total_reasoning_tokens'],
            'avg_tokens_per_query': self.stats['avg_tokens_per_query'],
            'total_queries': self.stats['successful_queries']
        }
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}({self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', config={self.config})"
