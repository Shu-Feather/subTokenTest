"""
HuggingFace model implementation for the benchmark.
"""

import os
from typing import Dict, Any, Optional
import logging
from .base_model import BaseModel

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers package not available. Install with: pip install transformers torch")

logger = logging.getLogger(__name__)


class HuggingFaceModel(BaseModel):
    """HuggingFace transformer model implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HuggingFace model.
        
        Args:
            config (Dict[str, Any]): Model configuration
        """
        super().__init__(config)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package is required. Install with: pip install transformers torch")
        
        self.model_name = config.get('model_name', 'microsoft/DialoGPT-medium')
        self.device = config.get('device', 'auto')
        self.max_new_tokens = config.get('max_new_tokens', 100)
        self.do_sample = config.get('do_sample', False)
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Initialize model lazily
        self._initialize_model()
        
        logger.info(f"Initialized HuggingFace model: {self.model_name}")
    
    def _initialize_model(self):
        """Initialize the HuggingFace model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device if self.device != 'auto' else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=self.device if self.device != 'auto' else None
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            raise
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using HuggingFace model.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional parameters
            
        Returns:
            str: Generated response
        """
        if not self.pipeline:
            raise RuntimeError("Model not properly initialized")
        
        try:
            # Prepare generation parameters
            generation_params = {
                'max_new_tokens': kwargs.get('max_new_tokens', self.max_new_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
                'do_sample': kwargs.get('do_sample', self.do_sample),
                'pad_token_id': self.tokenizer.eos_token_id,
                'return_full_text': False,
                'clean_up_tokenization_spaces': True
            }
            
            # Generate response
            outputs = self.pipeline(prompt, **generation_params)
            
            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0]['generated_text']
                return generated_text.strip()
            else:
                logger.warning("No output generated from model")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating response with HuggingFace model: {e}")
            raise
    
    def is_available(self) -> bool:
        """
        Check if HuggingFace model is available.
        
        Returns:
            bool: True if available, False otherwise
        """
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        # Check if model is loaded
        return self.model is not None and self.tokenizer is not None
    
    async def test_connection(self) -> bool:
        """
        Test the model by generating a simple response.
        
        Returns:
            bool: True if test successful, False otherwise
        """
        try:
            test_prompt = "Hello"
            response = await self.generate_response(test_prompt)
            return bool(response)
        except Exception as e:
            logger.error(f"HuggingFace model test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            'model_name': self.model_name,
            'device': self.device,
            'available': self.is_available()
        }
        
        if self.model:
            info['num_parameters'] = self.model.num_parameters()
            info['device_map'] = getattr(self.model, 'hf_device_map', None)
        
        return info
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("HuggingFace model resources cleaned up")