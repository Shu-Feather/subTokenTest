import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional, Dict, Any
from typewriter.models.base_model import BaseModel
from configs.typewriter.model_config import ModelConfig

class HuggingFaceModel(BaseModel):
    """HuggingFace model implementation for open-source models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()
        
    def _load_model(self):
        """Load the HuggingFace model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        except Exception as e:
            raise Exception(f"Failed to load HuggingFace model: {str(e)}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using HuggingFace model"""
        try:
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            temperature = kwargs.get("temperature", self.config.temperature)
            
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            raise Exception(f"HuggingFace generation error: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if HuggingFace model is available"""
        return self.model is not None and self.tokenizer is not None
