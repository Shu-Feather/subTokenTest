import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from typing import Optional, Dict, Any
from typewriter.models.base_model import BaseModel
from configs.typewriter.model_config import ModelConfig
import logging

logger = logging.getLogger(__name__)

class LocalModel(BaseModel):
    """Local model implementation (no usage tracking)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if not config.local_path or not os.path.exists(config.local_path):
            raise ValueError(f"Local model path does not exist: {config.local_path}")
        
        self._load_model()
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration"""
        if self.config.load_in_4bit:
            logger.info("Loading model with 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.config.load_in_8bit:
            logger.info("Loading model with 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        return None
    
    def _load_model(self):
        """Load the local model and tokenizer"""
        try:
            logger.info(f"Loading model from: {self.config.local_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.local_path,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            quantization_config = self._get_quantization_config()
            
            if quantization_config is not None:
                torch_dtype = None
                device_map = self.config.device_map
            else:
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                device_map = self.config.device_map if self.device == "cuda" else None
            
            logger.info(f"Loading model with dtype: {torch_dtype}, device_map: {device_map}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.local_path,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map=device_map
            )
            
        except Exception as e:
            logger.error(f"Failed to load local model: {str(e)}")
            raise Exception(f"Failed to load local model: {str(e)}")
    
    def _format_prompt_llama3(self, prompt: str) -> str:
        """Format prompt for Llama 3 models"""
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def _format_prompt_qwen(self, prompt: str) -> str:
        """Format prompt for Qwen models"""
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt based on model type"""
        model_id_lower = self.config.model_id.lower()
        
        if "llama-3" in model_id_lower or "llama3" in model_id_lower:
            return self._format_prompt_llama3(prompt)
        elif "qwen" in model_id_lower:
            return self._format_prompt_qwen(prompt)
        else:
            return prompt
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response using local model
        Note: Local models don't return usage info
        """
        try:
            max_tokens = kwargs.get("max_tokens", self.config.max_tokens)
            temperature = kwargs.get("temperature", self.config.temperature)
            
            formatted_prompt = self._format_prompt(prompt)
            
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=2048
            )
            
            if self.device == "cuda" and not self.config.load_in_4bit and not self.config.load_in_8bit:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=0.9 if temperature > 0 else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            if "llama-3" in self.config.model_id.lower() or "llama3" in self.config.model_id.lower():
                if "<|start_header_id|>assistant<|end_header_id|>" in full_output:
                    response = full_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                else:
                    response = full_output[len(formatted_prompt):]
            elif "qwen" in self.config.model_id.lower():
                if "<|im_start|>assistant" in full_output:
                    response = full_output.split("<|im_start|>assistant")[-1]
                    if response.startswith("\n"):
                        response = response[1:]
                else:
                    response = full_output[len(formatted_prompt):]
            else:
                response = full_output[len(formatted_prompt):]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Local model generation error: {str(e)}")
            raise Exception(f"Local model generation error: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if local model is available"""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            "local_path": self.config.local_path,
            "device": self.device,
            "quantization": "4-bit" if self.config.load_in_4bit else (
                "8-bit" if self.config.load_in_8bit else "none"
            )
        })
        return info
