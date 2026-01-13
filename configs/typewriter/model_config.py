import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for different model types"""
    name: str
    model_type: str  # 'openai', 'deepseek', 'huggingface', 'local'
    model_id: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 32768
    temperature: float = 0.6
    top_p: float = 0.95
    timeout: int = 300
    batch_size: int = 4
    additional_params: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None  # e.g., 'low', 'medium', 'high'
    
    # For local models
    local_path: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: str = "auto"
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 2
    enforce_eager: bool = True
    restricted_reasoning: bool = False

# Predefined model configurations
MODEL_CONFIGS = {
    # ==================== OpenAI Models ====================
    "gpt-5": ModelConfig(
        name="gpt-5",
        model_type="openai",
        model_id="gpt-5",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=6000,
        temperature=0.0
    ),
    "o4-mini": ModelConfig(
        name="o4-mini",
        model_type="openai",
        model_id="o4-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=60000,
        temperature=0.0,
        reasoning_effort="low" # Custom parameter for reasoning models
    ),
    "gpt-4": ModelConfig(
        name="gpt-4",
        model_type="openai",
        model_id="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=6000,
        temperature=0.0
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        model_type="openai",
        model_id="gpt-4-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=512,
        temperature=0.0
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        model_type="openai",
        model_id="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=512,
        temperature=0.0
    ),
    
    # ==================== DeepSeek Models ====================
    "deepseek-v3": ModelConfig(
        name="deepseek-chat",
        model_type="deepseek",
        model_id="deepseek-chat",   # DeepSeek V3 chatting model
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        max_tokens=6000,
        temperature=0.0
    ),
    "deepseek-r1": ModelConfig(
        name="deepseek-reasoner",
        model_type="deepseek",
        model_id="deepseek-reasoner",  # DeepSeek R1 reasoning model
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        max_tokens=60000,
        temperature=0.0,
        additional_params={
            "enable_reasoning": True  # Enable reasoning output
        }
    ),
    
    # ==================== Llama 3 Models (Local) ====================
    "llama3-8b": ModelConfig(
        name="llama3-8b",
        model_type="local",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        local_path=os.getenv("LLAMA3_8B_PATH", "/path/to/llama3-8b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=False,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True
    ),
    "llama3-70b": ModelConfig(
        name="llama3-70b",
        model_type="local",
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        local_path=os.getenv("LLAMA3_70B_PATH", "/path/to/llama3-70b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=True,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True
    ),
    "llama3-8b-4bit": ModelConfig(
        name="llama3-8b-4bit",
        model_type="local",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        local_path=os.getenv("LLAMA3_8B_PATH", "/path/to/llama3-8b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=False,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    ),
    "llama3-70b-4bit": ModelConfig(
        name="llama3-70b-4bit",
        model_type="local",
        model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        local_path=os.getenv("LLAMA3_70B_PATH", "/path/to/llama3-70b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=False,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True
    ),
    
    # ==================== Qwen Models (Local) ====================
    "qwen-7b": ModelConfig(
        name="qwen-7b",
        model_type="local",
        model_id="Qwen/Qwen-7B-Chat",
        local_path=os.getenv("QWEN_7B_PATH", "/path/to/qwen-7b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=False,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True
    ),
    "qwen-14b": ModelConfig(
        name="qwen-14b",
        model_type="local",
        model_id="Qwen/Qwen-14B-Chat",
        local_path=os.getenv("QWEN_14B_PATH", "/path/to/qwen-14b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=False,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True
    ),
    "qwen-72b": ModelConfig(
        name="qwen-72b",
        model_type="local",
        model_id="Qwen/Qwen-72B-Chat",
        local_path=os.getenv("QWEN_72B_PATH", "/path/to/qwen-72b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=True,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True
    ),
    "qwen1.5-7b": ModelConfig(
        name="qwen1.5-7b",
        model_type="local",
        model_id="Qwen/Qwen1.5-7B-Chat",
        local_path=os.getenv("QWEN1_5_7B_PATH", "/path/to/qwen1.5-7b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=False,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True
    ),
    "qwen2-7b": ModelConfig(
        name="qwen2-7b",
        model_type="local",
        model_id="Qwen/Qwen2-7B-Instruct",
        local_path=os.getenv("QWEN2_7B_PATH", "/path/to/qwen-7b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=False,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True
    ),
    "qwen2.5-32b": ModelConfig(
        name="qwen2.5-32b",
        model_type="local",
        model_id="Qwen/Qwen2.5-32B-Instruct",
        local_path="/path/to/local/qwen-32b",
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=True,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        enforce_eager=True
    ),
    "deepseek-r1-distill-qwen-32b": ModelConfig(
        name="deepseek-r1-distill-qwen-32b",
        model_type="local",
        model_id="DeepSeek-R1-Distill-Qwen-32B",
        local_path="/path/to/r1-distill-qwen-32b",
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=True,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        enforce_eager=True
    ),
    "qwen2.5-7b": ModelConfig(
        name="qwen2.5-7b",
        model_type="local",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        local_path="/path/to/qwen-7b",
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=True,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        enforce_eager=True
    ),
    "deepseek-r1-distill-qwen-7b": ModelConfig(
        name="deepseek-r1-distill-qwen-7b",
        model_type="local",
        model_id="DeepSeek-R1-Distill-Qwen-7B",
        local_path="/path/to/r1-distill-qwen-7b",
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=True,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=2,
        enforce_eager=True
    ),
    "qwen2-72b": ModelConfig(
        name="qwen2-72b",
        model_type="local",
        model_id="Qwen/Qwen2-72B-Instruct",
        local_path=os.getenv("QWEN2_72B_PATH", "/path/to/qwen2-72b"),
        max_tokens=32768,
        temperature=0.6,
        load_in_8bit=True,
        load_in_4bit=False,
        device_map="auto",
        trust_remote_code=True
    ),
    
    # ==================== HuggingFace Models (Download from Hub) ====================
    "llama2-7b": ModelConfig(
        name="llama2-7b",
        model_type="huggingface",
        model_id="meta-llama/Llama-2-7b-chat-hf",
        max_tokens=512,
        temperature=0.0,
        trust_remote_code=True
    ),
    "mistral-7b": ModelConfig(
        name="mistral-7b",
        model_type="huggingface",
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        max_tokens=512,
        temperature=0.0,
        trust_remote_code=True
    ),
}

def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found in configurations. "
                        f"Available models: {list_available_models()}")
    return MODEL_CONFIGS[model_name]

def list_available_models():
    """List all available model configurations"""
    return list(MODEL_CONFIGS.keys())

def add_custom_model(name: str, config: ModelConfig):
    """Add a custom model configuration at runtime"""
    MODEL_CONFIGS[name] = config
    print(f"Added custom model: {name}")

def list_models_by_type(model_type: str = None):
    """List models filtered by type"""
    if model_type is None:
        return MODEL_CONFIGS
    
    filtered = {name: config for name, config in MODEL_CONFIGS.items() 
                if config.model_type == model_type}
    return filtered

def print_model_info():
    """Print information about all available models"""
    print("\n=== Available Models ===\n")
    
    # Group by type
    types = {}
    for name, config in MODEL_CONFIGS.items():
        if config.model_type not in types:
            types[config.model_type] = []
        types[config.model_type].append((name, config))
    
    for model_type, models in types.items():
        print(f"[{model_type.upper()}]")
        for name, config in models:
            if config.model_type == "local":
                print(f"  - {name}: {config.model_id}")
                print(f"    Local path: {config.local_path}")
                if config.load_in_8bit:
                    print(f"    Quantization: 8-bit")
                elif config.load_in_4bit:
                    print(f"    Quantization: 4-bit")
            elif config.model_type == "deepseek":
                print(f"  - {name}: {config.model_id}")
                if config.additional_params and config.additional_params.get("enable_reasoning"):
                    print(f"    [Reasoning Model]")
            else:
                print(f"  - {name}: {config.model_id}")
        print()
