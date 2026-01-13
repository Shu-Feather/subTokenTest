MODEL_CONFIGS = {
    # OpenAI GPT-4 Models
    "gpt-4": {
        "provider": "openai",
        "model_name": "gpt-4",
        "max_tokens": 6000,
        "temperature": 0.0,
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "model_name": "gpt-4-turbo-preview",
        "max_tokens": 6000,
        "temperature": 0.0,
    },
    "gpt-4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "max_tokens": 6000,
        "temperature": 0.0,
    },
    
    # OpenAI GPT-5 Models
    "gpt-5": {
        "provider": "openai",
        "model_name": "gpt-5",
        "max_tokens": 40000,
        "temperature": 0.0,
    },
    
    # OpenAI GPT-3.5 Models
    "gpt-3.5-turbo": {
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
        "max_tokens": 6000,
        "temperature": 0.0,
    },
    
    # OpenAI o-series Models (use Responses API)
    "o1": {
        "provider": "openai",
        "model_name": "o1",
        "max_tokens": 6000,
        "reasoning_effort": "medium",  # 'low' | 'medium' | 'high'
    },
    "o1-mini": {
        "provider": "openai",
        "model_name": "o1-mini",
        "max_tokens": 6000,
        "reasoning_effort": "medium",
    },
    "o3-mini": {
        "provider": "openai",
        "model_name": "o3-mini",
        "max_tokens": 6000,
        "reasoning_effort": "medium",
    },
    "o4": {
        "provider": "openai",
        "model_name": "o4",
        "max_tokens": 40000,
        "reasoning_effort": "medium",
    },
    "o4-mini": {
        "provider": "openai",
        "model_name": "o4-mini",
        "max_tokens": 40000,
        "reasoning_effort": "high",
    },
    
    # DeepSeek Models
    "deepseek-chat": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "api_base": "https://api.deepseek.com",
        "max_tokens": 6000,
        "temperature": 0.0,
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "deepseek-v3": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "api_base": "https://api.deepseek.com",
        "max_tokens": 6000,
        "temperature": 0.0,
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "deepseek-reasoner": {
        "provider": "deepseek",
        "model_name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com",
        "max_tokens": 40000,
        "temperature": 0.0,
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "deepseek-r1": {
        "provider": "deepseek",
        "model_name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com",
        "max_tokens": 40000,
        "temperature": 0.0,
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}

# Models that support local deployment via vLLM
VLLM_SUPPORTED_MODELS = [
    "/path/to/local/llama-3-8b-instruct",
    "/path/to/local/r1-distill-qwen-32b",
    "/path/to/local/qwen-7b",
    "/path/to/local/r1-distill-qwen-7b",
]
